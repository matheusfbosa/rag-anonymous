import json
import logging
import urllib.request
from pathlib import Path

from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_anonymous.config import (
    DEFAULT_CORPUS_CACHE_SUBDIR,
    PROJECT_ROOT,
    Settings,
)

logger = logging.getLogger(__name__)

PROGRESS_EVERY = 10

_DATA_CACHE_DIR = PROJECT_ROOT / DEFAULT_CORPUS_CACHE_SUBDIR


def ingest_offline(corpus, anonymizer, collection_name="offline", dataset=None):
    return _ingest(corpus, collection_name, anonymizer=anonymizer, dataset=dataset)


def ingest_ondemand(corpus, collection_name="ondemand", dataset=None):
    return _ingest(corpus, collection_name, anonymizer=None, dataset=dataset)


def load_corpus(dataset: str) -> list[dict]:
    corpus = Settings.load().corpus
    if not corpus:
        raise SystemExit(
            "RAG_ANON_CORPUS env var is required "
            "(set it in .env, e.g. a URL or local path with optional {dataset})."
        )
    corpus = corpus.replace("{dataset}", dataset)

    if corpus.startswith(("http://", "https://")):
        path = _corpus_cache_path(dataset)
        if not path.exists():
            _download_corpus(corpus, path)
    else:
        path = Path(corpus)
        if not path.exists():
            raise FileNotFoundError(
                f"RAG_ANON_CORPUS points to a missing file: {path}"
            )

    logger.info("Loading corpus: path=%s", path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_documents(corpus, anonymizer=None, dataset=None):
    strategy = "offline" if anonymizer is not None else "ondemand"
    s = Settings.load()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        is_separator_regex=False,
    )

    documents = []
    ids = []
    chunk_count = 0

    for idx, item in enumerate(corpus):
        raw_text = item["text"]
        text_to_chunk = anonymizer.anonymize(raw_text) if anonymizer else raw_text
        chunks = splitter.split_text(text_to_chunk)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{item['doc_id']}_chunk_{chunk_idx}"
            metadata = {
                "doc_id": item["doc_id"],
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "strategy": strategy,
                "original_length": len(raw_text),
            }
            if dataset is not None:
                metadata["dataset"] = dataset
            if anonymizer is not None:
                metadata["anonymized_length"] = len(text_to_chunk)
            documents.append(Document(page_content=chunk, metadata=metadata))
            ids.append(chunk_id)
            chunk_count += 1

        if (idx + 1) % PROGRESS_EVERY == 0:
            logger.info(
                "  Processing documents: done=%d total=%d chunks=%d",
                idx + 1,
                len(corpus),
                chunk_count,
            )

    logger.info(
        "Ingested: documents=%d chunks=%d",
        len(corpus),
        chunk_count,
    )
    return documents, ids


def _corpus_cache_path(dataset: str) -> Path:
    corpus = Settings.load().corpus.replace("{dataset}", dataset)
    filename = corpus.rsplit("/", 1)[-1]
    return _DATA_CACHE_DIR / filename


def _download_corpus(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Ingesting download: url=%s dest=%s", url, dest)
    urllib.request.urlretrieve(url, dest)


def _embeddings() -> OllamaEmbeddings:
    s = Settings.load()
    return OllamaEmbeddings(
        model=s.embedding_model,
        base_url=s.ollama_base_url,
        num_ctx=s.embedding_num_ctx,
    )


def _ingest(corpus, collection_name, anonymizer, dataset=None):
    s = Settings.load()
    index = s.es_index(collection_name, dataset)
    label = "anonymized" if anonymizer is not None else "raw"

    logger.info(
        "Ingesting documents: label=%s corpus_items=%d",
        label,
        len(corpus),
    )
    documents, ids = _build_documents(corpus, anonymizer=anonymizer, dataset=dataset)

    embeddings = _embeddings()
    _wipe_existing_index(index, s.es_url)

    logger.info(
        "Ingesting chunks: chunk_count=%d es_url=%s index=%s",
        len(documents),
        s.es_url,
        index,
    )
    vectordb = ElasticsearchStore(
        index_name=index,
        embedding=embeddings,
        es_url=s.es_url,
    )
    vectordb.add_documents(documents, ids=ids)

    vectordb.client.indices.refresh(index=index)
    logger.info(
        "Ingested: chunks=%d",
        vectordb.client.count(index=index)["count"],
    )
    return vectordb


def _wipe_existing_index(index: str, es_url: str) -> None:
    from elasticsearch import Elasticsearch

    client = Elasticsearch(es_url)
    try:
        exists = client.indices.exists(index=index)
    except Exception as exc:
        logger.warning(
            "Could not check prior index: index=%s error=%s",
            index,
            exc,
        )
        return
    if exists:
        prior = client.count(index=index)["count"]
        logger.info(
            "Ingesting drop_index: index=%s prior_chunks=%d reason=reingest",
            index,
            prior,
        )
        client.indices.delete(index=index)
