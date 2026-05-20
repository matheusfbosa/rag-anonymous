import json
import logging
import os
import urllib.request
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

PROGRESS_EVERY = 10

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_CACHE_DIR = _PROJECT_ROOT / "data" / "input" / "text-anonymization-benchmark"
_TAB_BASE_URL = (
    "https://raw.githubusercontent.com/NorskRegnesentral/"
    "text-anonymization-benchmark/refs/heads/master"
)


def ingest_offline(
    corpus,
    anonymizer,
    collection_name="offline",
    persist_dir=None,
):
    return _ingest(corpus, collection_name, persist_dir, anonymizer=anonymizer)


def ingest_ondemand(corpus, collection_name="ondemand", persist_dir=None):
    return _ingest(corpus, collection_name, persist_dir, anonymizer=None)


def load_corpus(split: str) -> list[dict]:
    path = _corpus_cache_path(split)
    if not path.exists():
        _download_corpus(split)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ingest(corpus, collection_name, persist_dir, anonymizer):
    persist_dir = persist_dir or _chromadb_persist_dir()
    label = "anonymized" if anonymizer is not None else "raw"

    logger.info(
        "Ingesting documents: label=%s corpus_items=%d",
        label,
        len(corpus),
    )
    documents, ids = _build_documents(corpus, anonymizer=anonymizer)

    embeddings = _embeddings()
    _wipe_existing_collection(collection_name, persist_dir, embeddings)

    logger.info(
        "Ingesting chunks: chunk_count=%d persist_dir=%s",
        len(documents),
        persist_dir,
    )
    vectordb = Chroma.from_documents(
        documents=documents,
        ids=ids,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )

    logger.info(
        "Ingested: chunks=%d",
        vectordb._collection.count(),
    )
    return vectordb


def _wipe_existing_collection(collection_name: str, persist_dir: str, embeddings) -> None:
    existing = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    try:
        prior = existing._collection.count()
    except Exception as exc:
        logger.warning(
            "Could not read prior collection size: name=%s error=%s",
            collection_name,
            exc,
        )
        return
    if prior:
        logger.info(
            "Ingesting drop_collection: name=%s prior_chunks=%d reason=reingest",
            collection_name,
            prior,
        )
        existing.delete_collection()


def _build_documents(corpus, anonymizer=None):
    strategy = "offline" if anonymizer is not None else "ondemand"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("RAG_ANON_CHUNK_SIZE", "200")),
        chunk_overlap=int(os.getenv("RAG_ANON_CHUNK_OVERLAP", "0")),
        length_function=len,
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


def _download_corpus(split: str) -> None:
    url = f"{_TAB_BASE_URL}/echr_{split}.json"
    dest = _corpus_cache_path(split)
    _DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Ingesting download: url=%s dest=%s", url, dest)
    urllib.request.urlretrieve(url, dest)


def _corpus_cache_path(split: str) -> Path:
    return _DATA_CACHE_DIR / f"echr_{split}.json"


def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=os.getenv("RAG_ANON_EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("RAG_ANON_OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def _chromadb_persist_dir() -> str:
    return str(_PROJECT_ROOT / os.getenv("RAG_ANON_CHROMADB_PERSIST_DIR", "./chromadb"))
