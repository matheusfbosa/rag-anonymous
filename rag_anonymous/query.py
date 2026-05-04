import logging
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def query_rag(chain, vectordb, question, k_docs=None):
    if k_docs is None:
        k_docs = int(os.getenv("RAG_ANON_RETRIEVAL_K_DOCS", "5"))
    retriever = vectordb.as_retriever(search_kwargs={"k": k_docs})
    chunks = retriever.invoke(question)
    context_text = "\n\n---\n\n".join(doc.page_content for doc in chunks)

    response = chain.invoke({"context": context_text, "question": question})

    return {
        "question": question,
        "response": response,
        "retrieved_chunks": [
            {
                "content": doc.page_content,
                "doc_id": doc.metadata.get("doc_id"),
                "chunk_id": doc.metadata.get("chunk_id"),
            }
            for doc in chunks
        ],
        "num_unique_docs": len(set(c.metadata["doc_id"] for c in chunks)),
    }


def create_chain(llm=None):
    if llm is None:
        llm = build_llm()
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    return prompt | llm | StrOutputParser()


def load_vectordb(collection_name="offline", persist_dir=None):
    persist_dir = persist_dir or _chromadb_persist_dir()
    embeddings = OllamaEmbeddings(
        model=os.getenv("RAG_ANON_EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("RAG_ANON_OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    _log_collection_health(vectordb, collection_name)
    return vectordb


def build_llm(model_name: str | None = None) -> ChatOllama:
    """Build ChatOllama model."""
    return ChatOllama(
        model=model_name or os.getenv("RAG_ANON_LLM_MODEL", "qwen3:0.6b"),
        base_url=os.getenv("RAG_ANON_OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=float(os.getenv("RAG_ANON_LLM_TEMPERATURE", "0.0")),
        reasoning=reasoning_flag(),
    )


def reasoning_flag() -> bool:
    """Parse RAG_ANON_LLM_REASONING into a bool. Defaults to False to keep
    thinking disabled (faster + cleaner output) for non-reasoning workloads."""
    return os.getenv("RAG_ANON_LLM_REASONING", "false").strip().lower() in (
        "1", "true", "yes",
    )


def _log_collection_health(vectordb, collection_name):
    """Surface common ingestion problems (empty / duplicated chunks) at load time.

    Re-ingesting without first dropping the collection would silently append a
    full second copy of every chunk, which then makes top-K retrieval return the
    same chunk multiple times and inflates utility metrics. Comparing
    ``count`` against ``unique chunk_id`` count catches this drift cheaply.
    """
    try:
        count = vectordb._collection.count()
    except Exception as exc:
        logger.warning("Could not read count for collection '%s': %s", collection_name, exc)
        return

    if count == 0:
        logger.warning(
            "Collection '%s' is empty. Did you run `rag-anon ingest` "
            "with RAG_ANON_ANONYMIZER_STRATEGY=%s ?",
            collection_name, collection_name,
        )
        return

    try:
        result = vectordb._collection.get(include=["metadatas"])
        chunk_ids = [
            (m or {}).get("chunk_id") for m in (result.get("metadatas") or [])
        ]
        unique = len({c for c in chunk_ids if c})
    except Exception as exc:
        logger.warning("Could not inspect metadata for '%s': %s", collection_name, exc)
        logger.info("Collection '%s' loaded with %d chunks", collection_name, count)
        return

    if unique and count != unique:
        logger.warning(
            "Collection '%s' has %d chunks but only %d unique chunk_id values "
            "(%.1fx duplication). Re-run `rag-anon ingest` to clean it up.",
            collection_name, count, unique, count / unique,
        )
    else:
        logger.info("Collection '%s' loaded: %d chunks", collection_name, count)


def _chromadb_persist_dir() -> str:
    return str(_PROJECT_ROOT / os.getenv("RAG_ANON_CHROMADB_PERSIST_DIR", "./chromadb"))
