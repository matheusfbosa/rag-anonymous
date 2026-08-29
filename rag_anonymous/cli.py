import logging
import sys
import time

from rag_anonymous.anonymizer import Anonymizer
from rag_anonymous.config import Settings
from rag_anonymous.ingest import ingest_offline, ingest_ondemand, load_corpus
from rag_anonymous.log_config import configure_logging
from rag_anonymous.query import create_chain, load_retriever, query_rag

logger = logging.getLogger(__name__)

USAGE = "Usage: rag-anon {ingest|query}  (configure via env / .env)"


def cmd_ingest():
    s = Settings.load()
    strategy = s.anonymizer_strategy
    dataset = s.dataset

    logger.info("Ingesting: strategy=%s dataset=%s", strategy, dataset)
    corpus = load_corpus(dataset)
    if strategy == "ondemand":
        ingest_ondemand(corpus, collection_name=strategy, dataset=dataset)
    else:
        ingest_offline(corpus, Anonymizer(), collection_name=strategy, dataset=dataset)


def cmd_query():
    s = Settings.load()
    strategy = s.anonymizer_strategy
    question = s.query_question
    k_docs = s.retrieval_k_docs

    if not question:
        raise SystemExit(
            "RAG_ANON_QUERY_QUESTION env var is required for `rag-anon query` "
            "(set it in .env or inline: RAG_ANON_QUERY_QUESTION=\"...\" rag-anon query)."
        )

    logger.info("Querying: strategy=%s k_docs=%d", strategy, k_docs)
    logger.info("Querying: question=%s", question)

    retrieval_store = load_retriever(collection_name=strategy)
    chain = create_chain()
    result = query_rag(chain, retrieval_store, question, k_docs=k_docs)

    anonymize_sec = 0.0
    if strategy == "ondemand":
        started = time.perf_counter()
        result["response"] = Anonymizer().anonymize(_response_text(result["response"]))
        anonymize_sec = time.perf_counter() - started

    logger.info(
        "Queried: retrieval=%.2fs generation=%.2fs anonymize=%.3fs query=%.2fs",
        result["retrieval_sec"],
        result["generation_sec"],
        anonymize_sec,
        result["retrieval_sec"] + result["generation_sec"] + anonymize_sec,
    )
    logger.info("Queried: answer=%s", result["response"])
    logger.info(
        "Queried: retrieved_chunks=%d docs_unique=%d",
        len(result["retrieved_chunks"]),
        result["docs_unique"],
    )

    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        logger.info("Queried: chunk_index=%d doc_id=%s", i, chunk["doc_id"])
        logger.debug(
            "Queried: chunk_preview=%s...",
            chunk["content"][:120],
        )


COMMANDS = {
    "ingest": cmd_ingest,
    "query": cmd_query,
}


def main():
    configure_logging()

    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(USAGE)

    COMMANDS[sys.argv[1]]()


def _response_text(raw):
    if hasattr(raw, "content"):
        return raw.content
    return raw if isinstance(raw, str) else str(raw)


if __name__ == "__main__":
    main()
