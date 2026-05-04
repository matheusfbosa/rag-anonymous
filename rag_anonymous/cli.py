import logging
import os
import sys

from rag_anonymous.anonymizer import Anonymizer
from rag_anonymous.ingest import ingest_offline, ingest_ondemand, load_corpus
from rag_anonymous.log_config import configure_logging
from rag_anonymous.query import create_chain, load_vectordb, query_rag

logger = logging.getLogger(__name__)

USAGE = "Usage: rag-anon {ingest|query}  (configure via env / .env)"


def cmd_ingest():
    strategy = os.getenv("RAG_ANON_ANONYMIZER_STRATEGY", "offline")
    corpus_split = os.getenv("RAG_ANON_CORPUS_SPLIT", "dev")

    logger.info("[ingest] strategy=%s, corpus=%s", strategy, corpus_split)
    corpus = load_corpus(corpus_split)
    if strategy == "ondemand":
        ingest_ondemand(corpus, collection_name=strategy)
    else:
        ingest_offline(corpus, Anonymizer(), collection_name=strategy)


def cmd_query():
    strategy = os.getenv("RAG_ANON_ANONYMIZER_STRATEGY", "offline")
    question = os.getenv("RAG_ANON_QUERY_QUESTION", "")
    k_docs = int(os.getenv("RAG_ANON_RETRIEVAL_K_DOCS", "5"))

    if not question:
        raise SystemExit(
            "RAG_ANON_QUERY_QUESTION env var is required for `rag-anon query` "
            "(set it in .env or inline: RAG_ANON_QUERY_QUESTION=\"...\" rag-anon query)."
        )

    logger.info("[query] strategy=%s, k_docs=%d", strategy, k_docs)
    logger.info("[query] question: %s", question)

    vectordb = load_vectordb(collection_name=strategy)
    chain = create_chain()
    result = query_rag(chain, vectordb, question, k_docs=k_docs)

    if strategy == "ondemand":
        result["response"] = Anonymizer().anonymize(_response_text(result["response"]))

    logger.info("Answer: %s", result["response"])
    logger.info(
        "Retrieved %d chunks from %d unique documents:",
        len(result["retrieved_chunks"]),
        result["num_unique_docs"],
    )

    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        logger.info("  [%d] doc_id=%s", i, chunk["doc_id"])
        logger.debug("      %s...", chunk["content"][:120])


def _response_text(raw):
    """Extract plain string from chain response (LCEL may return message or str)."""
    if hasattr(raw, "content"):
        return raw.content
    return raw if isinstance(raw, str) else str(raw)


COMMANDS = {
    "ingest": cmd_ingest,
    "query": cmd_query,
}


def main():
    configure_logging()

    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(USAGE)

    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
