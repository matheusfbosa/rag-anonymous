import logging
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_elasticsearch import DenseVectorStrategy, ElasticsearchStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag_anonymous.config import Settings

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""

_CHARS_PER_TOKEN = 4
_CTX_WARN_RATIO = 0.9
_PROMPT_OVERHEAD_TOKENS = 64
_RETRIEVER_ATTEMPTS = 3
_RETRIEVER_RETRY_SLEEP_SEC = 1.0


def build_llm(
    model_name: str | None = None,
    *,
    num_predict: int | None = None,
    timeout: float | None = None,
) -> ChatOllama:
    s = Settings.load()
    return ChatOllama(
        model=model_name or s.llm_model,
        base_url=s.ollama_base_url,
        temperature=s.llm_temperature,
        reasoning=s.llm_reasoning,
        num_ctx=s.llm_num_ctx,
        num_predict=num_predict if num_predict is not None else s.llm_num_predict,
        timeout=timeout if timeout is not None else s.llm_timeout_sec,
    )


def create_chain(llm=None):
    if llm is None:
        llm = build_llm()
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    return prompt | llm | StrOutputParser()


def load_retriever(collection_name="offline", dataset: str | None = None):
    s = Settings.load()
    index = s.es_index(collection_name, dataset)
    embeddings = OllamaEmbeddings(
        model=s.embedding_model,
        base_url=s.ollama_base_url,
        num_ctx=s.embedding_num_ctx,
    )
    store = ElasticsearchStore(
        index_name=index,
        embedding=embeddings,
        es_url=s.es_url,
        strategy=DenseVectorStrategy(hybrid=True, rrf=False),
    )
    _log_collection_health(store, index)
    return store


def query_rag(chain, retrieval_store, question, k_docs=None):
    s = Settings.load()
    if k_docs is None:
        k_docs = s.retrieval_k_docs
    retriever = retrieval_store.as_retriever(search_kwargs={"k": k_docs})
    retrieval_started = time.perf_counter()
    chunks = _invoke_retriever(retriever, question)
    retrieval_sec = time.perf_counter() - retrieval_started
    context_text = "\n\n---\n\n".join(doc.page_content for doc in chunks)

    _warn_if_context_may_exceed_num_ctx(context_text, question, s.llm_num_ctx)

    generation_started = time.perf_counter()
    try:
        response = chain.invoke({"context": context_text, "question": question})
    except Exception as exc:
        generation_sec = time.perf_counter() - generation_started
        if is_recoverable_llm_error(exc):
            logger.warning(
                "LLM invoke failed after %.1fs: %s — %.80s",
                generation_sec,
                exc.__class__.__name__,
                question,
            )
            logger.debug("LLM invoke traceback", exc_info=True)
            response = ""
        else:
            logger.error(
                "LLM invoke failed after %.1fs: %s — %.80s",
                generation_sec,
                exc.__class__.__name__,
                question,
            )
            raise
    else:
        generation_sec = time.perf_counter() - generation_started
        if generation_sec > 60:
            logger.warning("LLM invoke slow: %.1fs — %.80s", generation_sec, question)
        else:
            logger.debug("LLM invoke completed in %.1fs", generation_sec)

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
        "docs_unique": len(set(c.metadata["doc_id"] for c in chunks)),
        "retrieval_sec": retrieval_sec,
        "generation_sec": generation_sec,
    }


def reasoning_flag() -> bool:
    return Settings.load().llm_reasoning


def is_llm_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    name = exc.__class__.__name__
    if "Timeout" in name:
        return True
    cause = exc.__cause__
    return cause is not None and is_llm_timeout(cause)


def is_llm_response_error(exc: BaseException) -> bool:
    if exc.__class__.__name__ == "ResponseError":
        return True
    cause = exc.__cause__
    return cause is not None and is_llm_response_error(cause)


def is_recoverable_llm_error(exc: BaseException) -> bool:
    return is_llm_timeout(exc) or is_llm_response_error(exc)


def _invoke_retriever(retriever, question: str):
    last_exc: BaseException | None = None
    for attempt in range(1, _RETRIEVER_ATTEMPTS + 1):
        try:
            return retriever.invoke(question)
        except Exception as exc:
            last_exc = exc
            if not is_recoverable_llm_error(exc) or attempt == _RETRIEVER_ATTEMPTS:
                raise
            logger.warning(
                "Retriever invoke failed (attempt %d/%d): %s — %.80s",
                attempt,
                _RETRIEVER_ATTEMPTS,
                exc.__class__.__name__,
                question,
            )
            time.sleep(_RETRIEVER_RETRY_SLEEP_SEC)
    assert last_exc is not None
    raise last_exc


def _warn_if_context_may_exceed_num_ctx(context_text, question, num_ctx):
    approx_tokens = (
        (len(context_text) + len(question)) // _CHARS_PER_TOKEN
        + _PROMPT_OVERHEAD_TOKENS
    )
    if approx_tokens > _CTX_WARN_RATIO * num_ctx:
        logger.warning(
            "Context may exceed num_ctx: approx_tokens=%d num_ctx=%d "
            "(Ollama truncates silently; raise RAG_ANON_LLM_NUM_CTX or lower "
            "RAG_ANON_RETRIEVAL_K_DOCS / RAG_ANON_CHUNK_SIZE)",
            approx_tokens,
            num_ctx,
        )


def _log_collection_health(store, index):
    try:
        exists = store.client.indices.exists(index=index)
    except Exception as exc:
        logger.warning(
            "Could not read index: index=%s error=%s",
            index,
            exc,
        )
        return

    if not exists:
        logger.warning(
            "Index missing: index=%s (run `rag-anon ingest` first)", index
        )
        return

    count = store.client.count(index=index)["count"]
    if count == 0:
        logger.warning("Index empty: index=%s", index)
        return

    logger.info(
        "Connected to Elasticsearch: index=%s chunks=%d",
        index,
        count,
    )
