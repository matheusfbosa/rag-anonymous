import logging

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
_PROMPT_OVERHEAD_TOKENS = 64
_CTX_WARN_RATIO = 0.9


def build_llm(model_name: str | None = None) -> ChatOllama:
    s = Settings.load()
    return ChatOllama(
        model=model_name or s.llm_model,
        base_url=s.ollama_base_url,
        temperature=s.llm_temperature,
        reasoning=s.llm_reasoning,
        num_ctx=s.llm_num_ctx,
    )


def create_chain(llm=None):
    if llm is None:
        llm = build_llm()
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    return prompt | llm | StrOutputParser()


def load_vectordb(collection_name="offline"):
    s = Settings.load()
    index = s.es_index(collection_name)
    embeddings = OllamaEmbeddings(
        model=s.embedding_model,
        base_url=s.ollama_base_url,
        num_ctx=s.embedding_num_ctx,
    )
    vectordb = ElasticsearchStore(
        index_name=index,
        embedding=embeddings,
        es_url=s.es_url,
        strategy=DenseVectorStrategy(hybrid=True, rrf=False),
    )
    _log_collection_health(vectordb, index)
    return vectordb


def query_rag(chain, vectordb, question, k_docs=None):
    s = Settings.load()
    if k_docs is None:
        k_docs = s.retrieval_k_docs
    retriever = vectordb.as_retriever(search_kwargs={"k": k_docs})
    chunks = retriever.invoke(question)
    context_text = "\n\n---\n\n".join(doc.page_content for doc in chunks)

    _warn_if_context_may_exceed_num_ctx(context_text, question, s.llm_num_ctx)

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
        "docs_unique": len(set(c.metadata["doc_id"] for c in chunks)),
    }


def reasoning_flag() -> bool:
    return Settings.load().llm_reasoning


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


def _log_collection_health(vectordb, index):
    try:
        exists = vectordb.client.indices.exists(index=index)
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

    count = vectordb.client.count(index=index)["count"]
    if count == 0:
        logger.warning("Index empty: index=%s", index)
        return

    logger.info(
        "Loaded vectordb: index=%s chunks=%d",
        index,
        count,
    )
