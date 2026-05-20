import logging

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag_anonymous.config import Settings

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""


def build_llm(model_name: str | None = None) -> ChatOllama:
    s = Settings.load()
    return ChatOllama(
        model=model_name or s.llm_model,
        base_url=s.ollama_base_url,
        temperature=s.llm_temperature,
        reasoning=s.llm_reasoning,
    )


def create_chain(llm=None):
    if llm is None:
        llm = build_llm()
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    return prompt | llm | StrOutputParser()


def load_vectordb(collection_name="offline", persist_dir=None):
    s = Settings.load()
    persist_dir = persist_dir or s.chromadb_persist_dir
    embeddings = OllamaEmbeddings(
        model=s.embedding_model,
        base_url=s.ollama_base_url,
    )
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    _log_collection_health(vectordb, collection_name)
    return vectordb


def query_rag(chain, vectordb, question, k_docs=None):
    if k_docs is None:
        k_docs = Settings.load().retrieval_k_docs
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
        "docs_unique": len(set(c.metadata["doc_id"] for c in chunks)),
    }


def reasoning_flag() -> bool:
    return Settings.load().llm_reasoning


def _log_collection_health(vectordb, collection_name):
    try:
        count = vectordb._collection.count()
    except Exception as exc:
        logger.warning(
            "Could not read collection count: name=%s error=%s",
            collection_name,
            exc,
        )
        return

    if count == 0:
        logger.warning("Collection empty: name=%s", collection_name)
        return

    try:
        result = vectordb._collection.get(include=["metadatas"])
        chunk_ids = [
            (m or {}).get("chunk_id") for m in (result.get("metadatas") or [])
        ]
        unique = len({c for c in chunk_ids if c})
    except Exception as exc:
        logger.warning(
            "Could not inspect collection metadata: name=%s error=%s",
            collection_name,
            exc,
        )
        logger.info(
            "Loaded vectordb: collection=%s chunks=%d metadata=skipped",
            collection_name,
            count,
        )
        return

    if unique and count != unique:
        logger.warning(
            "Collection chunk_id duplication: name=%s chunks=%d chunk_ids_unique=%d",
            collection_name,
            count,
            unique,
        )
    else:
        logger.info(
            "Loaded vectordb: collection=%s chunks=%d chunk_ids_unique=%d",
            collection_name,
            count,
            unique,
        )
