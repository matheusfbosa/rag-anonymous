from typing import TypedDict


class RetrievedChunk(TypedDict):
    content: str
    doc_id: str | None
    chunk_id: str | None


class QueryResult(TypedDict):
    question: str
    response: object
    retrieved_chunks: list[RetrievedChunk]
    docs_unique: int
    retrieval_sec: float
    generation_sec: float
