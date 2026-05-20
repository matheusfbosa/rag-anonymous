import pytest
from langchain_core.documents import Document

from rag_anonymous.query import query_rag, reasoning_flag


class FakeRetriever:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs
        self.invoked_with: str | None = None

    def invoke(self, question: str) -> list[Document]:
        self.invoked_with = question
        return self._docs


class FakeVectorDB:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs
        self.k: int | None = None
        self.retriever: FakeRetriever | None = None

    def as_retriever(self, search_kwargs: dict) -> FakeRetriever:
        self.k = search_kwargs["k"]
        self.retriever = FakeRetriever(self._docs)
        return self.retriever


class FakeChain:
    def __init__(self, response: str = "ANSWER") -> None:
        self._response = response
        self.payload: dict | None = None

    def invoke(self, payload: dict) -> str:
        self.payload = payload
        return self._response


@pytest.fixture
def retrieved_docs() -> list[Document]:
    return [
        Document(
            page_content="alpha content",
            metadata={"doc_id": "a", "chunk_id": "a_chunk_0"},
        ),
        Document(
            page_content="beta content",
            metadata={"doc_id": "a", "chunk_id": "a_chunk_1"},
        ),
        Document(
            page_content="gamma content",
            metadata={"doc_id": "b", "chunk_id": "b_chunk_0"},
        ),
    ]


class TestQueryRag:
    def test_response_round_trips(self, retrieved_docs):
        chain = FakeChain(response="generated answer")
        vectordb = FakeVectorDB(retrieved_docs)

        result = query_rag(chain, vectordb, "what is X?", k_docs=3)

        assert result["question"] == "what is X?"
        assert result["response"] == "generated answer"

    def test_retrieved_chunks_carry_doc_and_chunk_ids(self, retrieved_docs):
        result = query_rag(FakeChain(), FakeVectorDB(retrieved_docs), "q", k_docs=3)

        assert len(result["retrieved_chunks"]) == 3
        assert [c["doc_id"] for c in result["retrieved_chunks"]] == ["a", "a", "b"]
        assert [c["chunk_id"] for c in result["retrieved_chunks"]] == [
            "a_chunk_0",
            "a_chunk_1",
            "b_chunk_0",
        ]
        assert [c["content"] for c in result["retrieved_chunks"]] == [
            "alpha content",
            "beta content",
            "gamma content",
        ]

    def test_docs_unique_deduplicates(self, retrieved_docs):
        result = query_rag(FakeChain(), FakeVectorDB(retrieved_docs), "q", k_docs=3)
        assert result["docs_unique"] == 2

    def test_context_joined_with_separator(self, retrieved_docs):
        chain = FakeChain()
        query_rag(chain, FakeVectorDB(retrieved_docs), "q", k_docs=3)

        assert chain.payload is not None
        assert chain.payload["question"] == "q"
        assert chain.payload["context"] == (
            "alpha content\n\n---\n\nbeta content\n\n---\n\ngamma content"
        )

    def test_k_docs_argument_overrides_env(self, retrieved_docs, monkeypatch):
        monkeypatch.setenv("RAG_ANON_RETRIEVAL_K_DOCS", "99")
        vectordb = FakeVectorDB(retrieved_docs)

        query_rag(FakeChain(), vectordb, "q", k_docs=7)

        assert vectordb.k == 7

    def test_k_docs_falls_back_to_env(self, retrieved_docs, monkeypatch):
        monkeypatch.setenv("RAG_ANON_RETRIEVAL_K_DOCS", "11")
        vectordb = FakeVectorDB(retrieved_docs)

        query_rag(FakeChain(), vectordb, "q")

        assert vectordb.k == 11

    def test_k_docs_default_is_five(self, retrieved_docs, monkeypatch):
        monkeypatch.delenv("RAG_ANON_RETRIEVAL_K_DOCS", raising=False)
        vectordb = FakeVectorDB(retrieved_docs)

        query_rag(FakeChain(), vectordb, "q")

        assert vectordb.k == 5

    def test_retriever_receives_question(self, retrieved_docs):
        vectordb = FakeVectorDB(retrieved_docs)

        query_rag(FakeChain(), vectordb, "specific question", k_docs=3)

        assert vectordb.retriever is not None
        assert vectordb.retriever.invoked_with == "specific question"


class TestReasoningFlag:
    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "YES"])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("RAG_ANON_LLM_REASONING", value)
        assert reasoning_flag() is True

    @pytest.mark.parametrize("value", ["0", "false", "False", "no", "", "off"])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("RAG_ANON_LLM_REASONING", value)
        assert reasoning_flag() is False

    def test_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("RAG_ANON_LLM_REASONING", raising=False)
        assert reasoning_flag() is False

    def test_whitespace_is_stripped(self, monkeypatch):
        monkeypatch.setenv("RAG_ANON_LLM_REASONING", "  true  ")
        assert reasoning_flag() is True
