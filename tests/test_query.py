import pytest
from langchain_core.documents import Document

from rag_anonymous.query import (
    _warn_if_context_may_exceed_num_ctx,
    is_llm_response_error,
    is_llm_timeout,
    is_recoverable_llm_error,
    query_rag,
    reasoning_flag,
)


class FakeRetriever:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs
        self.invoked_with: str | None = None

    def invoke(self, question: str) -> list[Document]:
        self.invoked_with = question
        return self._docs


class FakeRetrievalStore:
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
        store = FakeRetrievalStore(retrieved_docs)

        result = query_rag(chain, store, "what is X?", k_docs=3)

        assert result["question"] == "what is X?"
        assert result["response"] == "generated answer"

    def test_retrieved_chunks_carry_doc_and_chunk_ids(self, retrieved_docs):
        result = query_rag(
            FakeChain(), FakeRetrievalStore(retrieved_docs), "q", k_docs=3
        )

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
        result = query_rag(
            FakeChain(), FakeRetrievalStore(retrieved_docs), "q", k_docs=3
        )
        assert result["docs_unique"] == 2

    def test_returns_retrieval_and_generation_timings(
        self, retrieved_docs, monkeypatch
    ):
        ticks = iter([10.0, 10.4, 20.0, 21.5])
        monkeypatch.setattr(
            "rag_anonymous.query.time.perf_counter", lambda: next(ticks)
        )

        result = query_rag(
            FakeChain(), FakeRetrievalStore(retrieved_docs), "q", k_docs=3
        )

        assert result["retrieval_sec"] == pytest.approx(0.4)
        assert result["generation_sec"] == pytest.approx(1.5)

    def test_generation_timing_includes_recoverable_failure(
        self, retrieved_docs, monkeypatch
    ):
        class SlowChain:
            def invoke(self, payload: dict) -> str:
                raise TimeoutError("deadline exceeded")

        ticks = iter([1.0, 1.1, 2.0, 5.0])
        monkeypatch.setattr(
            "rag_anonymous.query.time.perf_counter", lambda: next(ticks)
        )

        result = query_rag(
            SlowChain(), FakeRetrievalStore(retrieved_docs), "q", k_docs=3
        )

        assert result["response"] == ""
        assert result["retrieval_sec"] == pytest.approx(0.1)
        assert result["generation_sec"] == pytest.approx(3.0)

    def test_context_joined_with_separator(self, retrieved_docs):
        chain = FakeChain()
        query_rag(chain, FakeRetrievalStore(retrieved_docs), "q", k_docs=3)

        assert chain.payload is not None
        assert chain.payload["question"] == "q"
        assert chain.payload["context"] == (
            "alpha content\n\n---\n\nbeta content\n\n---\n\ngamma content"
        )

    def test_k_docs_argument_overrides_env(self, retrieved_docs, monkeypatch):
        monkeypatch.setenv("RAG_ANON_RETRIEVAL_K_DOCS", "99")
        store = FakeRetrievalStore(retrieved_docs)

        query_rag(FakeChain(), store, "q", k_docs=7)

        assert store.k == 7

    def test_k_docs_falls_back_to_env(self, retrieved_docs, monkeypatch):
        monkeypatch.setenv("RAG_ANON_RETRIEVAL_K_DOCS", "11")
        store = FakeRetrievalStore(retrieved_docs)

        query_rag(FakeChain(), store, "q")

        assert store.k == 11

    def test_k_docs_default_is_five(self, retrieved_docs, monkeypatch):
        monkeypatch.delenv("RAG_ANON_RETRIEVAL_K_DOCS", raising=False)
        store = FakeRetrievalStore(retrieved_docs)

        query_rag(FakeChain(), store, "q")

        assert store.k == 5

    def test_retriever_receives_question(self, retrieved_docs):
        store = FakeRetrievalStore(retrieved_docs)

        query_rag(FakeChain(), store, "specific question", k_docs=3)

        assert store.retriever is not None
        assert store.retriever.invoked_with == "specific question"


class TestContextFitGuard:
    def test_small_context_does_not_warn(self, caplog):
        with caplog.at_level("WARNING"):
            _warn_if_context_may_exceed_num_ctx("x" * 100, "q", 8192)
        assert caplog.records == []

    def test_oversized_context_warns(self, caplog):
        with caplog.at_level("WARNING"):
            _warn_if_context_may_exceed_num_ctx("x" * 40000, "q", 8192)
        assert any("Context may exceed num_ctx" in r.message for r in caplog.records)

    def test_warning_reports_estimated_and_limit_tokens(self, caplog):
        with caplog.at_level("WARNING"):
            _warn_if_context_may_exceed_num_ctx("x" * 40000, "q", 8192)
        msg = caplog.records[0].message
        assert "approx_tokens=10064" in msg
        assert "num_ctx=8192" in msg


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


class TestLlmTimeoutDetection:
    def test_timeout_error(self):
        assert is_llm_timeout(TimeoutError()) is True

    def test_named_timeout_exception(self):
        class ReadTimeout(Exception):
            pass

        assert is_llm_timeout(ReadTimeout("boom")) is True

    def test_wrapped_timeout(self):
        try:
            raise TimeoutError("inner")
        except TimeoutError as inner:
            outer = RuntimeError("outer")
            outer.__cause__ = inner
            assert is_llm_timeout(outer) is True

    def test_non_timeout(self):
        assert is_llm_timeout(ValueError("nope")) is False


class TestLlmResponseErrorDetection:
    def test_response_error(self):
        class ResponseError(Exception):
            pass

        assert is_llm_response_error(ResponseError("CUDA OOM")) is True

    def test_wrapped_response_error(self):
        class ResponseError(Exception):
            pass

        try:
            raise ResponseError("inner")
        except ResponseError as inner:
            outer = RuntimeError("outer")
            outer.__cause__ = inner
            assert is_llm_response_error(outer) is True

    def test_non_response_error(self):
        assert is_llm_response_error(ValueError("nope")) is False

    def test_recoverable_includes_timeout_and_response_error(self):
        class ResponseError(Exception):
            pass

        assert is_recoverable_llm_error(TimeoutError()) is True
        assert is_recoverable_llm_error(ResponseError("boom")) is True
        assert is_recoverable_llm_error(ValueError("nope")) is False


class TestQueryRagTimeout:
    def test_timeout_returns_empty_response(self, retrieved_docs, caplog):
        class SlowChain:
            def invoke(self, payload: dict) -> str:
                raise TimeoutError("deadline exceeded")

        with caplog.at_level("WARNING"):
            result = query_rag(
                SlowChain(), FakeRetrievalStore(retrieved_docs), "q", k_docs=3
            )

        assert result["response"] == ""
        assert result["retrieved_chunks"]
        assert any("LLM invoke failed" in r.message for r in caplog.records)


class TestQueryRagResponseError:
    def test_response_error_returns_empty_response(self, retrieved_docs, caplog):
        class ResponseError(Exception):
            pass

        class BrokenChain:
            def invoke(self, payload: dict) -> str:
                raise ResponseError("CUDA error: out of memory")

        with caplog.at_level("WARNING"):
            result = query_rag(
                BrokenChain(), FakeRetrievalStore(retrieved_docs), "q", k_docs=3
            )

        assert result["response"] == ""
        assert result["retrieved_chunks"]
        assert any("LLM invoke failed" in r.message for r in caplog.records)


class FlakyRetriever:
    def __init__(self, docs: list[Document], errors: list[Exception]) -> None:
        self._docs = docs
        self._errors = list(errors)
        self.calls = 0
        self.invoked_with: str | None = None

    def invoke(self, question: str) -> list[Document]:
        self.calls += 1
        self.invoked_with = question
        if self._errors:
            raise self._errors.pop(0)
        return self._docs


class FlakyStore:
    def __init__(self, retriever: FlakyRetriever) -> None:
        self.retriever = retriever
        self.k: int | None = None

    def as_retriever(self, search_kwargs: dict) -> FlakyRetriever:
        self.k = search_kwargs["k"]
        return self.retriever


class TestRetrieverRetry:
    def test_retries_recoverable_error_then_succeeds(
        self, retrieved_docs, monkeypatch, caplog
    ):
        class ResponseError(Exception):
            pass

        monkeypatch.setattr("rag_anonymous.query.time.sleep", lambda _s: None)
        retriever = FlakyRetriever(retrieved_docs, [ResponseError("CUDA busy")])
        store = FlakyStore(retriever)

        with caplog.at_level("WARNING"):
            result = query_rag(FakeChain(), store, "q", k_docs=3)

        assert retriever.calls == 2
        assert result["retrieved_chunks"]
        assert any("Retriever invoke failed" in r.message for r in caplog.records)

    def test_gives_up_after_retries(self, retrieved_docs, monkeypatch):
        class ResponseError(Exception):
            pass

        monkeypatch.setattr("rag_anonymous.query.time.sleep", lambda _s: None)
        retriever = FlakyRetriever(
            retrieved_docs,
            [ResponseError("eof"), ResponseError("eof"), ResponseError("eof")],
        )
        store = FlakyStore(retriever)

        with pytest.raises(ResponseError):
            query_rag(FakeChain(), store, "q", k_docs=3)

        assert retriever.calls == 3

    def test_non_recoverable_error_is_not_retried(self, retrieved_docs):
        retriever = FlakyRetriever(retrieved_docs, [ValueError("bad index")])
        store = FlakyStore(retriever)

        with pytest.raises(ValueError, match="bad index"):
            query_rag(FakeChain(), store, "q", k_docs=3)

        assert retriever.calls == 1
