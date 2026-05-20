import sys
from types import SimpleNamespace

import pytest

from rag_anonymous import cli
from rag_anonymous.cli import USAGE, _response_text, cmd_query, main


class TestResponseText:
    def test_message_like_object_returns_content(self):
        msg = SimpleNamespace(content="hello world")
        assert _response_text(msg) == "hello world"

    def test_plain_string_returned_unchanged(self):
        assert _response_text("hello") == "hello"

    def test_arbitrary_object_falls_back_to_str(self):
        class Custom:
            def __str__(self) -> str:
                return "stringified"

        assert _response_text(Custom()) == "stringified"

    def test_content_attribute_takes_precedence_over_str(self):
        obj = SimpleNamespace(content="from-content")
        assert _response_text(obj) == "from-content"


class TestMainDispatch:
    def test_no_args_raises_with_usage(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["rag-anon"])

        with pytest.raises(SystemExit) as exc:
            main()
        assert str(exc.value) == USAGE

    def test_unknown_command_raises_with_usage(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["rag-anon", "bogus"])

        with pytest.raises(SystemExit) as exc:
            main()
        assert str(exc.value) == USAGE

    def test_too_many_args_raises_with_usage(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["rag-anon", "ingest", "extra"])

        with pytest.raises(SystemExit) as exc:
            main()
        assert str(exc.value) == USAGE

    def test_ingest_command_dispatches_to_cmd_ingest(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setitem(cli.COMMANDS, "ingest", lambda: calls.append("ingest"))
        monkeypatch.setattr(sys, "argv", ["rag-anon", "ingest"])

        main()

        assert calls == ["ingest"]

    def test_query_command_dispatches_to_cmd_query(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setitem(cli.COMMANDS, "query", lambda: calls.append("query"))
        monkeypatch.setattr(sys, "argv", ["rag-anon", "query"])

        main()

        assert calls == ["query"]


class TestCmdQueryGuard:
    def test_missing_question_raises_systemexit(self, monkeypatch):
        monkeypatch.delenv("RAG_ANON_QUERY_QUESTION", raising=False)

        with pytest.raises(SystemExit) as exc:
            cmd_query()
        assert "RAG_ANON_QUERY_QUESTION" in str(exc.value)

    def test_empty_question_raises_systemexit(self, monkeypatch):
        monkeypatch.setenv("RAG_ANON_QUERY_QUESTION", "")

        with pytest.raises(SystemExit) as exc:
            cmd_query()
        assert "RAG_ANON_QUERY_QUESTION" in str(exc.value)
