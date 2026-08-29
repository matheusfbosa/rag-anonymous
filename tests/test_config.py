from unittest.mock import patch

import pytest

from rag_anonymous.config import Settings


class TestLlmLimitsSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("RAG_ANON_LLM_NUM_PREDICT", raising=False)
        monkeypatch.delenv("RAG_ANON_LLM_TIMEOUT_SEC", raising=False)

        s = Settings.load()

        assert s.llm_num_predict == 512
        assert s.llm_timeout_sec == 180.0

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("RAG_ANON_LLM_NUM_PREDICT", "256")
        monkeypatch.setenv("RAG_ANON_LLM_TIMEOUT_SEC", "90")

        s = Settings.load()

        assert s.llm_num_predict == 256
        assert s.llm_timeout_sec == 90.0


class TestPrivacyPromptSettings:
    def test_default_is_enabled(self, monkeypatch):
        monkeypatch.delenv("RAG_ANON_PRIVACY_PROMPT", raising=False)
        assert Settings.load().privacy_prompt is True

    @pytest.mark.parametrize("value", ["0", "false", "False", "no", "off"])
    def test_falsy_disables(self, monkeypatch, value):
        monkeypatch.setenv("RAG_ANON_PRIVACY_PROMPT", value)
        assert Settings.load().privacy_prompt is False

    @pytest.mark.parametrize("value", ["1", "true", "True", "yes"])
    def test_truthy_enables(self, monkeypatch, value):
        monkeypatch.setenv("RAG_ANON_PRIVACY_PROMPT", value)
        assert Settings.load().privacy_prompt is True


class TestBuildLlmLimits:
    def test_forwards_num_predict_and_timeout(self, monkeypatch):
        monkeypatch.setenv("RAG_ANON_LLM_NUM_PREDICT", "128")
        monkeypatch.setenv("RAG_ANON_LLM_TIMEOUT_SEC", "45")

        with patch("rag_anonymous.query.ChatOllama") as chat_cls:
            from rag_anonymous.query import build_llm

            build_llm("test-model")

        chat_cls.assert_called_once()
        kwargs = chat_cls.call_args.kwargs
        assert kwargs["model"] == "test-model"
        assert kwargs["num_predict"] == 128
        assert kwargs["timeout"] == 45.0

    def test_explicit_overrides_beat_settings(self, monkeypatch):
        monkeypatch.setenv("RAG_ANON_LLM_NUM_PREDICT", "128")
        monkeypatch.setenv("RAG_ANON_LLM_TIMEOUT_SEC", "45")

        with patch("rag_anonymous.query.ChatOllama") as chat_cls:
            from rag_anonymous.query import build_llm

            build_llm("judge-model", num_predict=2048, timeout=600.0)

        kwargs = chat_cls.call_args.kwargs
        assert kwargs["num_predict"] == 2048
        assert kwargs["timeout"] == 600.0


class TestLoadRetrieverDataset:
    def test_explicit_dataset_selects_index(self, monkeypatch):
        monkeypatch.setenv("RAG_ANON_CORPUS_DATASET", "test")

        with (
            patch("rag_anonymous.query.ElasticsearchStore") as store_cls,
            patch("rag_anonymous.query.OllamaEmbeddings"),
            patch("rag_anonymous.query._log_collection_health"),
        ):
            from rag_anonymous.query import load_retriever

            load_retriever("offline", dataset="dev")

        store_cls.assert_called_once()
        assert store_cls.call_args.kwargs["index_name"] == "offline_dev"
