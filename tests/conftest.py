import pytest


@pytest.fixture(autouse=True)
def _disable_startup_env_logging(monkeypatch):
    monkeypatch.setenv("RAG_ANON_LOG_STARTUP_ENV", "false")


class FakeAnonymizer:
    def anonymize(self, text: str) -> str:
        return text.replace("Alice", "<PERSON>")


@pytest.fixture
def fake_anonymizer() -> FakeAnonymizer:
    return FakeAnonymizer()


@pytest.fixture
def sample_corpus() -> list[dict]:
    return [
        {"doc_id": "doc1", "text": "Alice met Bob in Paris. " * 20},
        {"doc_id": "doc2", "text": "Short."},
    ]
