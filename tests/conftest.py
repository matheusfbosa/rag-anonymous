"""Shared fixtures for the rag-anonymous unit test suite."""

import pytest


class FakeAnonymizer:
    """Stand-in for ``Anonymizer`` that avoids loading Presidio/Spacy.

    Replaces ``Alice`` with ``<PERSON>`` so tests can assert that
    ``ingest._build_documents`` actually routes text through the anonymizer
    when one is provided.
    """

    def anonymize(self, text: str) -> str:
        return text.replace("Alice", "<PERSON>")


@pytest.fixture
def fake_anonymizer() -> FakeAnonymizer:
    return FakeAnonymizer()


@pytest.fixture
def sample_corpus() -> list[dict]:
    """Two-doc corpus where doc1 is large enough to be split into multiple chunks."""
    return [
        {"doc_id": "doc1", "text": "Alice met Bob in Paris. " * 20},
        {"doc_id": "doc2", "text": "Short."},
    ]
