from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_anonymous.ingest import _build_documents, _corpus_cache_path


@pytest.fixture(autouse=True)
def _force_small_chunks(monkeypatch):
    monkeypatch.setenv("RAG_ANON_CHUNK_SIZE", "50")
    monkeypatch.setenv("RAG_ANON_CHUNK_OVERLAP", "0")


class TestBuildDocumentsCommon:
    def test_ids_align_with_chunk_id_metadata(self, sample_corpus, fake_anonymizer):
        documents, ids = _build_documents(sample_corpus, anonymizer=fake_anonymizer)

        assert len(documents) == len(ids)
        for doc, chunk_id in zip(documents, ids):
            assert isinstance(doc, Document)
            assert doc.metadata["chunk_id"] == chunk_id

    def test_chunk_ids_are_zero_based_and_contiguous_per_doc(
        self, sample_corpus, fake_anonymizer
    ):
        documents, ids = _build_documents(sample_corpus, anonymizer=fake_anonymizer)

        per_doc_indices: dict[str, list[int]] = {}
        for chunk_id, doc in zip(ids, documents):
            doc_id = doc.metadata["doc_id"]
            assert chunk_id == f"{doc_id}_chunk_{doc.metadata['chunk_index']}"
            per_doc_indices.setdefault(doc_id, []).append(doc.metadata["chunk_index"])

        for doc_id, indices in per_doc_indices.items():
            assert indices == list(range(len(indices))), (
                f"chunk_index for {doc_id} should be 0..N-1, got {indices}"
            )

    def test_total_chunks_metadata_matches_per_doc_count(
        self, sample_corpus, fake_anonymizer
    ):
        documents, _ = _build_documents(sample_corpus, anonymizer=fake_anonymizer)

        per_doc = {item["doc_id"]: 0 for item in sample_corpus}
        for doc in documents:
            per_doc[doc.metadata["doc_id"]] += 1

        for doc in documents:
            assert doc.metadata["total_chunks"] == per_doc[doc.metadata["doc_id"]]

    def test_original_length_is_raw_text_length(self, sample_corpus, fake_anonymizer):
        documents, _ = _build_documents(sample_corpus, anonymizer=fake_anonymizer)
        raw_lengths = {item["doc_id"]: len(item["text"]) for item in sample_corpus}

        for doc in documents:
            assert doc.metadata["original_length"] == raw_lengths[doc.metadata["doc_id"]]

    def test_doc1_actually_chunked(self, sample_corpus, fake_anonymizer):
        documents, _ = _build_documents(sample_corpus, anonymizer=fake_anonymizer)
        doc1_chunks = [d for d in documents if d.metadata["doc_id"] == "doc1"]
        assert len(doc1_chunks) > 1


class TestBuildDocumentsOffline:
    def test_strategy_metadata_is_offline(self, sample_corpus, fake_anonymizer):
        documents, _ = _build_documents(sample_corpus, anonymizer=fake_anonymizer)
        assert all(doc.metadata["strategy"] == "offline" for doc in documents)

    def test_anonymized_length_is_present(self, sample_corpus, fake_anonymizer):
        documents, _ = _build_documents(sample_corpus, anonymizer=fake_anonymizer)
        assert all("anonymized_length" in doc.metadata for doc in documents)

    def test_text_is_routed_through_anonymizer(self, sample_corpus, fake_anonymizer):
        documents, _ = _build_documents(sample_corpus, anonymizer=fake_anonymizer)
        all_content = " ".join(doc.page_content for doc in documents)

        assert "Alice" not in all_content
        assert "<PERSON>" in all_content


class TestBuildDocumentsOndemand:
    def test_strategy_metadata_is_ondemand(self, sample_corpus):
        documents, _ = _build_documents(sample_corpus, anonymizer=None)
        assert all(doc.metadata["strategy"] == "ondemand" for doc in documents)

    def test_anonymized_length_is_absent(self, sample_corpus):
        documents, _ = _build_documents(sample_corpus, anonymizer=None)
        assert all("anonymized_length" not in doc.metadata for doc in documents)

    def test_raw_text_preserved(self, sample_corpus):
        documents, _ = _build_documents(sample_corpus, anonymizer=None)
        all_content = " ".join(doc.page_content for doc in documents)

        assert "Alice" in all_content
        assert "<PERSON>" not in all_content


class TestCorpusCachePath:
    @pytest.fixture(autouse=True)
    def _set_corpus(self, monkeypatch):
        monkeypatch.setenv(
            "RAG_ANON_CORPUS",
            "https://example.com/echr_{dataset}.json",
        )

    def test_path_layout(self):
        path = _corpus_cache_path("dev")
        assert isinstance(path, Path)
        assert path.name == "echr_dev.json"
        assert path.parent.name == "corpus"
        assert path.parent.parent.name == "input"
        assert path.parent.parent.parent.name == "data"

    @pytest.mark.parametrize("dataset", ["train", "dev", "test"])
    def test_dataset_is_interpolated(self, dataset):
        assert _corpus_cache_path(dataset).name == f"echr_{dataset}.json"
