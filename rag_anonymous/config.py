from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv()
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_ANONYMIZER_ENTITIES = "PERSON"

DEFAULT_CORPUS_CACHE_SUBDIR = "data/input/corpus"


@dataclass(frozen=True)
class Settings:
    anonymizer_strategy: str
    dataset: str
    corpus: str
    query_question: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    embedding_num_ctx: int
    ollama_base_url: str
    es_url: str
    retrieval_k_docs: int
    llm_model: str
    llm_temperature: float
    llm_reasoning: bool
    llm_num_ctx: int
    llm_num_predict: int
    llm_timeout_sec: float
    privacy_prompt: bool
    anonymizer_entities: tuple[str, ...]
    log_level: str | None
    log_level_http: str | None
    log_level_presidio: str | None
    log_startup_env: str | None

    def log_startup_env_enabled(self) -> bool:
        raw = (self.log_startup_env or "true").strip().lower()
        return raw not in ("0", "false", "no", "off")

    def es_index(self, collection_name: str, dataset: str | None = None) -> str:
        return f"{collection_name}_{dataset or self.dataset}".lower()

    @classmethod
    def load(cls) -> Settings:
        entities_csv = os.getenv(
            "RAG_ANON_ANONYMIZER_ENTITIES",
            DEFAULT_ANONYMIZER_ENTITIES,
        )
        return cls(
            anonymizer_strategy=os.getenv("RAG_ANON_ANONYMIZER_STRATEGY", "offline"),
            dataset=os.getenv("RAG_ANON_CORPUS_DATASET", "dev"),
            corpus=os.getenv("RAG_ANON_CORPUS", ""),
            query_question=os.getenv("RAG_ANON_QUERY_QUESTION", ""),
            chunk_size=int(os.getenv("RAG_ANON_CHUNK_SIZE", "7000")),
            chunk_overlap=int(os.getenv("RAG_ANON_CHUNK_OVERLAP", "0")),
            embedding_model=os.getenv("RAG_ANON_EMBEDDING_MODEL", "nomic-embed-text"),
            embedding_num_ctx=int(os.getenv("RAG_ANON_EMBEDDING_NUM_CTX", "8192")),
            ollama_base_url=os.getenv(
                "RAG_ANON_OLLAMA_BASE_URL", "http://localhost:11434"
            ),
            es_url=os.getenv("RAG_ANON_ES_URL", "http://localhost:9200"),
            retrieval_k_docs=int(os.getenv("RAG_ANON_RETRIEVAL_K_DOCS", "5")),
            llm_model=os.getenv("RAG_ANON_LLM_MODEL", "qwen3:0.6b"),
            llm_temperature=float(os.getenv("RAG_ANON_LLM_TEMPERATURE", "0.0")),
            llm_reasoning=os.getenv("RAG_ANON_LLM_REASONING", "false").strip().lower()
            in ("1", "true", "yes"),
            llm_num_ctx=int(os.getenv("RAG_ANON_LLM_NUM_CTX", "8192")),
            llm_num_predict=int(os.getenv("RAG_ANON_LLM_NUM_PREDICT", "512")),
            llm_timeout_sec=float(os.getenv("RAG_ANON_LLM_TIMEOUT_SEC", "180")),
            privacy_prompt=_env_flag("RAG_ANON_PRIVACY_PROMPT", default=True),
            anonymizer_entities=tuple(entities_csv.split(",")),
            log_level=os.getenv("RAG_ANON_LOG_LEVEL"),
            log_level_http=os.getenv("RAG_ANON_LOG_LEVEL_HTTP"),
            log_level_presidio=os.getenv("RAG_ANON_LOG_LEVEL_PRESIDIO"),
            log_startup_env=os.getenv("RAG_ANON_LOG_STARTUP_ENV"),
        )


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes")


__all__ = [
    "DEFAULT_ANONYMIZER_ENTITIES",
    "DEFAULT_CORPUS_CACHE_SUBDIR",
    "PROJECT_ROOT",
    "Settings",
]
