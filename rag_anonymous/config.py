from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv()
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_ANONYMIZER_ENTITIES = "PERSON,LOCATION,ORGANIZATION,DATE_TIME,MONEY"


@dataclass(frozen=True)
class Settings:
    anonymizer_strategy: str
    corpus_split: str
    query_question: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    ollama_base_url: str
    chromadb_persist_dir: str
    retrieval_k_docs: int
    llm_model: str
    llm_temperature: float
    llm_reasoning: bool
    anonymizer_entities: tuple[str, ...]
    log_level: str | None
    log_level_http: str | None
    log_level_presidio: str | None
    log_startup_env: str | None

    def log_startup_env_enabled(self) -> bool:
        raw = (self.log_startup_env or "true").strip().lower()
        return raw not in ("0", "false", "no", "off")

    @classmethod
    def load(cls) -> Settings:
        entities_csv = os.getenv(
            "RAG_ANON_ANONYMIZER_ENTITIES",
            DEFAULT_ANONYMIZER_ENTITIES,
        )
        return cls(
            anonymizer_strategy=os.getenv(
                "RAG_ANON_ANONYMIZER_STRATEGY", "offline"
            ),
            corpus_split=os.getenv("RAG_ANON_CORPUS_SPLIT", "dev"),
            query_question=os.getenv("RAG_ANON_QUERY_QUESTION", ""),
            chunk_size=int(os.getenv("RAG_ANON_CHUNK_SIZE", "200")),
            chunk_overlap=int(os.getenv("RAG_ANON_CHUNK_OVERLAP", "0")),
            embedding_model=os.getenv(
                "RAG_ANON_EMBEDDING_MODEL", "nomic-embed-text"
            ),
            ollama_base_url=os.getenv(
                "RAG_ANON_OLLAMA_BASE_URL", "http://localhost:11434"
            ),
            chromadb_persist_dir=str(
                PROJECT_ROOT
                / os.getenv("RAG_ANON_CHROMADB_PERSIST_DIR", "./chromadb")
            ),
            retrieval_k_docs=int(os.getenv("RAG_ANON_RETRIEVAL_K_DOCS", "5")),
            llm_model=os.getenv("RAG_ANON_LLM_MODEL", "qwen3:0.6b"),
            llm_temperature=float(os.getenv("RAG_ANON_LLM_TEMPERATURE", "0.0")),
            llm_reasoning=os.getenv("RAG_ANON_LLM_REASONING", "false")
            .strip()
            .lower()
            in ("1", "true", "yes"),
            anonymizer_entities=tuple(entities_csv.split(",")),
            log_level=os.getenv("RAG_ANON_LOG_LEVEL"),
            log_level_http=os.getenv("RAG_ANON_LOG_LEVEL_HTTP"),
            log_level_presidio=os.getenv("RAG_ANON_LOG_LEVEL_PRESIDIO"),
            log_startup_env=os.getenv("RAG_ANON_LOG_STARTUP_ENV"),
        )


__all__ = ["DEFAULT_ANONYMIZER_ENTITIES", "PROJECT_ROOT", "Settings"]
