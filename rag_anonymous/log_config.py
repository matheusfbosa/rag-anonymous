"""Shared logging configuration for rag-anonymous and downstream packages."""

import logging
import os

DEFAULT_LEVEL = "INFO"
DEFAULT_HTTP_LEVEL = "WARNING"
DEFAULT_PRESIDIO_LEVEL = "ERROR"

LOG_FORMAT = "%(asctime)s %(message)s"
DATE_FORMAT = "%H:%M:%S"

_HTTP_LOGGERS = ("httpx", "httpcore", "urllib3")
_PRESIDIO_LOGGERS = ("presidio-analyzer", "presidio-anonymizer")


def configure_logging() -> None:
    """Configure root + noisy third-party loggers using env-driven levels."""
    root_level = _resolve_level(os.getenv("RAG_ANON_LOG_LEVEL"), DEFAULT_LEVEL)
    http_level = _resolve_level(os.getenv("RAG_ANON_LOG_LEVEL_HTTP"), DEFAULT_HTTP_LEVEL)
    presidio_level = _resolve_level(
        os.getenv("RAG_ANON_LOG_LEVEL_PRESIDIO"), DEFAULT_PRESIDIO_LEVEL
    )

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=root_level, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    else:
        root.setLevel(root_level)

    for name in _HTTP_LOGGERS:
        logging.getLogger(name).setLevel(http_level)
    for name in _PRESIDIO_LOGGERS:
        logging.getLogger(name).setLevel(presidio_level)


def _resolve_level(raw: str | None, default: str) -> int:
    """Map a string/numeric level (or None) to a stdlib logging int."""
    value = (raw or default).strip()
    if value.isdigit():
        return int(value)
    level = logging.getLevelName(value.upper())
    if isinstance(level, int):
        return level
    return logging.getLevelName(default)
