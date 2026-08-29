import logging
import os

from dotenv import find_dotenv

from rag_anonymous.config import Settings

DEFAULT_LEVEL = "INFO"
DEFAULT_HTTP_LEVEL = "WARNING"
DEFAULT_PRESIDIO_LEVEL = "ERROR"

LOG_FORMAT = "%(asctime)s %(message)s"
DATE_FORMAT = "%H:%M:%S"

_HTTP_LOGGERS = ("httpx", "httpcore", "urllib3", "elastic_transport")
_PRESIDIO_LOGGERS = ("presidio-analyzer", "presidio-anonymizer")

_ENV_PREFIXES = ("RAG_ANON_", "RAG_METRICS_")

_SENSITIVE_KEY_MARKERS = (
    "SECRET",
    "PASSWORD",
    "TOKEN",
    "API_KEY",
    "PRIVATE_KEY",
)


def configure_logging() -> None:
    s = Settings.load()
    root_level = _resolve_level(s.log_level, DEFAULT_LEVEL)
    http_level = _resolve_level(s.log_level_http, DEFAULT_HTTP_LEVEL)
    presidio_level = _resolve_level(s.log_level_presidio, DEFAULT_PRESIDIO_LEVEL)

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=root_level, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    else:
        root.setLevel(root_level)

    for name in _HTTP_LOGGERS:
        logging.getLogger(name).setLevel(http_level)
    for name in _PRESIDIO_LOGGERS:
        logging.getLogger(name).setLevel(presidio_level)

    _log_startup_env_configuration()


def _is_sensitive_env_key(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _SENSITIVE_KEY_MARKERS)


def _log_startup_env_configuration() -> None:
    if not _startup_env_logging_enabled():
        return

    log = logging.getLogger(__name__)
    dotenv_path = find_dotenv()
    if dotenv_path:
        log.info("Loaded .env")
    else:
        log.info("No .env file found")

    keys = sorted(k for k in os.environ if k.startswith(_ENV_PREFIXES))
    if not keys:
        log.info(
            "No env variables starting with prefixes=%s",
            _ENV_PREFIXES,
        )
        return

    log.info(".env variables:")
    for key in keys:
        log.info(
            "  %s=%s",
            key,
            _mask_env_value(key, os.environ[key]),
        )


def _mask_env_value(key: str, value: str) -> str:
    if not _is_sensitive_env_key(key):
        return value
    if not value:
        return "***"
    return f"{value[:2]}***"


def _resolve_level(raw: str | None, default: str) -> int:
    value = (raw or default).strip()
    if value.isdigit():
        return int(value)
    level = logging.getLevelName(value.upper())
    if isinstance(level, int):
        return level
    fallback = logging.getLevelName(default.upper())
    return fallback if isinstance(fallback, int) else logging.INFO


def _startup_env_logging_enabled() -> bool:
    return Settings.load().log_startup_env_enabled()
