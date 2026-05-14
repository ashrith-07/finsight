"""Structured JSON logging with per-request correlation IDs.

Components call :func:`get_logger` and emit normal ``logger.info(...)`` /
``logger.error(...)`` lines. :func:`setup_logging` installs the JSON formatter
and silences noisy upstream libraries. The correlation ID rides on a
``ContextVar`` so it flows through async tasks (each request resets it).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextvars import ContextVar

correlation_id_var: ContextVar[str] = ContextVar(
    "correlation_id",
    default="no-correlation-id",
)

# LogRecord attributes that should never be echoed into the JSON envelope —
# everything else passed via ``extra=`` is surfaced as a structured field.
_LOG_RECORD_RESERVED: frozenset[str] = frozenset({
    "name", "msg", "args", "levelname", "levelno",
    "pathname", "filename", "module", "exc_info",
    "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread",
    "threadName", "processName", "process", "message",
    "taskName",
})


class JSONFormatter(logging.Formatter):
    """Render every LogRecord as a single-line JSON object."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        # ISO-8601 in UTC with millisecond precision and a trailing 'Z'.
        ts = time.gmtime(record.created)
        return f"{time.strftime('%Y-%m-%dT%H:%M:%S', ts)}.{int(record.msecs):03d}Z"

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, object] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "correlation_id": correlation_id_var.get(),
            "component": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _LOG_RECORD_RESERVED or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                log_data[key] = value
            except (TypeError, ValueError):
                log_data[key] = repr(value)
        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)


def setup_logging() -> None:
    """Install the JSON handler on the root logger. Idempotent."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]

    # Tame chatty third-party loggers so the JSON stream stays useful.
    for noisy in ("httpx", "httpcore", "yfinance", "urllib3", "openai", "groq"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger. Names starting with ``valura.`` are kept as-is."""
    if name.startswith("valura."):
        return logging.getLogger(name)
    return logging.getLogger(f"valura.{name}")


def new_correlation_id() -> str:
    """Short request-scoped identifier surfaced on every log line of a request."""
    return f"req_{uuid.uuid4().hex[:8]}"


__all__ = [
    "JSONFormatter",
    "correlation_id_var",
    "get_logger",
    "new_correlation_id",
    "setup_logging",
]
