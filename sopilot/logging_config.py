"""Structured JSON logging for SOPilot.

Usage
-----
In production (default)::

    setup_logging(level="INFO", use_json=True)

This formats every log record as a single-line JSON object, suitable for
log-aggregation stacks (ELK, CloudWatch, Loki, etc.).

Injecting per-request context::

    with log_context(job_id=42, task_id="pilot"):
        logger.info("scoring started")   # output includes job_id + task_id

Passing ad-hoc fields::

    logger.info("clip embedded", extra={"video_id": 7, "duration_ms": 38})
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# Per-task/request context carried via contextvars so it does not need to
# be threaded through every call site manually.
_LOG_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar("_log_context", default=None)

# Extra record attributes we promote to top-level JSON keys.
_EXTRA_KEYS = frozenset(
    {
        "request_id",
        "job_id",
        "video_id",
        "task_id",
        "embedder",
        "duration_ms",
        "clip_count",
        "score",
        "status",
        "video_path",
    }
)


class JsonFormatter(logging.Formatter):
    """Formats :class:`logging.LogRecord` objects as newline-delimited JSON.

    Each line contains at minimum: ``ts``, ``level``, ``logger``, ``msg``.
    Additional keys are merged from :func:`log_context` and from ``extra=``
    kwargs passed to the logging call.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Per-request context
        ctx = _LOG_CONTEXT.get() or {}
        if ctx:
            payload.update(ctx)

        # Extra fields passed via ``extra={"job_id": 5}``
        for key in _EXTRA_KEYS:
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        # Exception details
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exc"] = record.exc_text

        return json.dumps(payload, ensure_ascii=False)


class log_context:
    """Context manager that injects key/value pairs into all log records within
    its scope, even across async awaits (uses :mod:`contextvars`).

    Example::

        with log_context(job_id=42, task_id="pilot_task"):
            logger.info("score job started")
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self._token: Any = None

    def __enter__(self) -> log_context:
        existing = _LOG_CONTEXT.get() or {}
        self._token = _LOG_CONTEXT.set({**existing, **self._kwargs})
        return self

    def __exit__(self, *_: Any) -> None:
        if self._token is not None:
            _LOG_CONTEXT.reset(self._token)


def setup_logging(level: str = "INFO", use_json: bool = True) -> None:
    """Configure the root logger.

    Parameters
    ----------
    level:
        Log level name, e.g. ``"INFO"``, ``"DEBUG"``.
    use_json:
        When ``True`` (default), all output is newline-delimited JSON.
        When ``False``, uses a human-readable text format (useful for local
        development when piping to a terminal).
    """
    formatter: logging.Formatter
    formatter = JsonFormatter() if use_json else logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Suppress noisy access logs and HTTP client debug chatter.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
