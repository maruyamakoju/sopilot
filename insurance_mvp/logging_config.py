"""Structured JSON logging for Insurance MVP.

Provides a JSON formatter and correlation ID support using stdlib logging.
No external dependencies required.

Usage:
    from insurance_mvp.logging_config import configure_logging, set_correlation_id

    configure_logging(level="INFO", json_output=True)
    set_correlation_id(claim_id="claim_123", video_id="v_abc")
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone

# Correlation context — propagated to all log records in the current async task / thread.
_correlation: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar("_correlation", default=None)


def set_correlation_id(**kwargs: str) -> contextvars.Token:
    """Set correlation IDs (claim_id, video_id, etc.) for the current context."""
    current = (_correlation.get() or {}).copy()
    current.update(kwargs)
    return _correlation.set(current)


def clear_correlation_id() -> None:
    """Clear all correlation IDs."""
    _correlation.set(None)


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects.

    Fields: timestamp, level, logger, message, correlation IDs, exc_info.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach correlation IDs
        corr = _correlation.get()
        if corr:
            entry["correlation"] = corr

        # Extra fields (e.g. logger.info("msg", extra={"duration_sec": 1.2}))
        for key in ("duration_sec", "stage", "video_id", "claim_id", "clip_id"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val

        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, ensure_ascii=False, default=str)


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure root logger for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_output: If True, emit JSON; otherwise human-readable.
        log_file: Optional file path for log output (in addition to stderr).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates on re-configure
    for h in root.handlers[:]:
        root.removeHandler(h)

    if json_output:
        fmt = JSONFormatter()
    else:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    root.addHandler(stderr_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
