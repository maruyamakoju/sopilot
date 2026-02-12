"""
Production-grade structured logging configuration.

Features:
- JSON output for production (parseable by log aggregators)
- Human-readable output for development
- Request correlation IDs
- Performance timing
- Contextual fields (job_id, task_id, video_id)

Usage:
    from sopilot.logging_config import configure_logging, get_logger

    configure_logging(json_logs=True)  # Production
    logger = get_logger(__name__)
    logger.info("processing_video", video_id=123, task_id="task1")
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

try:
    import structlog

    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None  # type: ignore


def configure_logging(
    json_logs: bool | None = None,
    log_level: str | None = None,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        json_logs: If True, output JSON logs. If False, human-readable.
                   If None, auto-detect from SOPILOT_LOG_FORMAT env var.
        log_level: Log level (DEBUG, INFO, WARNING, ERROR). If None, use SOPILOT_LOG_LEVEL env.
    """
    if not STRUCTLOG_AVAILABLE:
        # Fallback to standard logging if structlog not installed
        level_str = log_level or os.getenv("SOPILOT_LOG_LEVEL", "INFO")
        level = getattr(logging, level_str.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            stream=sys.stdout,
        )
        return

    # Determine output format
    if json_logs is None:
        format_env = os.getenv("SOPILOT_LOG_FORMAT", "console").strip().lower()
        json_logs = format_env == "json"

    # Determine log level
    level_str = log_level or os.getenv("SOPILOT_LOG_LEVEL", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)

    # Configure stdlib logging to play nice with structlog
    logging.basicConfig(
        format="%(message)s",
        level=level,
        stream=sys.stdout,
    )

    # Structlog processors
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        # Production: JSON output
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Human-readable console output
        processors = shared_processors + [
            structlog.processors.ExceptionPrettyPrinter(),
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance (structlog.BoundLogger or stdlib logging.Logger)
    """
    if STRUCTLOG_AVAILABLE and structlog is not None:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


# Context manager for request correlation
class LogContext:
    """
    Context manager for adding contextual fields to all logs within a scope.

    Usage:
        with LogContext(request_id="req123", user="alice"):
            logger.info("processing_request")
            # Logs will include request_id and user automatically
    """

    def __init__(self, **kwargs):
        self.context = kwargs
        self._token = None

    def __enter__(self):
        if STRUCTLOG_AVAILABLE and structlog is not None:
            self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if STRUCTLOG_AVAILABLE and structlog is not None and self._token is not None:
            structlog.contextvars.unbind_contextvars(*self.context.keys())
        return False


# Decorator for timing function execution
def log_execution_time(logger: Any = None, event_name: str | None = None):
    """
    Decorator to log function execution time.

    Usage:
        @log_execution_time(logger=logger, event_name="score_job")
        def run_score_job(job_id):
            ...
    """
    import time
    from functools import wraps

    def decorator(func):
        nonlocal logger, event_name
        if logger is None:
            logger = get_logger(func.__module__)
        if event_name is None:
            event_name = f"{func.__name__}_execution"

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                logger.info(
                    event_name,
                    duration_sec=round(duration, 3),
                    success=True,
                )
                return result
            except Exception as e:
                duration = time.perf_counter() - start
                logger.error(
                    event_name,
                    duration_sec=round(duration, 3),
                    success=False,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        return wrapper

    return decorator
