"""Tests for logging_config module (P2-3).

Covers configure_logging, get_logger, LogContext, log_execution_time.
"""

import logging

import pytest

from sopilot.logging_config import (
    LogContext,
    configure_logging,
    get_logger,
    log_execution_time,
    STRUCTLOG_AVAILABLE,
)


class TestConfigureLogging:
    """Test configure_logging function."""

    def test_configure_default(self):
        """Default configuration succeeds."""
        configure_logging()

    def test_configure_json(self):
        """JSON format configuration succeeds."""
        configure_logging(json_logs=True)

    def test_configure_console(self):
        """Console format configuration succeeds."""
        configure_logging(json_logs=False)

    def test_configure_log_level_debug(self):
        """DEBUG level configuration succeeds."""
        configure_logging(log_level="DEBUG")

    def test_configure_log_level_warning(self):
        """WARNING level configuration succeeds."""
        configure_logging(log_level="WARNING")

    def test_configure_log_level_error(self):
        """ERROR level configuration succeeds."""
        configure_logging(log_level="ERROR")

    def test_configure_with_env_var(self, monkeypatch):
        """Configuration respects environment variables."""
        monkeypatch.setenv("SOPILOT_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("SOPILOT_LOG_FORMAT", "json")
        configure_logging()

    def test_configure_env_console(self, monkeypatch):
        """Console format via env var."""
        monkeypatch.setenv("SOPILOT_LOG_FORMAT", "console")
        configure_logging()


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_returns_logger(self):
        """get_logger returns a usable logger object."""
        logger = get_logger("test_module")
        assert logger is not None

    def test_get_logger_can_log(self):
        """Logger can log messages without error."""
        logger = get_logger("test_can_log")
        logger.info("test message")

    def test_get_logger_different_names(self):
        """Different names produce different loggers."""
        logger1 = get_logger("module_a")
        logger2 = get_logger("module_b")
        assert logger1 is not logger2


class TestLogContext:
    """Test LogContext context manager."""

    def test_context_manager_enters(self):
        """LogContext can be used as context manager."""
        with LogContext(request_id="test123"):
            pass

    def test_context_manager_with_kwargs(self):
        """LogContext accepts arbitrary kwargs."""
        with LogContext(request_id="req1", user="alice", job_id="j1"):
            pass

    def test_context_manager_returns_self(self):
        """LogContext returns self on enter."""
        ctx = LogContext(key="value")
        with ctx as c:
            assert c is ctx

    def test_context_manager_stores_context(self):
        """LogContext stores provided context."""
        ctx = LogContext(request_id="req1", user="bob")
        assert ctx.context == {"request_id": "req1", "user": "bob"}

    def test_nested_contexts(self):
        """Nested LogContexts don't crash."""
        with LogContext(request_id="outer"):
            with LogContext(job_id="inner"):
                pass

    def test_context_exit_on_exception(self):
        """LogContext cleans up on exception."""
        with pytest.raises(ValueError):
            with LogContext(request_id="test"):
                raise ValueError("test error")


class TestLogExecutionTime:
    """Test log_execution_time decorator."""

    def test_decorator_preserves_return(self):
        """Decorated function returns its value."""
        @log_execution_time()
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_decorator_preserves_exception(self):
        """Decorated function re-raises exceptions."""
        @log_execution_time()
        def fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            fail()

    def test_decorator_with_custom_logger(self):
        """Custom logger can be passed."""
        logger = get_logger("custom")

        @log_execution_time(logger=logger, event_name="custom_op")
        def noop():
            return 42

        assert noop() == 42

    def test_decorator_with_event_name(self):
        """Custom event name is accepted."""
        @log_execution_time(event_name="my_event")
        def noop():
            pass

        noop()

    def test_decorator_preserves_function_name(self):
        """Decorated function retains its name."""
        @log_execution_time()
        def my_function():
            pass

        assert my_function.__name__ == "my_function"
