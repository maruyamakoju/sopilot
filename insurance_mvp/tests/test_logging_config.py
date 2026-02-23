"""Tests for insurance_mvp.logging_config module."""

import json
import logging

from insurance_mvp.logging_config import (
    JSONFormatter,
    clear_correlation_id,
    configure_logging,
    set_correlation_id,
)


class TestJSONFormatter:
    def test_basic_format(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello %s", args=("world",), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "hello world"
        assert "timestamp" in data

    def test_correlation_ids_included(self):
        formatter = JSONFormatter()
        token = set_correlation_id(video_id="v123", claim_id="c456")
        try:
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg="test", args=(), exc_info=None,
            )
            output = formatter.format(record)
            data = json.loads(output)
            assert data["correlation"]["video_id"] == "v123"
            assert data["correlation"]["claim_id"] == "c456"
        finally:
            clear_correlation_id()

    def test_no_correlation_when_empty(self):
        clear_correlation_id()
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "correlation" not in data

    def test_exception_info_included(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="", lineno=0,
                msg="failed", args=(), exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        data = json.loads(output)
        assert "exception" in data
        assert "ValueError" in data["exception"]


class TestConfigureLogging:
    def test_configure_json_output(self):
        configure_logging(level="WARNING", json_output=True)
        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert any(isinstance(h.formatter, JSONFormatter) for h in root.handlers)
        # Restore default
        configure_logging(level="INFO", json_output=False)

    def test_configure_human_output(self):
        configure_logging(level="DEBUG", json_output=False)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        configure_logging(level="INFO", json_output=False)

    def test_reconfigure_removes_old_handlers(self):
        configure_logging(level="INFO")
        count1 = len(logging.getLogger().handlers)
        configure_logging(level="INFO")
        count2 = len(logging.getLogger().handlers)
        assert count1 == count2  # Should not accumulate


class TestCorrelationID:
    def test_set_and_clear(self):
        set_correlation_id(video_id="v1")
        set_correlation_id(claim_id="c1")  # Should merge
        clear_correlation_id()
        # After clear, no correlation should exist
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test", args=(), exc_info=None,
        )
        data = json.loads(formatter.format(record))
        assert "correlation" not in data
