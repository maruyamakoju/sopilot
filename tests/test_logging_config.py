"""Tests for sopilot.logging_config — JsonFormatter, log_context, setup_logging."""
import json
import logging

from sopilot.logging_config import JsonFormatter, log_context, setup_logging

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(msg: str = "hello", level: int = logging.INFO, **extra) -> logging.LogRecord:
    """Create a LogRecord with optional extra attributes."""
    record = logging.LogRecord(
        name="test.logger",
        level=level,
        pathname="test.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


# ---------------------------------------------------------------------------
# JsonFormatter
# ---------------------------------------------------------------------------

class TestJsonFormatter:
    def test_outputs_valid_json(self):
        fmt = JsonFormatter()
        record = _make_record("test message")
        output = fmt.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_includes_required_fields(self):
        fmt = JsonFormatter()
        record = _make_record("test message")
        output = fmt.format(record)
        parsed = json.loads(output)
        assert "ts" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "msg" in parsed

    def test_msg_field_value(self):
        fmt = JsonFormatter()
        record = _make_record("specific text")
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["msg"] == "specific text"

    def test_level_field_value(self):
        fmt = JsonFormatter()
        record = _make_record("x", level=logging.WARNING)
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "WARNING"

    def test_logger_field_value(self):
        fmt = JsonFormatter()
        record = _make_record("x")
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["logger"] == "test.logger"

    def test_extra_known_keys_promoted(self):
        fmt = JsonFormatter()
        record = _make_record("x", job_id=42, video_id=7)
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["job_id"] == 42
        assert parsed["video_id"] == 7

    def test_unknown_extra_keys_not_included(self):
        fmt = JsonFormatter()
        record = _make_record("x", random_key="val")
        output = fmt.format(record)
        parsed = json.loads(output)
        assert "random_key" not in parsed

    def test_ts_is_iso_format(self):
        fmt = JsonFormatter()
        record = _make_record("x")
        output = fmt.format(record)
        parsed = json.loads(output)
        ts = parsed["ts"]
        # ISO timestamps contain 'T' separator
        assert "T" in ts


# ---------------------------------------------------------------------------
# log_context
# ---------------------------------------------------------------------------

class TestLogContext:
    def test_injects_context_fields(self):
        fmt = JsonFormatter()
        with log_context(job_id=99, task_id="my_task"):
            record = _make_record("inside context")
            output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["job_id"] == 99
        assert parsed["task_id"] == "my_task"

    def test_restores_previous_context_on_exit(self):
        fmt = JsonFormatter()
        with log_context(job_id=10):
            with log_context(job_id=20):
                record_inner = _make_record("inner")
                inner = json.loads(fmt.format(record_inner))
                assert inner["job_id"] == 20
            record_outer = _make_record("outer")
            outer = json.loads(fmt.format(record_outer))
            assert outer["job_id"] == 10
        record_after = _make_record("after")
        after = json.loads(fmt.format(record_after))
        # After exiting all contexts, job_id should not be present
        assert "job_id" not in after

    def test_nested_context_merges(self):
        fmt = JsonFormatter()
        with log_context(task_id="a"), log_context(job_id=5):
            record = _make_record("nested")
            output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["task_id"] == "a"
        assert parsed["job_id"] == 5

    def test_none_values_filtered(self):
        fmt = JsonFormatter()
        with log_context(job_id=None, task_id="real"):
            record = _make_record("x")
            output = fmt.format(record)
        parsed = json.loads(output)
        assert "job_id" not in parsed
        assert parsed["task_id"] == "real"


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_setup_json_mode(self):
        setup_logging(level="DEBUG", use_json=True)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) >= 1
        handler = root.handlers[0]
        assert isinstance(handler.formatter, JsonFormatter)

    def test_setup_text_mode(self):
        setup_logging(level="INFO", use_json=False)
        root = logging.getLogger()
        assert root.level == logging.INFO
        assert len(root.handlers) >= 1
        handler = root.handlers[0]
        assert not isinstance(handler.formatter, JsonFormatter)

    def test_setup_sets_level(self):
        setup_logging(level="ERROR", use_json=True)
        root = logging.getLogger()
        assert root.level == logging.ERROR

    def test_suppresses_noisy_loggers(self):
        setup_logging(level="DEBUG", use_json=True)
        uvicorn_access = logging.getLogger("uvicorn.access")
        assert uvicorn_access.level >= logging.WARNING
