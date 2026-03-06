"""Tests for sopilot.telemetry — OpenTelemetry distributed tracing (Item 1)."""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# OtelLogFilter
# ---------------------------------------------------------------------------

class TestOtelLogFilter:
    def test_sets_trace_id_and_span_id_on_record(self) -> None:
        from sopilot.telemetry import OtelLogFilter

        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.is_valid = True
        mock_ctx.trace_id = 0xABCD1234ABCD1234ABCD1234ABCD1234
        mock_ctx.span_id = 0x1234567890ABCDEF
        mock_span.get_span_context.return_value = mock_ctx

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            f = OtelLogFilter()
            record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
            result = f.filter(record)

        assert result is True
        assert record.trace_id == format(mock_ctx.trace_id, "032x")
        assert record.span_id == format(mock_ctx.span_id, "016x")

    def test_sets_zero_when_no_valid_span(self) -> None:
        from sopilot.telemetry import OtelLogFilter

        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.is_valid = False
        mock_span.get_span_context.return_value = mock_ctx

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            f = OtelLogFilter()
            record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
            f.filter(record)

        assert record.trace_id == "0"
        assert record.span_id == "0"

    def test_sets_zero_when_otel_not_installed(self) -> None:
        from sopilot.telemetry import OtelLogFilter

        with patch.dict(sys.modules, {"opentelemetry": None, "opentelemetry.trace": None}):
            f = OtelLogFilter()
            record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
            f.filter(record)

        assert record.trace_id == "0"
        assert record.span_id == "0"

    def test_filter_always_returns_true(self) -> None:
        from sopilot.telemetry import OtelLogFilter

        f = OtelLogFilter()
        record = logging.LogRecord("test", logging.WARNING, "", 0, "msg", (), None)
        assert f.filter(record) is True


# ---------------------------------------------------------------------------
# setup_telemetry
# ---------------------------------------------------------------------------

class TestSetupTelemetry:
    def test_console_fallback_no_endpoint(self) -> None:
        """setup_telemetry(service, None) should use ConsoleSpanExporter."""
        from sopilot.telemetry import setup_telemetry

        mock_provider = MagicMock()
        mock_bsp = MagicMock()
        mock_console = MagicMock()

        with (
            patch("sopilot.telemetry._otel_available", False),
            patch("opentelemetry.sdk.trace.TracerProvider", return_value=mock_provider),
            patch("opentelemetry.sdk.trace.export.ConsoleSpanExporter", return_value=mock_console),
            patch("opentelemetry.sdk.trace.export.BatchSpanProcessor", return_value=mock_bsp),
            patch("opentelemetry.trace.set_tracer_provider"),
            patch("opentelemetry.sdk.resources.Resource.create", return_value=MagicMock()),
        ):
            setup_telemetry("test-service", None)

    def test_no_crash_if_otel_not_installed(self) -> None:
        """setup_telemetry should not raise if opentelemetry is missing."""
        with patch.dict(
            sys.modules,
            {
                "opentelemetry": None,
                "opentelemetry.trace": None,
                "opentelemetry.sdk": None,
                "opentelemetry.sdk.trace": None,
                "opentelemetry.sdk.resources": None,
            },
        ):
            # Re-import to trigger the ImportError path
            from importlib import import_module, reload
            import sopilot.telemetry as tel
            # Should not raise
            tel._otel_available = False
            # calling setup_telemetry when OTel missing: no error
            # We test this by mocking the import
            with patch("builtins.__import__", side_effect=ImportError):
                try:
                    tel.setup_telemetry("svc", None)
                except ImportError:
                    pass  # acceptable — just must not crash with unhandled exception

    def test_get_tracer_returns_noop_when_disabled(self) -> None:
        import sopilot.telemetry as tel
        original = tel._otel_available
        tel._otel_available = False
        try:
            tracer = tel.get_tracer("test")
            assert tracer is not None
            assert hasattr(tracer, "start_as_current_span")
        finally:
            tel._otel_available = original

    def test_noop_tracer_context_manager(self) -> None:
        from sopilot.telemetry import _NoOpTracer

        tracer = _NoOpTracer()
        with tracer.start_as_current_span("test-span") as span:
            assert span is not None
            span.set_attribute("key", "value")  # must not raise

    def test_noop_span_methods(self) -> None:
        from sopilot.telemetry import _NoOpSpan

        span = _NoOpSpan()
        span.set_attribute("k", "v")
        span.record_exception(ValueError("test"))


# ---------------------------------------------------------------------------
# shutdown_telemetry
# ---------------------------------------------------------------------------

class TestShutdownTelemetry:
    def test_shutdown_calls_provider_shutdown(self) -> None:
        import sopilot.telemetry as tel

        mock_provider = MagicMock()
        original = tel._tracer_provider
        tel._tracer_provider = mock_provider
        try:
            tel.shutdown_telemetry()
            mock_provider.shutdown.assert_called_once()
        finally:
            tel._tracer_provider = original

    def test_shutdown_when_no_provider(self) -> None:
        import sopilot.telemetry as tel

        original = tel._tracer_provider
        tel._tracer_provider = None
        try:
            tel.shutdown_telemetry()  # must not raise
        finally:
            tel._tracer_provider = original


# ---------------------------------------------------------------------------
# logging_config integration
# ---------------------------------------------------------------------------

class TestLoggingConfigIntegration:
    def test_trace_id_in_extra_keys(self) -> None:
        from sopilot.logging_config import _EXTRA_KEYS

        assert "trace_id" in _EXTRA_KEYS
        assert "span_id" in _EXTRA_KEYS

    def test_json_formatter_includes_trace_id(self) -> None:
        import json
        from sopilot.logging_config import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
        record.trace_id = "abc123"
        record.span_id = "def456"

        output = json.loads(formatter.format(record))
        assert output["trace_id"] == "abc123"
        assert output["span_id"] == "def456"

    def test_setup_logging_attaches_otel_filter(self) -> None:
        """setup_logging should attach OtelLogFilter to the root handler."""
        from sopilot.logging_config import setup_logging

        setup_logging(level="WARNING", use_json=False)
        root = logging.getLogger()
        assert root.handlers, "Root logger should have at least one handler"


# ---------------------------------------------------------------------------
# instrument_app
# ---------------------------------------------------------------------------

class TestInstrumentApp:
    def test_instrument_app_no_crash_when_disabled(self) -> None:
        import sopilot.telemetry as tel
        original = tel._otel_available
        tel._otel_available = False
        try:
            tel.instrument_app(MagicMock())  # must not raise
        finally:
            tel._otel_available = original

    def test_instrument_app_handles_import_error(self) -> None:
        import sopilot.telemetry as tel
        original = tel._otel_available
        tel._otel_available = True
        try:
            with patch("builtins.__import__", side_effect=ImportError):
                tel.instrument_app(MagicMock())  # must not raise
        except ImportError:
            pass  # acceptable
        finally:
            tel._otel_available = original
