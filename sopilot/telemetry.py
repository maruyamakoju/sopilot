"""OpenTelemetry distributed tracing for SOPilot.

Provides:
- ``setup_telemetry(service_name, endpoint)`` — initialise OTel SDK
- ``get_tracer(name)`` — get a named tracer
- ``OtelLogFilter`` — injects active span's trace_id/span_id into LogRecords

If ``opentelemetry`` packages are not installed, all functions degrade
gracefully to no-ops so the core service continues to run without OTel.

Configuration via environment variables (set in Settings):
    SOPILOT_OTEL_ENABLED=true       — enable OTel (default: true)
    SOPILOT_OTEL_ENDPOINT=...       — OTLP gRPC endpoint (default: None → ConsoleSpanExporter)
    SOPILOT_OTEL_SERVICE_NAME=...   — service name tag (default: sopilot)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-import flag
_otel_available = False
_tracer_provider: Any = None


def setup_telemetry(service_name: str, endpoint: str | None) -> None:
    """Initialise the OpenTelemetry SDK.

    Parameters
    ----------
    service_name:
        Service name used in span/trace metadata.
    endpoint:
        OTLP gRPC exporter endpoint (e.g. ``"http://jaeger:4317"``).
        When ``None``, falls back to ``ConsoleSpanExporter`` so existing
        deployments without a collector still work.
    """
    global _otel_available, _tracer_provider

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        logger.info("opentelemetry packages not installed; tracing disabled")
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=endpoint)
            logger.info("OTel OTLP exporter → %s", endpoint)
        except ImportError:
            logger.warning("OTLP exporter not available; falling back to ConsoleSpanExporter")
            exporter = ConsoleSpanExporter()  # type: ignore[assignment]
    else:
        exporter = ConsoleSpanExporter()  # type: ignore[assignment]
        logger.info("OTel ConsoleSpanExporter active (no OTLP endpoint configured)")

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _tracer_provider = provider
    _otel_available = True
    logger.info("OpenTelemetry tracing initialised (service=%s)", service_name)


def shutdown_telemetry() -> None:
    """Flush and shut down the tracer provider on application exit."""
    global _tracer_provider
    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
        except Exception:
            logger.exception("Error shutting down OTel tracer provider")
        finally:
            _tracer_provider = None


def get_tracer(name: str) -> Any:
    """Return a named tracer (no-op tracer if OTel is not available)."""
    if not _otel_available:
        return _NoOpTracer()
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except Exception:
        return _NoOpTracer()


class _NoOpTracer:
    """Minimal no-op tracer for when OTel is disabled."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> Any:
        from contextlib import contextmanager

        @contextmanager  # type: ignore[misc]
        def _noop() -> Any:
            yield _NoOpSpan()

        return _noop()


class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass


class OtelLogFilter(logging.Filter):
    """Inject ``trace_id`` and ``span_id`` from the active OTel span.

    Attach to the root log handler so every LogRecord emitted during
    an active span includes the trace context, enabling correlation
    between structured logs and distributed traces.

    When OTel is not available or no span is active, ``trace_id`` and
    ``span_id`` are set to ``"0"`` so downstream JSON still has the keys.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from opentelemetry import trace as otel_trace

            span = otel_trace.get_current_span()
            ctx = span.get_span_context()
            if ctx and ctx.is_valid:
                record.trace_id = format(ctx.trace_id, "032x")
                record.span_id = format(ctx.span_id, "016x")
            else:
                record.trace_id = "0"
                record.span_id = "0"
        except Exception:
            record.trace_id = "0"
            record.span_id = "0"
        return True


def instrument_app(app: Any) -> None:
    """Instrument a FastAPI app and HTTPX client with OTel if available."""
    if not _otel_available:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPIInstrumentor attached")
    except ImportError:
        logger.debug("FastAPIInstrumentor not available")
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTPXClientInstrumentor attached")
    except ImportError:
        logger.debug("HTTPXClientInstrumentor not available")
