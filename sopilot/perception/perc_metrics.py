"""Perception Engine metrics registry.

Thread-safe counters and histograms that the PerceptionEngine increments
during operation. The /metrics endpoint polls these to produce Prometheus
text-format output.

Usage::
    from sopilot.perception.perc_metrics import get_registry
    reg = get_registry()
    reg.record_frame(processing_ms=18.9)
    reg.record_violation(severity="warning")
    reg.record_detection()
    reg.record_vlm_call()
    reg.record_anomaly(detector="behavioral")
"""
from __future__ import annotations
import threading


class PercMetricsRegistry:
    """Holds all perception-engine metric counters (no external deps)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Counters
        self.frames_total: int = 0
        self.detections_total: int = 0
        self.vlm_calls_total: int = 0
        # Histogram buckets (processing latency)
        self._processing_seconds_sum: float = 0.0
        self._processing_seconds_count: int = 0
        # Labeled counters (dict[label, count])
        self.violations_by_severity: dict[str, int] = {}
        self.anomaly_events_by_detector: dict[str, int] = {}
        self.events_by_type: dict[str, int] = {}

    def record_frame(self, processing_ms: float = 0.0) -> None:
        with self._lock:
            self.frames_total += 1
            self._processing_seconds_sum += processing_ms / 1000.0
            self._processing_seconds_count += 1

    def record_detection(self, count: int = 1) -> None:
        with self._lock:
            self.detections_total += count

    def record_vlm_call(self) -> None:
        with self._lock:
            self.vlm_calls_total += 1

    def record_violation(self, severity: str = "warning") -> None:
        with self._lock:
            self.violations_by_severity[severity] = self.violations_by_severity.get(severity, 0) + 1

    def record_anomaly(self, detector: str = "unknown") -> None:
        with self._lock:
            self.anomaly_events_by_detector[detector] = self.anomaly_events_by_detector.get(detector, 0) + 1

    def record_event(self, event_type: str = "unknown") -> None:
        with self._lock:
            self.events_by_type[event_type] = self.events_by_type.get(event_type, 0) + 1

    def get_processing_seconds_avg(self) -> float:
        with self._lock:
            if self._processing_seconds_count == 0:
                return 0.0
            return self._processing_seconds_sum / self._processing_seconds_count

    def get_snapshot(self) -> dict:
        """Return a consistent snapshot for metrics export."""
        with self._lock:
            return {
                "frames_total": self.frames_total,
                "detections_total": self.detections_total,
                "vlm_calls_total": self.vlm_calls_total,
                "processing_seconds_sum": self._processing_seconds_sum,
                "processing_seconds_count": self._processing_seconds_count,
                "violations_by_severity": dict(self.violations_by_severity),
                "anomaly_events_by_detector": dict(self.anomaly_events_by_detector),
                "events_by_type": dict(self.events_by_type),
            }

    def reset(self) -> None:
        """Reset all counters (useful in tests)."""
        with self._lock:
            self.frames_total = 0
            self.detections_total = 0
            self.vlm_calls_total = 0
            self._processing_seconds_sum = 0.0
            self._processing_seconds_count = 0
            self.violations_by_severity.clear()
            self.anomaly_events_by_detector.clear()
            self.events_by_type.clear()


_REGISTRY: PercMetricsRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_registry() -> PercMetricsRegistry:
    """Return the global singleton PercMetricsRegistry (created on first call)."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = PercMetricsRegistry()
        return _REGISTRY


def reset_registry() -> None:
    """Reset the global registry (used in tests to isolate state)."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is not None:
            _REGISTRY.reset()
        else:
            _REGISTRY = PercMetricsRegistry()
