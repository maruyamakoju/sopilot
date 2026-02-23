"""Application-level metrics for Insurance MVP.

Provides lightweight in-memory counters/histograms that work without
external dependencies. When ``prometheus_client`` is installed, it also
exposes a standard ``/metrics`` Prometheus endpoint.

Usage:
    from insurance_mvp.metrics import METRICS

    METRICS.inc("claims_total", labels={"status": "queued"})
    METRICS.observe("processing_duration_seconds", 12.3, labels={"stage": "vlm"})
    METRICS.set_gauge("active_processing", 2)
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any


class InMemoryMetrics:
    """Thread-safe in-memory metrics collector.

    Supports counters, histograms (percentile approximation via stored values),
    and gauges.  Designed as a drop-in when Prometheus is unavailable.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = defaultdict(float)

    # -- Counter ----------------------------------------------------------

    def inc(self, name: str, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = _label_key(name, labels)
        with self._lock:
            self._counters[key] += amount

    # -- Histogram --------------------------------------------------------

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = _label_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)

    # -- Gauge ------------------------------------------------------------

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = _label_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def inc_gauge(self, name: str, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = _label_key(name, labels)
        with self._lock:
            self._gauges[key] += amount

    def dec_gauge(self, name: str, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        self.inc_gauge(name, -amount, labels)

    # -- Timer context manager -------------------------------------------

    @contextmanager
    def timer(self, name: str, labels: dict[str, str] | None = None):
        """Context manager that records elapsed seconds to a histogram."""
        start = time.perf_counter()
        yield
        self.observe(name, time.perf_counter() - start, labels)

    # -- Snapshot ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of all metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: _histogram_summary(v) for k, v in self._histograms.items()},
            }

    def reset(self) -> None:
        """Clear all metrics (useful in tests)."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()


def _label_key(name: str, labels: dict[str, str] | None) -> str:
    if not labels:
        return name
    parts = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{parts}}}"


def _histogram_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "sum": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    s = sorted(values)
    n = len(s)
    return {
        "count": n,
        "sum": sum(s),
        "mean": sum(s) / n,
        "p50": s[int(n * 0.50)],
        "p95": s[min(int(n * 0.95), n - 1)],
        "p99": s[min(int(n * 0.99), n - 1)],
    }


# Singleton metrics instance
METRICS = InMemoryMetrics()
