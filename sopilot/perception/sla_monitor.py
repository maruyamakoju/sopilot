"""SLA Observatory — per-stage latency SLO compliance monitoring.

Phase 48: Sliding-window P50/P95/P99 per pipeline stage, SLO definition and
compliance evaluation. Pure Python + stdlib — no external dependencies.

Architecture::

    SLAMonitor.record("detection", 12.5)   # ms per frame
    SLAMonitor.record("frame_total", 45.0)
        ↓
    SLAMonitor.check_slos()       → list[SLOStatus]
    SLAMonitor.get_all_stage_stats() → {stage: {p50, p95, p99, ...}}
    SLAMonitor.get_stats()        → summary dict

Default SLOs (all configurable):
    - frame_total  P95 < 200 ms
    - frame_total  P99 < 500 ms
    - detection    P95 < 80 ms
    - tracking     P95 < 30 ms
    - zone_analysis P95 < 15 ms
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SLODefinition:
    """A single Service Level Objective for one pipeline stage."""

    name: str               # e.g. "frame_total_p95"
    stage: str              # stage name to look up in the monitor
    percentile: float       # 0.50, 0.95, or 0.99
    threshold_ms: float     # alert if current percentile exceeds this
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "percentile": self.percentile,
            "threshold_ms": self.threshold_ms,
            "description": self.description,
        }


@dataclass
class SLOStatus:
    """Live compliance status of one SLO."""

    slo: SLODefinition
    current_value_ms: float  # current percentile value for the stage
    compliant: bool
    sample_count: int
    violation_count: int     # cumulative violations since monitor creation

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.slo.name,
            "stage": self.slo.stage,
            "percentile": self.slo.percentile,
            "threshold_ms": self.slo.threshold_ms,
            "current_value_ms": round(self.current_value_ms, 3),
            "compliant": self.compliant,
            "sample_count": self.sample_count,
            "violation_count": self.violation_count,
        }


# ---------------------------------------------------------------------------
# Default SLO set
# ---------------------------------------------------------------------------

_DEFAULT_SLOS: list[SLODefinition] = [
    SLODefinition(
        name="frame_total_p95",
        stage="frame_total",
        percentile=0.95,
        threshold_ms=200.0,
        description="End-to-end frame processing P95 < 200 ms",
    ),
    SLODefinition(
        name="frame_total_p99",
        stage="frame_total",
        percentile=0.99,
        threshold_ms=500.0,
        description="End-to-end frame processing P99 < 500 ms",
    ),
    SLODefinition(
        name="detection_p95",
        stage="detection_ms",
        percentile=0.95,
        threshold_ms=80.0,
        description="Object detection stage P95 < 80 ms",
    ),
    SLODefinition(
        name="tracking_p95",
        stage="tracking_ms",
        percentile=0.95,
        threshold_ms=30.0,
        description="Multi-object tracking stage P95 < 30 ms",
    ),
    SLODefinition(
        name="zone_analysis_p95",
        stage="zone_analysis_ms",
        percentile=0.95,
        threshold_ms=15.0,
        description="Zone analysis stage P95 < 15 ms",
    ),
]


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------


def _percentile(sorted_data: list[float], p: float) -> float:
    """Linear-interpolation percentile on pre-sorted ascending data."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_data[0]
    idx = p * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac


# ---------------------------------------------------------------------------
# SLAMonitor
# ---------------------------------------------------------------------------


class SLAMonitor:
    """Sliding-window latency tracking and SLO compliance evaluation.

    Parameters
    ----------
    window_size:
        Maximum latency samples per stage to retain (FIFO eviction).
    slos:
        List of :class:`SLODefinition` to evaluate. Defaults to
        :data:`_DEFAULT_SLOS`.
    """

    def __init__(
        self,
        window_size: int = 1000,
        slos: list[SLODefinition] | None = None,
    ) -> None:
        self._window_size = max(10, window_size)
        self._slos: list[SLODefinition] = slos if slos is not None else list(_DEFAULT_SLOS)
        self._lock = threading.Lock()
        self._windows: dict[str, list[float]] = {}   # stage → latency samples (ms)
        self._violation_counts: dict[str, int] = {}  # slo.name → cumulative violations
        self._total_recorded = 0
        self._created_at = time.time()

    # ── Recording ──────────────────────────────────────────────────────────

    def record(self, stage: str, latency_ms: float) -> None:
        """Record a latency sample for the given stage.

        Parameters
        ----------
        stage:
            Stage identifier (e.g. ``"detection_ms"``, ``"frame_total"``).
        latency_ms:
            Latency for this frame in milliseconds.
        """
        with self._lock:
            if stage not in self._windows:
                self._windows[stage] = []
            w = self._windows[stage]
            w.append(float(latency_ms))
            if len(w) > self._window_size:
                # Trim to window; keep a list for O(1) append
                self._windows[stage] = w[-self._window_size:]
            self._total_recorded += 1

    # ── Percentile query ───────────────────────────────────────────────────

    def get_percentiles(self, stage: str) -> dict[str, float]:
        """P50/P95/P99/mean/max for a single stage.

        Returns an empty dict if no samples exist for the stage.
        """
        with self._lock:
            samples = list(self._windows.get(stage, []))
        if not samples:
            return {}
        s = sorted(samples)
        return {
            "p50_ms": round(_percentile(s, 0.50), 3),
            "p95_ms": round(_percentile(s, 0.95), 3),
            "p99_ms": round(_percentile(s, 0.99), 3),
            "mean_ms": round(sum(s) / len(s), 3),
            "max_ms": round(s[-1], 3),
            "min_ms": round(s[0], 3),
            "count": len(s),
        }

    def get_all_stage_stats(self) -> dict[str, dict[str, float]]:
        """Return percentile stats for every recorded stage."""
        with self._lock:
            stages = list(self._windows.keys())
        return {stage: self.get_percentiles(stage) for stage in stages if self.get_percentiles(stage)}

    # ── SLO evaluation ─────────────────────────────────────────────────────

    def check_slos(self) -> list[SLOStatus]:
        """Evaluate all SLOs against the current window.

        Updates cumulative :attr:`_violation_counts` for each violated SLO.

        Returns
        -------
        list[SLOStatus]
            One entry per SLO, sorted by compliance (violated first).
        """
        results: list[SLOStatus] = []
        with self._lock:
            windows_copy = {s: list(v) for s, v in self._windows.items()}
            vc = dict(self._violation_counts)

        for slo in self._slos:
            samples = windows_copy.get(slo.stage, [])
            count = len(samples)
            if count == 0:
                current = 0.0
                compliant = True  # no data → assume compliant
            else:
                s = sorted(samples)
                current = _percentile(s, slo.percentile)
                compliant = current <= slo.threshold_ms

            if not compliant:
                vc[slo.name] = vc.get(slo.name, 0) + 1

            results.append(SLOStatus(
                slo=slo,
                current_value_ms=round(current, 3),
                compliant=compliant,
                sample_count=count,
                violation_count=vc.get(slo.name, 0),
            ))

        with self._lock:
            self._violation_counts = vc

        results.sort(key=lambda s: (s.compliant, -s.current_value_ms))
        return results

    # ── Summary ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return a top-level summary of SLA state."""
        statuses = self.check_slos()
        violated = [s for s in statuses if not s.compliant]
        with self._lock:
            stages_tracked = list(self._windows.keys())
            total_recorded = self._total_recorded
        return {
            "slos_total": len(self._slos),
            "slos_compliant": len(statuses) - len(violated),
            "slos_violated": len(violated),
            "violations_by_slo": {s.slo.name: s.violation_count for s in statuses},
            "stages_tracked": stages_tracked,
            "total_samples_recorded": total_recorded,
            "window_size": self._window_size,
            "uptime_seconds": round(time.time() - self._created_at, 1),
        }

    def get_prometheus_lines(self) -> list[str]:
        """Return Prometheus text-format lines for all stage percentiles."""
        lines: list[str] = []
        all_stats = self.get_all_stage_stats()
        if not all_stats:
            return lines

        lines.append("# HELP sla_stage_latency_ms Stage processing latency in milliseconds")
        lines.append("# TYPE sla_stage_latency_ms gauge")
        for stage, pcts in sorted(all_stats.items()):
            for quantile_key, q_val in (("p50_ms", "0.5"), ("p95_ms", "0.95"), ("p99_ms", "0.99")):
                val = pcts.get(quantile_key, 0.0)
                lines.append(f'sla_stage_latency_ms{{stage="{stage}",quantile="{q_val}"}} {val:.3f}')

        statuses = self.check_slos()
        lines.append("# HELP sla_slo_compliant 1 if SLO is currently met, 0 if violated")
        lines.append("# TYPE sla_slo_compliant gauge")
        for s in statuses:
            lines.append(f'sla_slo_compliant{{slo="{s.slo.name}"}} {1 if s.compliant else 0}')

        return lines

    def reset(self) -> None:
        """Clear all windows and violation counts."""
        with self._lock:
            self._windows.clear()
            self._violation_counts.clear()
            self._total_recorded = 0
