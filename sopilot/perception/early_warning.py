"""Phase 15: 予兆検知 (Early Warning Risk Score).

EarlyWarningEngine は過去の学習シグナルから detector ごとのリスクスコア [0,1] を計算し、
異常が発生する「前」に警告を提供する。

リスクスコアの構成要素 (3軸):
    1. sigma ドリフト速度 (σ/min) — σ値の急激な変化は感度が不安定な証拠
    2. FP 率 (AnomalyTuner pair_stats 由来) — 誤検知が多い detector は信頼性が低い
    3. 異常バースト頻度 (anomalies/min) — 短期間の異常集中は予兆になりやすい

スコア:
    risk = 0.40 × drift_norm + 0.40 × fp_norm + 0.20 × burst_norm

リスクレベル:
    low    : 0.0 ≤ risk < 0.3
    medium : 0.3 ≤ risk < 0.6
    high   : 0.6 ≤ risk ≤ 1.0
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_DETECTORS = ("behavioral", "spatial", "temporal", "interaction")


# ---------------------------------------------------------------------------
# EarlyWarningEngine
# ---------------------------------------------------------------------------


class EarlyWarningEngine:
    """Per-detector risk score engine for early anomaly warning (Phase 15).

    Args:
        burst_window_seconds: Sliding window for anomaly burst counting (default 300s = 5min).
        drift_ema_alpha: EMA smoothing factor for sigma drift velocity (0–1).
        max_drift_rate: Drift velocity (σ/min) at which the drift component reaches 1.0.
        max_fp_rate: FP rate at which the FP component reaches 1.0.
        max_burst_rate: Anomaly burst rate (per min) at which the burst component reaches 1.0.
    """

    W_SIGMA: float = 0.40
    W_FP: float = 0.40
    W_BURST: float = 0.20

    DRIFT_EMA_ALPHA: float = 0.30
    MAX_DRIFT_RATE: float = 0.50   # σ/min → norm=1.0
    MAX_FP_RATE: float = 0.70      # FP fraction → norm=1.0
    MAX_BURST_RATE: float = 5.0    # anomalies/min → norm=1.0

    BURST_WINDOW_SECONDS: int = 300  # 5-minute sliding window

    def __init__(
        self,
        burst_window_seconds: int = BURST_WINDOW_SECONDS,
        drift_ema_alpha: float = DRIFT_EMA_ALPHA,
        max_drift_rate: float = MAX_DRIFT_RATE,
        max_fp_rate: float = MAX_FP_RATE,
        max_burst_rate: float = MAX_BURST_RATE,
    ) -> None:
        self._burst_window_seconds = burst_window_seconds
        self._drift_ema_alpha = drift_ema_alpha
        self._max_drift_rate = max_drift_rate
        self._max_fp_rate = max_fp_rate
        self._max_burst_rate = max_burst_rate

        # detector → EMA of drift velocity (σ/min)
        self._drift_velocity: dict[str, float] = {}
        # detector → timestamp of last sigma change
        self._drift_last_ts: dict[str, float] = {}
        # detector → list of anomaly event timestamps (for burst window)
        self._anomaly_timestamps: dict[str, list[float]] = {}

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Feed signals
    # ------------------------------------------------------------------

    def observe_sigma_change(
        self,
        detector: str,
        old_sigma: float,
        new_sigma: float,
        timestamp: float | None = None,
    ) -> None:
        """Record a sigma adjustment event.

        Updates the EMA drift velocity for the given detector.

        Args:
            detector: Detector name (e.g. "behavioral").
            old_sigma: Previous sigma value.
            new_sigma: New sigma value.
            timestamp: Unix timestamp of the change (defaults to now).
        """
        ts = timestamp if timestamp is not None else time.time()
        delta = abs(new_sigma - old_sigma)

        with self._lock:
            last_ts = self._drift_last_ts.get(detector, ts)
            dt_min = max((ts - last_ts) / 60.0, 1e-6)
            velocity = delta / dt_min
            alpha = self._drift_ema_alpha
            prev = self._drift_velocity.get(detector, 0.0)
            self._drift_velocity[detector] = alpha * velocity + (1.0 - alpha) * prev
            self._drift_last_ts[detector] = ts

    def observe_anomaly(
        self,
        detector: str,
        timestamp: float | None = None,
    ) -> None:
        """Record an anomaly event for the given detector.

        Used to compute anomaly burst frequency within the sliding window.

        Args:
            detector: Detector name.
            timestamp: Unix timestamp (defaults to now).
        """
        ts = timestamp if timestamp is not None else time.time()
        cutoff = ts - self._burst_window_seconds
        with self._lock:
            buf = self._anomaly_timestamps.setdefault(detector, [])
            # Prune expired entries
            self._anomaly_timestamps[detector] = [t for t in buf if t >= cutoff]
            self._anomaly_timestamps[detector].append(ts)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_risk_score(
        self,
        detector: str,
        tuner_stats: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Compute composite risk score for a single detector.

        Args:
            detector: Detector name.
            tuner_stats: AnomalyTuner.get_stats() output (used for FP rate).

        Returns:
            (risk_score, detail_dict) — risk_score is [0, 1].
        """
        now = time.time()
        cutoff = now - self._burst_window_seconds

        with self._lock:
            # ── Drift component ──────────────────────────────────
            drift_vel = self._drift_velocity.get(detector, 0.0)
            drift_norm = min(drift_vel / self._max_drift_rate, 1.0)

            # ── Burst component ──────────────────────────────────
            buf = self._anomaly_timestamps.get(detector, [])
            recent = [t for t in buf if t >= cutoff]
            # Store pruned list back
            self._anomaly_timestamps[detector] = recent
            burst_rate = len(recent) / (self._burst_window_seconds / 60.0)
            burst_norm = min(burst_rate / self._max_burst_rate, 1.0)

        # ── FP rate component (from tuner_stats, outside lock) ──
        fp_rate = 0.0
        if tuner_stats:
            for ps in tuner_stats.get("pair_stats", []):
                if ps.get("detector") == detector:
                    total = int(ps.get("total", 0))
                    denied = int(ps.get("denied", 0))
                    if total > 0:
                        fp_rate = max(fp_rate, denied / total)
        fp_norm = min(fp_rate / self._max_fp_rate, 1.0)

        # ── Composite ────────────────────────────────────────────
        risk = (
            self.W_SIGMA * drift_norm
            + self.W_FP * fp_norm
            + self.W_BURST * burst_norm
        )
        risk = max(0.0, min(1.0, risk))
        level = "high" if risk >= 0.6 else ("medium" if risk >= 0.3 else "low")

        detail: dict[str, Any] = {
            "detector": detector,
            "risk_score": round(risk, 4),
            "risk_level": level,
            "sigma_drift_velocity": round(drift_vel, 4),
            "sigma_drift_norm": round(drift_norm, 4),
            "fp_rate": round(fp_rate, 4),
            "fp_rate_norm": round(fp_norm, 4),
            "anomaly_burst_rate": round(burst_rate, 4),
            "anomaly_burst_norm": round(burst_norm, 4),
        }
        return risk, detail

    def get_all_risks(
        self,
        tuner_stats: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Compute risk scores for all four detectors.

        Args:
            tuner_stats: AnomalyTuner.get_stats() output.

        Returns:
            dict mapping detector name → detail dict (same as get_risk_score).
        """
        results: dict[str, dict[str, Any]] = {}
        for det in _DETECTORS:
            _, detail = self.get_risk_score(det, tuner_stats)
            results[det] = detail
        return results

    def get_state(
        self,
        tuner_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return full early warning state for API responses.

        Args:
            tuner_stats: AnomalyTuner.get_stats() output.

        Returns:
            dict with keys: overall_risk, overall_level, detectors, computed_at,
            burst_window_seconds.
        """
        risks = self.get_all_risks(tuner_stats)
        overall = max((d["risk_score"] for d in risks.values()), default=0.0)
        overall_level = "high" if overall >= 0.6 else ("medium" if overall >= 0.3 else "low")
        return {
            "overall_risk": round(overall, 4),
            "overall_level": overall_level,
            "detectors": risks,
            "computed_at": time.time(),
            "burst_window_seconds": self._burst_window_seconds,
        }

    def reset(self) -> None:
        """Clear all accumulated signals."""
        with self._lock:
            self._drift_velocity.clear()
            self._drift_last_ts.clear()
            self._anomaly_timestamps.clear()
