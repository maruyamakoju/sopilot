"""Phase 57: Predictive Maintenance — Anomaly Forecaster.

時系列の異常スコアに Holt-Winters 指数平滑化を適用し、
機器障害・安全インシデントの発生確率を予測する。

主要機能:
  - **Holt-Winters 二重指数平滑化**: レベル + トレンド成分
  - **故障確率推定**: sigmoid(normalized_forecast)
  - **予知保全アラート**: 3段階 (info / warning / critical)
  - **カメラ別予測**: 独立した時系列管理
  - **トレンド検出**: 上昇 / 下降 / 安定の自動判定

Architecture::

    AnomalyForecaster
      ├── observe(timestamp, anomaly_score, violation_count, camera_id)
      ├── forecast(camera_id, horizon_steps)   → ForecastResult
      ├── get_maintenance_alerts()             → list[MaintenanceAlert]
      ├── get_trend(camera_id)                 → TrendResult
      ├── get_stats()                          → dict
      └── reset(camera_id?)

Algorithm:
    Holt-Winters (additive, no seasonality):
        L_t = α * y_t + (1-α) * (L_{t-1} + T_{t-1})
        T_t = β * (L_t - L_{t-1}) + (1-β) * T_{t-1}
        forecast(h) = L_t + h * T_t

    Failure probability:
        p = sigmoid((forecast - threshold) / scale)
        sigmoid(x) = 1 / (1 + exp(-x))

Design decisions:
  - 純 NumPy — GPU / sklearn 不要
  - スレッドセーフ: RLock
  - Best-effort: 例外は swallow + logger.debug
  - 最低 3 サンプルで予測開始
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_SAMPLES_FOR_FORECAST = 3  # minimum observations before forecasting
_ALERT_HISTORY_MAX = 100       # maximum stored alerts per camera
_OBS_BUFFER_MAX = 2000         # rolling observation buffer


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ForecastResult:
    """Output of AnomalyForecaster.forecast()."""
    camera_id: str
    horizon_steps: int
    current_level: float       # latest smoothed level (L_t)
    current_trend: float       # latest trend (T_t), positive = rising
    forecast_values: list[float]  # predicted scores for steps 1..horizon
    failure_probabilities: list[float]  # sigmoid(normalized) per step
    peak_failure_prob: float   # max failure probability in horizon
    alert_level: str           # "none" / "info" / "warning" / "critical"
    samples_used: int
    last_observation_ts: float | None


@dataclass
class TrendResult:
    """Trend summary for a camera."""
    camera_id: str
    direction: str     # "rising" / "falling" / "stable" / "unknown"
    trend_per_step: float   # T_t value
    current_score: float    # latest raw observation
    smoothed_score: float   # L_t


@dataclass
class MaintenanceAlert:
    """A predictive maintenance alert."""
    camera_id: str
    alert_level: str   # "info" / "warning" / "critical"
    failure_prob: float
    forecast_score: float
    steps_ahead: int
    message_ja: str
    message_en: str
    created_at: float


# ---------------------------------------------------------------------------
# Per-camera state
# ---------------------------------------------------------------------------


@dataclass
class CameraForecastState:
    """Per-camera Holt-Winters state."""
    camera_id: str

    # Observation buffer: (timestamp, anomaly_score, violation_count)
    obs: deque = field(default_factory=lambda: deque(maxlen=_OBS_BUFFER_MAX))

    # Holt-Winters state
    level: float | None = None   # L_t
    trend: float | None = None   # T_t (initialized from first two points)

    # Alert history
    alerts: list = field(default_factory=list)

    # Tracking
    total_observed: int = 0
    last_ts: float | None = None


# ---------------------------------------------------------------------------
# AnomalyForecaster
# ---------------------------------------------------------------------------


class AnomalyForecaster:
    """Holt-Winters exponential smoothing for anomaly score time series.

    Predicts future anomaly scores and converts them to failure probabilities
    using a sigmoid with configurable threshold and scale.
    """

    def __init__(
        self,
        alpha: float = 0.3,        # level smoothing factor
        beta: float = 0.1,         # trend smoothing factor
        threshold: float = 0.6,    # anomaly score triggering high alert
        sigmoid_scale: float = 0.2,
        default_horizon_steps: int = 10,
        alert_info_prob: float = 0.3,
        alert_warning_prob: float = 0.6,
        alert_critical_prob: float = 0.85,
    ) -> None:
        self._alpha = alpha
        self._beta = beta
        self._threshold = threshold
        self._sigmoid_scale = sigmoid_scale
        self._default_horizon_steps = default_horizon_steps
        self._alert_info_prob = alert_info_prob
        self._alert_warning_prob = alert_warning_prob
        self._alert_critical_prob = alert_critical_prob

        self._lock = threading.RLock()
        self._cameras: dict[str, CameraForecastState] = {}
        self._total_observed: int = 0
        self._total_alerts_generated: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def observe(
        self,
        anomaly_score: float,
        violation_count: int = 0,
        camera_id: str = "cam",
        timestamp: float | None = None,
    ) -> None:
        """Record a new observation and update Holt-Winters state.

        Args:
            anomaly_score:   Normalized anomaly score [0, 1].
            violation_count: Number of safety violations in this frame.
            camera_id:       Camera identifier.
            timestamp:       Unix timestamp (defaults to now).
        """
        if timestamp is None:
            timestamp = time.time()
        anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))

        with self._lock:
            st = self._get_or_create(camera_id)
            st.obs.append((timestamp, anomaly_score, violation_count))
            st.total_observed += 1
            st.last_ts = timestamp
            self._total_observed += 1
            self._hw_update(st, anomaly_score)

    def forecast(
        self,
        camera_id: str = "cam",
        horizon_steps: int | None = None,
    ) -> ForecastResult:
        """Forecast future anomaly scores using current Holt-Winters state.

        Returns a ForecastResult with per-step predictions and failure probs.
        """
        if horizon_steps is None:
            horizon_steps = self._default_horizon_steps

        with self._lock:
            st = self._cameras.get(camera_id)
            if st is None or st.level is None or len(st.obs) < _MIN_SAMPLES_FOR_FORECAST:
                return ForecastResult(
                    camera_id=camera_id,
                    horizon_steps=horizon_steps,
                    current_level=0.0,
                    current_trend=0.0,
                    forecast_values=[],
                    failure_probabilities=[],
                    peak_failure_prob=0.0,
                    alert_level="none",
                    samples_used=len(st.obs) if st else 0,
                    last_observation_ts=st.last_ts if st else None,
                )

            L, T = st.level, st.trend if st.trend is not None else 0.0
            forecasts = [float(np.clip(L + h * T, 0.0, 1.0)) for h in range(1, horizon_steps + 1)]
            probs = [self._failure_prob(f) for f in forecasts]
            peak_prob = max(probs) if probs else 0.0
            alert_level = self._prob_to_level(peak_prob)

            return ForecastResult(
                camera_id=camera_id,
                horizon_steps=horizon_steps,
                current_level=round(L, 4),
                current_trend=round(T, 6),
                forecast_values=[round(f, 4) for f in forecasts],
                failure_probabilities=[round(p, 4) for p in probs],
                peak_failure_prob=round(peak_prob, 4),
                alert_level=alert_level,
                samples_used=len(st.obs),
                last_observation_ts=st.last_ts,
            )

    def get_maintenance_alerts(self, camera_id: str | None = None) -> list[dict]:
        """Return pending maintenance alerts for one or all cameras."""
        with self._lock:
            cameras = (
                [self._cameras[camera_id]] if camera_id and camera_id in self._cameras
                else list(self._cameras.values())
            )
            alerts = []
            for st in cameras:
                alerts.extend(st.alerts[-10:])  # latest 10 per camera
            alerts.sort(key=lambda a: a["created_at"], reverse=True)
            return alerts

    def get_trend(self, camera_id: str = "cam") -> TrendResult:
        """Return trend summary for a camera."""
        with self._lock:
            st = self._cameras.get(camera_id)
            if st is None or st.level is None:
                return TrendResult(
                    camera_id=camera_id,
                    direction="unknown",
                    trend_per_step=0.0,
                    current_score=0.0,
                    smoothed_score=0.0,
                )
            T = st.trend if st.trend is not None else 0.0
            if T > 0.01:
                direction = "rising"
            elif T < -0.01:
                direction = "falling"
            else:
                direction = "stable"

            latest_score = float(st.obs[-1][1]) if st.obs else 0.0
            return TrendResult(
                camera_id=camera_id,
                direction=direction,
                trend_per_step=round(T, 6),
                current_score=round(latest_score, 4),
                smoothed_score=round(st.level, 4),
            )

    def get_camera_states(self) -> list[dict]:
        """Return per-camera state summary."""
        with self._lock:
            return [
                {
                    "camera_id": st.camera_id,
                    "samples": len(st.obs),
                    "level": round(st.level, 4) if st.level is not None else None,
                    "trend": round(st.trend, 6) if st.trend is not None else None,
                    "total_observed": st.total_observed,
                    "alert_count": len(st.alerts),
                    "last_ts": st.last_ts,
                }
                for st in self._cameras.values()
            ]

    def get_stats(self) -> dict:
        """Return overall forecaster statistics."""
        with self._lock:
            return {
                "enabled": True,
                "total_observed": self._total_observed,
                "total_alerts_generated": self._total_alerts_generated,
                "camera_count": len(self._cameras),
                "alpha": self._alpha,
                "beta": self._beta,
                "threshold": self._threshold,
                "default_horizon_steps": self._default_horizon_steps,
                "cameras": self.get_camera_states(),
            }

    def reset(self, camera_id: str | None = None) -> None:
        """Reset one or all cameras."""
        with self._lock:
            if camera_id is not None:
                self._cameras.pop(camera_id, None)
            else:
                self._cameras.clear()
                self._total_observed = 0
                self._total_alerts_generated = 0

    # ── Internal helpers ───────────────────────────────────────────────────

    def _get_or_create(self, camera_id: str) -> CameraForecastState:
        if camera_id not in self._cameras:
            self._cameras[camera_id] = CameraForecastState(camera_id=camera_id)
        return self._cameras[camera_id]

    def _hw_update(self, st: CameraForecastState, y: float) -> None:
        """Update Holt-Winters level and trend (caller holds lock)."""
        if st.level is None:
            # First observation: initialize level = y, trend = 0
            st.level = y
            st.trend = 0.0
            return

        L_prev = st.level
        T_prev = st.trend if st.trend is not None else 0.0

        # Holt-Winters update
        L_new = self._alpha * y + (1.0 - self._alpha) * (L_prev + T_prev)
        T_new = self._beta * (L_new - L_prev) + (1.0 - self._beta) * T_prev

        st.level = float(np.clip(L_new, 0.0, 1.0))
        st.trend = float(T_new)

        # Check if we should emit an alert (1-step-ahead forecast)
        forecast_1 = float(np.clip(L_new + T_new, 0.0, 1.0))
        prob = self._failure_prob(forecast_1)
        level = self._prob_to_level(prob)
        if level != "none" and len(st.obs) >= _MIN_SAMPLES_FOR_FORECAST:
            self._emit_alert(st, level, prob, forecast_1, steps_ahead=1)

    def _failure_prob(self, score: float) -> float:
        """Convert forecast score to failure probability via sigmoid."""
        x = (score - self._threshold) / (self._sigmoid_scale + 1e-8)
        return float(1.0 / (1.0 + math.exp(-x)))

    def _prob_to_level(self, prob: float) -> str:
        if prob >= self._alert_critical_prob:
            return "critical"
        if prob >= self._alert_warning_prob:
            return "warning"
        if prob >= self._alert_info_prob:
            return "info"
        return "none"

    def _emit_alert(
        self,
        st: CameraForecastState,
        level: str,
        prob: float,
        forecast_score: float,
        steps_ahead: int,
    ) -> None:
        """Create and store a maintenance alert (caller holds lock)."""
        if level == "critical":
            ja = f"【緊急】カメラ {st.camera_id}: 異常スコアが臨界レベルに達する予測 (確率 {prob:.0%})"
            en = f"[CRITICAL] Camera {st.camera_id}: anomaly score projected to reach critical level (prob {prob:.0%})"
        elif level == "warning":
            ja = f"【警告】カメラ {st.camera_id}: 異常スコアが上昇傾向 (確率 {prob:.0%})"
            en = f"[WARNING] Camera {st.camera_id}: anomaly score trending upward (prob {prob:.0%})"
        else:
            ja = f"【情報】カメラ {st.camera_id}: 軽度の異常スコア上昇を検知 (確率 {prob:.0%})"
            en = f"[INFO] Camera {st.camera_id}: mild anomaly score increase detected (prob {prob:.0%})"

        alert = {
            "camera_id": st.camera_id,
            "alert_level": level,
            "failure_prob": round(prob, 4),
            "forecast_score": round(forecast_score, 4),
            "steps_ahead": steps_ahead,
            "message_ja": ja,
            "message_en": en,
            "created_at": st.last_ts or time.time(),
        }
        st.alerts.append(alert)
        if len(st.alerts) > _ALERT_HISTORY_MAX:
            st.alerts.pop(0)
        self._total_alerts_generated += 1
