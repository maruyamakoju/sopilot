"""Adaptive learning and concept drift detection for anomaly detection.

Uses the Page-Hinkley (PH) test to detect distribution shifts in anomaly
score streams. On drift detection, auto-recalibrates the ensemble's
sigma_threshold using a moving average of recent scores.

No external dependencies beyond stdlib + numpy.
Thread-safe under RLock.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PageHinkleyDetector:
    """Page-Hinkley change-point detector.

    Detects an upward drift (mean increase) in a univariate stream.

    Parameters
    ----------
    delta : float
        Minimum acceptable change magnitude (reduces sensitivity to noise).
        Default 0.005.
    lambda_ : float
        Detection threshold. Larger -> fewer false alarms. Default 50.0.
    alpha : float
        Running mean forgetting factor (closer to 1 = longer memory).
        Default 0.9999.
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
        alpha: float = 0.9999,
    ) -> None:
        self._delta = delta
        self._lambda = lambda_
        self._alpha = alpha
        self._lock = threading.Lock()
        self._x_mean: float = 0.0
        self._sum: float = 0.0
        self._min_sum: float = 0.0
        self._n: int = 0

    def update(self, x: float) -> bool:
        """Process one observation. Returns True if drift is detected."""
        with self._lock:
            self._n += 1
            # Update running mean with forgetting
            self._x_mean = self._alpha * self._x_mean + (1 - self._alpha) * x
            # PH statistic
            self._sum += x - self._x_mean - self._delta
            self._min_sum = min(self._min_sum, self._sum)
            ph_stat = self._sum - self._min_sum
            return ph_stat > self._lambda

    def reset(self) -> None:
        with self._lock:
            self._x_mean = 0.0
            self._sum = 0.0
            self._min_sum = 0.0
            self._n = 0

    def get_state(self) -> dict:
        with self._lock:
            return {
                "n": self._n,
                "x_mean": round(self._x_mean, 6),
                "ph_stat": round(self._sum - self._min_sum, 6),
                "threshold": self._lambda,
            }


@dataclass
class RecalibrationRecord:
    timestamp: float
    reason: str                # "drift" | "manual" | "feedback"
    old_threshold: float
    new_threshold: float
    observations_used: int
    drift_score: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "reason": self.reason,
            "old_threshold": round(self.old_threshold, 4),
            "new_threshold": round(self.new_threshold, 4),
            "observations_used": self.observations_used,
            "drift_score": round(self.drift_score, 4),
        }


class AdaptiveLearner:
    """Auto-recalibrates anomaly detection thresholds on concept drift.

    Monitors a stream of anomaly scores (floats) from the ensemble.
    When PageHinkleyDetector fires, re-estimates the sigma_threshold
    as percentile of recent score window and applies it to the ensemble.

    Parameters
    ----------
    ensemble : Any
        AnomalyDetectorEnsemble with .config.sigma_threshold attribute.
        If None, recalibration is logged but not applied.
    min_observations : int
        Minimum score observations before recalibration is allowed.
    recalibration_cooldown_s : float
        Minimum seconds between recalibrations (prevents rapid oscillation).
    score_window_size : int
        Number of recent scores to use for threshold estimation.
    new_threshold_percentile : float
        Percentile of recent scores used as new sigma_threshold (default 75).
    threshold_min : float
        Minimum allowed sigma_threshold after recalibration. Default 1.5.
    threshold_max : float
        Maximum allowed sigma_threshold after recalibration. Default 8.0.
    drift_delta : float
        PH delta parameter.
    drift_lambda : float
        PH detection threshold.
    """

    def __init__(
        self,
        ensemble: Any = None,
        min_observations: int = 50,
        recalibration_cooldown_s: float = 300.0,
        score_window_size: int = 200,
        new_threshold_percentile: float = 75.0,
        threshold_min: float = 1.5,
        threshold_max: float = 8.0,
        drift_delta: float = 0.005,
        drift_lambda: float = 50.0,
    ) -> None:
        self._ensemble = ensemble
        self._min_obs = min_observations
        self._cooldown = recalibration_cooldown_s
        self._window_size = score_window_size
        self._percentile = new_threshold_percentile
        self._thresh_min = threshold_min
        self._thresh_max = threshold_max
        self._lock = threading.RLock()
        self._ph = PageHinkleyDetector(delta=drift_delta, lambda_=drift_lambda)
        self._scores: list[float] = []           # rolling window
        self._total_observed: int = 0
        self._last_recal_ts: float = 0.0
        self._recal_history: list[RecalibrationRecord] = []
        self._drift_count: int = 0

    def observe(self, score: float, timestamp: float | None = None) -> bool:
        """Record an anomaly score. Returns True if recalibration was triggered."""
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            self._scores.append(score)
            if len(self._scores) > self._window_size:
                self._scores = self._scores[-self._window_size:]
            self._total_observed += 1

            drift_detected = self._ph.update(score)
            if drift_detected:
                self._drift_count += 1
                # Only recalibrate if enough observations and not in cooldown
                if (
                    self._total_observed >= self._min_obs
                    and (ts - self._last_recal_ts) >= self._cooldown
                ):
                    record = self._do_recalibrate(ts, reason="drift", drift_score=score)
                    if record:
                        self._ph.reset()  # reset after applying recalibration
                        return True
        return False

    def force_recalibrate(self, timestamp: float | None = None) -> RecalibrationRecord | None:
        """Manual recalibration. Returns RecalibrationRecord or None if insufficient data."""
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            return self._do_recalibrate(ts, reason="manual", drift_score=0.0)

    def _do_recalibrate(
        self, ts: float, reason: str, drift_score: float
    ) -> RecalibrationRecord | None:
        """Internal: compute new threshold and apply to ensemble."""
        if len(self._scores) < max(10, self._min_obs // 5):
            return None

        new_thresh = float(np.percentile(self._scores, self._percentile))
        new_thresh = max(self._thresh_min, min(self._thresh_max, new_thresh))

        # Current threshold
        old_thresh = self._thresh_min  # default
        if self._ensemble is not None:
            cfg = getattr(self._ensemble, "config", None)
            if cfg is not None:
                old_thresh = float(getattr(cfg, "sigma_threshold", self._thresh_min))
            # Apply to ensemble config
            if cfg is not None:
                try:
                    cfg.sigma_threshold = new_thresh
                except AttributeError:
                    pass  # frozen or read-only config

        record = RecalibrationRecord(
            timestamp=ts,
            reason=reason,
            old_threshold=round(old_thresh, 4),
            new_threshold=round(new_thresh, 4),
            observations_used=len(self._scores),
            drift_score=round(drift_score, 4),
        )
        self._recal_history.append(record)
        self._last_recal_ts = ts
        logger.info(
            "AdaptiveLearner recalibration: reason=%s old=%.2f new=%.2f obs=%d",
            reason, old_thresh, new_thresh, len(self._scores),
        )
        return record

    def get_recalibration_history(self, n: int = 10) -> list[RecalibrationRecord]:
        with self._lock:
            return list(self._recal_history[-n:])

    def get_state_dict(self) -> dict:
        with self._lock:
            ph_state = self._ph.get_state()
            last_rec = self._recal_history[-1].to_dict() if self._recal_history else None
            score_arr = np.array(self._scores) if self._scores else np.array([0.0])
            return {
                "total_observed": self._total_observed,
                "score_window_size": len(self._scores),
                "score_mean": round(float(np.mean(score_arr)), 4),
                "score_std": round(float(np.std(score_arr)), 4),
                "drift_count": self._drift_count,
                "recalibration_count": len(self._recal_history),
                "last_recalibration": last_rec,
                "ph_state": ph_state,
            }
