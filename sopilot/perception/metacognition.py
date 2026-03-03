"""Metacognitive monitoring: self-awareness and auto-calibration.

The MetacognitiveMonitor observes the perception engine's own outputs
and feedback signals to:
1. Estimate the quality of ongoing perception
2. Detect degradation patterns (high FP rate, low coverage, confidence drift)
3. Generate calibration recommendations
4. Optionally apply auto-calibrations within safe bounds

Quality is graded A–F based on three signals:
  - confidence_avg: average detection confidence (higher = better)
  - tracking_stability: fraction of frames where entities were tracked (higher = better)
  - fp_rate_estimate: estimated false-positive rate from feedback (lower = better)
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

from sopilot.perception.types import FrameResult, PerceptionConfig, ViolationSeverity


@dataclass
class PerceptionHealthReport:
    """Snapshot of perception quality at a point in time."""

    timestamp: float
    frames_observed: int
    detection_confidence_avg: float    # 0–1 rolling avg
    tracking_stability: float          # 0–1 fraction of frames with >=1 entity
    event_rate_per_minute: float       # events / minute (recent)
    fp_rate_estimate: float            # 0–1 estimated false positive rate
    coverage: float                    # fraction of frames where >=1 detection
    quality_grade: str                 # "A" | "B" | "C" | "D" | "F"
    quality_score: float               # 0–100 composite score
    issues: list[str]                  # detected issues (Japanese)
    recommendations: list[str]         # calibration suggestions (Japanese)
    auto_adjustments: list[str]        # what was auto-tuned (Japanese)


class MetacognitiveMonitor:
    """Self-monitoring layer for the Perception Engine.

    Call `observe_frame(frame_result)` after every frame.
    Call `record_feedback(event_type, confirmed)` when operator confirms/denies events.
    Call `get_health_report()` periodically to get quality status.
    Call `apply_calibration(config)` to get an auto-tuned config.
    """

    # Quality grade thresholds (quality_score 0-100)
    _GRADE_THRESHOLDS = {"A": 85, "B": 70, "C": 55, "D": 40}

    # Auto-calibration safe bounds
    _MIN_SIGMA = 1.5
    _MAX_SIGMA = 4.0
    _MIN_CONFIDENCE = 0.15
    _MAX_CONFIDENCE = 0.60
    _FEEDBACK_WINDOW = 200       # feedback records to keep

    def __init__(self, window_size: int = 100) -> None:
        """
        Args:
            window_size: Number of recent frames to keep for rolling statistics.
        """
        self._window_size = window_size
        # Rolling buffers
        self._confidence_history: deque[float] = deque(maxlen=window_size)
        self._tracked_history: deque[bool] = deque(maxlen=window_size)    # True if >=1 entity tracked
        self._detection_history: deque[bool] = deque(maxlen=window_size)  # True if >=1 detection
        self._event_timestamps: deque[float] = deque(maxlen=1000)
        # Feedback
        self._feedback_confirmed: deque[bool] = deque(maxlen=self._FEEDBACK_WINDOW)
        # Stats
        self._frames_observed: int = 0
        self._last_report_time: float = 0.0
        # Accumulated auto-adjustments (cleared each call to apply_calibration)
        self._pending_adjustments: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def observe_frame(self, frame_result: FrameResult, timestamp: float | None = None) -> None:
        """Update rolling statistics from one frame's results.

        Extracts:
        - Violation confidence values -> confidence_history
        - Whether any violations present -> tracked_history (proxy for tracking)
        - Whether FrameResult has violations -> detection_history
        - Event timestamps for event_rate calculation
        """
        ts = timestamp if timestamp is not None else time.monotonic()
        self._frames_observed += 1

        violations = frame_result.violations if frame_result.violations else []

        # Confidence history: collect all violation confidences
        for v in violations:
            self._confidence_history.append(float(v.confidence))

        has_violations = len(violations) > 0

        # tracked_history: True if >=1 entity tracked (proxy via violations)
        self._tracked_history.append(has_violations)

        # detection_history: True if >=1 detection (same proxy)
        self._detection_history.append(has_violations)

        # Record event timestamps for event rate calculation
        if has_violations:
            self._event_timestamps.append(ts)

    def record_feedback(self, confirmed: bool) -> None:
        """Record operator feedback: True=correct detection, False=false positive."""
        self._feedback_confirmed.append(confirmed)

    def get_health_report(self, current_time: float | None = None) -> PerceptionHealthReport:
        """Compute and return current perception health."""
        ts = current_time if current_time is not None else time.monotonic()

        confidence = self._compute_confidence_avg()
        stability = self._compute_tracking_stability()
        fp_rate = self._estimate_fp_rate()
        coverage = self._compute_coverage()
        event_rate = self._compute_event_rate(ts)
        score = self._compute_quality_score(confidence, stability, fp_rate, coverage)
        grade = self._grade(score)
        issues = self._generate_issues(confidence, stability, fp_rate, coverage)
        recommendations = self._generate_recommendations(confidence, stability, fp_rate)

        return PerceptionHealthReport(
            timestamp=ts,
            frames_observed=self._frames_observed,
            detection_confidence_avg=confidence,
            tracking_stability=stability,
            event_rate_per_minute=event_rate,
            fp_rate_estimate=fp_rate,
            coverage=coverage,
            quality_grade=grade,
            quality_score=score,
            issues=issues,
            recommendations=recommendations,
            auto_adjustments=list(self._pending_adjustments),
        )

    def get_calibration_delta(self, config: PerceptionConfig) -> dict[str, Any]:
        """Return a dict of suggested config changes (field_name -> new_value).
        Does NOT modify config in place.
        """
        changes: dict[str, Any] = {}
        confidence = self._compute_confidence_avg()
        fp_rate = self._estimate_fp_rate()

        if fp_rate > 0.4:
            new_sigma = min(config.anomaly_sigma_threshold + 0.3, self._MAX_SIGMA)
            if new_sigma != config.anomaly_sigma_threshold:
                changes["anomaly_sigma_threshold"] = new_sigma
        elif fp_rate < 0.1 and confidence > 0.7 and self._frames_observed > 50:
            new_sigma = max(config.anomaly_sigma_threshold - 0.2, self._MIN_SIGMA)
            if new_sigma != config.anomaly_sigma_threshold:
                changes["anomaly_sigma_threshold"] = new_sigma

        if confidence < 0.35:
            new_threshold = max(
                config.detection_confidence_threshold - 0.05, self._MIN_CONFIDENCE
            )
            if new_threshold != config.detection_confidence_threshold:
                changes["detection_confidence_threshold"] = new_threshold

        return changes

    def apply_calibration(self, config: PerceptionConfig) -> PerceptionConfig:
        """Return a new PerceptionConfig with auto-calibrated parameters.
        Records what was changed in self._pending_adjustments.
        """
        self._pending_adjustments.clear()
        confidence = self._compute_confidence_avg()
        fp_rate = self._estimate_fp_rate()
        changes: dict[str, Any] = {}

        # Anomaly sigma: increase to suppress FP, decrease to raise sensitivity
        if fp_rate > 0.4:
            old_sigma = config.anomaly_sigma_threshold
            new_sigma = min(old_sigma + 0.3, self._MAX_SIGMA)
            if new_sigma != old_sigma:
                changes["anomaly_sigma_threshold"] = new_sigma
                self._pending_adjustments.append(
                    f"anomaly_sigma_threshold: {old_sigma:.1f} → {new_sigma:.1f} (誤検知抑制)"
                )
        elif fp_rate < 0.1 and confidence > 0.7 and self._frames_observed > 50:
            old_sigma = config.anomaly_sigma_threshold
            new_sigma = max(old_sigma - 0.2, self._MIN_SIGMA)
            if new_sigma != old_sigma:
                changes["anomaly_sigma_threshold"] = new_sigma
                self._pending_adjustments.append(
                    f"anomaly_sigma_threshold: {old_sigma:.1f} → {new_sigma:.1f} (感度向上)"
                )

        # Detector confidence: lower to catch more detections when confidence is low
        if confidence < 0.35:
            old_thr = config.detection_confidence_threshold
            new_thr = max(old_thr - 0.05, self._MIN_CONFIDENCE)
            if new_thr != old_thr:
                changes["detection_confidence_threshold"] = new_thr
                self._pending_adjustments.append("detection_confidence_threshold を下げました")

        if changes:
            return replace(config, **changes)
        return config

    def get_state_dict(self) -> dict[str, Any]:
        """JSON-serializable current state for the API."""
        confidence = self._compute_confidence_avg()
        stability = self._compute_tracking_stability()
        fp_rate = self._estimate_fp_rate()
        coverage = self._compute_coverage()
        score = self._compute_quality_score(confidence, stability, fp_rate, coverage)
        grade = self._grade(score)

        return {
            "frames_observed": self._frames_observed,
            "detection_confidence_avg": round(confidence, 4),
            "tracking_stability": round(stability, 4),
            "fp_rate_estimate": round(fp_rate, 4),
            "coverage": round(coverage, 4),
            "quality_score": round(score, 2),
            "quality_grade": grade,
            "confidence_history_size": len(self._confidence_history),
            "feedback_samples": len(self._feedback_confirmed),
            "pending_adjustments": list(self._pending_adjustments),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_confidence_avg(self) -> float:
        """Rolling average confidence. Returns 0.5 if no data."""
        if not self._confidence_history:
            return 0.5
        return sum(self._confidence_history) / len(self._confidence_history)

    def _compute_tracking_stability(self) -> float:
        """Fraction of recent frames with >=1 entity tracked. Returns 0.5 if no data."""
        if not self._tracked_history:
            return 0.5
        return sum(1 for v in self._tracked_history if v) / len(self._tracked_history)

    def _compute_coverage(self) -> float:
        """Fraction of recent frames with >=1 detection. Returns 0.5 if no data."""
        if not self._detection_history:
            return 0.5
        return sum(1 for v in self._detection_history if v) / len(self._detection_history)

    def _estimate_fp_rate(self) -> float:
        """Estimate false positive rate from feedback. Returns 0.0 if < 10 samples."""
        if len(self._feedback_confirmed) < 10:
            return 0.0
        fp_count = sum(1 for c in self._feedback_confirmed if not c)
        return fp_count / len(self._feedback_confirmed)

    def _compute_event_rate(self, current_time: float) -> float:
        """Events per minute in the last 5 minutes."""
        cutoff = current_time - 300.0  # 5 minutes
        recent = [t for t in self._event_timestamps if t >= cutoff]
        if not recent:
            return 0.0
        # Events per minute
        return len(recent) / 5.0

    def _compute_quality_score(
        self,
        confidence: float,
        stability: float,
        fp_rate: float,
        coverage: float,
    ) -> float:
        """Composite quality score 0–100.

        Formula:
          score = 40 * confidence_component
                + 30 * stability_component
                + 30 * (1 - fp_rate)
        where confidence_component = clamp((confidence - 0.3) / 0.5, 0, 1)
              stability_component = clamp(stability, 0, 1)
        Then scale to [0, 100].
        """
        confidence_component = max(0.0, min(1.0, (confidence - 0.3) / 0.5))
        stability_component = max(0.0, min(1.0, stability))
        fp_component = max(0.0, min(1.0, 1.0 - fp_rate))

        raw = (
            40.0 * confidence_component
            + 30.0 * stability_component
            + 30.0 * fp_component
        )
        # raw already in [0, 100] since max is 40+30+30=100
        return max(0.0, min(100.0, raw))

    def _grade(self, score: float) -> str:
        """Map score 0–100 to grade A/B/C/D/F."""
        if score >= self._GRADE_THRESHOLDS["A"]:
            return "A"
        if score >= self._GRADE_THRESHOLDS["B"]:
            return "B"
        if score >= self._GRADE_THRESHOLDS["C"]:
            return "C"
        if score >= self._GRADE_THRESHOLDS["D"]:
            return "D"
        return "F"

    def _generate_issues(
        self,
        confidence: float,
        stability: float,
        fp_rate: float,
        coverage: float,
    ) -> list[str]:
        """Generate list of issue descriptions in Japanese.

        Thresholds:
        - confidence < 0.4  -> "検出信頼度が低い (平均 {:.0%})"
        - stability < 0.5   -> "トラッキングが不安定 (安定率 {:.0%})"
        - fp_rate > 0.4     -> "誤検知率が高い (推定 {:.0%})"
        - coverage < 0.3    -> "検出カバレッジが低い (カバレッジ {:.0%})"
        """
        issues: list[str] = []
        if confidence < 0.4:
            issues.append(f"検出信頼度が低い (平均 {confidence:.0%})")
        if stability < 0.5:
            issues.append(f"トラッキングが不安定 (安定率 {stability:.0%})")
        if fp_rate > 0.4:
            issues.append(f"誤検知率が高い (推定 {fp_rate:.0%})")
        if coverage < 0.3:
            issues.append(f"検出カバレッジが低い (カバレッジ {coverage:.0%})")
        return issues

    def _generate_recommendations(
        self,
        confidence: float,
        stability: float,
        fp_rate: float,
    ) -> list[str]:
        """Generate calibration recommendations in Japanese.

        - confidence < 0.4                          -> "信頼度閾値を下げることを推奨します"
        - fp_rate > 0.4                             -> "異常検知の感度を下げることを推奨します (sigma閾値を上げてください)"
        - fp_rate < 0.1 and confidence > 0.7        -> "感度をわずかに上げられる可能性があります"
        - stability < 0.3                           -> "トラッカーの min_hits パラメータを下げることを推奨します"
        """
        recommendations: list[str] = []
        if confidence < 0.4:
            recommendations.append("信頼度閾値を下げることを推奨します")
        if fp_rate > 0.4:
            recommendations.append(
                "異常検知の感度を下げることを推奨します (sigma閾値を上げてください)"
            )
        if fp_rate < 0.1 and confidence > 0.7:
            recommendations.append("感度をわずかに上げられる可能性があります")
        if stability < 0.3:
            recommendations.append(
                "トラッカーの min_hits パラメータを下げることを推奨します"
            )
        return recommendations
