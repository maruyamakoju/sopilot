"""Tests for MetacognitiveMonitor — self-awareness and auto-calibration layer.

Covers:
- TestInit              (3 tests)  — initial state
- TestObserveFrame      (8 tests)  — rolling statistics from frames
- TestRecordFeedback    (4 tests)  — operator feedback / FP rate
- TestHealthReport      (8 tests)  — PerceptionHealthReport fields
- TestQualityGrading    (5 tests)  — grade / score computation
- TestIssues            (4 tests)  — Japanese issue generation
- TestCalibration       (8 tests)  — auto-calibration logic
- TestStateDict         (3 tests)  — get_state_dict

Run: python -m pytest tests/test_metacognition.py -v
"""

from __future__ import annotations

import time
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from sopilot.perception.metacognition import MetacognitiveMonitor, PerceptionHealthReport
from sopilot.perception.types import FrameResult, PerceptionConfig, ViolationSeverity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame_result(violation_count: int = 0, confidence: float = 0.7):
    """Build a MagicMock FrameResult with the given number of violations."""
    fr = MagicMock(spec=FrameResult)
    fr.violations = []
    for _ in range(violation_count):
        v = MagicMock()
        v.confidence = confidence
        v.severity = ViolationSeverity.WARNING
        v.source = "rule"
        fr.violations.append(v)
    fr.timing = {}
    fr.pose_results = []
    return fr


def _make_config(**kwargs) -> PerceptionConfig:
    """Create a PerceptionConfig, overriding fields with kwargs."""
    cfg = PerceptionConfig()
    return replace(cfg, **kwargs) if kwargs else cfg


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_initial_frames_observed(self):
        mon = MetacognitiveMonitor()
        assert mon._frames_observed == 0

    def test_initial_confidence_history_empty(self):
        mon = MetacognitiveMonitor()
        assert len(mon._confidence_history) == 0

    def test_initial_feedback_empty(self):
        mon = MetacognitiveMonitor()
        assert len(mon._feedback_confirmed) == 0


# ---------------------------------------------------------------------------
# TestObserveFrame
# ---------------------------------------------------------------------------


class TestObserveFrame:
    def test_increments_frames_observed(self):
        mon = MetacognitiveMonitor()
        mon.observe_frame(_make_frame_result(0))
        assert mon._frames_observed == 1

    def test_adds_to_confidence_history_on_violation(self):
        mon = MetacognitiveMonitor()
        mon.observe_frame(_make_frame_result(violation_count=1, confidence=0.8))
        assert len(mon._confidence_history) == 1
        assert abs(mon._confidence_history[0] - 0.8) < 1e-9

    def test_high_confidence_raises_avg(self):
        mon = MetacognitiveMonitor()
        mon.observe_frame(_make_frame_result(1, confidence=0.9))
        mon.observe_frame(_make_frame_result(1, confidence=0.9))
        avg = mon._compute_confidence_avg()
        assert avg > 0.8

    def test_frame_with_no_violations_tracked_false(self):
        mon = MetacognitiveMonitor()
        mon.observe_frame(_make_frame_result(0))
        assert mon._tracked_history[-1] is False

    def test_frame_with_violations_tracked_true(self):
        mon = MetacognitiveMonitor()
        mon.observe_frame(_make_frame_result(1, confidence=0.7))
        assert mon._tracked_history[-1] is True

    def test_detection_history_updated(self):
        mon = MetacognitiveMonitor()
        mon.observe_frame(_make_frame_result(0))
        assert len(mon._detection_history) == 1

    def test_multiple_frames_accumulate(self):
        mon = MetacognitiveMonitor()
        for _ in range(5):
            mon.observe_frame(_make_frame_result(1, confidence=0.6))
        assert mon._frames_observed == 5
        assert len(mon._confidence_history) == 5

    def test_timestamp_recorded_for_event_rate(self):
        mon = MetacognitiveMonitor()
        ts = time.monotonic()
        mon.observe_frame(_make_frame_result(1, confidence=0.7), timestamp=ts)
        assert len(mon._event_timestamps) == 1
        assert abs(mon._event_timestamps[0] - ts) < 1e-6


# ---------------------------------------------------------------------------
# TestRecordFeedback
# ---------------------------------------------------------------------------


class TestRecordFeedback:
    def test_record_true_confirmed(self):
        mon = MetacognitiveMonitor()
        mon.record_feedback(True)
        assert list(mon._feedback_confirmed) == [True]

    def test_record_false_false_positive(self):
        mon = MetacognitiveMonitor()
        mon.record_feedback(False)
        assert list(mon._feedback_confirmed) == [False]

    def test_fp_rate_after_10_fp_feedbacks(self):
        mon = MetacognitiveMonitor()
        # 10 false positives
        for _ in range(10):
            mon.record_feedback(False)
        rate = mon._estimate_fp_rate()
        assert abs(rate - 1.0) < 1e-9

    def test_fp_rate_mixed_feedback(self):
        mon = MetacognitiveMonitor()
        # 5 TP, 5 FP = 50% FP rate
        for _ in range(5):
            mon.record_feedback(True)
        for _ in range(5):
            mon.record_feedback(False)
        rate = mon._estimate_fp_rate()
        assert abs(rate - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# TestHealthReport
# ---------------------------------------------------------------------------


class TestHealthReport:
    def test_returns_perception_health_report(self):
        mon = MetacognitiveMonitor()
        report = mon.get_health_report()
        assert isinstance(report, PerceptionHealthReport)

    def test_quality_grade_valid(self):
        mon = MetacognitiveMonitor()
        report = mon.get_health_report()
        assert report.quality_grade in {"A", "B", "C", "D", "F"}

    def test_quality_score_range(self):
        mon = MetacognitiveMonitor()
        report = mon.get_health_report()
        assert 0.0 <= report.quality_score <= 100.0

    def test_detection_confidence_avg_range(self):
        mon = MetacognitiveMonitor()
        mon.observe_frame(_make_frame_result(1, confidence=0.75))
        report = mon.get_health_report()
        assert 0.0 <= report.detection_confidence_avg <= 1.0

    def test_tracking_stability_range(self):
        mon = MetacognitiveMonitor()
        for _ in range(10):
            mon.observe_frame(_make_frame_result(1, confidence=0.6))
        report = mon.get_health_report()
        assert 0.0 <= report.tracking_stability <= 1.0

    def test_fp_rate_estimate_range(self):
        mon = MetacognitiveMonitor()
        for _ in range(10):
            mon.record_feedback(False)
        report = mon.get_health_report()
        assert 0.0 <= report.fp_rate_estimate <= 1.0

    def test_issues_is_list(self):
        mon = MetacognitiveMonitor()
        report = mon.get_health_report()
        assert isinstance(report.issues, list)

    def test_recommendations_is_list(self):
        mon = MetacognitiveMonitor()
        report = mon.get_health_report()
        assert isinstance(report.recommendations, list)


# ---------------------------------------------------------------------------
# TestQualityGrading
# ---------------------------------------------------------------------------


class TestQualityGrading:
    def test_high_confidence_low_fp_grade_a_or_b(self):
        mon = MetacognitiveMonitor()
        # Feed high-confidence violations + all-positive feedback
        for _ in range(20):
            mon.observe_frame(_make_frame_result(1, confidence=0.9))
        for _ in range(10):
            mon.record_feedback(True)
        report = mon.get_health_report()
        assert report.quality_grade in {"A", "B"}

    def test_low_confidence_gives_lower_grade(self):
        mon = MetacognitiveMonitor()
        # Very low confidence
        for _ in range(20):
            mon.observe_frame(_make_frame_result(1, confidence=0.15))
        high_conf_mon = MetacognitiveMonitor()
        for _ in range(20):
            high_conf_mon.observe_frame(_make_frame_result(1, confidence=0.9))
        low_report = mon.get_health_report()
        high_report = high_conf_mon.get_health_report()
        assert low_report.quality_score < high_report.quality_score

    def test_high_fp_rate_gives_grade_d_or_f(self):
        mon = MetacognitiveMonitor()
        for _ in range(20):
            mon.record_feedback(False)
        report = mon.get_health_report()
        assert report.quality_grade in {"D", "F"}

    def test_grade_changes_with_window(self):
        mon = MetacognitiveMonitor(window_size=10)
        # Fill with good data then overwrite with bad data
        for _ in range(10):
            mon.observe_frame(_make_frame_result(1, confidence=0.9))
        good_report = mon.get_health_report()
        for _ in range(10):
            mon.observe_frame(_make_frame_result(1, confidence=0.1))
        bad_report = mon.get_health_report()
        assert bad_report.quality_score < good_report.quality_score

    def test_quality_score_correlates_with_grade(self):
        mon = MetacognitiveMonitor()
        # A-grade scenario
        for _ in range(30):
            mon.observe_frame(_make_frame_result(1, confidence=0.95))
        for _ in range(15):
            mon.record_feedback(True)
        report = mon.get_health_report()
        if report.quality_grade == "A":
            assert report.quality_score >= 85
        elif report.quality_grade == "B":
            assert report.quality_score >= 70
        elif report.quality_grade == "C":
            assert report.quality_score >= 55
        elif report.quality_grade == "D":
            assert report.quality_score >= 40
        else:
            assert report.quality_score < 40


# ---------------------------------------------------------------------------
# TestIssues
# ---------------------------------------------------------------------------


class TestIssues:
    def test_no_issues_when_quality_good(self):
        mon = MetacognitiveMonitor()
        for _ in range(20):
            mon.observe_frame(_make_frame_result(1, confidence=0.9))
        for _ in range(15):
            mon.record_feedback(True)
        report = mon.get_health_report()
        # With high confidence and low FP, no issues for confidence/FP/coverage
        # Stability might be 1.0 (all frames have violations)
        assert "検出信頼度が低い" not in " ".join(report.issues)
        assert "誤検知率が高い" not in " ".join(report.issues)

    def test_issue_generated_for_low_confidence(self):
        mon = MetacognitiveMonitor()
        for _ in range(20):
            mon.observe_frame(_make_frame_result(1, confidence=0.2))
        report = mon.get_health_report()
        issues_text = " ".join(report.issues)
        assert "検出信頼度が低い" in issues_text

    def test_issue_generated_for_high_fp_rate(self):
        mon = MetacognitiveMonitor()
        for _ in range(20):
            mon.record_feedback(False)
        report = mon.get_health_report()
        issues_text = " ".join(report.issues)
        assert "誤検知率が高い" in issues_text

    def test_issue_generated_for_low_coverage(self):
        mon = MetacognitiveMonitor()
        # Mostly empty frames -> low coverage
        for _ in range(20):
            mon.observe_frame(_make_frame_result(0))
        report = mon.get_health_report()
        issues_text = " ".join(report.issues)
        assert "検出カバレッジが低い" in issues_text


# ---------------------------------------------------------------------------
# TestCalibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_get_calibration_delta_returns_dict(self):
        mon = MetacognitiveMonitor()
        cfg = _make_config()
        delta = mon.get_calibration_delta(cfg)
        assert isinstance(delta, dict)

    def test_apply_calibration_returns_perception_config(self):
        mon = MetacognitiveMonitor()
        cfg = _make_config()
        result = mon.apply_calibration(cfg)
        assert isinstance(result, PerceptionConfig)

    def test_high_fp_rate_increases_sigma(self):
        mon = MetacognitiveMonitor()
        for _ in range(20):
            mon.record_feedback(False)
        cfg = _make_config(anomaly_sigma_threshold=2.0)
        result = mon.apply_calibration(cfg)
        assert result.anomaly_sigma_threshold > cfg.anomaly_sigma_threshold

    def test_low_fp_high_confidence_decreases_sigma(self):
        mon = MetacognitiveMonitor()
        # Feed enough frames so frames_observed > 50
        for _ in range(60):
            mon.observe_frame(_make_frame_result(1, confidence=0.9))
        for _ in range(15):
            mon.record_feedback(True)
        cfg = _make_config(anomaly_sigma_threshold=3.0)
        result = mon.apply_calibration(cfg)
        assert result.anomaly_sigma_threshold < cfg.anomaly_sigma_threshold

    def test_low_confidence_decreases_detector_threshold(self):
        mon = MetacognitiveMonitor()
        for _ in range(20):
            mon.observe_frame(_make_frame_result(1, confidence=0.2))
        cfg = _make_config(detection_confidence_threshold=0.35)
        result = mon.apply_calibration(cfg)
        assert result.detection_confidence_threshold < cfg.detection_confidence_threshold

    def test_calibration_stays_within_safe_bounds_max_sigma(self):
        mon = MetacognitiveMonitor()
        # Drive FP very high
        for _ in range(50):
            mon.record_feedback(False)
        # Start near max
        cfg = _make_config(anomaly_sigma_threshold=3.9)
        result = mon.apply_calibration(cfg)
        assert result.anomaly_sigma_threshold <= MetacognitiveMonitor._MAX_SIGMA

    def test_calibration_stays_within_safe_bounds_min_sigma(self):
        mon = MetacognitiveMonitor()
        for _ in range(60):
            mon.observe_frame(_make_frame_result(1, confidence=0.9))
        for _ in range(15):
            mon.record_feedback(True)
        # Start near min
        cfg = _make_config(anomaly_sigma_threshold=1.6)
        result = mon.apply_calibration(cfg)
        assert result.anomaly_sigma_threshold >= MetacognitiveMonitor._MIN_SIGMA

    def test_original_config_not_modified(self):
        mon = MetacognitiveMonitor()
        for _ in range(20):
            mon.record_feedback(False)
        cfg = _make_config(anomaly_sigma_threshold=2.0)
        original_sigma = cfg.anomaly_sigma_threshold
        mon.apply_calibration(cfg)
        assert cfg.anomaly_sigma_threshold == original_sigma

    def test_auto_adjustments_in_health_report(self):
        mon = MetacognitiveMonitor()
        for _ in range(20):
            mon.record_feedback(False)
        cfg = _make_config(anomaly_sigma_threshold=2.0)
        mon.apply_calibration(cfg)
        report = mon.get_health_report()
        assert isinstance(report.auto_adjustments, list)
        assert len(report.auto_adjustments) > 0


# ---------------------------------------------------------------------------
# TestStateDict
# ---------------------------------------------------------------------------


class TestStateDict:
    def test_get_state_dict_returns_dict(self):
        mon = MetacognitiveMonitor()
        state = mon.get_state_dict()
        assert isinstance(state, dict)

    def test_state_dict_contains_frames_observed(self):
        mon = MetacognitiveMonitor()
        mon.observe_frame(_make_frame_result(1, confidence=0.7))
        state = mon.get_state_dict()
        assert "frames_observed" in state
        assert state["frames_observed"] == 1

    def test_state_dict_contains_quality_keys(self):
        mon = MetacognitiveMonitor()
        state = mon.get_state_dict()
        assert "quality_score" in state
        assert "quality_grade" in state
        assert "detection_confidence_avg" in state
