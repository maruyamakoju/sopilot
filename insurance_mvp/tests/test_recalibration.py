"""Tests for Phase 2: Post-VLM severity recalibration.

Validates that mining signal scores correctly adjust VLM severity
predictions to catch misclassifications.
"""

import pytest
from insurance_mvp.pipeline.stages.recalibration import (
    SEVERITY_ORDER,
    RecalibrationConfig,
    recalibrate_severity,
)


class TestRecalibrationBasic:
    """Core recalibration rules."""

    def test_low_bumped_to_medium_on_high_danger(self):
        """HIGH danger score + VLM says LOW → bumped to MEDIUM."""
        severity, conf, reason = recalibrate_severity(
            vlm_severity="LOW", vlm_confidence=0.8, danger_score=0.85
        )
        assert severity == "MEDIUM"
        assert "bumped" in reason

    def test_none_bumped_to_low_on_high_danger(self):
        """HIGH danger score + VLM says NONE → bumped to LOW."""
        severity, conf, reason = recalibrate_severity(
            vlm_severity="NONE", vlm_confidence=0.9, danger_score=0.75
        )
        assert severity == "LOW"
        assert "bumped" in reason

    def test_low_bumped_to_medium_on_high_motion_proximity(self):
        """HIGH motion + proximity but LOW VLM → bumped to MEDIUM."""
        severity, conf, reason = recalibrate_severity(
            vlm_severity="LOW",
            vlm_confidence=0.8,
            danger_score=0.5,  # not high enough for rule 1
            motion_score=0.7,
            proximity_score=0.6,
        )
        assert severity == "MEDIUM"
        assert "motion" in reason

    def test_high_downgraded_on_low_danger(self):
        """LOW danger score + VLM says HIGH → downgraded to MEDIUM."""
        severity, conf, reason = recalibrate_severity(
            vlm_severity="HIGH", vlm_confidence=0.9, danger_score=0.1
        )
        assert severity == "MEDIUM"
        assert "downgraded" in reason

    def test_no_change_when_signals_agree(self):
        """No adjustment when signals agree with VLM."""
        severity, conf, reason = recalibrate_severity(
            vlm_severity="MEDIUM", vlm_confidence=0.85, danger_score=0.5
        )
        assert severity == "MEDIUM"
        assert reason == "no_adjustment"

    def test_no_change_for_high_with_high_danger(self):
        """HIGH VLM + high danger → no change (already correct)."""
        severity, conf, reason = recalibrate_severity(
            vlm_severity="HIGH", vlm_confidence=0.9, danger_score=0.9
        )
        assert severity == "HIGH"
        assert reason == "no_adjustment"


class TestMaxBumpLimit:
    """Bump limits and very-high-danger escalation."""

    def test_none_very_high_danger_jumps_to_high(self):
        """NONE + danger > very_high_threshold → jumps directly to HIGH (Rule 1b)."""
        severity, conf, reason = recalibrate_severity(
            vlm_severity="NONE", vlm_confidence=0.9, danger_score=0.9
        )
        assert severity == "HIGH"  # Rule 1b: skip directly to HIGH
        assert "very_high_threshold" in reason

    def test_none_moderate_danger_bumps_one_level(self):
        """NONE + danger in (high_threshold, very_high_threshold] → bumps to LOW only."""
        config = RecalibrationConfig(very_high_danger_threshold=0.85)
        severity, conf, reason = recalibrate_severity(
            vlm_severity="NONE", vlm_confidence=0.9, danger_score=0.75, config=config
        )
        assert severity == "LOW"  # Rule 1: 1 level bump

    def test_custom_max_bump_below_very_high(self):
        """max_bump_levels=2 with danger below very_high_threshold → NONE→MEDIUM."""
        config = RecalibrationConfig(max_bump_levels=2, very_high_danger_threshold=0.95)
        severity, conf, reason = recalibrate_severity(
            vlm_severity="NONE", vlm_confidence=0.9, danger_score=0.9, config=config
        )
        assert severity == "MEDIUM"  # Rule 1 with max_bump=2, very_high=0.95 not triggered


class TestConfidencePenalty:
    """Confidence must be reduced on recalibration bumps."""

    def test_confidence_reduced_on_bump(self):
        """Bumped severity → confidence reduced by penalty."""
        _, conf, _ = recalibrate_severity(
            vlm_severity="LOW", vlm_confidence=0.8, danger_score=0.85
        )
        assert conf == pytest.approx(0.65, abs=0.01)

    def test_confidence_not_below_zero(self):
        """Confidence never goes negative."""
        _, conf, _ = recalibrate_severity(
            vlm_severity="LOW", vlm_confidence=0.1, danger_score=0.85
        )
        assert conf >= 0.0

    def test_confidence_unchanged_on_no_adjustment(self):
        """No adjustment → confidence unchanged."""
        _, conf, _ = recalibrate_severity(
            vlm_severity="MEDIUM", vlm_confidence=0.85, danger_score=0.5
        )
        assert conf == 0.85


class TestDisabledConfig:
    """Recalibration can be disabled via config."""

    def test_disabled_returns_original(self):
        """Disabled config → no adjustment even with high danger."""
        config = RecalibrationConfig(enabled=False)
        severity, conf, reason = recalibrate_severity(
            vlm_severity="LOW", vlm_confidence=0.8, danger_score=0.9, config=config
        )
        assert severity == "LOW"
        assert conf == 0.8
        assert reason == "recalibration_disabled"


class TestMissingScores:
    """Handle missing or default signal scores gracefully."""

    def test_default_scores_no_crash(self):
        """Default motion/proximity = 0.0 → no crash."""
        severity, conf, reason = recalibrate_severity(
            vlm_severity="MEDIUM", vlm_confidence=0.8, danger_score=0.5
        )
        assert severity == "MEDIUM"

    def test_zero_danger_score(self):
        """danger_score=0.0 with HIGH VLM → downgraded (rule 3)."""
        severity, conf, reason = recalibrate_severity(
            vlm_severity="HIGH", vlm_confidence=0.9, danger_score=0.0
        )
        assert severity == "MEDIUM"


class TestSeverityOrder:
    """SEVERITY_ORDER constant must be correct."""

    def test_severity_order(self):
        assert SEVERITY_ORDER == ["NONE", "LOW", "MEDIUM", "HIGH"]

    def test_unknown_severity_defaults_to_low(self):
        """Unknown severity string defaults to LOW index."""
        severity, _, _ = recalibrate_severity(
            vlm_severity="UNKNOWN", vlm_confidence=0.5, danger_score=0.5
        )
        # Unknown maps to index 1 (LOW), with no rule triggering
        assert severity == "UNKNOWN"
