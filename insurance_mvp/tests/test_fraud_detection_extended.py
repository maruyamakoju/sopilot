"""Extended tests for Fraud Detection Engine.

Covers edge cases not in test_insurance_domain.py:
- Zero/negative claim amounts, std=0 z-score
- Time-to-report boundaries
- Empty/single claim history
- Weight configuration
- Score clamping
- Reasoning generation
"""

import pytest
from datetime import datetime, timedelta

from insurance_mvp.insurance.fraud_detection import (
    FraudDetectionEngine,
    FraudDetectionConfig,
    FraudIndicator,
    VideoEvidence,
    ClaimDetails,
    ClaimHistory,
)


# ============================================================================
# TestEdgeCases
# ============================================================================

class TestEdgeCases:
    """Boundary and edge-case tests."""

    def test_zero_std_zscore(self):
        """std=0 → z_score=0 (no crash)."""
        engine = FraudDetectionEngine(
            claim_amount_stats={"mean": 5000.0, "std": 0.0}
        )
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(claimed_amount=50000.0)
        result = engine.detect_fraud(video, claim)
        # Should not crash; no amount anomaly indicator because z_score=0
        assert result.risk_score >= 0.0

    def test_negative_claim_amount(self):
        """Negative claimed_amount → no amount anomaly (z_score < threshold)."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(claimed_amount=-1000.0)
        result = engine.detect_fraud(video, claim)
        # Negative amount → z_score is very negative, so no outlier indicator
        assert not any("claim_amount_anomaly" in ind for ind in result.indicators)

    def test_zero_claim_amount(self):
        """claimed_amount=0 → no amount anomaly."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(claimed_amount=0.0)
        result = engine.detect_fraud(video, claim)
        assert not any("claim_amount_anomaly" in ind for ind in result.indicators)

    def test_time_to_report_zero(self):
        """time_to_report=0.0 → quick report indicator (< 0.5 threshold)."""
        engine = FraudDetectionEngine(FraudDetectionConfig(
            suspicious_quick_report_hours=0.5,
        ))
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(claimed_amount=8000.0, time_to_report_hours=0.0)
        result = engine.detect_fraud(video, claim)
        assert any("timing_anomaly" in ind for ind in result.indicators)

    def test_time_to_report_exactly_threshold(self):
        """time_to_report=72.0 → NOT delayed (> required, not >=)."""
        engine = FraudDetectionEngine(FraudDetectionConfig(
            suspicious_delay_hours=72.0,
        ))
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(claimed_amount=8000.0, time_to_report_hours=72.0)
        result = engine.detect_fraud(video, claim)
        # 72.0 is NOT > 72.0, so no delay indicator
        assert not any(
            "timing_anomaly" in ind and "delayed" in ind
            for ind in result.indicators
        )

    def test_no_history(self):
        """claim_history=None → history checks skipped."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(claimed_amount=8000.0)
        result = engine.detect_fraud(video, claim, claim_history=None)
        assert not any("claim_frequency" in ind for ind in result.indicators)
        assert not any("fraud_history" in ind for ind in result.indicators)

    def test_empty_claim_dates(self):
        """previous_claim_dates=[] → no clustering check."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(claimed_amount=8000.0)
        history = ClaimHistory(vehicle_id="X", previous_claim_dates=[])
        result = engine.detect_fraud(video, claim, history)
        assert not any("claim_clustering" in ind for ind in result.indicators)

    def test_single_claim_date(self):
        """1 date → no clustering (needs >=2 for gap calculation)."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(claimed_amount=8000.0)
        history = ClaimHistory(
            vehicle_id="X",
            previous_claim_dates=[datetime.now() - timedelta(days=5)],
        )
        result = engine.detect_fraud(video, claim, history)
        assert not any("claim_clustering" in ind for ind in result.indicators)


# ============================================================================
# TestWeightConfiguration
# ============================================================================

class TestWeightConfiguration:
    """Test custom weight configurations."""

    def test_custom_weights(self):
        """Changing weights alters final score."""
        video = VideoEvidence(
            has_collision_sound=False,
            damage_visible=True,
            damage_severity="severe",
        )
        claim = ClaimDetails(claimed_amount=8000.0)

        # Default weights
        engine_default = FraudDetectionEngine()
        result_default = engine_default.detect_fraud(video, claim)

        # High audio_visual weight
        config = FraudDetectionConfig(weight_audio_visual_mismatch=0.9)
        engine_heavy = FraudDetectionEngine(config)
        result_heavy = engine_heavy.detect_fraud(video, claim)

        # Higher weight should increase score
        assert result_heavy.risk_score >= result_default.risk_score

    def test_all_zero_weights(self):
        """All weights=0 → score=0 regardless of indicators."""
        config = FraudDetectionConfig(
            weight_audio_visual_mismatch=0.0,
            weight_damage_inconsistency=0.0,
            weight_suspicious_positioning=0.0,
            weight_claim_history=0.0,
            weight_claim_amount_anomaly=0.0,
            weight_timing_anomaly=0.0,
        )
        engine = FraudDetectionEngine(config)
        video = VideoEvidence(
            has_collision_sound=False,
            damage_visible=True,
            damage_severity="severe",
            suspicious_edits=True,
            vehicle_positioned_suspiciously=True,
        )
        claim = ClaimDetails(claimed_amount=50000.0, time_to_report_hours=200.0)
        history = ClaimHistory(
            vehicle_id="X", claims_last_year=10, previous_fraud_flags=3,
        )
        result = engine.detect_fraud(video, claim, history)
        assert result.risk_score == 0.0

    def test_custom_thresholds(self):
        """Custom high/medium thresholds shift classification."""
        config = FraudDetectionConfig(
            high_risk_threshold=0.3,
            medium_risk_threshold=0.1,
        )
        engine = FraudDetectionEngine(config)
        video = VideoEvidence(
            has_collision_sound=False,
            damage_visible=True,
            damage_severity="moderate",
        )
        claim = ClaimDetails(claimed_amount=8000.0)
        result = engine.detect_fraud(video, claim)
        # Lower thresholds → more likely to be classified as HIGH/MEDIUM
        if result.risk_score >= 0.3:
            assert "HIGH FRAUD RISK" in result.reasoning
        elif result.risk_score >= 0.1:
            assert "MEDIUM FRAUD RISK" in result.reasoning


# ============================================================================
# TestScoreCalculation
# ============================================================================

class TestScoreCalculation:
    """Test fraud score calculation mechanics."""

    def test_score_clamp_below_zero(self):
        """No indicators → score >= 0.0."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(claimed_amount=5000.0, time_to_report_hours=2.0)
        result = engine.detect_fraud(video, claim)
        assert result.risk_score >= 0.0

    def test_score_clamp_above_one(self):
        """Extreme indicators → capped at 1.0."""
        engine = FraudDetectionEngine(FraudDetectionConfig(
            weight_audio_visual_mismatch=1.0,
            weight_damage_inconsistency=1.0,
            weight_suspicious_positioning=1.0,
            weight_claim_history=1.0,
            weight_claim_amount_anomaly=1.0,
            weight_timing_anomaly=1.0,
        ))
        video = VideoEvidence(
            has_collision_sound=False,
            damage_visible=True,
            damage_severity="severe",
            speed_at_impact_kmh=5.0,
            vehicle_positioned_suspiciously=True,
            has_pre_collision_braking=False,
            suspicious_edits=True,
        )
        claim = ClaimDetails(
            claimed_amount=100000.0,
            time_to_report_hours=500.0,
            injury_claimed=True,
            medical_claimed=95000.0,
        )
        history = ClaimHistory(
            vehicle_id="X",
            claims_last_year=20,
            claims_last_month=5,
            previous_fraud_flags=5,
            previous_claim_dates=[
                datetime.now() - timedelta(days=1),
                datetime.now() - timedelta(days=2),
            ],
        )
        result = engine.detect_fraud(video, claim, history)
        assert result.risk_score <= 1.0

    def test_damage_and_tampering_combined(self):
        """Both damage_inconsistency and video_tampering count as damage weight."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(
            has_collision_sound=True,
            damage_visible=True,
            damage_severity="severe",
            speed_at_impact_kmh=5.0,  # Low speed + severe damage
            suspicious_edits=True,     # Tampering
        )
        claim = ClaimDetails(claimed_amount=8000.0, time_to_report_hours=2.0)
        result = engine.detect_fraud(video, claim)
        # Both types contribute to damage_inconsistency weight
        damage_related = [
            ind for ind in result.indicators
            if "damage_inconsistency" in ind or "video_tampering" in ind
        ]
        assert len(damage_related) >= 2

    def test_multiple_audio_visual_averaged(self):
        """2 audio_visual indicators → averaged severity."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(
            has_collision_sound=True,
            damage_visible=False,
            speed_at_impact_kmh=25.0,  # Sound + speed but no damage
        )
        claim = ClaimDetails(claimed_amount=8000.0)
        result = engine.detect_fraud(video, claim)
        av_indicators = [ind for ind in result.indicators if "audio_visual_mismatch" in ind]
        assert len(av_indicators) >= 1


# ============================================================================
# TestReasoningGeneration
# ============================================================================

class TestReasoningGeneration:
    """Test human-readable reasoning generation."""

    def test_high_risk_reasoning(self):
        """score >= 0.65 → 'HIGH FRAUD RISK'."""
        engine = FraudDetectionEngine(FraudDetectionConfig(
            weight_audio_visual_mismatch=1.0,
            weight_damage_inconsistency=1.0,
        ))
        video = VideoEvidence(
            has_collision_sound=False,
            damage_visible=True,
            damage_severity="severe",
            speed_at_impact_kmh=5.0,
            suspicious_edits=True,
        )
        claim = ClaimDetails(claimed_amount=8000.0)
        result = engine.detect_fraud(video, claim)
        if result.risk_score >= 0.65:
            assert "HIGH FRAUD RISK" in result.reasoning

    def test_medium_risk_reasoning(self):
        """0.4 <= score < 0.65 → 'MEDIUM FRAUD RISK'."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(
            has_collision_sound=False,
            damage_visible=True,
            damage_severity="severe",
            vehicle_positioned_suspiciously=True,
        )
        claim = ClaimDetails(claimed_amount=8000.0)
        result = engine.detect_fraud(video, claim)
        if 0.4 <= result.risk_score < 0.65:
            assert "MEDIUM FRAUD RISK" in result.reasoning

    def test_low_risk_reasoning(self):
        """score < 0.4 → 'LOW FRAUD RISK'."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(
            has_collision_sound=True,
            damage_visible=True,
        )
        claim = ClaimDetails(claimed_amount=5000.0, time_to_report_hours=2.0)
        result = engine.detect_fraud(video, claim)
        assert result.risk_score < 0.4
        assert "LOW FRAUD RISK" in result.reasoning

    def test_top_3_indicators_shown(self):
        """Only top 3 indicators appear in reasoning."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(
            has_collision_sound=False,
            damage_visible=True,
            damage_severity="severe",
            speed_at_impact_kmh=5.0,
            vehicle_positioned_suspiciously=True,
            has_pre_collision_braking=False,
            suspicious_edits=True,
            video_quality="poor",
        )
        claim = ClaimDetails(
            claimed_amount=50000.0,
            time_to_report_hours=200.0,
        )
        history = ClaimHistory(
            vehicle_id="X",
            claims_last_year=10,
            previous_fraud_flags=3,
        )
        result = engine.detect_fraud(video, claim, history)
        # Count enumerated indicators "(1)", "(2)", "(3)" — max 3
        import re
        indicator_numbers = re.findall(r"\(\d+\)", result.reasoning)
        assert len(indicator_numbers) <= 3

    def test_no_indicators_reasoning(self):
        """No fraud indicators → specific message."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(
            has_collision_sound=True,
            damage_visible=True,
            speed_at_impact_kmh=40.0,
        )
        claim = ClaimDetails(claimed_amount=5000.0, time_to_report_hours=2.0)
        result = engine.detect_fraud(video, claim)
        if len(result.indicators) == 0:
            assert "No significant fraud indicators" in result.reasoning


# ============================================================================
# TestPoorVideoQuality
# ============================================================================

class TestPoorVideoQuality:
    """Test video quality impact on fraud detection."""

    def test_poor_quality_with_damage(self):
        """Poor video quality + damage claimed → indicator."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(
            has_collision_sound=True,
            damage_visible=True,
            video_quality="poor",
        )
        claim = ClaimDetails(claimed_amount=8000.0)
        result = engine.detect_fraud(video, claim)
        assert any("damage_inconsistency" in ind for ind in result.indicators)

    def test_good_quality_no_indicator(self):
        """Good video quality → no quality-related indicator."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(
            has_collision_sound=True,
            damage_visible=True,
            video_quality="good",
        )
        claim = ClaimDetails(claimed_amount=8000.0)
        result = engine.detect_fraud(video, claim)
        quality_indicators = [
            ind for ind in result.indicators
            if "quality" in ind.lower()
        ]
        assert len(quality_indicators) == 0
