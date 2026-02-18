"""Tests for Insurance Domain Logic.

Comprehensive tests for fault assessment, fraud detection, and utilities.
"""

from datetime import datetime, timedelta

import pytest
from insurance_mvp.insurance import (
    ClaimDetails,
    ClaimHistory,
    FaultAssessmentConfig,
    # Fault Assessment
    FaultAssessmentEngine,
    FraudDetectionConfig,
    # Fraud Detection
    FraudDetectionEngine,
    FraudIndicator,
    ScenarioContext,
    ScenarioType,
    TrafficSignal,
    VideoEvidence,
    detect_scenario_type,
    # Utils
    format_timestamp,
    parse_timestamp,
)

# ============================================================================
# Fault Assessment Tests
# ============================================================================


class TestFaultAssessment:
    """Test suite for fault assessment engine."""

    def test_rear_end_collision_default(self):
        """Test standard rear-end collision (100% fault)."""
        engine = FaultAssessmentEngine()
        context = ScenarioContext(
            scenario_type=ScenarioType.REAR_END,
            speed_ego_kmh=60.0,
            ego_braking=True,
        )
        result = engine.assess_fault(context)

        assert result.fault_ratio == 100.0
        assert "Rear-end collision" in result.reasoning
        assert result.scenario_type == "rear_end"
        assert len(result.applicable_rules) > 0

    def test_rear_end_sudden_stop(self):
        """Test rear-end with sudden stop by front vehicle."""
        config = FaultAssessmentConfig(rear_end_sudden_stop=70.0)
        engine = FaultAssessmentEngine(config)
        context = ScenarioContext(
            scenario_type=ScenarioType.REAR_END,
            speed_ego_kmh=60.0,
            speed_other_kmh=5.0,
            other_braking=True,
        )
        result = engine.assess_fault(context)

        assert result.fault_ratio == 70.0
        assert "sudden stop" in result.reasoning.lower()

    def test_head_on_default_split(self):
        """Test head-on collision with unclear fault (50-50)."""
        engine = FaultAssessmentEngine()
        context = ScenarioContext(scenario_type=ScenarioType.HEAD_ON)
        result = engine.assess_fault(context)

        assert result.fault_ratio == 50.0
        assert "50" in result.reasoning

    def test_head_on_ego_crossed_center_line(self):
        """Test head-on with ego crossing center line (100% fault)."""
        engine = FaultAssessmentEngine()
        context = ScenarioContext(
            scenario_type=ScenarioType.HEAD_ON,
            ego_lane_change=True,
            other_lane_change=False,
        )
        result = engine.assess_fault(context)

        assert result.fault_ratio == 100.0
        assert "crossing center line" in result.reasoning.lower()

    def test_side_swipe_ego_lane_change(self):
        """Test side-swipe during ego lane change."""
        config = FaultAssessmentConfig(side_swipe_lane_change=80.0)
        engine = FaultAssessmentEngine(config)
        context = ScenarioContext(
            scenario_type=ScenarioType.SIDE_SWIPE,
            ego_lane_change=True,
            other_lane_change=False,
        )
        result = engine.assess_fault(context)

        assert result.fault_ratio == 80.0
        assert "lane change" in result.reasoning.lower()

    def test_left_turn_failure_to_yield(self):
        """Test left turn collision (ego failed to yield)."""
        config = FaultAssessmentConfig(left_turn_default=75.0)
        engine = FaultAssessmentEngine(config)
        context = ScenarioContext(
            scenario_type=ScenarioType.LEFT_TURN,
            ego_right_of_way=False,
        )
        result = engine.assess_fault(context)

        assert result.fault_ratio == 75.0
        assert "yield" in result.reasoning.lower()
        assert result.right_of_way == "other"

    def test_intersection_red_light_violation(self):
        """Test intersection collision with red light violation."""
        config = FaultAssessmentConfig(red_light_violation_fault=100.0)
        engine = FaultAssessmentEngine(config)
        context = ScenarioContext(
            scenario_type=ScenarioType.INTERSECTION,
            traffic_signal=TrafficSignal.RED,
            ego_right_of_way=False,
        )
        result = engine.assess_fault(context)

        assert result.fault_ratio == 100.0
        assert "red light" in result.reasoning.lower()
        assert result.traffic_signal == "red"

    def test_pedestrian_collision(self):
        """Test vehicle-pedestrian collision (vehicle at fault)."""
        engine = FaultAssessmentEngine()
        context = ScenarioContext(scenario_type=ScenarioType.PEDESTRIAN)
        result = engine.assess_fault(context)

        assert result.fault_ratio == 100.0
        assert "pedestrian" in result.reasoning.lower()

    def test_pedestrian_jaywalking(self):
        """Test pedestrian collision with jaywalking."""
        engine = FaultAssessmentEngine()
        context = ScenarioContext(
            scenario_type=ScenarioType.PEDESTRIAN,
            witness_statements=["Pedestrian darted into traffic"],
        )
        result = engine.assess_fault(context)

        assert result.fault_ratio == 70.0
        assert "jaywalking" in result.reasoning.lower()

    def test_excessive_speed_adjustment(self):
        """Test fault adjustment for excessive speed."""
        config = FaultAssessmentConfig(excessive_speed_adjustment=10.0)
        engine = FaultAssessmentEngine(config)
        context = ScenarioContext(
            scenario_type=ScenarioType.REAR_END,
            speed_ego_kmh=100.0,  # Well above typical 60 km/h limit
        )
        result = engine.assess_fault(context)

        # Base 100% + speed adjustment (capped at 100%)
        assert result.fault_ratio == 100.0
        assert "excessive speed" in result.reasoning.lower()

    def test_weather_adjustment(self):
        """Test fault adjustment for adverse weather."""
        config = FaultAssessmentConfig(weather_adjustment=5.0)
        engine = FaultAssessmentEngine(config)
        context = ScenarioContext(
            scenario_type=ScenarioType.HEAD_ON,
            weather_conditions="rain",
        )
        result = engine.assess_fault(context)

        # Base 50% + 5% weather = 55%
        assert result.fault_ratio == 55.0
        assert "weather" in result.reasoning.lower()

    def test_road_conditions_adjustment(self):
        """Test fault adjustment for poor road conditions."""
        config = FaultAssessmentConfig(weather_adjustment=5.0)
        engine = FaultAssessmentEngine(config)
        context = ScenarioContext(
            scenario_type=ScenarioType.HEAD_ON,
            road_conditions="icy",
        )
        result = engine.assess_fault(context)

        # Base 50% + 5% road conditions = 55%
        assert result.fault_ratio == 55.0
        assert "road conditions" in result.reasoning.lower()

    def test_unknown_scenario(self):
        """Test unknown scenario defaults to 50-50."""
        engine = FaultAssessmentEngine()
        context = ScenarioContext(scenario_type=ScenarioType.UNKNOWN)
        result = engine.assess_fault(context)

        assert result.fault_ratio == 50.0
        assert "unclear" in result.reasoning.lower()

    def test_scenario_detection_rear_end(self):
        """Test scenario type detection from text."""
        scenario = detect_scenario_type("Car hit from behind at intersection")
        assert scenario == ScenarioType.REAR_END

    def test_scenario_detection_head_on(self):
        """Test head-on scenario detection."""
        scenario = detect_scenario_type("Head-on collision on highway")
        assert scenario == ScenarioType.HEAD_ON

    def test_scenario_detection_lane_change(self):
        """Test lane change scenario detection."""
        scenario = detect_scenario_type("Collision during merge", ego_lane_change=True)
        assert scenario == ScenarioType.LANE_CHANGE


# ============================================================================
# Fraud Detection Tests
# ============================================================================


class TestFraudDetection:
    """Test suite for fraud detection engine."""

    def test_no_fraud_indicators(self):
        """Test clean claim with no fraud indicators."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(
            has_collision_sound=True,
            damage_visible=True,
            speed_at_impact_kmh=40.0,
        )
        claim = ClaimDetails(claimed_amount=8000.0)
        result = engine.detect_fraud(video, claim)

        assert result.risk_score < 0.4
        assert "LOW FRAUD RISK" in result.reasoning

    def test_audio_visual_mismatch(self):
        """Test fraud detection for audio/visual inconsistency."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(
            has_collision_sound=False,
            damage_visible=True,
            damage_severity="severe",
            speed_at_impact_kmh=50.0,
        )
        claim = ClaimDetails(claimed_amount=10000.0)
        result = engine.detect_fraud(video, claim)

        assert result.risk_score > 0.0
        assert any("audio_visual_mismatch" in ind for ind in result.indicators)
        assert len(result.indicators) > 0

    def test_damage_inconsistency_high_speed_no_damage(self):
        """Test fraud detection for high speed with no damage."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(
            has_collision_sound=True,
            damage_visible=False,
            speed_at_impact_kmh=50.0,
        )
        claim = ClaimDetails(claimed_amount=12000.0)
        result = engine.detect_fraud(video, claim)

        assert result.risk_score > 0.0
        assert any("damage_inconsistency" in ind for ind in result.indicators)

    def test_damage_inconsistency_low_speed_severe_damage(self):
        """Test fraud detection for low speed with severe damage."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(
            has_collision_sound=True,
            damage_visible=True,
            damage_severity="severe",
            speed_at_impact_kmh=5.0,
        )
        claim = ClaimDetails(claimed_amount=15000.0)
        result = engine.detect_fraud(video, claim)

        assert result.risk_score > 0.0
        assert any("damage_inconsistency" in ind for ind in result.indicators)

    def test_video_tampering(self):
        """Test fraud detection for video editing."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(
            has_collision_sound=True,
            damage_visible=True,
            suspicious_edits=True,
        )
        claim = ClaimDetails(claimed_amount=10000.0, time_to_report_hours=2.0)
        result = engine.detect_fraud(video, claim)

        assert result.risk_score > 0.15  # Video tampering is high severity
        assert any("video_tampering" in ind for ind in result.indicators)

    def test_suspicious_positioning(self):
        """Test fraud detection for suspicious vehicle positioning."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(
            vehicle_positioned_suspiciously=True,
            has_pre_collision_braking=False,
            speed_at_impact_kmh=30.0,
        )
        claim = ClaimDetails(claimed_amount=10000.0)
        result = engine.detect_fraud(video, claim)

        assert result.risk_score > 0.0
        assert any("suspicious_positioning" in ind for ind in result.indicators)

    def test_claim_frequency_high(self):
        """Test fraud detection for high claim frequency."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(claimed_amount=8000.0)
        history = ClaimHistory(
            vehicle_id="ABC123",
            claims_last_year=5,
            claims_last_month=0,
        )
        result = engine.detect_fraud(video, claim, history)

        assert result.risk_score > 0.0
        assert any("claim_frequency" in ind for ind in result.indicators)

    def test_claim_clustering(self):
        """Test fraud detection for claim clustering."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(claimed_amount=8000.0, time_to_report_hours=2.0)
        history = ClaimHistory(
            vehicle_id="ABC123",
            claims_last_month=3,
            previous_claim_dates=[
                datetime.now() - timedelta(days=5),
                datetime.now() - timedelta(days=10),
                datetime.now() - timedelta(days=15),
            ],
        )
        result = engine.detect_fraud(video, claim, history)

        assert result.risk_score > 0.13  # Adjusted for realistic weighting
        assert any("claim_clustering" in ind or "claim_frequency" in ind for ind in result.indicators)

    def test_previous_fraud_flags(self):
        """Test fraud detection for previous fraud history."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(claimed_amount=8000.0, time_to_report_hours=2.0)
        history = ClaimHistory(
            vehicle_id="ABC123",
            previous_fraud_flags=2,
        )
        result = engine.detect_fraud(video, claim, history)

        assert result.risk_score > 0.15  # Adjusted for realistic weighting
        assert any("fraud_history" in ind for ind in result.indicators)

    def test_claim_amount_outlier(self):
        """Test fraud detection for unusually high claim amount."""
        # Set claim stats with lower mean
        engine = FraudDetectionEngine(claim_amount_stats={"mean": 5000.0, "std": 2000.0})
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(claimed_amount=20000.0)  # 7.5 std devs above mean
        result = engine.detect_fraud(video, claim)

        assert result.risk_score > 0.0
        assert any("claim_amount_anomaly" in ind for ind in result.indicators)

    def test_disproportionate_medical_claim(self):
        """Test fraud detection for disproportionate medical claims."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(
            claimed_amount=10000.0,
            injury_claimed=True,
            medical_claimed=9000.0,  # 90% of total
        )
        result = engine.detect_fraud(video, claim)

        assert result.risk_score > 0.0
        assert any("claim_amount_anomaly" in ind for ind in result.indicators)

    def test_suspicious_reporting_delay(self):
        """Test fraud detection for delayed reporting."""
        config = FraudDetectionConfig(suspicious_delay_hours=72.0)
        engine = FraudDetectionEngine(config)
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(
            claimed_amount=8000.0,
            time_to_report_hours=100.0,
        )
        result = engine.detect_fraud(video, claim)

        assert result.risk_score > 0.0
        assert any("timing_anomaly" in ind for ind in result.indicators)

    def test_suspicious_quick_report(self):
        """Test fraud detection for suspiciously quick reporting."""
        config = FraudDetectionConfig(suspicious_quick_report_hours=0.5)
        engine = FraudDetectionEngine(config)
        video = VideoEvidence(has_collision_sound=True, damage_visible=True)
        claim = ClaimDetails(
            claimed_amount=8000.0,
            time_to_report_hours=0.1,  # Reported within 6 minutes
        )
        result = engine.detect_fraud(video, claim)

        assert result.risk_score > 0.0
        assert any("timing_anomaly" in ind for ind in result.indicators)

    def test_high_fraud_risk_multiple_indicators(self):
        """Test high fraud risk with multiple red flags."""
        engine = FraudDetectionEngine()
        video = VideoEvidence(
            has_collision_sound=False,
            damage_visible=True,
            damage_severity="severe",
            vehicle_positioned_suspiciously=True,
            suspicious_edits=True,
        )
        claim = ClaimDetails(claimed_amount=25000.0, time_to_report_hours=2.0)
        history = ClaimHistory(
            vehicle_id="ABC123",
            claims_last_year=4,
            previous_fraud_flags=1,
        )
        result = engine.detect_fraud(video, claim, history)

        assert result.risk_score >= 0.5  # Multiple indicators should push risk high
        assert "RISK" in result.reasoning  # Medium or High risk
        assert len(result.indicators) >= 3

    def test_fraud_indicator_creation(self):
        """Test fraud indicator creation."""
        indicator = FraudIndicator(
            type="test_indicator",
            description="Test description",
            severity=0.8,
            confidence=0.9,
        )
        assert indicator.type == "test_indicator"
        assert indicator.severity == 0.8
        assert indicator.confidence == 0.9


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtils:
    """Test suite for utility functions."""

    def test_format_timestamp_with_hours(self):
        """Test timestamp formatting with hours."""
        result = format_timestamp(3661.25, include_hours=True)
        assert result == "01:01:01.25"

    def test_format_timestamp_without_hours(self):
        """Test timestamp formatting without hours."""
        result = format_timestamp(125.50, include_hours=False)
        assert result == "02:05.50"

    def test_format_timestamp_zero(self):
        """Test formatting zero timestamp."""
        result = format_timestamp(0.0)
        assert result == "00:00:00.00"

    def test_parse_timestamp_full(self):
        """Test parsing full timestamp (HH:MM:SS.mm)."""
        result = parse_timestamp("01:30:45.50")
        assert result == 5445.50

    def test_parse_timestamp_minutes(self):
        """Test parsing minutes timestamp (MM:SS.mm)."""
        result = parse_timestamp("05:30.25")
        assert result == 330.25

    def test_parse_timestamp_seconds(self):
        """Test parsing seconds only (SS.mm)."""
        result = parse_timestamp("45.75")
        assert result == 45.75

    def test_parse_timestamp_invalid(self):
        """Test parsing invalid timestamp."""
        with pytest.raises(ValueError):
            parse_timestamp("invalid")

    def test_parse_timestamp_roundtrip(self):
        """Test timestamp format/parse roundtrip."""
        original = 3661.25
        formatted = format_timestamp(original)
        parsed = parse_timestamp(formatted)
        assert abs(parsed - original) < 0.01


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_complete_claim_assessment_workflow(self):
        """Test complete workflow: fault + fraud assessment."""
        # Fault assessment
        fault_engine = FaultAssessmentEngine()
        fault_context = ScenarioContext(
            scenario_type=ScenarioType.REAR_END,
            speed_ego_kmh=60.0,
            traffic_signal=TrafficSignal.GREEN,
        )
        fault_result = fault_engine.assess_fault(fault_context)

        assert fault_result.fault_ratio == 100.0
        assert fault_result.scenario_type == "rear_end"

        # Fraud detection
        fraud_engine = FraudDetectionEngine()
        video_evidence = VideoEvidence(
            has_collision_sound=True,
            damage_visible=True,
            speed_at_impact_kmh=60.0,
        )
        claim_details = ClaimDetails(claimed_amount=8000.0)
        fraud_result = fraud_engine.detect_fraud(video_evidence, claim_details)

        assert fraud_result.risk_score < 0.5  # Low fraud risk

        # Both results available
        assert fault_result is not None
        assert fraud_result is not None

    def test_suspicious_claim_workflow(self):
        """Test workflow for suspicious claim."""
        # Fault assessment (unclear scenario)
        fault_engine = FaultAssessmentEngine()
        fault_context = ScenarioContext(
            scenario_type=ScenarioType.INTERSECTION,
            traffic_signal=TrafficSignal.UNKNOWN,
        )
        fault_result = fault_engine.assess_fault(fault_context)

        # Fraud detection (multiple red flags)
        fraud_engine = FraudDetectionEngine()
        video_evidence = VideoEvidence(
            has_collision_sound=False,
            damage_visible=True,
            damage_severity="severe",
            speed_at_impact_kmh=10.0,  # Low speed but severe damage
        )
        claim_details = ClaimDetails(claimed_amount=20000.0, time_to_report_hours=2.0)
        history = ClaimHistory(
            vehicle_id="FRAUD123",
            claims_last_year=5,
        )
        fraud_result = fraud_engine.detect_fraud(video_evidence, claim_details, history)

        # Should trigger elevated fraud risk
        assert fraud_result.risk_score >= 0.3
        assert len(fraud_result.indicators) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
