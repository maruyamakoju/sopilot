"""Tests for Insurance Schema (Pydantic models).

Validates field constraints, serialization, and factory functions.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from insurance_mvp.insurance.schema import (
    Evidence,
    HazardDetail,
    FaultAssessment,
    FraudRisk,
    ClaimAssessment,
    ReviewDecision,
    AuditLog,
    create_default_claim_assessment,
)

# Use the cosmos schema for completeness (it's the same models re-exported)
from insurance_mvp.cosmos.schema import (
    ClaimAssessment as CosmosClaimAssessment,
    create_default_claim_assessment as cosmos_default,
)


# ============================================================================
# TestEvidence
# ============================================================================

class TestEvidence:
    """Evidence model validation."""

    def test_basic_creation(self):
        e = Evidence(timestamp_sec=5.0, description="Impact detected")
        assert e.timestamp_sec == 5.0
        assert e.description == "Impact detected"
        assert e.frame_path is None

    def test_with_frame_path(self):
        e = Evidence(timestamp_sec=10.0, description="Hazard", frame_path="/tmp/frame.jpg")
        assert e.frame_path == "/tmp/frame.jpg"


# ============================================================================
# TestHazardDetail
# ============================================================================

class TestHazardDetail:
    """HazardDetail model validation."""

    def test_basic_creation(self):
        h = HazardDetail(
            type="collision",
            actors=["car", "truck"],
            spatial_relation="front",
            timestamp_sec=20.0,
        )
        assert h.type == "collision"
        assert len(h.actors) == 2
        assert h.spatial_relation == "front"

    def test_empty_actors(self):
        h = HazardDetail(
            type="near_miss",
            actors=[],
            spatial_relation="left",
            timestamp_sec=15.0,
        )
        assert h.actors == []


# ============================================================================
# TestFaultAssessment
# ============================================================================

class TestFaultAssessment:
    """FaultAssessment model validation."""

    def test_valid_creation(self):
        fa = FaultAssessment(
            fault_ratio=75.0,
            reasoning="Left turn failure to yield",
            applicable_rules=["Must yield to oncoming traffic"],
            scenario_type="left_turn",
            traffic_signal="green",
            right_of_way="other",
        )
        assert fa.fault_ratio == 75.0

    def test_fault_ratio_bounds_zero(self):
        fa = FaultAssessment(
            fault_ratio=0.0,
            reasoning="Other fully at fault",
            scenario_type="head_on",
        )
        assert fa.fault_ratio == 0.0

    def test_fault_ratio_bounds_hundred(self):
        fa = FaultAssessment(
            fault_ratio=100.0,
            reasoning="Ego fully at fault",
            scenario_type="rear_end",
        )
        assert fa.fault_ratio == 100.0

    def test_fault_ratio_negative_rejected(self):
        with pytest.raises(ValidationError):
            FaultAssessment(
                fault_ratio=-1.0,
                reasoning="Invalid",
                scenario_type="test",
            )

    def test_fault_ratio_over_100_rejected(self):
        with pytest.raises(ValidationError):
            FaultAssessment(
                fault_ratio=100.1,
                reasoning="Invalid",
                scenario_type="test",
            )

    def test_default_applicable_rules(self):
        fa = FaultAssessment(
            fault_ratio=50.0,
            reasoning="Unknown",
            scenario_type="unknown",
        )
        assert fa.applicable_rules == []

    def test_optional_fields_none(self):
        fa = FaultAssessment(
            fault_ratio=50.0,
            reasoning="Test",
            scenario_type="unknown",
        )
        assert fa.traffic_signal is None
        assert fa.right_of_way is None


# ============================================================================
# TestFraudRisk
# ============================================================================

class TestFraudRisk:
    """FraudRisk model validation."""

    def test_risk_score_bounds_zero(self):
        fr = FraudRisk(risk_score=0.0, reasoning="Clean")
        assert fr.risk_score == 0.0

    def test_risk_score_bounds_one(self):
        fr = FraudRisk(risk_score=1.0, reasoning="Maximum fraud")
        assert fr.risk_score == 1.0

    def test_risk_score_over_one_rejected(self):
        with pytest.raises(ValidationError):
            FraudRisk(risk_score=1.1, reasoning="Invalid")

    def test_risk_score_negative_rejected(self):
        with pytest.raises(ValidationError):
            FraudRisk(risk_score=-0.1, reasoning="Invalid")

    def test_empty_indicators(self):
        fr = FraudRisk(risk_score=0.0, reasoning="Clean")
        assert fr.indicators == []

    def test_with_indicators(self):
        fr = FraudRisk(
            risk_score=0.8,
            indicators=["audio_visual_mismatch: No sound", "video_tampering: Edits"],
            reasoning="HIGH FRAUD RISK",
        )
        assert len(fr.indicators) == 2


# ============================================================================
# TestClaimAssessment
# ============================================================================

class TestClaimAssessment:
    """ClaimAssessment model validation."""

    def _make_valid(self, **overrides):
        defaults = dict(
            severity="MEDIUM",
            confidence=0.75,
            prediction_set={"MEDIUM"},
            review_priority="STANDARD",
            fault_assessment=FaultAssessment(
                fault_ratio=50.0,
                reasoning="Test",
                scenario_type="test",
            ),
            fraud_risk=FraudRisk(
                risk_score=0.0,
                reasoning="Clean",
            ),
            causal_reasoning="Test reasoning",
            recommended_action="REVIEW",
            video_id="test_vid",
            processing_time_sec=1.0,
        )
        defaults.update(overrides)
        return ClaimAssessment(**defaults)

    def test_valid_creation(self):
        ca = self._make_valid()
        assert ca.severity == "MEDIUM"
        assert ca.confidence == 0.75

    @pytest.mark.parametrize("severity", ["NONE", "LOW", "MEDIUM", "HIGH"])
    def test_severity_values(self, severity):
        ca = self._make_valid(severity=severity)
        assert ca.severity == severity

    def test_prediction_set_type(self):
        ca = self._make_valid(prediction_set={"LOW", "MEDIUM", "HIGH"})
        assert isinstance(ca.prediction_set, set)
        assert len(ca.prediction_set) == 3

    def test_default_timestamp(self):
        ca = self._make_valid()
        assert isinstance(ca.timestamp, datetime)

    def test_serialization_roundtrip(self):
        ca = self._make_valid()
        data = ca.model_dump()
        # Sets are serialized; reconstruct
        data["prediction_set"] = set(data["prediction_set"])
        ca2 = ClaimAssessment(**data)
        assert ca2.severity == ca.severity
        assert ca2.confidence == ca.confidence
        assert ca2.video_id == ca.video_id

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            self._make_valid(confidence=1.5)

    def test_confidence_negative(self):
        with pytest.raises(ValidationError):
            self._make_valid(confidence=-0.1)


# ============================================================================
# TestReviewDecision
# ============================================================================

class TestReviewDecision:
    """ReviewDecision model validation."""

    def test_basic_creation(self):
        rd = ReviewDecision(
            claim_id="CLM-001",
            reviewer_id="REV-001",
            decision="APPROVE",
            reasoning="Looks legitimate",
            review_time_sec=120.0,
        )
        assert rd.decision == "APPROVE"
        assert rd.severity_override is None
        assert rd.fault_ratio_override is None

    def test_with_overrides(self):
        rd = ReviewDecision(
            claim_id="CLM-002",
            reviewer_id="REV-002",
            decision="REJECT",
            severity_override="HIGH",
            fault_ratio_override=100.0,
            fraud_override=True,
            reasoning="Multiple red flags",
            comments="Investigated thoroughly",
            review_time_sec=300.0,
        )
        assert rd.severity_override == "HIGH"
        assert rd.fault_ratio_override == 100.0
        assert rd.fraud_override is True


# ============================================================================
# TestAuditLog
# ============================================================================

class TestAuditLog:
    """AuditLog model validation."""

    def test_basic_creation(self):
        al = AuditLog(
            claim_id="CLM-001",
            event_type="AI_ASSESSMENT",
            actor_type="AI",
            actor_id="qwen2.5-vl-7b-v1",
            explanation="Initial automated assessment",
        )
        assert al.event_type == "AI_ASSESSMENT"
        assert al.before_state is None
        assert al.after_state is None

    def test_with_state(self):
        al = AuditLog(
            claim_id="CLM-001",
            event_type="DECISION_CHANGE",
            before_state={"severity": "MEDIUM"},
            after_state={"severity": "HIGH"},
            actor_type="HUMAN",
            actor_id="reviewer-001",
            explanation="Upgraded severity after review",
        )
        assert al.before_state["severity"] == "MEDIUM"
        assert al.after_state["severity"] == "HIGH"


# ============================================================================
# TestCreateDefaultClaimAssessment
# ============================================================================

class TestCreateDefaultClaimAssessment:
    """Test factory function defaults."""

    def test_default_values(self):
        ca = create_default_claim_assessment("vid-123")
        assert ca.severity == "LOW"
        assert ca.confidence == 0.0
        assert ca.review_priority == "URGENT"
        assert ca.fault_assessment.fault_ratio == 50.0
        assert ca.fraud_risk.risk_score == 0.0
        assert ca.video_id == "vid-123"
        assert "LOW" in ca.prediction_set
        assert "MEDIUM" in ca.prediction_set
        assert "HIGH" in ca.prediction_set

    def test_cosmos_schema_matches(self):
        """Both schema modules export equivalent factory."""
        ca1 = create_default_claim_assessment("test")
        ca2 = cosmos_default("test")
        assert ca1.severity == ca2.severity
        assert ca1.confidence == ca2.confidence
        assert ca1.review_priority == ca2.review_priority
