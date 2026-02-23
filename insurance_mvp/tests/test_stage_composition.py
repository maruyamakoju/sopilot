"""Integration tests for pipeline stage composition.

Tests that stages chain correctly and data flows through the full pipeline.
"""

from unittest.mock import Mock

from insurance_mvp.conformal.split_conformal import SplitConformal
from insurance_mvp.insurance.schema import ClaimAssessment
from insurance_mvp.pipeline.stages.conformal_stage import apply_conformal
from insurance_mvp.pipeline.stages.ranking import rank_by_severity
from insurance_mvp.pipeline.stages.review_priority import assign_review_priority
from insurance_mvp.tests.conftest import make_claim_assessment


class TestStageChaining:
    """Test that stages compose correctly in sequence."""

    def _make_assessments(self):
        """Create a varied set of assessments for testing."""
        return [
            make_claim_assessment(severity="LOW", confidence=0.6),
            make_claim_assessment(severity="HIGH", confidence=0.9),
            make_claim_assessment(severity="MEDIUM", confidence=0.7),
            make_claim_assessment(severity="HIGH", confidence=0.85),
        ]

    def test_ranking_then_review_priority(self):
        """Stage 3 → Stage 5: Ranking feeds into review priority."""
        assessments = self._make_assessments()
        ranked = rank_by_severity(assessments)
        with_priority = assign_review_priority(ranked)

        # Verify ordering maintained after Stage 5
        severities = [a.severity for a in with_priority]
        assert severities[0] == "HIGH"
        assert severities[-1] == "LOW"

        # Verify priorities assigned
        for a in with_priority:
            assert a.review_priority in ("URGENT", "STANDARD", "LOW_PRIORITY")

    def test_ranking_then_conformal_then_review(self):
        """Stage 3 → Stage 4 → Stage 5: Full downstream chain."""
        assessments = self._make_assessments()

        # Stage 3
        ranked = rank_by_severity(assessments)

        # Stage 4 with mock predictor
        predictor = Mock(spec=SplitConformal)
        predictor._calibrated = True
        predictor.predict_set_single.return_value = {"HIGH", "MEDIUM"}
        with_conformal = apply_conformal(ranked, predictor)

        # Stage 5
        final = assign_review_priority(with_conformal)

        assert len(final) == 4
        # All should have prediction sets now
        for a in final:
            assert isinstance(a.prediction_set, set)
            assert len(a.prediction_set) >= 1

    def test_empty_assessments_chain(self):
        """Empty list flows through all stages without error."""
        ranked = rank_by_severity([])
        assert ranked == []

        with_priority = assign_review_priority(ranked)
        assert with_priority == []

    def test_single_assessment_chain(self):
        """Single assessment flows through without issues."""
        assessments = [make_claim_assessment(severity="HIGH", confidence=0.95)]
        ranked = rank_by_severity(assessments)
        final = assign_review_priority(ranked)
        assert len(final) == 1
        assert final[0].severity == "HIGH"

    def test_stage_ordering_preserves_data(self):
        """Data integrity across stages — no field corruption."""
        original = make_claim_assessment(
            severity="MEDIUM",
            confidence=0.77,
            fault_ratio=65.0,
            fraud_score=0.3,
            video_id="integrity_test",
        )
        ranked = rank_by_severity([original])
        final = assign_review_priority(ranked)

        result = final[0]
        assert result.severity == "MEDIUM"
        assert result.confidence == 0.77
        assert result.fault_assessment.fault_ratio == 65.0
        assert result.fraud_risk.risk_score == 0.3
        assert result.video_id == "integrity_test"


class TestFaultFraudIntegration:
    """Test fault assessment + fraud detection used together."""

    def test_both_engines_produce_valid_assessment(self):
        from insurance_mvp.insurance.fault_assessment import (
            FaultAssessmentEngine,
            ScenarioContext,
            ScenarioType,
        )
        from insurance_mvp.insurance.fraud_detection import (
            FraudDetectionEngine,
        )
        from insurance_mvp.tests.conftest import make_claim_details, make_video_evidence

        # Run fault assessment
        fault_engine = FaultAssessmentEngine()
        context = ScenarioContext(scenario_type=ScenarioType.REAR_END)
        fault_result = fault_engine.assess_fault(context)

        # Run fraud detection
        fraud_engine = FraudDetectionEngine()
        evidence = make_video_evidence("clean")
        details = make_claim_details()
        fraud_result = fraud_engine.detect_fraud(evidence, details)

        # Both produce valid outputs
        assert 0 <= fault_result.fault_ratio <= 100
        assert 0 <= fraud_result.risk_score <= 1.0

        # Can be combined into a ClaimAssessment
        assessment = ClaimAssessment(
            severity="HIGH",
            confidence=0.85,
            prediction_set={"HIGH"},
            review_priority="URGENT",
            fault_assessment=fault_result,
            fraud_risk=fraud_result,
            hazards=[],
            evidence=[],
            causal_reasoning="Integration test",
            recommended_action="REVIEW",
            video_id="integration_test",
            processing_time_sec=1.0,
        )
        assert assessment.severity == "HIGH"
        assert assessment.fault_assessment.fault_ratio == 100.0
