"""Insurance Domain Models

Pydantic models for insurance claim assessment.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """Evidence from video clip"""

    timestamp_sec: float = Field(description="Timestamp in seconds")
    description: str = Field(description="What happened at this timestamp")
    frame_path: str | None = Field(default=None, description="Path to keyframe image")


class HazardDetail(BaseModel):
    """Hazard detected in video"""

    type: str = Field(description="Hazard type: collision, near_miss, traffic_violation, etc.")
    actors: list[str] = Field(description="Actors involved: car, pedestrian, bicycle, etc.")
    spatial_relation: str = Field(description="Spatial relationship: front, left, right, etc.")
    timestamp_sec: float = Field(description="When hazard occurred")


class FaultAssessment(BaseModel):
    """Fault ratio assessment"""

    fault_ratio: float = Field(ge=0.0, le=100.0, description="Fault percentage (0-100%)")
    reasoning: str = Field(description="Why this fault ratio")
    applicable_rules: list[str] = Field(default_factory=list, description="Traffic rules applied")

    # Context
    scenario_type: str = Field(description="rear_end, head_on, side_swipe, etc.")
    traffic_signal: str | None = Field(default=None, description="red, yellow, green")
    right_of_way: str | None = Field(default=None, description="Who had right of way")


class FraudRisk(BaseModel):
    """Fraud risk assessment"""

    risk_score: float = Field(ge=0.0, le=1.0, description="Fraud risk (0.0-1.0)")
    indicators: list[str] = Field(default_factory=list, description="Fraud indicators detected")
    reasoning: str = Field(description="Why suspicious")


class ClaimAssessment(BaseModel):
    """Complete insurance claim assessment"""

    # Core assessment
    severity: Literal["NONE", "LOW", "MEDIUM", "HIGH"] = Field(description="Incident severity level")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence (0.0-1.0)")

    # Conformal prediction (uncertainty quantification)
    prediction_set: set[str] = Field(description="Conformal prediction set, e.g., {MEDIUM, HIGH}")
    review_priority: Literal["URGENT", "STANDARD", "LOW_PRIORITY"] = Field(description="Human review priority")

    # Fault assessment
    fault_assessment: FaultAssessment

    # Fraud detection
    fraud_risk: FraudRisk

    # Hazards and evidence
    hazards: list[HazardDetail] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    # Reasoning
    causal_reasoning: str = Field(description="Why this severity")
    recommended_action: str = Field(description="APPROVE, REVIEW, REJECT, REQUEST_MORE_INFO")

    # Metadata
    video_id: str
    processing_time_sec: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReviewDecision(BaseModel):
    """Human review decision"""

    claim_id: str
    reviewer_id: str
    decision: str = Field(description="APPROVE, REJECT, REQUEST_MORE_INFO")

    # Overrides
    severity_override: str | None = None
    fault_ratio_override: float | None = None
    fraud_override: bool | None = None

    # Reasoning
    reasoning: str
    comments: str | None = None

    # Metadata
    review_time_sec: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AuditLog(BaseModel):
    """Audit log entry for regulatory compliance"""

    claim_id: str
    event_type: str = Field(description="AI_ASSESSMENT, HUMAN_REVIEW, DECISION_CHANGE, etc.")

    # Before/After
    before_state: dict | None = None
    after_state: dict | None = None

    # Actor
    actor_type: str = Field(description="AI, HUMAN")
    actor_id: str = Field(description="Model version or reviewer ID")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    explanation: str


# Convenience functions


def create_default_claim_assessment(video_id: str) -> ClaimAssessment:
    """Create default assessment with safe values"""
    return ClaimAssessment(
        severity="LOW",
        confidence=0.0,
        prediction_set={"LOW", "MEDIUM", "HIGH"},  # Maximum uncertainty
        review_priority="URGENT",  # Default to human review
        fault_assessment=FaultAssessment(
            fault_ratio=50.0,  # Neutral
            reasoning="Default assessment, requires human review",
            applicable_rules=[],
            scenario_type="unknown",
        ),
        fraud_risk=FraudRisk(risk_score=0.0, indicators=[], reasoning="Not evaluated"),
        hazards=[],
        evidence=[],
        causal_reasoning="Automated assessment pending",
        recommended_action="REVIEW",
        video_id=video_id,
        processing_time_sec=0.0,
    )
