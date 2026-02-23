"""Insurance Domain Models — re-exported from canonical location.

All models are defined in ``insurance_mvp.insurance.schema``.
This module re-exports them for backward compatibility.
"""

from insurance_mvp.insurance.schema import (
    AuditLog,
    ClaimAssessment,
    Evidence,
    FaultAssessment,
    FraudRisk,
    HazardDetail,
    ReviewDecision,
    create_default_claim_assessment,
)

__all__ = [
    "AuditLog",
    "ClaimAssessment",
    "Evidence",
    "FaultAssessment",
    "FraudRisk",
    "HazardDetail",
    "ReviewDecision",
    "create_default_claim_assessment",
]
