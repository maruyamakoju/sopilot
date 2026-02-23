"""Stage 3: Severity Ranking."""

from __future__ import annotations

from insurance_mvp.insurance.schema import ClaimAssessment

_SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}


def rank_by_severity(assessments: list[ClaimAssessment]) -> list[ClaimAssessment]:
    """Sort assessments by severity (HIGH first) then by confidence (desc)."""
    return sorted(
        assessments,
        key=lambda a: (_SEVERITY_ORDER.get(a.severity, 4), -a.confidence),
    )
