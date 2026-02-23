"""Stage 5: Review Priority Assignment."""

from __future__ import annotations

from insurance_mvp.conformal.split_conformal import compute_review_priority
from insurance_mvp.insurance.schema import ClaimAssessment


def assign_review_priority(assessments: list[ClaimAssessment]) -> list[ClaimAssessment]:
    """Compute review priority from severity + prediction set uncertainty.

    Updates ``review_priority`` on each assessment in-place.
    """
    for assessment in assessments:
        assessment.review_priority = compute_review_priority(
            assessment.severity, assessment.prediction_set
        )
    return assessments
