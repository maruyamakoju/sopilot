"""Stage 4: Conformal Prediction Sets."""

from __future__ import annotations

import logging

import numpy as np

from insurance_mvp.conformal.split_conformal import (
    SplitConformal,
    severity_to_ordinal,
)
from insurance_mvp.insurance.schema import ClaimAssessment

logger = logging.getLogger(__name__)


def apply_conformal(
    assessments: list[ClaimAssessment],
    predictor: SplitConformal,
) -> list[ClaimAssessment]:
    """Add conformal prediction sets to each assessment.

    Args:
        assessments: List of assessments to augment.
        predictor: A calibrated SplitConformal predictor.

    Returns:
        The same list with ``prediction_set`` updated in-place.
    """
    if not predictor or not predictor._calibrated:
        logger.warning("Conformal predictor not calibrated. Skipping.")
        return assessments

    for assessment in assessments:
        severity_idx = severity_to_ordinal(assessment.severity)
        scores = np.zeros(4)
        scores[severity_idx] = assessment.confidence
        scores += (1 - assessment.confidence) / 4
        scores = scores / scores.sum()

        pred_set = predictor.predict_set_single(scores)
        assessment.prediction_set = pred_set

    return assessments
