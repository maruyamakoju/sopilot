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


def _build_scores(severity: str, confidence: float) -> np.ndarray:
    """Build a 4-class score vector from a severity string + confidence."""
    severity_idx = severity_to_ordinal(severity)
    scores = np.zeros(4)
    scores[severity_idx] = confidence
    scores += (1.0 - confidence) / 4.0
    scores /= scores.sum()
    return scores


def apply_conformal_single(
    raw_severity: str,
    raw_confidence: float,
    predictor: SplitConformal,
) -> set[str]:
    """Compute a conformal prediction set from **raw** VLM output.

    This must be called **before** any recalibration so that the prediction
    set reflects the original VLM distribution and the exchangeability
    assumption required for valid coverage guarantees is maintained.

    Args:
        raw_severity: VLM severity label before any recalibration.
        raw_confidence: VLM confidence before any recalibration.
        predictor: A calibrated SplitConformal predictor.

    Returns:
        Conformal prediction set (set of severity strings).
    """
    if not predictor or not predictor._calibrated:
        return {raw_severity}
    scores = _build_scores(raw_severity, raw_confidence)
    return predictor.predict_set_single(scores)


def apply_conformal(
    assessments: list[ClaimAssessment],
    predictor: SplitConformal,
) -> list[ClaimAssessment]:
    """Add conformal prediction sets to each assessment (batch version).

    NOTE: When conformal is applied per-clip inside ``_process_single_clip``
    (the preferred path), this function is a no-op because the prediction_set
    is already populated from the raw VLM scores.  It is retained for backward
    compatibility with tests that call it directly.

    Args:
        assessments: List of assessments to augment.
        predictor: A calibrated SplitConformal predictor.

    Returns:
        The same list with ``prediction_set`` updated in-place (skipped if
        prediction_set already contains more than one element, indicating
        it was set from raw VLM output earlier in the pipeline).
    """
    if not predictor or not predictor._calibrated:
        logger.warning("Conformal predictor not calibrated. Skipping.")
        return assessments

    for assessment in assessments:
        # Skip if conformal was already applied per-clip (prediction_set != singleton)
        if len(assessment.prediction_set) > 1:
            continue
        # Fallback: apply conformal on current (possibly recalibrated) severity
        scores = _build_scores(assessment.severity, assessment.confidence)
        pred_set = predictor.predict_set_single(scores)
        assessment.prediction_set = pred_set

    return assessments
