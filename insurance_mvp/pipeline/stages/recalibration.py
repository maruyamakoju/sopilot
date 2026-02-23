"""Stage 2.5: Post-VLM Severity Recalibration.

Adjusts VLM severity using mining signal scores (danger, motion, proximity)
to catch common misclassifications like near-miss → LOW instead of MEDIUM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SEVERITY_ORDER = ["NONE", "LOW", "MEDIUM", "HIGH"]


@dataclass
class RecalibrationConfig:
    """Configuration for post-VLM severity recalibration."""

    enabled: bool = True
    high_danger_threshold: float = 0.7
    low_danger_threshold: float = 0.2
    high_motion_threshold: float = 0.6
    high_proximity_threshold: float = 0.5
    max_bump_levels: int = 1
    confidence_penalty: float = 0.15


def _severity_index(severity: str) -> int:
    """Return ordinal index for severity level."""
    try:
        return SEVERITY_ORDER.index(severity.upper())
    except ValueError:
        return 1  # default to LOW for unknown


def recalibrate_severity(
    vlm_severity: str,
    vlm_confidence: float,
    danger_score: float,
    motion_score: float = 0.0,
    proximity_score: float = 0.0,
    config: RecalibrationConfig | None = None,
) -> tuple[str, float, str]:
    """Adjust VLM severity using mining signals.

    Rules:
    1. danger_score > high_danger_threshold AND VLM says LOW/NONE → bump up 1 level
    2. motion_score > 0.6 AND proximity_score > 0.5 AND VLM says LOW → bump to MEDIUM
    3. danger_score < low_danger_threshold AND VLM says HIGH → downgrade to MEDIUM
    4. When bumped, reduce confidence by confidence_penalty
    5. Never bump more than max_bump_levels

    Args:
        vlm_severity: Severity from VLM inference.
        vlm_confidence: Confidence from VLM inference.
        danger_score: Composite danger score from mining (0-1).
        motion_score: Motion signal score (0-1).
        proximity_score: Proximity signal score (0-1).
        config: Recalibration settings (uses defaults if None).

    Returns:
        Tuple of (adjusted_severity, adjusted_confidence, adjustment_reason).
    """
    if config is None:
        config = RecalibrationConfig()

    if not config.enabled:
        return vlm_severity, vlm_confidence, "recalibration_disabled"

    severity = vlm_severity.upper()
    idx = _severity_index(severity)
    reason = "no_adjustment"

    # Rule 1: High danger but VLM says LOW or NONE → bump up
    if danger_score > config.high_danger_threshold and idx <= 1:
        new_idx = min(idx + config.max_bump_levels, len(SEVERITY_ORDER) - 1)
        if new_idx != idx:
            severity = SEVERITY_ORDER[new_idx]
            vlm_confidence = max(0.0, vlm_confidence - config.confidence_penalty)
            reason = f"danger_score={danger_score:.2f}>threshold, bumped {vlm_severity}→{severity}"
            logger.info("Recalibration: %s", reason)
            return severity, vlm_confidence, reason

    # Rule 2: High motion + proximity but VLM says LOW → bump to MEDIUM
    if (
        motion_score > config.high_motion_threshold
        and proximity_score > config.high_proximity_threshold
        and idx == 1  # LOW
    ):
        severity = "MEDIUM"
        vlm_confidence = max(0.0, vlm_confidence - config.confidence_penalty)
        reason = f"motion={motion_score:.2f}+proximity={proximity_score:.2f} high, bumped LOW→MEDIUM"
        logger.info("Recalibration: %s", reason)
        return severity, vlm_confidence, reason

    # Rule 3: Low danger but VLM says HIGH → downgrade to MEDIUM
    if danger_score < config.low_danger_threshold and idx == 3:
        severity = "MEDIUM"
        vlm_confidence = max(0.0, vlm_confidence - config.confidence_penalty)
        reason = f"danger_score={danger_score:.2f}<threshold, downgraded HIGH→MEDIUM"
        logger.info("Recalibration: %s", reason)
        return severity, vlm_confidence, reason

    return severity, vlm_confidence, reason
