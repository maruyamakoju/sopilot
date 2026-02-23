"""Adapt VLM output to fault assessment and fraud detection engines."""

from __future__ import annotations

import logging
from typing import Any

from insurance_mvp.insurance.fault_assessment import (
    FaultAssessmentEngine,
    ScenarioContext,
    detect_scenario_type,
)
from insurance_mvp.insurance.fraud_detection import (
    ClaimDetails,
    FraudDetectionEngine,
    VideoEvidence,
)
from insurance_mvp.insurance.schema import FaultAssessment, FraudRisk

logger = logging.getLogger(__name__)


def assess_fault_from_vlm(
    vlm_result: dict[str, Any],
    clip: Any,
    engine: FaultAssessmentEngine,
) -> FaultAssessment:
    """Translate VLM output into a fault assessment.

    Args:
        vlm_result: Raw VLM inference dict.
        clip: Danger clip metadata dict.
        engine: Initialized FaultAssessmentEngine.

    Returns:
        FaultAssessment with ratio and reasoning.
    """
    causal = vlm_result.get("causal_reasoning", vlm_result.get("reasoning", ""))
    scenario_type = detect_scenario_type(causal)

    context = ScenarioContext(
        scenario_type=scenario_type,
        speed_ego_kmh=clip.get("speed_kmh"),
        ego_braking=clip.get("has_braking", False),
    )

    return engine.assess_fault(context)


def detect_fraud_from_vlm(
    vlm_result: dict[str, Any],
    clip: Any,
    engine: FraudDetectionEngine,
) -> FraudRisk:
    """Translate VLM output into a fraud risk assessment.

    Args:
        vlm_result: Raw VLM inference dict.
        clip: Danger clip metadata dict.
        engine: Initialized FraudDetectionEngine.

    Returns:
        FraudRisk with score and indicators.
    """
    severity = vlm_result.get("severity", "LOW")
    has_collision_sound = clip.get("has_crash_sound", False)

    damage_map = {"NONE": "none", "LOW": "minor", "MEDIUM": "moderate", "HIGH": "severe"}

    video_evidence = VideoEvidence(
        has_collision_sound=has_collision_sound,
        damage_visible=(severity in ["MEDIUM", "HIGH"]),
        damage_severity=damage_map.get(severity, "none"),
        speed_at_impact_kmh=clip.get("speed_kmh", 40.0),
        video_duration_sec=clip.get("duration_sec", 5.0),
    )

    claim_details = ClaimDetails(claimed_amount=10000.0)

    return engine.detect_fraud(video_evidence=video_evidence, claim_details=claim_details)
