"""Stage 2: Video-LLM Inference → ClaimAssessment per clip."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from insurance_mvp.insurance.schema import (
    ClaimAssessment,
    FaultAssessment,
    FraudRisk,
)

logger = logging.getLogger(__name__)


def mock_vlm_result(clip: Any) -> dict[str, Any]:
    """Generate scenario-aware mock VLM result for testing.

    Infers scenario from video filename to return appropriate severity,
    confidence, and reasoning.
    """
    video_path = str(clip.get("video_path", "")).lower()
    filename = Path(video_path).stem if video_path else ""

    if "collision" in filename or "crash" in filename:
        return {
            "severity": "HIGH",
            "confidence": 0.92,
            "reasoning": "Mock: Rear-end collision detected with significant impact",
            "causal_reasoning": "rear-end collision: ego vehicle struck the vehicle ahead",
            "hazards": [{"type": "collision", "actors": ["car", "car"]}],
            "evidence": [{"timestamp_sec": 20.0, "description": "Impact detected"}],
        }
    elif "near_miss" in filename or "near-miss" in filename:
        return {
            "severity": "MEDIUM",
            "confidence": 0.80,
            "reasoning": "Mock: Near-miss event with pedestrian avoidance",
            "causal_reasoning": "near-miss: ego vehicle braked to avoid pedestrian",
            "hazards": [{"type": "near_miss", "actors": ["car", "pedestrian"]}],
            "evidence": [{"timestamp_sec": 15.0, "description": "Emergency braking"}],
        }
    elif "normal" in filename or "safe" in filename:
        return {
            "severity": "NONE",
            "confidence": 0.95,
            "reasoning": "Mock: Normal driving with no incidents detected",
            "causal_reasoning": "normal driving conditions, no hazards observed",
            "hazards": [],
            "evidence": [],
        }
    else:
        return {
            "severity": "MEDIUM",
            "confidence": 0.75,
            "reasoning": "Mock: Unable to determine scenario from filename",
            "hazards": [],
            "evidence": [],
        }


def create_error_assessment(clip: Any, video_id: str, error_msg: str) -> ClaimAssessment:
    """Create default assessment for a clip that failed processing."""
    return ClaimAssessment(
        severity="LOW",
        confidence=0.0,
        prediction_set={"LOW", "MEDIUM", "HIGH"},
        review_priority="URGENT",
        fault_assessment=FaultAssessment(
            fault_ratio=50.0,
            reasoning=f"Error during processing: {error_msg}",
            applicable_rules=[],
            scenario_type="error",
        ),
        fraud_risk=FraudRisk(risk_score=0.0, indicators=[], reasoning="Not evaluated due to error"),
        hazards=[],
        evidence=[],
        causal_reasoning=f"Processing error: {error_msg}",
        recommended_action="REVIEW",
        video_id=video_id,
        processing_time_sec=0.0,
    )
