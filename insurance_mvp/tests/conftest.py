"""Shared test fixtures for Insurance MVP tests.

Provides factory functions and fixtures used across all test modules.
"""

import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import pytest
import numpy as np

from insurance_mvp.config import (
    PipelineConfig,
    CosmosBackend,
    CosmosConfig,
    MiningConfig,
    ConformalConfig as PipelineConformalConfig,
)
from insurance_mvp.insurance.schema import (
    ClaimAssessment,
    FaultAssessment,
    FraudRisk,
    Evidence,
    HazardDetail,
)
from insurance_mvp.insurance.fraud_detection import (
    VideoEvidence,
    ClaimDetails,
    ClaimHistory,
)


# ---------------------------------------------------------------------------
# Factory helpers (importable, not fixtures)
# ---------------------------------------------------------------------------

def make_test_config(**overrides) -> PipelineConfig:
    """Create a PipelineConfig suitable for testing.

    Uses mock backend, temp output dir, and disables heavy features by default.
    """
    defaults = dict(
        output_dir=tempfile.mkdtemp(prefix="ins_test_"),
        log_level="WARNING",
        parallel_workers=1,
        enable_conformal=False,
        enable_transcription=False,
        continue_on_error=True,
    )
    defaults.update(overrides)
    config = PipelineConfig(**defaults)
    config.cosmos.backend = CosmosBackend.MOCK
    return config


def make_video_evidence(preset: str = "clean", **overrides) -> VideoEvidence:
    """Create VideoEvidence with preset profiles.

    Presets:
        clean      - legitimate claim, consistent evidence
        suspicious - multiple inconsistencies
        staged     - strong indicators of staged accident
    """
    presets = {
        "clean": dict(
            has_collision_sound=True,
            has_pre_collision_braking=True,
            damage_visible=True,
            damage_severity="moderate",
            vehicle_positioned_suspiciously=False,
            speed_at_impact_kmh=40.0,
            impact_force_estimated=5.0,
            video_quality="good",
            video_duration_sec=30.0,
            suspicious_edits=False,
        ),
        "suspicious": dict(
            has_collision_sound=False,
            has_pre_collision_braking=False,
            damage_visible=True,
            damage_severity="severe",
            vehicle_positioned_suspiciously=False,
            speed_at_impact_kmh=5.0,
            impact_force_estimated=2.0,
            video_quality="fair",
            video_duration_sec=15.0,
            suspicious_edits=False,
        ),
        "staged": dict(
            has_collision_sound=False,
            has_pre_collision_braking=False,
            damage_visible=True,
            damage_severity="severe",
            vehicle_positioned_suspiciously=True,
            speed_at_impact_kmh=10.0,
            impact_force_estimated=1.0,
            video_quality="poor",
            video_duration_sec=10.0,
            suspicious_edits=True,
        ),
    }
    base = presets.get(preset, presets["clean"]).copy()
    base.update(overrides)
    return VideoEvidence(**base)


def make_claim_details(**overrides) -> ClaimDetails:
    """Create ClaimDetails with sensible defaults."""
    defaults = dict(
        claimed_amount=8000.0,
        injury_claimed=False,
        property_damage_claimed=0.0,
        medical_claimed=0.0,
        claimant_name="Test Claimant",
        claimant_phone="090-1234-5678",
        time_to_report_hours=2.0,
    )
    defaults.update(overrides)
    return ClaimDetails(**defaults)


def make_claim_history(**overrides) -> ClaimHistory:
    """Create ClaimHistory with sensible defaults."""
    defaults = dict(
        vehicle_id="TEST-001",
        num_previous_claims=0,
        claims_last_year=0,
        claims_last_month=0,
        previous_claim_dates=[],
        previous_fraud_flags=0,
        total_claimed_amount=0.0,
        average_claim_amount=0.0,
    )
    defaults.update(overrides)
    return ClaimHistory(**defaults)


def make_claim_assessment(
    severity: str = "MEDIUM",
    confidence: float = 0.75,
    prediction_set: set = None,
    review_priority: str = "STANDARD",
    fault_ratio: float = 50.0,
    fraud_score: float = 0.0,
    video_id: str = "test_video",
    **overrides,
) -> ClaimAssessment:
    """Create ClaimAssessment for testing."""
    if prediction_set is None:
        prediction_set = {severity}

    data = dict(
        severity=severity,
        confidence=confidence,
        prediction_set=prediction_set,
        review_priority=review_priority,
        fault_assessment=FaultAssessment(
            fault_ratio=fault_ratio,
            reasoning="Test fault reasoning",
            applicable_rules=["Test rule"],
            scenario_type="test",
        ),
        fraud_risk=FraudRisk(
            risk_score=fraud_score,
            indicators=[],
            reasoning="Test fraud reasoning",
        ),
        hazards=[],
        evidence=[],
        causal_reasoning="Test causal reasoning",
        recommended_action="REVIEW",
        video_id=video_id,
        processing_time_sec=1.0,
    )
    data.update(overrides)
    return ClaimAssessment(**data)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_output_dir():
    """Temporary output directory, cleaned up after test."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="ins_test_"))
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)
