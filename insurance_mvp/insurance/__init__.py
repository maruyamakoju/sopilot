"""Insurance Domain Logic Module.

Production-grade insurance claim assessment components:
- Fault assessment based on traffic scenarios and rules
- Fraud detection with multi-signal analysis
- Video analysis utilities

Example usage:

    from insurance_mvp.insurance import (
        FaultAssessmentEngine,
        ScenarioContext,
        ScenarioType,
        FraudDetectionEngine,
        VideoEvidence,
        ClaimDetails,
        extract_video_metadata,
        extract_keyframes,
    )

    # Fault Assessment
    fault_engine = FaultAssessmentEngine()
    context = ScenarioContext(
        scenario_type=ScenarioType.REAR_END,
        speed_ego_kmh=45.0,
        ego_braking=True,
    )
    fault_result = fault_engine.assess_fault(context)
    print(f"Fault: {fault_result.fault_ratio}%")

    # Fraud Detection
    fraud_engine = FraudDetectionEngine()
    video_evidence = VideoEvidence(
        has_collision_sound=False,
        damage_visible=True,
        speed_at_impact_kmh=40.0,
    )
    claim_details = ClaimDetails(claimed_amount=15000.0)
    fraud_result = fraud_engine.detect_fraud(video_evidence, claim_details)
    print(f"Fraud risk: {fraud_result.risk_score:.2f}")

    # Video Utilities
    metadata = extract_video_metadata("dashcam.mp4")
    print(metadata)

    evidence = extract_keyframes("dashcam.mp4", [5.0, 10.0, 15.0], output_dir="evidence/")
"""

from .schema import (
    Evidence,
    HazardDetail,
    FaultAssessment,
    FraudRisk,
    ClaimAssessment,
    ReviewDecision,
    AuditLog,
    create_default_claim_assessment,
)

from .fault_assessment import (
    FaultAssessmentEngine,
    FaultAssessmentConfig,
    ScenarioContext,
    ScenarioType,
    TrafficSignal,
    detect_scenario_type,
)

from .fraud_detection import (
    FraudDetectionEngine,
    FraudDetectionConfig,
    FraudIndicator,
    VideoEvidence,
    ClaimHistory,
    ClaimDetails,
)

from .utils import (
    VideoMetadata,
    extract_video_metadata,
    extract_keyframes,
    format_timestamp,
    parse_timestamp,
    calculate_frame_difference,
    detect_scene_changes,
    estimate_motion_intensity,
    calculate_video_quality_score,
)


__all__ = [
    # Schema
    "Evidence",
    "HazardDetail",
    "FaultAssessment",
    "FraudRisk",
    "ClaimAssessment",
    "ReviewDecision",
    "AuditLog",
    "create_default_claim_assessment",
    # Fault Assessment
    "FaultAssessmentEngine",
    "FaultAssessmentConfig",
    "ScenarioContext",
    "ScenarioType",
    "TrafficSignal",
    "detect_scenario_type",
    # Fraud Detection
    "FraudDetectionEngine",
    "FraudDetectionConfig",
    "FraudIndicator",
    "VideoEvidence",
    "ClaimHistory",
    "ClaimDetails",
    # Utils
    "VideoMetadata",
    "extract_video_metadata",
    "extract_keyframes",
    "format_timestamp",
    "parse_timestamp",
    "calculate_frame_difference",
    "detect_scene_changes",
    "estimate_motion_intensity",
    "calculate_video_quality_score",
]
