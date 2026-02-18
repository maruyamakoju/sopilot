"""Video-LLM inference module for insurance claim assessment.

This module provides production-ready Video-LLM integration with:
- NVIDIA Cosmos Reason 2 (when available)
- Qwen2.5-VL-7B-Instruct (current default)
- Mock mode for testing

Key features:
- Robust 7-step JSON parsing pipeline
- Model caching (singleton pattern)
- GPU memory management
- Timeout handling (1200 seconds)
- Graceful degradation on failures

Example usage:
    ```python
    from insurance_mvp.cosmos import create_client

    # Create client (loads model once, caches for future use)
    client = create_client(model_name="qwen2.5-vl-7b", device="cuda")

    # Assess claim
    assessment = client.assess_claim(
        video_path="dashcam.mp4",
        video_id="claim_12345",
        start_sec=10.0,
        end_sec=30.0
    )

    print(f"Severity: {assessment.severity}")
    print(f"Fault ratio: {assessment.fault_assessment.fault_ratio}%")
    print(f"Fraud risk: {assessment.fraud_risk.risk_score}")
    ```

For testing without GPU:
    ```python
    client = create_client(model_name="mock", device="cpu")
    ```
"""

from .client import ModelName, VideoLLMClient, VLMConfig, create_client
from .prompt import (
    get_claim_assessment_prompt,
    get_fault_assessment_prompt,
    get_fraud_detection_prompt,
    get_quick_severity_prompt,
)
from .schema import (
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
    # Client
    "VideoLLMClient",
    "VLMConfig",
    "ModelName",
    "create_client",
    # Schema
    "ClaimAssessment",
    "FaultAssessment",
    "FraudRisk",
    "HazardDetail",
    "Evidence",
    "ReviewDecision",
    "AuditLog",
    "create_default_claim_assessment",
    # Prompts
    "get_claim_assessment_prompt",
    "get_quick_severity_prompt",
    "get_fault_assessment_prompt",
    "get_fraud_detection_prompt",
]
