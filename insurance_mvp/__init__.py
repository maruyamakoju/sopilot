"""Insurance MVP - 保険映像レビュー自動化システム

Phase 1: 損保ジャパン PoC向けMVP

Main Components:
- Pipeline: End-to-end orchestration (config.py, pipeline.py)
- Mining: Multimodal signal analysis (mining/)
- Cosmos: Video-LLM inference (cosmos/)
- Insurance: Domain logic (insurance/)
- Conformal: Uncertainty quantification (conformal/)
- API: REST API server (api/)
"""

__version__ = "0.1.0"

# Export main components
from insurance_mvp.config import (
    ConformalConfig,
    CosmosBackend,
    CosmosConfig,
    DeviceType,
    MiningConfig,
    PipelineConfig,
    VideoConfig,
    WhisperBackend,
    WhisperConfig,
    load_config,
    save_config,
)
from insurance_mvp.conformal.split_conformal import (
    SplitConformal,
    compute_review_priority,
    ordinal_to_severity,
    severity_to_ordinal,
)
from insurance_mvp.insurance.schema import (
    AuditLog,
    ClaimAssessment,
    Evidence,
    FaultAssessment,
    FraudRisk,
    HazardDetail,
    ReviewDecision,
    create_default_claim_assessment,
)
from insurance_mvp.pipeline import (
    InsurancePipeline,
    PipelineMetrics,
    VideoResult,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "PipelineConfig",
    "load_config",
    "save_config",
    "VideoConfig",
    "MiningConfig",
    "CosmosConfig",
    "ConformalConfig",
    "WhisperConfig",
    "CosmosBackend",
    "DeviceType",
    "WhisperBackend",
    # Pipeline
    "InsurancePipeline",
    "PipelineMetrics",
    "VideoResult",
    # Insurance Domain
    "ClaimAssessment",
    "FaultAssessment",
    "FraudRisk",
    "Evidence",
    "HazardDetail",
    "ReviewDecision",
    "AuditLog",
    "create_default_claim_assessment",
    # Conformal
    "SplitConformal",
    "ConformalConfig",
    "compute_review_priority",
    "severity_to_ordinal",
    "ordinal_to_severity",
]
