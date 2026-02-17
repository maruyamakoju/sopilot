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
    PipelineConfig,
    load_config,
    save_config,
    VideoConfig,
    MiningConfig,
    CosmosConfig,
    ConformalConfig,
    WhisperConfig,
    CosmosBackend,
    DeviceType,
    WhisperBackend,
)

from insurance_mvp.pipeline import (
    InsurancePipeline,
    PipelineMetrics,
    VideoResult,
)

from insurance_mvp.insurance.schema import (
    ClaimAssessment,
    FaultAssessment,
    FraudRisk,
    Evidence,
    HazardDetail,
    ReviewDecision,
    AuditLog,
    create_default_claim_assessment,
)

from insurance_mvp.conformal.split_conformal import (
    SplitConformal,
    ConformalConfig,
    compute_review_priority,
    severity_to_ordinal,
    ordinal_to_severity,
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
