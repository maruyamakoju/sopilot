"""Insurance MVP Pipeline Package.

Re-exports all public symbols for backward compatibility:
  from insurance_mvp.pipeline import InsurancePipeline, VideoResult, ...
"""

# Re-export mining symbols used by patch() in tests
from insurance_mvp.mining.fuse import SignalFuser  # noqa: F401

# CLI entry point
from insurance_mvp.pipeline.cli import main  # noqa: F401
from insurance_mvp.pipeline.orchestrator import (
    InsurancePipeline,
    PipelineMetrics,
    VideoResult,
)

__all__ = [
    "InsurancePipeline",
    "PipelineMetrics",
    "VideoResult",
    "main",
]
