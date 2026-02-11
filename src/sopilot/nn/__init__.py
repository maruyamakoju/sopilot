"""Neural network components for SOPilot.

Provides learned replacements for heuristic pipeline components:
- ProjectionHead: contrastive projection (replaces Z-score adapter)
- SoftDTW / SoftDTWCuda: differentiable alignment (replaces vanilla DTW)
- SinkhornDistance / GromovWassersteinDistance: optimal transport alignment
- ASFormer: transformer-based temporal action segmentation
- NeuralStepSegmenter: MS-TCN++ temporal segmentation (fallback)
- ScoringHead: learned scoring with uncertainty (replaces penalty formula)
- SplitConformalPredictor: distribution-free uncertainty quantification
- DILATELoss / SOPDilateLoss: shape + temporal distortion loss
- IntegratedGradientsExplainer: principled attribution method
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)


def _torch_available() -> bool:
    try:
        importlib.import_module("torch")
        return True
    except ImportError:
        return False


__all__ = [
    # Core
    "ProjectionHead",
    "SoftDTW",
    "SoftDTWAlignment",
    "NeuralStepSegmenter",
    "ScoringHead",
    "IsotonicCalibrator",
    # CUDA-accelerated Soft-DTW
    "SoftDTWCuda",
    "SoftDTWAlignmentCuda",
    "MultiScaleSoftDTW",
    # Optimal Transport
    "SinkhornDistance",
    "GromovWassersteinDistance",
    "FusedGromovWasserstein",
    "HierarchicalOTAlignment",
    # ASFormer
    "ASFormer",
    "ASFormerLoss",
    # Conformal prediction
    "SplitConformalPredictor",
    "ConformizedQuantileRegression",
    "AdaptiveConformalInference",
    "MondrianConformal",
    "ConformalMCDropout",
    # DILATE loss
    "DILATELoss",
    "SOPDilateLoss",
    # Explainability
    "IntegratedGradientsExplainer",
    "WachterCounterfactualExplainer",
]


def __getattr__(name: str):
    """Lazy imports so non-torch environments don't crash on import."""
    _map = {
        # Core modules
        "ProjectionHead": ".projection_head",
        "SoftDTW": ".soft_dtw",
        "SoftDTWAlignment": ".soft_dtw",
        "NeuralStepSegmenter": ".step_segmenter",
        "ScoringHead": ".scoring_head",
        "IsotonicCalibrator": ".scoring_head",
        # CUDA Soft-DTW
        "SoftDTWCuda": ".soft_dtw_cuda",
        "SoftDTWAlignmentCuda": ".soft_dtw_cuda",
        "MultiScaleSoftDTW": ".soft_dtw_cuda",
        # Optimal Transport
        "SinkhornDistance": ".optimal_transport",
        "GromovWassersteinDistance": ".optimal_transport",
        "FusedGromovWasserstein": ".optimal_transport",
        "HierarchicalOTAlignment": ".optimal_transport",
        # ASFormer
        "ASFormer": ".asformer",
        "ASFormerLoss": ".asformer",
        # Conformal prediction
        "SplitConformalPredictor": ".conformal",
        "ConformizedQuantileRegression": ".conformal",
        "AdaptiveConformalInference": ".conformal",
        "MondrianConformal": ".conformal",
        "ConformalMCDropout": ".conformal",
        # DILATE loss
        "DILATELoss": ".dilate_loss",
        "SOPDilateLoss": ".dilate_loss",
        # Explainability
        "IntegratedGradientsExplainer": ".explainability",
        "WachterCounterfactualExplainer": ".explainability",
    }
    if name in _map:
        mod = importlib.import_module(_map[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
