"""Neural Step Segmenter — replaces threshold-based boundary detection.

Implements a lightweight MS-TCN++ inspired architecture (Li et al. 2020)
for temporal action segmentation. Two-stage prediction-refinement with
dilated causal convolutions.

Architecture:
    Stage 1 (prediction):
        Conv1d(D, 64, 1) -> 10x DilatedResidualBlock(64, dilation=[1,2,4,...,512])
        -> Conv1d(64, 2, 1)  # binary boundary / non-boundary

    Stage 2 (refinement):
        Conv1d(2+D, 64, 1) -> 10x DilatedResidualBlock(64, dilation=[1,2,4,...,512])
        -> Conv1d(64, 2, 1)

    Receptive field: 1023 frames per stage (2046 total) with dilations up to 512.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DilatedResidualBlock(nn.Module):
    """Single dilated temporal convolution with residual connection."""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, C, T)"""
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out + x  # Residual


class _SegmentationStage(nn.Module):
    """Single MS-TCN stage: input projection + dilated residual blocks + output."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int = 64,
        n_classes: int = 2,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dilations = dilations
        self.input_proj = nn.Conv1d(d_in, d_hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [DilatedResidualBlock(d_hidden, d, dropout) for d in dilations]
        )
        self.output_proj = nn.Conv1d(d_hidden, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D_in, T) -> (B, n_classes, T)"""
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class NeuralStepSegmenter(nn.Module):
    """Two-stage temporal segmentation network for step boundary detection.

    Classifies each temporal position as boundary (1) or non-boundary (0).
    Stage 2 refines Stage 1's predictions by concatenating them with input features.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int = 64,
        n_classes: int = 2,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.dilations = dilations
        self.stage1 = _SegmentationStage(d_in, d_hidden, n_classes, dilations, dropout)
        self.stage2 = _SegmentationStage(
            n_classes + d_in, d_hidden, n_classes, dilations, dropout
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both stages.

        Args:
            x: (B, D, T) temporal feature sequence.

        Returns:
            (stage1_logits, stage2_logits) each (B, 2, T).
        """
        logits1 = self.stage1(x)
        # Concatenate stage1 softmax predictions with input features
        probs1 = F.softmax(logits1, dim=1)
        combined = torch.cat([probs1, x], dim=1)  # (B, 2+D, T)
        logits2 = self.stage2(combined)
        return logits1, logits2

    def predict_boundaries(
        self,
        embeddings: np.ndarray,
        min_step_clips: int = 2,
        threshold: float = 0.5,
    ) -> list[int]:
        """Predict step boundaries from embeddings.

        Args:
            embeddings: (T, D) clip embeddings.
            min_step_clips: Minimum clips between boundaries.
            threshold: Classification threshold for boundary class.

        Returns:
            List of boundary indices [0, b1, b2, ..., T].
        """
        t, d = embeddings.shape
        if t <= 1:
            return [0, t]

        device = next(self.parameters()).device
        x = torch.from_numpy(embeddings.astype(np.float32)).to(device)
        x = x.T.unsqueeze(0)  # (1, D, T)

        with torch.no_grad():
            _, logits2 = self(x)
            probs = F.softmax(logits2, dim=1)  # (1, 2, T)
            boundary_probs = probs[0, 1, :].cpu().numpy()  # (T,)

        # Extract boundaries above threshold with minimum spacing
        raw_points = [i for i in range(1, t) if boundary_probs[i] >= threshold]

        filtered: list[int] = []
        last = 0
        for point in raw_points:
            if point - last >= max(1, min_step_clips):
                filtered.append(point)
                last = point

        boundaries = [0] + filtered
        if boundaries[-1] != t:
            boundaries.append(t)
        return boundaries

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Segmentation loss: BCE + temporal smoothing
# ---------------------------------------------------------------------------


class SegmentationLoss(nn.Module):
    """Combined BCE + temporal smoothing loss for step segmentation.

    The smoothing term penalizes noisy predictions by encouraging
    neighboring frames to have similar classifications.
    """

    def __init__(self, smoothing_weight: float = 0.15) -> None:
        super().__init__()
        self.smoothing_weight = smoothing_weight

    def forward(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute segmentation loss.

        Args:
            logits1: (B, 2, T) stage 1 logits.
            logits2: (B, 2, T) stage 2 logits.
            targets: (B, T) long tensor, 0=non-boundary, 1=boundary.

        Returns:
            Scalar loss.
        """
        loss1 = F.cross_entropy(logits1, targets)
        loss2 = F.cross_entropy(logits2, targets)
        bce = loss1 + loss2

        # Temporal smoothing on stage 2 predictions
        probs2 = F.softmax(logits2, dim=1)
        smooth = torch.mean(torch.abs(probs2[:, :, 1:] - probs2[:, :, :-1]))

        return bce + self.smoothing_weight * smooth


# ---------------------------------------------------------------------------
# Pseudo-label generation from DTW alignment
# ---------------------------------------------------------------------------


def generate_pseudo_labels(
    gold_boundaries: list[int],
    n_trainee_clips: int,
    alignment_path: list[tuple[int, int, float]],
) -> np.ndarray:
    """Generate pseudo boundary labels for trainee from gold boundaries + DTW.

    Maps gold boundary positions through the DTW alignment path to find
    corresponding positions in the trainee sequence.

    Args:
        gold_boundaries: Gold video step boundaries [0, b1, ..., M].
        n_trainee_clips: Number of clips in trainee.
        alignment_path: DTW path [(gold_i, trainee_j, sim), ...].

    Returns:
        (n_trainee_clips,) binary array, 1 at predicted boundaries.
    """
    labels = np.zeros(n_trainee_clips, dtype=np.int64)
    gold_boundary_set = set(gold_boundaries[1:-1])  # Exclude first/last

    if not alignment_path or not gold_boundary_set:
        return labels

    # For each gold boundary, find the mapped trainee position
    for gi, tj, _ in alignment_path:
        if gi in gold_boundary_set:
            if 0 < tj < n_trainee_clips:
                labels[tj] = 1

    return labels


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_segmenter(model: NeuralStepSegmenter, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dilations = model.dilations if hasattr(model, "dilations") else (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    torch.save(
        {
            "d_in": model.d_in,
            "dilations": dilations,
            "state_dict": model.state_dict(),
        },
        path,
    )
    logger.info("Saved NeuralStepSegmenter (%d params) to %s", model.num_parameters, path)


def load_segmenter(path: Path, device: str = "cpu") -> NeuralStepSegmenter:
    data = torch.load(path, map_location=device, weights_only=False)
    dilations = data.get("dilations", (1, 2, 4, 8, 16, 32, 64, 128, 256, 512))
    if isinstance(dilations, list):
        dilations = tuple(dilations)
    model = NeuralStepSegmenter(d_in=data["d_in"], dilations=dilations)
    model.load_state_dict(data["state_dict"])
    model.to(device)
    model.eval()
    logger.info("Loaded NeuralStepSegmenter from %s", path)
    return model
