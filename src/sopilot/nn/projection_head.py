"""Contrastive Projection Head — replaces Z-score feature adaptation.

Implements a 3-layer MLP projection with NT-Xent (InfoNCE) contrastive loss,
following SimCLR (Chen et al. 2020).  Clips from the same gold step form
positive pairs; clips from different steps form negatives.

Architecture:
    Linear(D_in, 512) -> BatchNorm1d -> GELU -> Dropout(0.1)
    Linear(512, 256)  -> BatchNorm1d -> GELU -> Dropout(0.1)
    Linear(256, D_out) -> L2Normalize
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """3-layer contrastive projection head with L2-normalized output."""

    def __init__(
        self,
        d_in: int,
        d_out: int = 128,
        d_hidden: int = 512,
        d_mid: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.layers = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_mid),
            nn.BatchNorm1d(d_mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mid, d_out),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize: (B, D_in) -> (B, D_out)."""
        z = self.layers(x)
        return F.normalize(z, dim=-1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# NT-Xent (InfoNCE) contrastive loss
# ---------------------------------------------------------------------------


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR).

    Given a batch of N embeddings with step labels, treats same-step clips
    as positives and different-step clips as negatives.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        step_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NT-Xent loss.

        Args:
            embeddings: (N, D) L2-normalized embeddings.
            step_labels: (N,) integer step assignment for each clip.

        Returns:
            Scalar loss.
        """
        device = embeddings.device
        n = embeddings.size(0)
        if n < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Similarity matrix
        sim = embeddings @ embeddings.T / self.temperature  # (N, N)

        # Mask: same step = positive (excluding self)
        labels = step_labels.unsqueeze(0)  # (1, N)
        positive_mask = (labels == labels.T).float()  # (N, N)
        positive_mask.fill_diagonal_(0.0)

        # Check there are positives
        positives_per_row = positive_mask.sum(dim=1)
        has_positive = positives_per_row > 0
        if not has_positive.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Self-mask: exclude diagonal from denominator
        self_mask = torch.eye(n, device=device, dtype=torch.bool)
        sim.masked_fill_(self_mask, -1e9)

        # Log-sum-exp over all non-self entries (denominator)
        log_denom = torch.logsumexp(sim, dim=1)  # (N,)

        # Mean of positive similarities (numerator)
        # For stability, use log-sum-exp over positives then subtract log(count)
        neg_inf = torch.full_like(sim, -1e9)
        pos_sim = torch.where(positive_mask.bool(), sim, neg_inf)
        log_numer = torch.logsumexp(pos_sim, dim=1) - torch.log(positives_per_row.clamp(min=1))

        # Loss: -log(positive / all) = -(log_numer - log_denom)
        loss = -(log_numer - log_denom)

        # Only average over rows that have at least one positive
        loss = loss[has_positive].mean()
        return loss


# ---------------------------------------------------------------------------
# Step-pair mining utility
# ---------------------------------------------------------------------------


class StepPairMiner:
    """Assigns step labels to clips given step boundaries.

    Given step boundaries [0, b1, b2, ..., N], clip i belongs to step k
    if boundaries[k] <= i < boundaries[k+1].
    """

    @staticmethod
    def assign_step_labels(n_clips: int, boundaries: list[int]) -> torch.Tensor:
        """Return (n_clips,) integer tensor of step assignments."""
        labels = torch.zeros(n_clips, dtype=torch.long)
        for k in range(len(boundaries) - 1):
            start = boundaries[k]
            end = boundaries[k + 1]
            labels[start:end] = k
        return labels


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_projection_head(model: ProjectionHead, path: Path) -> None:
    """Save projection head weights."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "d_in": model.d_in,
            "d_out": model.d_out,
            "state_dict": model.state_dict(),
        },
        path,
    )
    logger.info("Saved ProjectionHead (%d params) to %s", model.num_parameters, path)


def load_projection_head(path: Path, device: str = "cpu") -> ProjectionHead:
    """Load projection head from saved checkpoint."""
    data = torch.load(path, map_location=device, weights_only=True)
    model = ProjectionHead(d_in=data["d_in"], d_out=data["d_out"])
    model.load_state_dict(data["state_dict"])
    model.to(device)
    model.eval()
    logger.info("Loaded ProjectionHead from %s", path)
    return model
