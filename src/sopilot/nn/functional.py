"""Shared low-level functions for nn/ modules.

Avoids duplication of common primitives across soft_dtw, soft_dtw_cuda,
dilate_loss, and optimal_transport modules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def softmin3(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    gamma: torch.Tensor | float,
) -> torch.Tensor:
    """Soft-minimum of three values using negative log-sum-exp.

    softmin_γ(a, b, c) = -γ * log(exp(-a/γ) + exp(-b/γ) + exp(-c/γ))

    Used by Soft-DTW DP recurrences (Cuturi & Blondel 2017).
    """
    if isinstance(gamma, torch.Tensor):
        g = gamma
    else:
        g = torch.tensor(gamma, dtype=a.dtype, device=a.device)
    stacked = torch.stack([a, b, c], dim=-1) / (-g)
    return -g * torch.logsumexp(stacked, dim=-1)


def pairwise_euclidean_sq(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Squared-Euclidean pairwise distance matrix.

    Args:
        x: (M, D) or (B, M, D) embeddings.
        y: (N, D) or (B, N, D) embeddings.

    Returns:
        (M, N) or (B, M, N) squared Euclidean distances.
    """
    if x.dim() == 2 and y.dim() == 2:
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (M, N, D)
        return torch.sum(diff ** 2, dim=-1)
    # Batched: use expansion trick for efficiency
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)
    xx = (x * x).sum(dim=-1, keepdim=True)
    yy = (y * y).sum(dim=-1, keepdim=True)
    xy = torch.bmm(x, y.transpose(1, 2))
    return (xx + yy.transpose(1, 2) - 2.0 * xy).clamp(min=0.0)


def pairwise_cosine_dist(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Cosine distance cost matrix: 1 - cosine_similarity.

    Args:
        x: (M, D) or (B, M, D) embeddings.
        y: (N, D) or (B, N, D) embeddings.

    Returns:
        (M, N) or (B, M, N) cosine distances in [0, 2].
    """
    if x.dim() == 2 and y.dim() == 2:
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        return 1.0 - x_norm @ y_norm.T
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    return 1.0 - torch.bmm(x_norm, y_norm.transpose(1, 2))
