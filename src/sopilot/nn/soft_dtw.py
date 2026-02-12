"""Soft-DTW — differentiable Dynamic Time Warping.

Implements Cuturi & Blondel (2017), "Soft-DTW: a Differentiable Loss Function
for Time-Series".  Uses log-sum-exp soft-minimum over DP cells with smoothing
parameter gamma (γ).

Key properties:
    - γ → 0: recovers hard DTW (discrete path)
    - γ → ∞: uniform alignment (all paths equally weighted)
    - Gradients flow through alignment, enabling end-to-end training

The forward pass uses the same anti-diagonal wavefront pattern as our existing
DTW for cache-friendly vectorized computation.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sopilot.nn.constants import GAMMA_MIN
from sopilot.nn.functional import softmin3 as _softmin3

logger = logging.getLogger(__name__)


class SoftDTW(nn.Module):
    """Differentiable Soft-DTW distance.

    Computes the soft-DTW distance between two time series represented
    as embedding matrices.  The cost matrix is (1 - cosine_similarity).
    """

    def __init__(self, gamma: float = 1.0, normalize: bool = True) -> None:
        """
        Args:
            gamma: Smoothing parameter. Lower = closer to hard DTW.
            normalize: If True, return normalized Soft-DTW (subtract diagonal costs).
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.normalize = normalize

    def _compute_cost_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cosine distance cost matrix: (1 - cosine_sim).

        Args:
            x: (M, D) time series 1.
            y: (N, D) time series 2.

        Returns:
            (M, N) cost matrix.
        """
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        return 1.0 - x_norm @ y_norm.T

    def _dp_forward(self, cost: torch.Tensor) -> torch.Tensor:
        """Anti-diagonal wavefront DP for soft-DTW.

        Args:
            cost: (M, N) cost matrix.

        Returns:
            Scalar soft-DTW distance.
        """
        m, n = cost.shape
        gamma = self.gamma.abs().clamp(min=GAMMA_MIN)

        # DP table — use list-of-rows to avoid in-place mutation on a single tensor
        # which breaks autograd.  We keep a flat list and accumulate.
        INF = torch.tensor(float("inf"), dtype=cost.dtype, device=cost.device)
        # dp_vals[(i, j)] = accumulated cost
        # Use sequential accumulation to maintain grad graph
        dp: dict[tuple[int, int], torch.Tensor] = {}
        dp[(0, 0)] = torch.tensor(0.0, dtype=cost.dtype, device=cost.device)

        for d in range(2, m + n + 2):
            i_start = max(1, d - n)
            i_end = min(m, d - 1)
            for i in range(i_start, i_end + 1):
                j = d - i
                diag = dp.get((i - 1, j - 1), INF)
                above = dp.get((i - 1, j), INF)
                left = dp.get((i, j - 1), INF)
                dp[(i, j)] = cost[i - 1, j - 1] + _softmin3(
                    diag.unsqueeze(0), above.unsqueeze(0), left.unsqueeze(0), gamma
                ).squeeze(0)

        return dp[(m, n)]

    def _dp_forward_diagonal(self, cost: torch.Tensor) -> torch.Tensor:
        """Soft-DTW along diagonal (for normalization)."""
        n = min(cost.shape[0], cost.shape[1])
        if n == 0:
            return torch.tensor(0.0, dtype=cost.dtype, device=cost.device)
        return cost.diagonal()[:n].sum()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute (optionally normalized) Soft-DTW distance.

        Args:
            x: (M, D) first time series embeddings.
            y: (N, D) second time series embeddings.

        Returns:
            Scalar distance.
        """
        cost = self._compute_cost_matrix(x, y)

        sdtw = self._dp_forward(cost)

        if self.normalize:
            # Normalized Soft-DTW: subtract self-alignment costs
            sdtw_xx = self._dp_forward(self._compute_cost_matrix(x, x))
            sdtw_yy = self._dp_forward(self._compute_cost_matrix(y, y))
            sdtw = sdtw - 0.5 * (sdtw_xx + sdtw_yy)

        return sdtw


class SoftDTWAlignment(nn.Module):
    """Differentiable alignment module that returns a soft alignment matrix.

    Uses the Soft-DTW forward and backward DP values to compute the
    probability that cell (i,j) lies on the optimal alignment path.
    All computation stays in PyTorch tensors to maintain gradient flow.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))

    def _compute_cost_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        return 1.0 - x_norm @ y_norm.T

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute soft alignment matrix and distance.

        All computation stays in PyTorch tensors, maintaining gradient flow
        through both the alignment matrix and the distance.

        Args:
            x: (M, D) first time series.
            y: (N, D) second time series.

        Returns:
            (alignment_matrix, distance) where alignment_matrix is (M, N)
            with values in [0, 1] summing to ~path_length along each row.
        """
        cost = self._compute_cost_matrix(x, y)
        m, n = cost.shape
        gamma = self.gamma.abs().clamp(min=GAMMA_MIN)
        INF = torch.tensor(float("inf"), dtype=cost.dtype, device=cost.device)

        # Forward DP (dict-based to avoid in-place mutation, all in PyTorch)
        dp_f: dict[tuple[int, int], torch.Tensor] = {}
        dp_f[(0, 0)] = torch.tensor(0.0, dtype=cost.dtype, device=cost.device)

        for d in range(2, m + n + 2):
            i_start = max(1, d - n)
            i_end = min(m, d - 1)
            for i in range(i_start, i_end + 1):
                j = d - i
                diag = dp_f.get((i - 1, j - 1), INF)
                above = dp_f.get((i - 1, j), INF)
                left = dp_f.get((i, j - 1), INF)
                dp_f[(i, j)] = cost[i - 1, j - 1] + _softmin3(
                    diag.unsqueeze(0), above.unsqueeze(0), left.unsqueeze(0), gamma
                ).squeeze(0)

        distance = dp_f[(m, n)]

        # Backward DP (also in PyTorch, maintaining grad graph)
        dp_b: dict[tuple[int, int], torch.Tensor] = {}
        dp_b[(m, n)] = torch.tensor(0.0, dtype=cost.dtype, device=cost.device)

        for d in range(m + n, 0, -1):
            i_start = max(1, d - n)
            i_end = min(m, d - 1)
            for i in range(i_start, i_end + 1):
                j = d - i
                candidates = []
                if i + 1 <= m and j + 1 <= n and (i + 1, j + 1) in dp_b:
                    candidates.append(dp_b[(i + 1, j + 1)] + cost[i, j])
                if i + 1 <= m and j >= 1 and (i + 1, j) in dp_b:
                    candidates.append(dp_b[(i + 1, j)] + cost[i, j - 1])
                if j + 1 <= n and i >= 1 and (i, j + 1) in dp_b:
                    candidates.append(dp_b[(i, j + 1)] + cost[i - 1, j])
                if candidates:
                    stacked = torch.stack(candidates) / (-gamma)
                    dp_b[(i, j)] = -gamma * torch.logsumexp(stacked, dim=0)

        # Build alignment matrix (differentiable)
        rows = []
        for i in range(1, m + 1):
            cols = []
            for j in range(1, n + 1):
                f_val = dp_f.get((i, j), INF)
                b_val = dp_b.get((i, j), INF)
                log_p = -(f_val + cost[i - 1, j - 1] + b_val - distance) / gamma
                cols.append(log_p.unsqueeze(0))
            rows.append(torch.cat(cols, dim=0).unsqueeze(0))
        log_prob = torch.cat(rows, dim=0)  # (m, n)

        log_prob = torch.clamp(log_prob, max=0.0)
        alignment = torch.exp(log_prob)
        total = alignment.sum().clamp(min=1e-8)
        alignment = alignment / total * min(m, n)

        return alignment, distance


# ---------------------------------------------------------------------------
# Numpy-compatible wrapper for integration with existing step_engine
# ---------------------------------------------------------------------------


def soft_dtw_align_numpy(
    gold: np.ndarray,
    trainee: np.ndarray,
    gamma: float = 1.0,
) -> tuple[list[tuple[int, int, float]], float, np.ndarray]:
    """Soft-DTW alignment with numpy I/O for backward compatibility.

    Args:
        gold: (M, D) gold embeddings.
        trainee: (N, D) trainee embeddings.
        gamma: Smoothing parameter.

    Returns:
        (path, mean_cost, alignment_matrix) where:
        - path: list of (gold_idx, trainee_idx, similarity) tuples
        - mean_cost: average alignment cost
        - alignment_matrix: (M, N) soft alignment probabilities
    """
    if gold.ndim != 2 or trainee.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got gold.shape={gold.shape}, trainee.shape={trainee.shape}")
    if gold.shape[1] != trainee.shape[1]:
        raise ValueError(f"Embedding dim mismatch: gold has {gold.shape[1]}, trainee has {trainee.shape[1]}")
    if gold.shape[0] == 0 or trainee.shape[0] == 0:
        raise ValueError("Input sequences must not be empty")
    device = "cpu"
    x = torch.from_numpy(gold.astype(np.float32)).to(device)
    y = torch.from_numpy(trainee.astype(np.float32)).to(device)

    with torch.no_grad():
        model = SoftDTWAlignment(gamma=gamma).to(device)
        alignment, distance = model(x, y)

    alignment_np = alignment.cpu().numpy()
    m, n = alignment_np.shape

    # Extract hard path from soft alignment (argmax per gold frame)
    path: list[tuple[int, int, float]] = []
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    sim_matrix = (x_norm @ y_norm.T).cpu().numpy()

    for i in range(m):
        j = int(np.argmax(alignment_np[i]))
        sim = float(sim_matrix[i, j])
        path.append((i, j, sim))

    mean_cost = float(distance.item()) / max(1, len(path))

    return path, mean_cost, alignment_np
