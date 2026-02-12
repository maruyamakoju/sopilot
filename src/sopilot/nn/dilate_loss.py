"""DILATE: Shape and Time Distortion Loss for temporal alignment.

Decomposes alignment quality into two interpretable, independently
optimizable components:
    1. Shape loss: Are the right things happening? (Soft-DTW)
    2. Temporal loss: Are they happening at the right time? (TDI)

This principled decomposition enables targeted optimization and
interpretable diagnostics that monolithic losses cannot provide.

References:
    Le Guen, V. & Thome, N. (2019). "Shape and Time Distortion Loss
    for Training Deep Time Series Forecasting Models", ICML 2019.

    Cuturi, M. & Blondel, M. (2017). "Soft-DTW: a Differentiable Loss
    Function for Time-Series", ICML 2017.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from sopilot.nn.constants import GAMMA_MIN as _GAMMA_MIN
from sopilot.nn.constants import INF as _INF
from sopilot.nn.functional import pairwise_euclidean_sq
from sopilot.nn.functional import softmin3 as _softmin3

logger = logging.getLogger(__name__)

__all__ = [
    "ShapeDTWLoss",
    "TemporalDistortionLoss",
    "DILATELoss",
    "SOPDilateLoss",
]


# ---------------------------------------------------------------------------
# Soft-DTW forward pass (self-contained for loss computation)
# ---------------------------------------------------------------------------


def _soft_dtw_forward(cost: torch.Tensor, gamma: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Soft-DTW distance and store DP table.

    Args:
        cost: (M, N) pairwise cost matrix.
        gamma: Smoothing parameter.

    Returns:
        (distance, R) where R is the (M+1, N+1) DP table.
    """
    m, n = cost.shape
    device = cost.device
    dtype = cost.dtype
    g = torch.tensor(gamma, dtype=dtype, device=device).clamp(min=_GAMMA_MIN)

    # Padded DP table
    R = torch.full((m + 1, n + 1), _INF, dtype=dtype, device=device)
    R[0, 0] = 0.0

    for d in range(2, m + n + 2):
        i_start = max(1, d - n)
        i_end = min(m, d - 1)
        if i_start > i_end:
            continue
        ii = torch.arange(i_start, i_end + 1, device=device)
        jj = d - ii

        diag = R[ii - 1, jj - 1]
        above = R[ii - 1, jj]
        left = R[ii, jj - 1]

        sm = _softmin3(diag, above, left, g)
        R[ii, jj] = cost[ii - 1, jj - 1] + sm

    return R[m, n], R


def _soft_dtw_alignment(cost: torch.Tensor, gamma: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Soft-DTW soft alignment matrix via forward-backward.

    The alignment matrix A[i,j] gives the probability mass that cell (i,j)
    is on the optimal alignment path.

    Args:
        cost: (M, N) pairwise cost matrix.
        gamma: Smoothing parameter.

    Returns:
        (alignment, distance) where alignment is (M, N).
    """
    m, n = cost.shape
    device = cost.device
    dtype = cost.dtype
    g = max(gamma, _GAMMA_MIN)

    # Forward pass
    distance, R_fwd = _soft_dtw_forward(cost, gamma)

    # Backward pass
    R_bwd = torch.full((m + 2, n + 2), _INF, dtype=dtype, device=device)
    R_bwd[m, n] = 0.0
    g_t = torch.tensor(g, dtype=dtype, device=device)

    for d in range(m + n, 0, -1):
        i_start = max(1, d - n)
        i_end = min(m, d - 1)
        if i_start > i_end:
            continue
        ii = torch.arange(i_start, i_end + 1, device=device)
        jj = d - ii

        candidates = []
        weights = []

        # Diagonal successor (i+1, j+1)
        diag_cost = torch.where(
            (ii + 1 <= m) & (jj + 1 <= n),
            R_bwd[ii + 1, jj + 1] + cost[ii.clamp(max=m - 1), jj.clamp(max=n - 1)],
            torch.tensor(_INF, dtype=dtype, device=device),
        )
        # Below successor (i+1, j)
        below_cost = torch.where(
            ii + 1 <= m,
            R_bwd[ii + 1, jj] + cost[ii.clamp(max=m - 1), (jj - 1).clamp(min=0)],
            torch.tensor(_INF, dtype=dtype, device=device),
        )
        # Right successor (i, j+1)
        right_cost = torch.where(
            jj + 1 <= n,
            R_bwd[ii, jj + 1] + cost[(ii - 1).clamp(min=0), jj.clamp(max=n - 1)],
            torch.tensor(_INF, dtype=dtype, device=device),
        )

        sm = _softmin3(diag_cost, below_cost, right_cost, g_t)
        R_bwd[ii, jj] = sm

    # Alignment matrix: exp(-(forward + cost + backward - total) / gamma)
    f_vals = R_fwd[1 : m + 1, 1 : n + 1]
    b_vals = R_bwd[1 : m + 1, 1 : n + 1]
    log_prob = -(f_vals + cost + b_vals - distance) / g
    log_prob = log_prob.clamp(max=0.0)
    alignment = torch.exp(log_prob)
    total = alignment.sum()
    if total > 1e-8:
        alignment = alignment / total * min(m, n)

    return alignment, distance


# ---------------------------------------------------------------------------
# Shape Loss
# ---------------------------------------------------------------------------


class ShapeDTWLoss(nn.Module):
    """Shape component of DILATE: differentiable Soft-DTW distance.

    Measures shape similarity independent of temporal alignment.
    Lower values indicate better shape match.
    """

    def __init__(self, gamma: float = 0.1) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Soft-DTW shape loss.

        Args:
            pred: (T1, D) or (B, T1, D) predicted sequence.
            target: (T2, D) or (B, T2, D) target sequence.

        Returns:
            Scalar shape loss.
        """
        if pred.dim() == 3:
            # Batched
            losses = []
            for i in range(pred.shape[0]):
                cost = pairwise_euclidean_sq(pred[i], target[i])
                dist, _ = _soft_dtw_forward(cost, self.gamma)
                losses.append(dist)
            return torch.stack(losses).mean()
        else:
            cost = pairwise_euclidean_sq(pred, target)
            dist, _ = _soft_dtw_forward(cost, self.gamma)
            return dist


# ---------------------------------------------------------------------------
# Temporal Distortion Index
# ---------------------------------------------------------------------------


class TemporalDistortionLoss(nn.Module):
    """Temporal component of DILATE: Temporal Distortion Index (TDI).

    Penalizes temporal distortion in the alignment path. The TDI measures
    how much the soft alignment deviates from the diagonal:

        TDI = sum_{i,j} A[i,j] * |i/M - j/N|^2

    where A is the soft alignment matrix from Soft-DTW.
    A diagonal alignment (perfect timing) gives TDI = 0.
    """

    def __init__(self, gamma: float = 0.1) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alignment: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute TDI loss.

        Args:
            pred: (T1, D) or (B, T1, D) predicted sequence.
            target: (T2, D) or (B, T2, D) target sequence.
            alignment: Optional pre-computed alignment matrix.

        Returns:
            Scalar TDI loss.
        """
        if pred.dim() == 3:
            losses = []
            for i in range(pred.shape[0]):
                losses.append(self._single(pred[i], target[i]))
            return torch.stack(losses).mean()
        return self._single(pred, target, alignment)

    def _single(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alignment: torch.Tensor | None = None,
    ) -> torch.Tensor:
        M, N = pred.shape[0], target.shape[0]
        device = pred.device

        if alignment is None:
            cost = pairwise_euclidean_sq(pred, target)
            alignment, _ = _soft_dtw_alignment(cost, self.gamma)

        # Normalized position grid
        i_pos = torch.arange(M, device=device, dtype=pred.dtype) / max(M - 1, 1)
        j_pos = torch.arange(N, device=device, dtype=pred.dtype) / max(N - 1, 1)

        # Squared temporal distortion at each cell
        distortion = (i_pos.unsqueeze(1) - j_pos.unsqueeze(0)) ** 2  # (M, N)

        # Weighted sum using alignment probabilities
        tdi = (alignment * distortion).sum()
        return tdi


# ---------------------------------------------------------------------------
# DILATE Loss (Combined)
# ---------------------------------------------------------------------------


class DILATELoss(nn.Module):
    """DILATE: Shape and Time Distortion Loss (Le Guen & Thome, ICML 2019).

    L = alpha * L_shape + (1-alpha) * L_temporal

    Decomposes alignment quality into two interpretable components:
        1. Shape: Are the right things happening? (Soft-DTW distance)
        2. Temporal: Are they happening at the right time? (TDI)

    Args:
        alpha: Balance between shape (1.0) and temporal (0.0). Default 0.5.
        gamma: Soft-DTW smoothing parameter. Default 0.1.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute DILATE loss.

        Args:
            pred: (T1, D) or (B, T1, D) predicted sequence.
            target: (T2, D) or (B, T2, D) target sequence.

        Returns:
            (total_loss, components) where components has:
                shape: shape loss, temporal: TDI loss, alignment: soft alignment
        """
        if pred.dim() == 3:
            total_losses, shape_losses, temporal_losses = [], [], []
            for i in range(pred.shape[0]):
                loss, comp = self._single(pred[i], target[i])
                total_losses.append(loss)
                shape_losses.append(comp["shape"])
                temporal_losses.append(comp["temporal"])
            return torch.stack(total_losses).mean(), {
                "shape": torch.stack(shape_losses).mean(),
                "temporal": torch.stack(temporal_losses).mean(),
            }
        return self._single(pred, target)

    def _single(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        M, N = pred.shape[0], target.shape[0]
        cost = pairwise_euclidean_sq(pred, target)

        # Shape loss (Soft-DTW distance)
        alignment, shape_loss = _soft_dtw_alignment(cost, self.gamma)

        # Temporal loss (TDI using alignment matrix)
        device = pred.device
        i_pos = torch.arange(M, device=device, dtype=pred.dtype) / max(M - 1, 1)
        j_pos = torch.arange(N, device=device, dtype=pred.dtype) / max(N - 1, 1)
        distortion = (i_pos.unsqueeze(1) - j_pos.unsqueeze(0)) ** 2
        temporal_loss = (alignment * distortion).sum()

        # Combined
        total = self.alpha * shape_loss + (1.0 - self.alpha) * temporal_loss

        return total, {
            "shape": shape_loss,
            "temporal": temporal_loss,
            "alignment": alignment.detach(),
        }


# ---------------------------------------------------------------------------
# Extended DILATE for SOP evaluation
# ---------------------------------------------------------------------------


class SOPDilateLoss(nn.Module):
    """Extended DILATE loss specialized for SOP compliance evaluation.

    Adds three SOP-specific components to the base DILATE loss:

    1. Step boundary alignment: penalizes misaligned step boundaries
       L_boundary = sum_b min_j |b/M - j/N|^2 * A[b, j]
       where b are gold step boundaries

    2. Step ordering: penalizes out-of-order step execution
       L_order = sum_{b1 < b2} max(0, median_match(b2) - median_match(b1))
       Differentiable via soft-argmax of alignment rows

    3. Coverage: penalizes missing steps (rows with low alignment mass)
       L_coverage = sum_b (1 - sum_j A[b_start:b_end, :])

    Total: w1*L_shape + w2*L_temporal + w3*L_boundary + w4*L_order + w5*L_coverage
    """

    def __init__(
        self,
        alpha_shape: float = 0.3,
        alpha_temporal: float = 0.2,
        alpha_boundary: float = 0.2,
        alpha_order: float = 0.15,
        alpha_coverage: float = 0.15,
        gamma: float = 0.1,
    ) -> None:
        super().__init__()
        self.alpha_shape = alpha_shape
        self.alpha_temporal = alpha_temporal
        self.alpha_boundary = alpha_boundary
        self.alpha_order = alpha_order
        self.alpha_coverage = alpha_coverage
        self.gamma = gamma

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        gold_boundaries: list[int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute SOP-DILATE loss.

        Args:
            pred: (T1, D) predicted (trainee) sequence.
            target: (T2, D) target (gold) sequence.
            gold_boundaries: Step boundaries [0, b1, b2, ..., M].

        Returns:
            (total_loss, component_dict)
        """
        M, N = target.shape[0], pred.shape[0]
        device = pred.device
        dtype = pred.dtype

        # Cost matrix and alignment
        cost = pairwise_euclidean_sq(target, pred)
        alignment, shape_loss = _soft_dtw_alignment(cost, self.gamma)

        # Temporal distortion
        i_pos = torch.arange(M, device=device, dtype=dtype) / max(M - 1, 1)
        j_pos = torch.arange(N, device=device, dtype=dtype) / max(N - 1, 1)
        distortion = (i_pos.unsqueeze(1) - j_pos.unsqueeze(0)) ** 2
        temporal_loss = (alignment * distortion).sum()

        # Step boundary alignment loss
        boundary_loss = torch.tensor(0.0, device=device, dtype=dtype)
        internal_bounds = [b for b in gold_boundaries if 0 < b < M]
        if internal_bounds:
            for b in internal_bounds:
                # Weight of alignment at boundary row
                row_weights = alignment[b, :]  # (N,)
                # Expected trainee position should be proportional
                expected_j = b / max(M - 1, 1)
                j_positions = torch.arange(N, device=device, dtype=dtype) / max(N - 1, 1)
                deviation = (j_positions - expected_j) ** 2
                boundary_loss = boundary_loss + (row_weights * deviation).sum()
            boundary_loss = boundary_loss / max(len(internal_bounds), 1)

        # Step ordering loss (soft)
        order_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if len(gold_boundaries) > 2:
            # Compute soft median position for each step
            step_positions = []
            for s in range(len(gold_boundaries) - 1):
                start, end = gold_boundaries[s], gold_boundaries[s + 1]
                if start >= M or end > M or start >= end:
                    continue
                # Soft argmax: weighted average of trainee positions
                step_alignment = alignment[start:end, :]  # (step_len, N)
                total_mass = step_alignment.sum().clamp(min=1e-8)
                weighted_pos = (step_alignment.sum(dim=0) * j_pos).sum() / total_mass
                step_positions.append(weighted_pos)

            # Penalize inversions
            for k in range(1, len(step_positions)):
                order_loss = order_loss + F.relu(step_positions[k - 1] - step_positions[k])
            if len(step_positions) > 1:
                order_loss = order_loss / (len(step_positions) - 1)

        # Coverage loss
        coverage_loss = torch.tensor(0.0, device=device, dtype=dtype)
        n_steps = len(gold_boundaries) - 1
        if n_steps > 0:
            for s in range(n_steps):
                start = gold_boundaries[s]
                end = min(gold_boundaries[s + 1], M)
                if start >= end:
                    continue
                step_mass = alignment[start:end, :].sum()
                expected_mass = end - start  # Expected mass proportional to step length
                coverage_loss = coverage_loss + F.relu(expected_mass * 0.5 - step_mass)
            coverage_loss = coverage_loss / n_steps

        # Total
        total = (
            self.alpha_shape * shape_loss
            + self.alpha_temporal * temporal_loss
            + self.alpha_boundary * boundary_loss
            + self.alpha_order * order_loss
            + self.alpha_coverage * coverage_loss
        )

        return total, {
            "shape": shape_loss,
            "temporal": temporal_loss,
            "boundary": boundary_loss,
            "order": order_loss,
            "coverage": coverage_loss,
            "alignment": alignment.detach(),
        }
