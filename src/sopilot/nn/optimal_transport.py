"""Optimal Transport for Procedure Alignment in SOP Video Evaluation.

Novel application of entropy-regularized optimal transport to temporal
procedure alignment.  Unlike DTW which constrains to monotonic warping
paths, OT relaxes this to allow many-to-many soft correspondences -- better
modelling real-world procedure variations where steps may overlap, repeat,
or execute concurrently.

Implements:

- **SinkhornDistance**: Log-domain Sinkhorn iterations with epsilon-scaling
  for entropy-regularized OT (Cuturi, "Sinkhorn Distances: Lightspeed
  Computation of Optimal Transport", NeurIPS 2013).

- **GromovWassersteinDistance**: Structure-preserving alignment via
  Frank-Wolfe conditional gradient (Peyre, Cuturi & Solomon, "Gromov-
  Wasserstein Averaging of Kernel and Distance Matrices", ICML 2016).

- **FusedGromovWasserstein**: Joint feature + structure alignment combining
  Sinkhorn and GW objectives (Vayer, Chapel, Flamary, Tavenard & Courty,
  "Optimal Transport for structured data with application on graphs",
  ICML 2019).

- **HierarchicalOTAlignment**: Three-level coarse-to-fine alignment using
  OT at phase, step, and frame granularity.  Coarse alignment constrains
  finer levels, reducing computation and improving coherence.

- **WassersteinBarycenter**: Fixed-point Sinkhorn barycenter for computing
  the "average" SOP demonstration (Cuturi & Doucet, "Fast Computation of
  Wasserstein Barycenters", ICML 2014).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from sopilot.nn.functional import pairwise_cosine_dist, pairwise_euclidean_sq

logger = logging.getLogger(__name__)

__all__ = [
    "SinkhornDistance",
    "GromovWassersteinDistance",
    "FusedGromovWasserstein",
    "HierarchicalOTAlignment",
    "WassersteinBarycenter",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_safe(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log, clamping to avoid -inf."""
    return torch.log(x.clamp(min=1e-30))


def _uniform_marginal(n: int, batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return uniform distribution of shape (batch, n)."""
    return torch.full((batch, n), 1.0 / n, dtype=dtype, device=device)


def _ensure_batched_3d(x: torch.Tensor) -> torch.Tensor:
    """Promote a 2-D tensor to (1, M, N)."""
    if x.dim() == 2:
        return x.unsqueeze(0)
    return x


def _ensure_batched_2d(x: torch.Tensor) -> torch.Tensor:
    """Promote a 1-D tensor to (1, N)."""
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


def _cosine_cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batched cosine distance cost matrix (always 3D)."""
    c = pairwise_cosine_dist(x, y)
    return c.unsqueeze(0) if c.dim() == 2 else c


def _euclidean_cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batched squared-Euclidean cost matrix (always 3D)."""
    c = pairwise_euclidean_sq(x, y)
    return c.unsqueeze(0) if c.dim() == 2 else c


def _intra_distance_matrix(x: torch.Tensor) -> torch.Tensor:
    """Compute intra-sequence pairwise distance matrix (always 3D)."""
    c = pairwise_euclidean_sq(x, x)
    return c.unsqueeze(0) if c.dim() == 2 else c


# ---------------------------------------------------------------------------
# SinkhornDistance
# ---------------------------------------------------------------------------


class SinkhornDistance(nn.Module):
    """Log-domain Sinkhorn algorithm for entropy-regularized optimal transport.

    Solves:
        min_{P >= 0}  <P, C> - epsilon * H(P)
        s.t.  P @ 1_N = a,   P^T @ 1_M = b

    where H(P) = -sum_{ij} P_{ij} (log P_{ij} - 1) is the entropic barrier.

    Uses epsilon-scaling (starting from a large epsilon and annealing) for
    improved convergence, following Schmitzer (2019) "Stabilized Sparse
    Scaling Algorithms for Entropy Regularized Transport Problems".

    Fully differentiable: gradients flow through the Sinkhorn iterations
    via implicit differentiation of the fixed-point equations.

    Args:
        epsilon: Entropy regularization strength.  Larger values produce
            smoother (more spread-out) transport plans.
        max_iter: Maximum number of Sinkhorn iterations.
        tol: Convergence tolerance on marginal violation (L1 norm).
        epsilon_scaling: If True, use epsilon-scaling schedule for faster
            convergence.  Starts at 10 * epsilon and anneals geometrically.
        scaling_steps: Number of epsilon-scaling stages.
        unbalanced_tau: If not None, use unbalanced OT with KL divergence
            penalty of strength tau on both marginals.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6,
        epsilon_scaling: bool = False,
        scaling_steps: int = 5,
        unbalanced_tau: float | None = None,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon_scaling = epsilon_scaling
        self.scaling_steps = scaling_steps
        self.unbalanced_tau = unbalanced_tau

    def _sinkhorn_log(
        self,
        C: torch.Tensor,
        log_a: torch.Tensor,
        log_b: torch.Tensor,
        epsilon: float,
        max_iter: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core log-domain Sinkhorn iterations.

        Args:
            C: (B, M, N) cost matrix.
            log_a: (B, M) log of source marginal.
            log_b: (B, N) log of target marginal.
            epsilon: Current epsilon value.
            max_iter: Maximum iterations for this stage.

        Returns:
            (log_P, u, v) where log_P is (B, M, N) log-transport plan,
            u is (B, M) and v is (B, N) dual variables.
        """
        B, M, N = C.shape
        log_K = -C / epsilon  # (B, M, N)

        u = torch.zeros(B, M, dtype=C.dtype, device=C.device)
        v = torch.zeros(B, N, dtype=C.dtype, device=C.device)

        for iteration in range(max_iter):
            u_prev = u.clone()

            if self.unbalanced_tau is not None:
                tau = self.unbalanced_tau
                rho = tau / (tau + epsilon)
                u = rho * (log_a - torch.logsumexp(log_K + v.unsqueeze(1), dim=2))
                v = rho * (log_b - torch.logsumexp(log_K + u.unsqueeze(2), dim=1))
            else:
                u = log_a - torch.logsumexp(log_K + v.unsqueeze(1), dim=2)
                v = log_b - torch.logsumexp(log_K + u.unsqueeze(2), dim=1)

            log_P = u.unsqueeze(2) + log_K + v.unsqueeze(1)
            row_sums = torch.logsumexp(log_P, dim=2)
            err = (torch.exp(row_sums) - torch.exp(log_a)).abs().sum(dim=1).max()

            if err.item() < self.tol:
                logger.debug(
                    "Sinkhorn converged at iteration %d (err=%.2e, eps=%.4f)",
                    iteration,
                    err.item(),
                    epsilon,
                )
                break

        log_P = u.unsqueeze(2) + log_K + v.unsqueeze(1)
        return log_P, u, v

    def forward(
        self,
        C: torch.Tensor,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the Sinkhorn distance and optimal transport plan.

        Args:
            C: (B, M, N) or (M, N) cost matrix.  If 2-D, a batch dimension
                is added automatically.
            a: (B, M) or (M,) source marginal (must sum to 1 per batch).
                Defaults to uniform.
            b: (B, N) or (N,) target marginal (must sum to 1 per batch).
                Defaults to uniform.

        Returns:
            (distance, P) where:
                distance: (B,) Sinkhorn distances.
                P: (B, M, N) optimal transport plan.
        """
        squeeze_batch = C.dim() == 2
        C = _ensure_batched_3d(C)
        B, M, N = C.shape

        if a is None:
            a = _uniform_marginal(M, B, C.dtype, C.device)
        else:
            a = _ensure_batched_2d(a)

        if b is None:
            b = _uniform_marginal(N, B, C.dtype, C.device)
        else:
            b = _ensure_batched_2d(b)

        log_a = _log_safe(a)
        log_b = _log_safe(b)

        if self.epsilon_scaling:
            eps_values = [
                self.epsilon * (10.0 ** ((self.scaling_steps - 1 - i) / max(1, self.scaling_steps - 1)))
                for i in range(self.scaling_steps)
            ]
            eps_values[-1] = self.epsilon
            iters_per_stage = max(1, self.max_iter // self.scaling_steps)

            for eps in eps_values:
                log_P, u, v = self._sinkhorn_log(C, log_a, log_b, eps, iters_per_stage)
        else:
            log_P, u, v = self._sinkhorn_log(C, log_a, log_b, self.epsilon, self.max_iter)

        P = torch.exp(log_P)
        distance = (P * C).sum(dim=(1, 2))

        if squeeze_batch:
            distance = distance.squeeze(0)
            P = P.squeeze(0)

        return distance, P


# ---------------------------------------------------------------------------
# GromovWassersteinDistance
# ---------------------------------------------------------------------------


class GromovWassersteinDistance(nn.Module):
    """Gromov-Wasserstein distance for structure-preserving alignment.

    Solves:
        min_P  sum_{i,j,k,l} |D1[i,k] - D2[j,l]|^2 * P[i,j] * P[k,l]
        s.t.  P @ 1 = a,  P^T @ 1 = b

    where D1 and D2 are intra-distance matrices for the two sequences.
    Solved via Frank-Wolfe (conditional gradient) iterations where each
    inner step solves a linear OT problem using Sinkhorn.

    Reference: Peyre, Cuturi & Solomon, "Gromov-Wasserstein Averaging of
    Kernel and Distance Matrices", ICML 2016.

    Args:
        epsilon: Entropy regularization for inner Sinkhorn solver.
        max_outer_iter: Maximum Frank-Wolfe outer iterations.
        max_inner_iter: Maximum Sinkhorn iterations per FW step.
        tol: Convergence tolerance on transport plan change.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        max_outer_iter: int = 50,
        max_inner_iter: int = 50,
        tol: float = 1e-5,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.tol = tol
        self.sinkhorn = SinkhornDistance(epsilon=epsilon, max_iter=max_inner_iter, tol=tol)

    @staticmethod
    def _compute_gw_gradient(D1: torch.Tensor, D2: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Efficient O(M^2*N + M*N^2) GW gradient computation.

        G[i,j] = sum_k D1[i,k]^2 * a[k] + sum_l D2[j,l]^2 * b[l]
                - 2 * (D1 @ P @ D2^T)[i,j]

        Args:
            D1: (B, M, M) intra-distance matrix for sequence 1.
            D2: (B, N, N) intra-distance matrix for sequence 2.
            P: (B, M, N) current transport plan.

        Returns:
            (B, M, N) gradient matrix.
        """
        D1_sq = D1 * D1
        D2_sq = D2 * D2
        row_marginals = P.sum(dim=2)
        term1 = torch.bmm(D1_sq, row_marginals.unsqueeze(2))
        col_marginals = P.sum(dim=1)
        term2 = torch.bmm(col_marginals.unsqueeze(1), D2_sq)
        term3 = -2.0 * torch.bmm(torch.bmm(D1, P), D2.transpose(1, 2))
        return term1 + term2 + term3

    @staticmethod
    def _compute_gw_gradient_naive(D1: torch.Tensor, D2: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Naive O(M^2 * N^2) gradient computation for verification.

        Computes G[i,j] = sum_{k,l} (D1[i,k] - D2[j,l])^2 * P[k,l]
        by explicit iteration.  Only suitable for small problems.
        """
        B, M, _ = D1.shape
        N = D2.shape[1]
        G = torch.zeros(B, M, N, dtype=D1.dtype, device=D1.device)
        for b_idx in range(B):
            for i in range(M):
                for j in range(N):
                    total = torch.tensor(0.0, dtype=D1.dtype, device=D1.device)
                    for k in range(M):
                        for l_idx in range(N):
                            diff = D1[b_idx, i, k] - D2[b_idx, j, l_idx]
                            total = total + diff * diff * P[b_idx, k, l_idx]
                    G[b_idx, i, j] = total
        return G

    @staticmethod
    def _compute_gw_cost(D1: torch.Tensor, D2: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Compute the full GW objective value.

        Efficient form:
            GW(P) = a^T D1^2 a + b^T D2^2 b - 2 trace(D1 P D2 P^T)

        where a = P 1, b = P^T 1.
        """
        a = P.sum(dim=2)
        b = P.sum(dim=1)
        D1_sq = D1 * D1
        D2_sq = D2 * D2
        term1 = torch.bmm(a.unsqueeze(1), torch.bmm(D1_sq, a.unsqueeze(2))).squeeze(2).squeeze(1)
        term2 = torch.bmm(b.unsqueeze(1), torch.bmm(D2_sq, b.unsqueeze(2))).squeeze(2).squeeze(1)
        D1_P = torch.bmm(D1, P)
        P_D2T = torch.bmm(P, D2.transpose(1, 2))
        cross_term = (D1_P * P_D2T).sum(dim=(1, 2))
        return term1 + term2 - 2.0 * cross_term

    def forward(
        self,
        D1: torch.Tensor,
        D2: torch.Tensor,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GW distance and transport plan via Frank-Wolfe.

        Args:
            D1: (B, M, M) or (M, M) intra-distance matrix for sequence 1.
            D2: (B, N, N) or (N, N) intra-distance matrix for sequence 2.
            a: (B, M) or (M,) source marginal.  Defaults to uniform.
            b: (B, N) or (N,) target marginal.  Defaults to uniform.

        Returns:
            (cost, P) where cost is (B,) GW distances and
            P is (B, M, N) transport plan.
        """
        squeeze_batch = D1.dim() == 2
        D1 = _ensure_batched_3d(D1)
        D2 = _ensure_batched_3d(D2)
        B = D1.shape[0]
        M = D1.shape[1]
        N = D2.shape[1]
        if a is None:
            a = _uniform_marginal(M, B, D1.dtype, D1.device)
        else:
            a = _ensure_batched_2d(a)
        if b is None:
            b = _uniform_marginal(N, B, D1.dtype, D1.device)
        else:
            b = _ensure_batched_2d(b)
        P = a.unsqueeze(2) * b.unsqueeze(1)
        for step in range(self.max_outer_iter):
            P_prev = P.clone()
            G = self._compute_gw_gradient(D1, D2, P)
            _, P_star = self.sinkhorn(G, a, b)
            alpha = 2.0 / (step + 2)
            P = (1.0 - alpha) * P + alpha * P_star
            change = (P - P_prev).abs().sum(dim=(1, 2)).max()
            if change.item() < self.tol:
                logger.debug("GW converged at iteration %d (change=%.2e)", step, change.item())
                break
        cost = self._compute_gw_cost(D1, D2, P)
        if squeeze_batch:
            cost = cost.squeeze(0)
            P = P.squeeze(0)
        return cost, P


# ---------------------------------------------------------------------------
# FusedGromovWasserstein
# ---------------------------------------------------------------------------


class FusedGromovWasserstein(nn.Module):
    """Fused Gromov-Wasserstein distance combining feature and structure OT.

    Solves:
        min_P  alpha * <P, C_features> + (1 - alpha) * GW_cost(D1, D2, P)
               - epsilon * H(P)
        s.t.  P @ 1 = a,  P^T @ 1 = b

    Reference: Vayer et al., "Optimal Transport for structured data with
    application on graphs", ICML 2019.

    Args:
        alpha: Trade-off between feature OT (alpha=1) and structure GW
            (alpha=0).  Default 0.5 gives equal weight.
        epsilon: Entropy regularization.
        max_outer_iter: Maximum outer iterations.
        max_inner_iter: Maximum inner Sinkhorn iterations.
        tol: Convergence tolerance.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        epsilon: float = 0.05,
        max_outer_iter: int = 50,
        max_inner_iter: int = 50,
        tol: float = 1e-5,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.tol = tol
        self.sinkhorn = SinkhornDistance(epsilon=epsilon, max_iter=max_inner_iter, tol=tol)

    def forward(
        self,
        C: torch.Tensor,
        D1: torch.Tensor,
        D2: torch.Tensor,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Fused GW distance.

        Args:
            C: (B, M, N) or (M, N) feature-space cost matrix.
            D1: (B, M, M) or (M, M) intra-distance matrix for sequence 1.
            D2: (B, N, N) or (N, N) intra-distance matrix for sequence 2.
            a: (B, M) or (M,) source marginal.  Defaults to uniform.
            b: (B, N) or (N,) target marginal.  Defaults to uniform.

        Returns:
            (cost, P) where cost is (B,) Fused GW distances and
            P is (B, M, N) transport plan.
        """
        squeeze_batch = C.dim() == 2
        C = _ensure_batched_3d(C)
        D1 = _ensure_batched_3d(D1)
        D2 = _ensure_batched_3d(D2)
        B, M, N = C.shape
        if a is None:
            a = _uniform_marginal(M, B, C.dtype, C.device)
        else:
            a = _ensure_batched_2d(a)
        if b is None:
            b = _uniform_marginal(N, B, C.dtype, C.device)
        else:
            b = _ensure_batched_2d(b)
        P = a.unsqueeze(2) * b.unsqueeze(1)
        for step in range(self.max_outer_iter):
            P_prev = P.clone()
            G_gw = GromovWassersteinDistance._compute_gw_gradient(D1, D2, P)
            G_combined = self.alpha * C + (1.0 - self.alpha) * G_gw
            _, P_star = self.sinkhorn(G_combined, a, b)
            alpha_fw = 2.0 / (step + 2)
            P = (1.0 - alpha_fw) * P + alpha_fw * P_star
            change = (P - P_prev).abs().sum(dim=(1, 2)).max()
            if change.item() < self.tol:
                logger.debug("Fused GW converged at iteration %d (change=%.2e)", step, change.item())
                break
        feature_cost = (P * C).sum(dim=(1, 2))
        gw_cost = GromovWassersteinDistance._compute_gw_cost(D1, D2, P)
        cost = self.alpha * feature_cost + (1.0 - self.alpha) * gw_cost
        if squeeze_batch:
            cost = cost.squeeze(0)
            P = P.squeeze(0)
        return cost, P


# ---------------------------------------------------------------------------
# HierarchicalOTAlignment
# ---------------------------------------------------------------------------


class HierarchicalOTAlignment(nn.Module):
    """Three-level hierarchical alignment using Optimal Transport.

    Level 1 (coarse): Align procedure *phases* using GW on temporal structure.
    Level 2 (medium): Within each phase, align *steps* using Sinkhorn OT.
    Level 3 (fine): Within each step, align *frames* using cosine similarity.

    Each level constrains the next, reducing computation and improving
    temporal coherence of the overall alignment.

    Args:
        n_phases: Number of phases to divide each sequence into at Level 1.
        epsilon_coarse: Regularization for coarse (GW) level.
        epsilon_fine: Regularization for fine (Sinkhorn) levels.
        max_iter: Maximum iterations at each level.
        constraint_temperature: Temperature for soft constraint propagation.
    """

    def __init__(
        self,
        n_phases: int = 4,
        epsilon_coarse: float = 0.1,
        epsilon_fine: float = 0.05,
        max_iter: int = 50,
        constraint_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_phases = n_phases
        self.epsilon_coarse = epsilon_coarse
        self.epsilon_fine = epsilon_fine
        self.max_iter = max_iter
        self.constraint_temperature = constraint_temperature
        self.gw_solver = GromovWassersteinDistance(
            epsilon=epsilon_coarse,
            max_outer_iter=max_iter,
            max_inner_iter=max_iter,
            tol=1e-5,
        )
        self.sinkhorn_fine = SinkhornDistance(
            epsilon=epsilon_fine,
            max_iter=max_iter,
            tol=1e-6,
        )

    @staticmethod
    def _segment_embeddings(x: torch.Tensor, n_segments: int) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Divide a sequence into segments and compute segment-level embeddings.

        Args:
            x: (B, T, D) frame embeddings.
            n_segments: Number of segments.

        Returns:
            (segment_embs, boundaries) where segment_embs is (B, K, D).
        """
        B, T, D = x.shape
        K = min(n_segments, T)
        boundaries: list[tuple[int, int]] = []
        seg_size = T / K
        for i in range(K):
            start = int(round(i * seg_size))
            end = int(round((i + 1) * seg_size))
            end = max(end, start + 1)
            end = min(end, T)
            boundaries.append((start, end))
        segments = torch.zeros(B, K, D, dtype=x.dtype, device=x.device)
        for i, (s, e) in enumerate(boundaries):
            segments[:, i, :] = x[:, s:e, :].mean(dim=1)
        return segments, boundaries

    def _coarse_alignment(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[tuple[int, int]]]:
        """Level 1: Phase-level GW alignment."""
        x_seg, x_bounds = self._segment_embeddings(x, self.n_phases)
        y_seg, y_bounds = self._segment_embeddings(y, self.n_phases)
        D1 = _intra_distance_matrix(x_seg)
        D2 = _intra_distance_matrix(y_seg)
        _, P_coarse = self.gw_solver(D1, D2)
        return P_coarse, x_bounds, y_bounds

    def _build_constraint_mask(
        self,
        P_coarse: torch.Tensor,
        x_bounds: list[tuple[int, int]],
        y_bounds: list[tuple[int, int]],
        T1: int,
        T2: int,
    ) -> torch.Tensor:
        """Build frame-level soft constraint mask from phase alignment."""
        B = P_coarse.shape[0]
        mask = torch.zeros(B, T1, T2, dtype=P_coarse.dtype, device=P_coarse.device)
        for k, (xs, xe) in enumerate(x_bounds):
            for l_idx, (ys, ye) in enumerate(y_bounds):
                mask[:, xs:xe, ys:ye] = P_coarse[:, k, l_idx].view(B, 1, 1)
        mask_max = mask.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        mask = mask / mask_max
        mask = torch.sigmoid((mask - 0.5) / self.constraint_temperature)
        return mask

    def _fine_alignment(
        self,
        C: torch.Tensor,
        constraint_mask: torch.Tensor,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Level 2+3: Constrained Sinkhorn alignment."""
        large_cost = C.max().item() * 3.0
        C_constrained = C + large_cost * (1.0 - constraint_mask)
        return self.sinkhorn_fine(C_constrained, a, b)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cost_type: str = "cosine",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute hierarchical OT alignment.

        Args:
            x: (B, T1, D) or (T1, D) source embeddings.
            y: (B, T2, D) or (T2, D) target embeddings.
            cost_type: "cosine" or "euclidean" for frame-level cost.

        Returns:
            (distance, P_fine, P_coarse).
        """
        squeeze_batch = x.dim() == 2
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if y.dim() == 2:
            y = y.unsqueeze(0)
        B, T1, D = x.shape
        T2 = y.shape[1]
        P_coarse, x_bounds, y_bounds = self._coarse_alignment(x, y)
        constraint_mask = self._build_constraint_mask(P_coarse, x_bounds, y_bounds, T1, T2)
        if cost_type == "cosine":
            C = _cosine_cost_matrix(x, y)
        elif cost_type == "euclidean":
            C = _euclidean_cost_matrix(x, y)
        else:
            raise ValueError(f"Unknown cost_type: {cost_type!r}")
        distance, P_fine = self._fine_alignment(C, constraint_mask)
        if squeeze_batch:
            distance = distance.squeeze(0)
            P_fine = P_fine.squeeze(0)
            P_coarse = P_coarse.squeeze(0)
        return distance, P_fine, P_coarse


# ---------------------------------------------------------------------------
# WassersteinBarycenter
# ---------------------------------------------------------------------------


class WassersteinBarycenter(nn.Module):
    """Wasserstein barycenter via fixed-point Sinkhorn iterations.

    Computes the weighted Frechet mean in Wasserstein space: the distribution
    that minimizes the sum of (weighted) Wasserstein distances to a set of
    input distributions.

    Reference: Cuturi & Doucet, "Fast Computation of Wasserstein
    Barycenters", ICML 2014.

    Args:
        epsilon: Entropy regularization.
        max_iter: Maximum Sinkhorn iterations per barycenter update.
        max_outer_iter: Maximum fixed-point iterations.
        tol: Convergence tolerance on barycenter change.
        support_size: If specified, barycenter support size.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        max_iter: int = 100,
        max_outer_iter: int = 20,
        tol: float = 1e-6,
        support_size: int | None = None,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.max_outer_iter = max_outer_iter
        self.tol = tol
        self.support_size = support_size

    def compute_barycenter(
        self,
        distributions: list[torch.Tensor],
        costs: list[torch.Tensor],
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the Wasserstein barycenter of a set of distributions.

        Uses the iterative Bregman projection algorithm from Benamou et al.
        (2015), which is more stable than the original Cuturi-Doucet (2014)
        fixed-point iteration.

        Args:
            distributions: List of K distributions, each a (N_k,) tensor
                summing to 1.
            costs: List of K cost matrices, each (N_k, M) where M is the
                barycenter support size.
            weights: (K,) barycenter weights summing to 1.

        Returns:
            (M,) barycenter distribution.
        """
        K = len(distributions)
        if K == 0:
            raise ValueError("Need at least one distribution")
        if len(costs) != K:
            raise ValueError(f"Got {K} distributions but {len(costs)} cost matrices")
        M = costs[0].shape[1]
        dtype = distributions[0].dtype
        device = distributions[0].device
        if weights is None:
            weights = torch.full((K,), 1.0 / K, dtype=dtype, device=device)
        # Precompute log-kernels and log-distributions
        log_Ks = [-C_k / self.epsilon for C_k in costs]
        log_as = [_log_safe(d) for d in distributions]
        # Initialize scaling vectors
        v_list = [torch.zeros(M, dtype=dtype, device=device) for _ in range(K)]
        # Initialize barycenter as uniform
        q = torch.full((M,), 1.0 / M, dtype=dtype, device=device)
        for outer_step in range(self.max_outer_iter):
            q_prev = q.clone()
            log_q_new = torch.zeros(M, dtype=dtype, device=device)
            for k in range(K):
                log_K_k = log_Ks[k]
                log_a_k = log_as[k]
                v_k = v_list[k]
                # Row update
                u_k = log_a_k - torch.logsumexp(log_K_k + v_k.unsqueeze(0), dim=1)
                # Column marginal in log domain
                log_col_k = torch.logsumexp(u_k.unsqueeze(1) + log_K_k, dim=0) + v_k
                log_q_new = log_q_new + weights[k] * log_col_k
            # Normalize
            log_q_new = log_q_new - torch.logsumexp(log_q_new, dim=0)
            q = torch.exp(log_q_new)
            log_q = _log_safe(q)
            # Update v_k for new barycenter
            for k in range(K):
                log_K_k = log_Ks[k]
                log_a_k = log_as[k]
                u_k = log_a_k - torch.logsumexp(log_K_k + v_list[k].unsqueeze(0), dim=1)
                v_list[k] = log_q - torch.logsumexp(log_K_k + u_k.unsqueeze(1), dim=0)
            change = (q - q_prev).abs().sum()
            if change.item() < self.tol:
                logger.debug("Barycenter converged at iteration %d (change=%.2e)", outer_step, change.item())
                break
        return q

    def forward(
        self,
        distributions: list[torch.Tensor],
        costs: list[torch.Tensor],
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Wasserstein barycenter (nn.Module interface)."""
        return self.compute_barycenter(distributions, costs, weights)
