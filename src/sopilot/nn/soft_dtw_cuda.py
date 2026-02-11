"""CUDA-accelerated Soft-DTW with custom autograd.

Implements Cuturi & Blondel (2017) with backward from Mensch & Blondel (2018).
Uses vectorized PyTorch ops with anti-diagonal wavefront parallelism.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sopilot.nn.constants import INF as _INF, GAMMA_MIN as _GAMMA_MIN
from sopilot.nn.functional import pairwise_euclidean_sq, pairwise_cosine_dist

logger = logging.getLogger(__name__)

__all__ = [
    "SoftDTWFunction",
    "SoftDTWCuda",
    "SoftDTWAlignmentCuda",
    "multi_scale_sdtw",
    "pairwise_soft_dtw",
]


def _compute_pairwise_cost(x, y, metric="cosine"):
    """Compute pairwise cost matrix (always returns batched 3D)."""
    if metric == "cosine":
        c = pairwise_cosine_dist(x, y)
    elif metric == "euclidean":
        c = pairwise_euclidean_sq(x, y)
    else:
        raise ValueError(f"Unknown metric: {metric!r}")
    return c.unsqueeze(0) if c.dim() == 2 else c


def _apply_bandwidth_mask(cost, bandwidth):
    """Sakoe-Chiba band constraint."""
    if bandwidth is None or bandwidth >= 1.0: return cost
    B, M, N = cost.shape
    bandwidth = max(0.0, bandwidth)
    ii = torch.arange(M, device=cost.device, dtype=cost.dtype).unsqueeze(1)
    jj = torch.arange(N, device=cost.device, dtype=cost.dtype).unsqueeze(0)
    mask = (ii / max(M - 1, 1) - jj / max(N - 1, 1)).abs() > bandwidth
    mc = cost.clone()
    mc[:, mask] = _INF
    return mc


class SoftDTWFunction(torch.autograd.Function):
    """Custom autograd for Soft-DTW (Mensch & Blondel 2018)."""

    @staticmethod
    def forward(ctx, cost, gamma):
        B, M, N = cost.shape
        dev, dt = cost.device, cost.dtype
        gv = gamma.detach().clamp(min=_GAMMA_MIN)
        R = torch.full((B, M + 1, N + 1), _INF, device=dev, dtype=dt)
        R[:, 0, 0] = 0.0
        for d in range(M + N - 1):
            i0, i1 = max(0, d - N + 1), min(d, M - 1)
            if i0 > i1: continue
            ii = torch.arange(i0, i1 + 1, device=dev)
            jj = d - ii
            ip, jp = ii + 1, jj + 1
            p = torch.stack([R[:, ip - 1, jp - 1], R[:, ip - 1, jp], R[:, ip, jp - 1]], dim=-1)
            R[:, ip, jp] = cost[:, ii, jj] + (-gv * torch.logsumexp(p / (-gv), dim=-1))
        result = R[:, M, N].clone()
        ctx.save_for_backward(cost, R, gv.unsqueeze(0))
        ctx.M, ctx.N = M, N
        return result

    @staticmethod
    def backward(ctx, grad_output):
        cost, R, gp = ctx.saved_tensors
        gv = gp[0]
        M, N, B = ctx.M, ctx.N, cost.shape[0]
        dev, dt = cost.device, cost.dtype
        E = torch.zeros((B, M + 2, N + 2), device=dev, dtype=dt)
        E[:, M, N] = 1.0
        Re = torch.full((B, M + 2, N + 2), -_INF, device=dev, dtype=dt)
        Re[:, :M + 1, :N + 1] = R
        Ce = torch.zeros((B, M + 2, N + 2), device=dev, dtype=dt)
        Ce[:, 1:M + 1, 1:N + 1] = cost
        for d in range(M + N - 3, -1, -1):
            i0, i1 = max(0, d - N + 1), min(d, M - 1)
            if i0 > i1: continue
            ii = torch.arange(i0, i1 + 1, device=dev)
            jj = d - ii
            ip, jp = ii + 1, jj + 1
            rc = Re[:, ip, jp]
            wd = torch.exp(((Re[:, ip + 1, jp + 1] - Ce[:, ip + 1, jp + 1] - rc) / gv).clamp(-50, 50))
            wn = torch.exp(((Re[:, ip + 1, jp] - Ce[:, ip + 1, jp] - rc) / gv).clamp(-50, 50))
            wr = torch.exp(((Re[:, ip, jp + 1] - Ce[:, ip, jp + 1] - rc) / gv).clamp(-50, 50))
            E[:, ip, jp] = E[:, ip + 1, jp + 1] * wd + E[:, ip + 1, jp] * wn + E[:, ip, jp + 1] * wr
        return E[:, 1:M + 1, 1:N + 1] * grad_output.view(B, 1, 1), None


class SoftDTWCuda(nn.Module):
    """Soft-DTW module with learnable gamma, normalization, and bandwidth."""

    def __init__(self, gamma=1.0, normalize=True, bandwidth=None, metric="cosine"):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.normalize = normalize
        self.bandwidth = bandwidth
        self.metric = metric

    def _effective_gamma(self):
        return self.gamma.abs().clamp(min=_GAMMA_MIN)

    def _sdtw_cost(self, cost):
        cost = _apply_bandwidth_mask(cost, self.bandwidth)
        return SoftDTWFunction.apply(cost, self._effective_gamma())

    def forward(self, x, y):
        was_unbatched = x.dim() == 2
        if x.dim() == 2: x = x.unsqueeze(0)
        if y.dim() == 2: y = y.unsqueeze(0)
        cost_xy = _compute_pairwise_cost(x, y, metric=self.metric)
        dist = self._sdtw_cost(cost_xy)
        if self.normalize:
            cost_xx = _compute_pairwise_cost(x, x, metric=self.metric)
            cost_yy = _compute_pairwise_cost(y, y, metric=self.metric)
            dist_xx = self._sdtw_cost(cost_xx)
            dist_yy = self._sdtw_cost(cost_yy)
            dist = dist - 0.5 * (dist_xx + dist_yy)
        if was_unbatched: return dist.squeeze(0)
        return dist


class SoftDTWAlignmentCuda(nn.Module):
    """Differentiable soft alignment matrix from Soft-DTW backward DP."""

    def __init__(self, gamma=1.0, bandwidth=None, metric="cosine"):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.bandwidth = bandwidth
        self.metric = metric

    def _effective_gamma(self):
        return self.gamma.abs().clamp(min=_GAMMA_MIN)

    def _compute_alignment(self, cost):
        cost = _apply_bandwidth_mask(cost, self.bandwidth)
        B, M, N = cost.shape
        dev, dt = cost.device, cost.dtype
        gv = self._effective_gamma()
        R = torch.full((B, M + 1, N + 1), _INF, device=dev, dtype=dt)
        R[:, 0, 0] = 0.0
        for d in range(M + N - 1):
            i0, i1 = max(0, d - N + 1), min(d, M - 1)
            if i0 > i1: continue
            ii = torch.arange(i0, i1 + 1, device=dev)
            jj = d - ii
            ip, jp = ii + 1, jj + 1
            p = torch.stack([R[:, ip - 1, jp - 1], R[:, ip - 1, jp], R[:, ip, jp - 1]], dim=-1)
            R[:, ip, jp] = cost[:, ii, jj] + (-gv * torch.logsumexp(p / (-gv), dim=-1))
        distance = R[:, M, N]
        E = torch.zeros((B, M + 2, N + 2), device=dev, dtype=dt)
        E[:, M, N] = 1.0
        Re = torch.full((B, M + 2, N + 2), -_INF, device=dev, dtype=dt)
        Re[:, :M + 1, :N + 1] = R
        Ce = torch.zeros((B, M + 2, N + 2), device=dev, dtype=dt)
        Ce[:, 1:M + 1, 1:N + 1] = cost
        for d in range(M + N - 3, -1, -1):
            i0, i1 = max(0, d - N + 1), min(d, M - 1)
            if i0 > i1: continue
            ii = torch.arange(i0, i1 + 1, device=dev)
            jj = d - ii
            ip, jp = ii + 1, jj + 1
            rc = Re[:, ip, jp]
            wd = torch.exp(((Re[:, ip+1, jp+1] - Ce[:, ip+1, jp+1] - rc) / gv).clamp(-50, 50))
            wn = torch.exp(((Re[:, ip+1, jp] - Ce[:, ip+1, jp] - rc) / gv).clamp(-50, 50))
            wr = torch.exp(((Re[:, ip, jp+1] - Ce[:, ip, jp+1] - rc) / gv).clamp(-50, 50))
            E[:, ip, jp] = E[:, ip+1, jp+1]*wd + E[:, ip+1, jp]*wn + E[:, ip, jp+1]*wr
        return E[:, 1:M+1, 1:N+1], distance

    def forward(self, x, y):
        was_unbatched = x.dim() == 2
        if x.dim() == 2: x = x.unsqueeze(0)
        if y.dim() == 2: y = y.unsqueeze(0)
        cost = _compute_pairwise_cost(x, y, metric=self.metric)
        alignment, distance = self._compute_alignment(cost)
        if was_unbatched: return alignment.squeeze(0), distance.squeeze(0)
        return alignment, distance

def _downsample_sequence(x, factor):
    """Downsample a sequence by averaging consecutive frames."""
    if factor <= 1:
        return x
    if x.dim() == 2:
        T, D = x.shape
        trim = (T // factor) * factor
        return x[:trim].reshape(-1, factor, D).mean(dim=1)
    B, T, D = x.shape
    trim = (T // factor) * factor
    return x[:, :trim].reshape(B, -1, factor, D).mean(dim=2)


def multi_scale_sdtw(x, y, gamma=1.0, normalize=True, bandwidth=None,
                      metric="cosine", scales=None, weights=None):
    """Compute Soft-DTW at multiple temporal resolutions.

    Computes Soft-DTW at each scale (downsampling factor) and returns
    a weighted combination. This captures both fine-grained and coarse
    temporal structure.

    Args:
        x: Query sequences (B, M, D) or (M, D).
        y: Reference sequences (B, N, D) or (N, D).
        gamma: Smoothing parameter.
        normalize: Whether to use normalized Soft-DTW.
        bandwidth: Optional Sakoe-Chiba bandwidth.
        metric: Distance metric ("cosine" or "euclidean").
        scales: List of downsampling factors (default [1, 2, 4]).
        weights: List of weights per scale (default uniform).

    Returns:
        Weighted sum of Soft-DTW distances across scales (B,) or scalar.
    """
    if scales is None:
        scales = [1, 2, 4]
    if weights is None:
        weights = [1.0 / len(scales)] * len(scales)
    assert len(scales) == len(weights), "scales and weights must have same length"

    was_unbatched = x.dim() == 2
    if x.dim() == 2: x = x.unsqueeze(0)
    if y.dim() == 2: y = y.unsqueeze(0)

    total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    sdtw = SoftDTWCuda(gamma=gamma, normalize=normalize,
                        bandwidth=bandwidth, metric=metric)
    # Move gamma to match device
    sdtw = sdtw.to(x.device)
    # Detach gamma so it does not double-count in multi-scale
    sdtw.gamma.requires_grad_(False)
    sdtw.gamma.data = torch.tensor(float(gamma), device=x.device, dtype=x.dtype)

    for scale, weight in zip(scales, weights):
        xs = _downsample_sequence(x, scale)
        ys = _downsample_sequence(y, scale)
        if xs.shape[1] < 2 or ys.shape[1] < 2:
            logger.debug("Skipping scale %d: sequence too short after downsampling", scale)
            continue
        d = sdtw(xs, ys)
        total = total + weight * d

    if was_unbatched:
        return total.squeeze(0)
    return total


def pairwise_soft_dtw(sequences, gamma=1.0, normalize=True, bandwidth=None,
                       metric="cosine"):
    """Compute pairwise Soft-DTW distance matrix for a list of sequences.

    Given K sequences, computes the (K, K) distance matrix where entry (i, j)
    is the Soft-DTW distance between sequences[i] and sequences[j].

    Args:
        sequences: List of K tensors, each (T_k, D). Sequence lengths may vary.
        gamma: Smoothing parameter.
        normalize: Whether to use normalized Soft-DTW.
        bandwidth: Optional Sakoe-Chiba bandwidth.
        metric: Distance metric ("cosine" or "euclidean").

    Returns:
        dist_matrix: (K, K) tensor of pairwise distances.
    """
    K = len(sequences)
    if K == 0:
        return torch.zeros((0, 0))

    dev = sequences[0].device
    dt = sequences[0].dtype
    dist_matrix = torch.zeros((K, K), device=dev, dtype=dt)

    sdtw = SoftDTWCuda(gamma=gamma, normalize=normalize,
                        bandwidth=bandwidth, metric=metric)
    sdtw = sdtw.to(dev)
    sdtw.gamma.requires_grad_(False)
    sdtw.gamma.data = torch.tensor(float(gamma), device=dev, dtype=dt)

    for i in range(K):
        for j in range(i + 1, K):
            d = sdtw(sequences[i].unsqueeze(0), sequences[j].unsqueeze(0))
            dist_matrix[i, j] = d.squeeze()
            dist_matrix[j, i] = d.squeeze()

    return dist_matrix

