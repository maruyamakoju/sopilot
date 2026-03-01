"""Soft-DTW: differentiable time-series alignment for SOP scoring.

Reference:
    Cuturi, M. & Blondel, M. (2017). Soft-DTW: a Differentiable Loss Function
    for Time-Series. Proceedings of ICML 2017.

Key notation:
    m, n  — sequence lengths (gold, trainee)
    D     — embedding dimension
    gamma — smoothing parameter (→0 recovers hard-DTW, large → Euclidean)
    R     — (m+2) × (n+2) accumulated cost matrix (padded with sentinels)
    E     — (m+2) × (n+2) gradient matrix (backward pass)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Optional Numba JIT — graceful pure-Python fallback
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit  # type: ignore[import]

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    def _njit(**_kw):  # type: ignore[misc]
        def _decorator(fn):  # type: ignore[misc]
            return fn
        return _decorator

    _NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Numerically-stable softmin helper
# ---------------------------------------------------------------------------

def _softmin3(a: float, b: float, c: float, gamma: float) -> float:
    """Compute softmin(a, b, c) with temperature gamma (pure Python).

    Numerically stable: uses max-shift trick + exponent clipping to prevent
    overflow/underflow for very small gamma values.
    """
    m = min(a, b, c)
    # Clip exponents to prevent overflow (exp(709) is float64 max)
    _EXP_CLIP = 500.0
    ea = max(min(-(a - m) / gamma, _EXP_CLIP), -_EXP_CLIP)
    eb = max(min(-(b - m) / gamma, _EXP_CLIP), -_EXP_CLIP)
    ec = max(min(-(c - m) / gamma, _EXP_CLIP), -_EXP_CLIP)
    sa = math.exp(ea)
    sb = math.exp(eb)
    sc = math.exp(ec)
    s = sa + sb + sc
    return -gamma * math.log(s) + m


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def _soft_dtw_forward_python(
    D: np.ndarray,
    gamma: float,
    band_width: int | None,
) -> np.ndarray:
    """Pure-Python forward pass filling (m+2)×(n+2) accumulated cost matrix."""
    m, n = D.shape
    INF = 1e15
    R = np.full((m + 2, n + 2), INF, dtype=np.float64)
    R[0, 0] = 0.0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if band_width is not None and abs(i - j) > band_width:
                continue
            r_diag = R[i - 1, j - 1]
            r_up = R[i - 1, j]
            r_left = R[i, j - 1]
            R[i, j] = D[i - 1, j - 1] + _softmin3(r_diag, r_up, r_left, gamma)

    return R


try:
    from numba import njit as _real_njit  # type: ignore[import]

    @_real_njit(cache=False)
    def _soft_dtw_forward_numba(  # type: ignore[misc]
        D: np.ndarray,
        gamma: float,
        band_width: int,  # -1 means no band
    ) -> np.ndarray:
        m, n = D.shape
        INF = 1e15
        R = np.full((m + 2, n + 2), INF)
        R[0, 0] = 0.0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if band_width >= 0 and abs(i - j) > band_width:
                    continue
                a = R[i - 1, j - 1]
                b = R[i - 1, j]
                c = R[i, j - 1]
                mn = min(a, b, c)
                _CLIP = 500.0
                ea = max(min(-(a - mn) / gamma, _CLIP), -_CLIP)
                eb = max(min(-(b - mn) / gamma, _CLIP), -_CLIP)
                ec = max(min(-(c - mn) / gamma, _CLIP), -_CLIP)
                sa = math.exp(ea)
                sb = math.exp(eb)
                sc = math.exp(ec)
                s = sa + sb + sc
                R[i, j] = D[i - 1, j - 1] + (-gamma * math.log(s) + mn)

        return R

    _NUMBA_FORWARD_AVAILABLE = True
except Exception:
    _NUMBA_FORWARD_AVAILABLE = False


def _soft_dtw_forward(
    D: np.ndarray,
    gamma: float,
    band_width: int | None,
) -> np.ndarray:
    if _NUMBA_FORWARD_AVAILABLE:
        bw = -1 if band_width is None else band_width
        return _soft_dtw_forward_numba(D.astype(np.float64), gamma, bw)  # type: ignore[misc]
    return _soft_dtw_forward_python(D, gamma, band_width)


# ---------------------------------------------------------------------------
# Backward pass (gradient w.r.t. local cost matrix D)
# ---------------------------------------------------------------------------

def _soft_dtw_backward(D: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    """Algorithm 2 from Cuturi & Blondel (2017).

    Returns E of shape (m+2)×(n+2).  The inner m×n block E[1:m+1, 1:n+1]
    is the gradient ∂(soft-DTW) / ∂D[i,j].
    """
    m, n = D.shape
    INF = 1e15
    E = np.zeros((m + 2, n + 2), dtype=np.float64)
    # Only initialise the terminal cell if it was actually reached
    E[m, n] = 1.0 if math.isfinite(R[m, n]) else 0.0

    # Sentinel row/col so boundary conditions work cleanly
    R_ext = R.copy()
    # Cells outside the DP table should not propagate gradients
    R_ext[m + 1, :] = -INF
    R_ext[:, n + 1] = -INF

    for j in range(n, 0, -1):
        for i in range(m, 0, -1):
            # Skip unreachable cells (e.g. outside a Sakoe-Chiba band)
            if R_ext[i, j] > 1e14:
                continue

            a = R_ext[i + 1, j + 1]  # came from diagonal
            b = R_ext[i + 1, j]      # came from below (i going up)
            c = R_ext[i, j + 1]      # came from right (j going left)
            current = R_ext[i, j]
            d = D[i - 1, j - 1]

            # Soft-min weights for each predecessor
            def _w(r_pred: float) -> float:
                # Sentinel cells do not contribute to the gradient
                if r_pred <= -1e14 or not math.isfinite(r_pred):
                    return 0.0
                exponent = (r_pred - current + d) / gamma
                if exponent > 500:
                    return 0.0
                if exponent < -500:
                    return math.exp(500.0)
                return math.exp(-exponent)

            wa = _w(a)
            wb = _w(b)
            wc = _w(c)
            total = wa + wb + wc + 1e-15

            E[i, j] = (
                E[i + 1, j + 1] * wa / total
                + E[i + 1, j] * wb / total
                + E[i, j + 1] * wc / total
            )

    return E


# ---------------------------------------------------------------------------
# Hard-DTW traceback (for alignment_path visualization)
# ---------------------------------------------------------------------------

def _hard_dtw_traceback(D: np.ndarray) -> list[tuple[int, int]]:
    """Run hard DTW and return the alignment path as (i, j) 0-indexed pairs."""
    m, n = D.shape
    INF = 1e15
    acc = np.full((m, n), INF)
    acc[0, 0] = D[0, 0]
    for i in range(1, m):
        acc[i, 0] = acc[i - 1, 0] + D[i, 0]
    for j in range(1, n):
        acc[0, j] = acc[0, j - 1] + D[0, j]
    for i in range(1, m):
        for j in range(1, n):
            acc[i, j] = D[i, j] + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])

    path: list[tuple[int, int]] = []
    i, j = m - 1, n - 1
    path.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            choice = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
            if choice == 0:
                i -= 1
                j -= 1
            elif choice == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class SoftDTWResult:
    """Result of a Soft-DTW computation."""

    distance: float
    """Soft-DTW distance (differentiable scalar)."""

    gamma: float
    """Smoothing parameter used."""

    cost_matrix: np.ndarray
    """Local cosine-distance cost matrix, shape (m, n), float32."""

    R: np.ndarray
    """Accumulated cost matrix (padded), shape (m+2, n+2), float64."""

    alignment_path: list[tuple[int, int]]
    """Hard-DTW traceback path for visualization, 0-indexed (i, j) pairs."""

    normalized_cost: float
    """distance / max(m, n) for sequence-length-independent comparisons."""

    gradient_gold: np.ndarray | None = field(default=None, repr=False)
    """Gradient ∂(soft-DTW)/∂gold, shape (m, D).  Set by soft_dtw_gradient()."""

    gradient_trainee: np.ndarray | None = field(default=None, repr=False)
    """Gradient ∂(soft-DTW)/∂trainee, shape (n, D).  Set by soft_dtw_gradient()."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def soft_dtw(
    gold: np.ndarray,
    trainee: np.ndarray,
    gamma: float = 1.0,
    band_width: int | None = None,
) -> SoftDTWResult:
    """Compute Soft-DTW between two embedding sequences.

    Args:
        gold:       Gold embedding array, shape (m, D).
        trainee:    Trainee embedding array, shape (n, D).
        gamma:      Smoothing temperature.  Smaller → closer to hard DTW.
        band_width: Sakoe-Chiba band constraint.  None = unconstrained.

    Returns:
        SoftDTWResult with distance and auxiliary data.

    References:
        Cuturi & Blondel (ICML 2017), https://arxiv.org/abs/1703.01541
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    gold_f = gold.astype(np.float32)
    trainee_f = trainee.astype(np.float32)

    # L2-normalize rows (unit sphere → cosine similarity via dot product)
    g_norms = np.linalg.norm(gold_f, axis=1, keepdims=True).clip(1e-8)
    t_norms = np.linalg.norm(trainee_f, axis=1, keepdims=True).clip(1e-8)
    gold_n = gold_f / g_norms
    trainee_n = trainee_f / t_norms

    # Local cost: 1 − cosine_similarity ∈ [0, 2]
    dot = gold_n @ trainee_n.T  # (m, n)
    D = (1.0 - np.clip(dot, -1.0, 1.0)).astype(np.float64)

    R = _soft_dtw_forward(D, gamma, band_width)
    m, n = D.shape
    distance = float(R[m, n])

    alignment_path = _hard_dtw_traceback(D)

    return SoftDTWResult(
        distance=distance,
        gamma=gamma,
        cost_matrix=D.astype(np.float32),
        R=R,
        alignment_path=alignment_path,
        normalized_cost=distance / max(m, n),
    )


def soft_dtw_gradient(
    gold: np.ndarray,
    trainee: np.ndarray,
    gamma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Soft-DTW and its gradients w.r.t. input embeddings.

    Enables gradient-based fine-tuning of the V-JEPA2 embedding space.

    Args:
        gold:    Gold sequence, shape (m, D).
        trainee: Trainee sequence, shape (n, D).
        gamma:   Smoothing temperature.

    Returns:
        (grad_gold, grad_trainee) — gradients w.r.t. the input arrays,
        shapes (m, D) and (n, D) respectively.

    The chain rule through D[i,j] = 1 − <g_i, t_j> (unit vectors) gives:
        ∂L/∂g_i = -∑_j E[i,j] * t_j
        ∂L/∂t_j = -∑_i E[i,j] * g_i
    """
    gold_f = gold.astype(np.float32)
    trainee_f = trainee.astype(np.float32)

    g_norms = np.linalg.norm(gold_f, axis=1, keepdims=True).clip(1e-8)
    t_norms = np.linalg.norm(trainee_f, axis=1, keepdims=True).clip(1e-8)
    gold_n = gold_f / g_norms
    trainee_n = trainee_f / t_norms

    dot = gold_n @ trainee_n.T
    D = (1.0 - np.clip(dot, -1.0, 1.0)).astype(np.float64)

    R = _soft_dtw_forward(D, gamma, None)
    E = _soft_dtw_backward(D, R, gamma)

    m, n = D.shape
    E_inner = E[1 : m + 1, 1 : n + 1]  # shape (m, n)

    # Chain rule through cosine distance: ∂D/∂g_i = -t_j (for unit vecs)
    grad_gold = -(E_inner @ trainee_n).astype(np.float32)     # (m, D)
    grad_trainee = -(E_inner.T @ gold_n).astype(np.float32)   # (n, D)

    return grad_gold, grad_trainee


def compare_soft_vs_hard(
    gold: np.ndarray,
    trainee: np.ndarray,
    gamma: float = 1.0,
) -> dict:
    """Compare Soft-DTW vs. Hard-DTW alignment on the same pair.

    Returns a dict with distances, path similarity (Jaccard), and timing.
    """
    import time

    # Soft-DTW
    t0 = time.perf_counter()
    soft_result = soft_dtw(gold, trainee, gamma=gamma)
    soft_ms = (time.perf_counter() - t0) * 1000

    # Hard-DTW (reuse cost matrix from soft result)
    t0 = time.perf_counter()
    hard_path = _hard_dtw_traceback(soft_result.cost_matrix.astype(np.float64))
    hard_ms = (time.perf_counter() - t0) * 1000

    m, n = soft_result.cost_matrix.shape
    hard_cost = float(
        sum(soft_result.cost_matrix[i, j] for i, j in hard_path)
    )
    hard_normalized = hard_cost / max(m, n)

    # Path similarity (Jaccard index)
    soft_set = set(soft_result.alignment_path)
    hard_set = set(hard_path)
    jaccard = len(soft_set & hard_set) / len(soft_set | hard_set) if (soft_set | hard_set) else 1.0

    return {
        "soft_distance": soft_result.distance,
        "hard_distance": hard_cost,
        "soft_normalized": soft_result.normalized_cost,
        "hard_normalized": hard_normalized,
        "path_similarity": jaccard,
        "soft_ms": soft_ms,
        "hard_ms": hard_ms,
        "gamma": gamma,
    }


def optimal_gamma(
    gold: np.ndarray,
    trainee: np.ndarray,
    n_trials: int = 10,
) -> float:
    """Estimate a good gamma as 10% of the mean local cost.

    Args:
        gold:     Gold embeddings, shape (m, D).
        trainee:  Trainee embeddings, shape (n, D).
        n_trials: Reserved for future line-search; currently unused.

    Returns:
        Estimated gamma clamped to [1e-4, 10.0].
    """
    gold_n = gold / np.linalg.norm(gold, axis=1, keepdims=True).clip(1e-8)
    trainee_n = trainee / np.linalg.norm(trainee, axis=1, keepdims=True).clip(1e-8)
    dot = (gold_n @ trainee_n.T).astype(np.float32)
    D = 1.0 - np.clip(dot, -1.0, 1.0)
    mean_cost = float(D.mean())
    if mean_cost < 1e-8:
        return 0.01
    gamma = mean_cost * 0.1
    return float(np.clip(gamma, 1e-4, 10.0))
