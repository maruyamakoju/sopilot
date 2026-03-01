"""Sinkhorn-based Optimal Transport alignment for video embedding sequences.

An alternative to Dynamic Time Warping (DTW) for temporal alignment of
SOP video clip embeddings.  Unlike DTW, which enforces monotonicity and
continuity constraints on the alignment path, Optimal Transport (OT)
finds a *coupling* between two discrete distributions that minimises the
total transport cost.  The entropic regularisation introduced by Cuturi
(2013) makes the optimisation differentiable, GPU-friendly, and
solvable via the Sinkhorn-Knopp matrix-scaling algorithm.

Theoretical background
----------------------
Let :math:`\\mu = \\sum_i a_i \\delta_{x_i}` and
:math:`\\nu = \\sum_j b_j \\delta_{y_j}` be two discrete measures
supported on embedding vectors.  The *entropic-regularised* OT cost is:

.. math::

    W_\\varepsilon(\\mu, \\nu) =
        \\min_{P \\in U(a,b)}
            \\langle C, P \\rangle + \\varepsilon H(P)

where :math:`C_{ij} = c(x_i, y_j)` is the pairwise cost matrix
(here: cosine distance), :math:`U(a,b)` is the set of couplings
with marginals *a* and *b*, and :math:`H(P)` is the entropy of the
transport plan.

**Approximation bound** (Cuturi 2013, Corollary 1):

    :math:`|W_\\varepsilon - W| \\leq \\varepsilon \\log n`

where *W* is the true (unregularised) Wasserstein distance and *n* is
the number of support points.  The Sinkhorn algorithm converges to a
:math:`\\delta`-approximate solution in :math:`O(n^2 / \\varepsilon
\\cdot \\log(1/\\delta))` iterations (Altschuler, Weed & Rigollet 2017).

**When to prefer OT over DTW:**

- When temporal monotonicity is *not* a strong prior (e.g. repeated
  actions, interchangeable sub-steps).
- When a soft, differentiable alignment signal is desired for
  fine-tuning embeddings (the transport plan gradient is smoother than
  DTW's).
- When comparing sequences of very different lengths where DTW's
  monotonicity constraint introduces excessive stretching artefacts.

References
----------
- Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of
  Optimal Transport. Advances in Neural Information Processing Systems
  (NeurIPS) 26.
- Peyre, G. & Cuturi, M. (2019). Computational Optimal Transport.
  Foundations and Trends in Machine Learning, 11(5-6), 355-607.
- Altschuler, J., Weed, J. & Rigollet, P. (2017). Near-linear time
  approximation algorithms for optimal transport via Sinkhorn iteration.
  NeurIPS 2017.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimum sequence length (consistent with DTW module)
# ---------------------------------------------------------------------------
_MIN_SEQUENCE_LENGTH = 2


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OTAlignment:
    """Result of an Optimal-Transport alignment computation.

    Attributes:
        wasserstein_distance: Entropic OT cost
            :math:`\\langle C, P^* \\rangle` (transport cost component only,
            excluding the entropy regularisation term).
        normalized_distance: wasserstein_distance normalised by the
            geometric mean of sequence lengths,
            :math:`W / \\sqrt{m \\cdot n}`.
        transport_plan: The optimal coupling matrix :math:`P^*` of shape
            ``(m, n)`` satisfying marginal constraints (up to convergence
            tolerance).
        alignment_path: Hard alignment path extracted from the soft
            transport plan by greedy row/column argmax.
        cost_matrix: Pairwise cosine-distance cost matrix ``C`` of shape
            ``(m, n)``.
        epsilon: Entropic regularisation strength used.
        converged: Whether Sinkhorn iterations converged within tolerance.
        n_iterations: Actual number of Sinkhorn iterations performed.
        gold_length: Number of gold clip embeddings (*m*).
        trainee_length: Number of trainee clip embeddings (*n*).
    """

    wasserstein_distance: float
    normalized_distance: float
    transport_plan: np.ndarray
    alignment_path: list[tuple[int, int]]
    cost_matrix: np.ndarray
    epsilon: float
    converged: bool
    n_iterations: int
    gold_length: int
    trainee_length: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine_cost_matrix(
    gold: np.ndarray,
    trainee: np.ndarray,
) -> np.ndarray:
    """Compute pairwise cosine-distance cost matrix.

    Both inputs are L2-normalised internally (consistent with the DTW
    module) so that the cost :math:`C_{ij} = 1 - \\langle g_i, t_j
    \\rangle \\in [0, 2]`.

    Args:
        gold: Embedding array of shape ``(m, D)``.
        trainee: Embedding array of shape ``(n, D)``.

    Returns:
        Cost matrix of shape ``(m, n)``, dtype float64.
    """
    gold_f = gold.astype(np.float64)
    trainee_f = trainee.astype(np.float64)

    # L2-normalise rows
    g_norms = np.linalg.norm(gold_f, axis=1, keepdims=True)
    t_norms = np.linalg.norm(trainee_f, axis=1, keepdims=True)
    gold_n = gold_f / np.maximum(g_norms, 1e-8)
    trainee_n = trainee_f / np.maximum(t_norms, 1e-8)

    # Cosine distance: 1 - cos_sim
    dot = gold_n @ trainee_n.T
    # Clip to [-1, 1] to avoid numerical issues with acos etc.
    dot = np.clip(dot, -1.0, 1.0)
    return (1.0 - dot).astype(np.float64)


def _log_sinkhorn(
    C: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, bool, int]:
    """Log-domain Sinkhorn-Knopp iterations for numerical stability.

    Operates entirely in log-space to avoid overflow/underflow that
    plagues the standard multiplicative Sinkhorn when epsilon is small.

    The algorithm maintains dual variables :math:`f \\in \\mathbb{R}^m`
    and :math:`g \\in \\mathbb{R}^n` and iterates:

    .. math::

        f_i &\\leftarrow \\varepsilon \\log a_i
              - \\varepsilon \\log \\sum_j
                \\exp\\bigl((g_j - C_{ij}) / \\varepsilon\\bigr)

        g_j &\\leftarrow \\varepsilon \\log b_j
              - \\varepsilon \\log \\sum_i
                \\exp\\bigl((f_i - C_{ij}) / \\varepsilon\\bigr)

    The transport plan is recovered as
    :math:`P_{ij} = \\exp\\bigl((f_i + g_j - C_{ij}) / \\varepsilon\\bigr)`.

    Args:
        C: Cost matrix, shape ``(m, n)``, float64.
        a: Source marginal, shape ``(m,)``, sums to 1.
        b: Target marginal, shape ``(n,)``, sums to 1.
        epsilon: Entropic regularisation parameter (> 0).
        max_iter: Maximum number of Sinkhorn iterations.
        tol: Convergence tolerance on the L1 marginal violation.

    Returns:
        Tuple of (transport_plan, converged, n_iterations).
    """
    m, n = C.shape

    # Log-domain kernel: K_ij = -C_ij / epsilon
    log_K = -C / epsilon  # (m, n)

    # Dual variables (initialised to zero)
    f = np.zeros(m, dtype=np.float64)  # (m,)
    g = np.zeros(n, dtype=np.float64)  # (n,)

    log_a = np.log(np.maximum(a, 1e-300))  # (m,)
    log_b = np.log(np.maximum(b, 1e-300))  # (n,)

    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        # f update: f_i = eps * log(a_i) - eps * logsumexp_j(log_K_ij + g_j / eps)
        # Equivalent: f_i = eps * log(a_i) - eps * logsumexp_j((-C_ij + g_j) / eps)
        M_f = log_K + g[np.newaxis, :] / epsilon  # (m, n)
        f_new = epsilon * log_a - epsilon * _logsumexp_rows(M_f)  # (m,)
        f = f_new

        # g update: g_j = eps * log(b_j) - eps * logsumexp_i(log_K_ij + f_i / eps)
        M_g = log_K + f[:, np.newaxis] / epsilon  # (m, n)
        g_new = epsilon * log_b - epsilon * _logsumexp_cols(M_g)  # (n,)
        g = g_new

        # Check marginal convergence every 5 iterations (and on last)
        if iteration % 5 == 0 or iteration == max_iter:
            # Recover plan in log-domain
            log_P = (f[:, np.newaxis] + g[np.newaxis, :] - C) / epsilon
            log_P_max = log_P.max()
            P = np.exp(log_P - log_P_max)
            P *= np.exp(log_P_max)

            # Marginal violation
            row_sums = P.sum(axis=1)
            col_sums = P.sum(axis=0)
            row_err = float(np.abs(row_sums - a).sum())
            col_err = float(np.abs(col_sums - b).sum())
            marginal_err = max(row_err, col_err)

            if marginal_err < tol:
                converged = True
                break

    # Final plan recovery
    log_P = (f[:, np.newaxis] + g[np.newaxis, :] - C) / epsilon
    # Stable exponentiation: shift by max to avoid overflow
    log_P_shift = log_P - log_P.max()
    P = np.exp(log_P_shift)
    # Normalise to ensure exact marginal satisfaction
    P *= np.exp(log_P.max())
    # Clip any tiny negative artefacts from floating point
    P = np.maximum(P, 0.0)

    return P, converged, iteration


def _logsumexp_rows(M: np.ndarray) -> np.ndarray:
    """Numerically stable logsumexp along axis=1 (rows).

    Args:
        M: 2D array, shape ``(m, n)``.

    Returns:
        1D array of shape ``(m,)``.
    """
    row_max = M.max(axis=1)
    # Guard against -inf rows (all entries are -inf)
    finite_mask = np.isfinite(row_max)
    result = np.full_like(row_max, -np.inf)
    if finite_mask.any():
        shifted = M[finite_mask] - row_max[finite_mask, np.newaxis]
        result[finite_mask] = row_max[finite_mask] + np.log(
            np.exp(shifted).sum(axis=1)
        )
    return result


def _logsumexp_cols(M: np.ndarray) -> np.ndarray:
    """Numerically stable logsumexp along axis=0 (columns).

    Args:
        M: 2D array, shape ``(m, n)``.

    Returns:
        1D array of shape ``(n,)``.
    """
    col_max = M.max(axis=0)
    finite_mask = np.isfinite(col_max)
    result = np.full_like(col_max, -np.inf)
    if finite_mask.any():
        shifted = M[:, finite_mask] - col_max[np.newaxis, finite_mask]
        result[finite_mask] = col_max[finite_mask] + np.log(
            np.exp(shifted).sum(axis=0)
        )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ot_align(
    gold: np.ndarray,
    trainee: np.ndarray,
    *,
    epsilon: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-6,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> OTAlignment:
    """Compute entropic-regularised Optimal Transport alignment.

    Finds the transport plan :math:`P^*` between the gold and trainee
    embedding sequences that minimises the regularised transport cost:

    .. math::

        \\min_{P \\in U(a,b)} \\langle C, P \\rangle
            + \\varepsilon H(P)

    where *C* is the cosine-distance cost matrix and *H(P)* is the
    Shannon entropy of the plan.

    **Approximation guarantee** (Cuturi 2013): the entropic OT cost
    satisfies :math:`|W_\\varepsilon - W| \\leq \\varepsilon \\log n`,
    where *W* is the true Wasserstein distance.

    **Convergence rate** (Altschuler, Weed & Rigollet 2017): Sinkhorn
    converges to a :math:`\\delta`-approximate solution in
    :math:`O(n^2 / \\varepsilon \\cdot \\log(1/\\delta))` iterations.

    Args:
        gold: Gold embedding array, shape ``(m, D)``.  Rows need not be
            pre-normalised; L2 normalisation is applied internally.
        trainee: Trainee embedding array, shape ``(n, D)``.
        epsilon: Entropic regularisation parameter.  Smaller values
            yield plans closer to the true OT solution but require more
            iterations and are numerically less stable.  Typical range:
            0.01 -- 1.0.  Use :func:`optimal_epsilon` for data-driven
            selection.
        max_iter: Maximum Sinkhorn iterations.
        tol: Convergence tolerance on L1 marginal violation.
        a: Source marginal, shape ``(m,)``, must sum to 1.  If ``None``,
            uses the uniform distribution ``1/m``.
        b: Target marginal, shape ``(n,)``, must sum to 1.  If ``None``,
            uses the uniform distribution ``1/n``.

    Returns:
        :class:`OTAlignment` with the transport plan, Wasserstein
        distance, hard alignment path, and auxiliary information.

    Raises:
        ValueError: If inputs are not 2D, have incompatible embedding
            dimensions, or have fewer than
            :data:`_MIN_SEQUENCE_LENGTH` clips.

    References:
        Cuturi (2013). Sinkhorn Distances. NeurIPS 26.
        Peyre & Cuturi (2019). Computational Optimal Transport.
    """
    # --- Input validation ---
    if gold.ndim != 2 or trainee.ndim != 2:
        raise ValueError("gold and trainee must be 2D arrays")
    m, d_gold = gold.shape
    n, d_trainee = trainee.shape
    if d_gold != d_trainee:
        raise ValueError(
            f"Embedding dimensions must match: gold has D={d_gold}, "
            f"trainee has D={d_trainee}"
        )
    if m < _MIN_SEQUENCE_LENGTH or n < _MIN_SEQUENCE_LENGTH:
        raise ValueError(
            f"Both sequences must have at least {_MIN_SEQUENCE_LENGTH} "
            f"clip embeddings, got gold={m}, trainee={n}"
        )
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    # --- Marginals ---
    if a is None:
        a_vec = np.full(m, 1.0 / m, dtype=np.float64)
    else:
        a_vec = np.asarray(a, dtype=np.float64).ravel()
        if len(a_vec) != m:
            raise ValueError(f"Source marginal 'a' must have length {m}, got {len(a_vec)}")
        a_sum = a_vec.sum()
        if abs(a_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Source marginal 'a' must sum to 1.0, got {a_sum:.8f}"
            )

    if b is None:
        b_vec = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        b_vec = np.asarray(b, dtype=np.float64).ravel()
        if len(b_vec) != n:
            raise ValueError(f"Target marginal 'b' must have length {n}, got {len(b_vec)}")
        b_sum = b_vec.sum()
        if abs(b_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Target marginal 'b' must sum to 1.0, got {b_sum:.8f}"
            )

    # --- Cost matrix ---
    C = _cosine_cost_matrix(gold, trainee)

    # --- Handle degenerate case: near-zero cost everywhere ---
    max_cost = float(C.max())
    if max_cost < 1e-12:
        # Identical (or near-identical) sequences — return trivial plan
        P = np.outer(a_vec, b_vec)
        path = extract_alignment_from_plan(P)
        return OTAlignment(
            wasserstein_distance=0.0,
            normalized_distance=0.0,
            transport_plan=P,
            alignment_path=path,
            cost_matrix=C,
            epsilon=epsilon,
            converged=True,
            n_iterations=0,
            gold_length=m,
            trainee_length=n,
        )

    # --- Sinkhorn iterations (log-domain) ---
    P, converged, n_iters = _log_sinkhorn(
        C, a_vec, b_vec, epsilon, max_iter, tol,
    )

    if not converged:
        logger.warning(
            "Sinkhorn did not converge after %d iterations "
            "(tol=%.2e, epsilon=%.4f, m=%d, n=%d). "
            "Consider increasing max_iter or epsilon.",
            n_iters, tol, epsilon, m, n,
        )

    # --- Wasserstein distance: <C, P> (transport cost, excluding entropy) ---
    w_dist = float(np.sum(C * P))

    # --- Normalised distance ---
    # Use geometric mean of sequence lengths for a length-invariant metric
    geo_mean = max(np.sqrt(m * n), 1.0)
    norm_dist = w_dist / geo_mean

    # --- Extract hard alignment path ---
    path = extract_alignment_from_plan(P)

    return OTAlignment(
        wasserstein_distance=w_dist,
        normalized_distance=norm_dist,
        transport_plan=P,
        alignment_path=path,
        cost_matrix=C,
        epsilon=epsilon,
        converged=converged,
        n_iterations=n_iters,
        gold_length=m,
        trainee_length=n,
    )


def extract_alignment_from_plan(
    plan: np.ndarray,
) -> list[tuple[int, int]]:
    """Convert a soft OT transport plan to a hard alignment path.

    Uses a greedy strategy: for each row *i* (gold frame), find the
    column *j* (trainee frame) with the largest transport mass.  Then
    sort the resulting pairs by gold index to produce a monotonic path.
    Ties are broken by selecting the column closest to the diagonal.

    This is analogous to extracting a hard alignment from an attention
    matrix in sequence-to-sequence models (Zenkel et al., WMT 2019).

    Args:
        plan: Transport plan of shape ``(m, n)`` with non-negative
            entries.

    Returns:
        List of ``(i, j)`` tuples sorted by *i*, representing the hard
        alignment path.  Length equals *m* (one match per gold frame).
    """
    if plan.ndim != 2:
        raise ValueError("plan must be a 2D array")
    m, n = plan.shape
    if m == 0 or n == 0:
        return []

    path: list[tuple[int, int]] = []
    for i in range(m):
        row = plan[i]
        max_val = row.max()
        if max_val <= 0:
            # Fallback: assign to nearest diagonal position
            j = min(i, n - 1)
        else:
            # Find all columns tied for the maximum
            candidates = np.where(np.abs(row - max_val) < 1e-12 * max(max_val, 1.0))[0]
            if len(candidates) == 1:
                j = int(candidates[0])
            else:
                # Break tie: pick column closest to diagonal position
                diagonal_pos = i * n / max(m, 1)
                j = int(candidates[np.argmin(np.abs(candidates - diagonal_pos))])
        path.append((i, j))

    # Sort by gold index (should already be sorted, but enforce)
    path.sort(key=lambda pair: pair[0])
    return path


def optimal_epsilon(
    gold: np.ndarray,
    trainee: np.ndarray,
    *,
    quantile: float = 0.5,
    scale: float = 0.05,
) -> float:
    """Data-driven selection of the entropic regularisation parameter.

    Computes the cost matrix and sets epsilon as a fraction of the
    median pairwise cost.  This heuristic ensures that the
    regularisation is scaled appropriately to the magnitude of the
    cost entries, avoiding both under-regularisation (numerical
    instability) and over-regularisation (blurred transport plan).

    The median (or other quantile) of the cost matrix is a robust
    statistic that is insensitive to outliers in the embedding space.

    Guideline from Peyre & Cuturi (2019, Section 4.2): epsilon should
    be of the order of the mean cost divided by a factor of 10-20 for
    a good trade-off between approximation quality and convergence
    speed.

    Args:
        gold: Gold embedding array, shape ``(m, D)``.
        trainee: Trainee embedding array, shape ``(n, D)``.
        quantile: Which quantile of the cost matrix to use as the
            reference scale.  Default 0.5 (median).
        scale: Multiplicative factor applied to the quantile cost.
            Default 0.05 (i.e. epsilon = 5 % of median cost).

    Returns:
        Estimated epsilon, clamped to ``[1e-4, 1.0]``.
    """
    C = _cosine_cost_matrix(gold, trainee)
    ref_cost = float(np.quantile(C, quantile))
    if ref_cost < 1e-8:
        return 0.01  # Identical or near-identical sequences
    eps = ref_cost * scale
    return float(np.clip(eps, 1e-4, 1.0))


def wasserstein_distance(
    gold: np.ndarray,
    trainee: np.ndarray,
    epsilon: float = 0.1,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """Compute the entropic Wasserstein distance between two sequences.

    Convenience function that returns only the scalar distance without
    the full :class:`OTAlignment` result.  Useful for quick pairwise
    comparisons.

    The returned value is the transport cost :math:`\\langle C, P^*
    \\rangle` (excluding the entropy regularisation term), which
    approximates the true Wasserstein distance with error bounded by
    :math:`\\varepsilon \\log n` (Cuturi 2013).

    Args:
        gold: Gold embedding array, shape ``(m, D)``.
        trainee: Trainee embedding array, shape ``(n, D)``.
        epsilon: Entropic regularisation parameter.
        max_iter: Maximum Sinkhorn iterations.
        tol: Convergence tolerance.

    Returns:
        Entropic Wasserstein distance (float, non-negative).
    """
    result = ot_align(
        gold, trainee, epsilon=epsilon, max_iter=max_iter, tol=tol,
    )
    return result.wasserstein_distance
