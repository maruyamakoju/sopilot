"""Dynamic Time Warping for temporal alignment of video embedding sequences.

Supports:
    - Vectorized cosine-distance cost matrix via matmul
    - Optional Numba JIT acceleration for the DP inner loop
    - Sakoe-Chiba band constraint for controlling alignment flexibility
    - Configurable normalization strategies (path-length, diagonal, max-length)

Reference:
    Sakoe, H. & Chiba, S. (1978). Dynamic programming algorithm optimization
    for spoken word recognition. IEEE Trans. Acoustics, Speech, and Signal
    Processing, 26(1), 43-49.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional numba JIT for the DP accumulation inner loop.
# Falls back to pure-Python if numba is not installed.
# ---------------------------------------------------------------------------
try:
    import numba as _nb

    @_nb.njit(cache=False)
    def _dtw_accumulate(local_cost: np.ndarray, band_width: int) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        """Numba-JIT DP fill with optional Sakoe-Chiba band; ~10-50x faster."""
        m, n = local_cost.shape
        cost = np.full((m + 1, n + 1), np.inf)
        direction = np.zeros((m + 1, n + 1), dtype=np.int8)
        cost[0, 0] = np.float64(0.0)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if band_width >= 0 and abs(i - j) > band_width:
                    continue
                d = cost[i - 1, j - 1]
                u = cost[i - 1, j]
                left = cost[i, j - 1]
                if d <= u and d <= left:
                    best_dir = 0
                    best = d
                elif u <= left:
                    best_dir = 1
                    best = u
                else:
                    best_dir = 2
                    best = left
                cost[i, j] = local_cost[i - 1, j - 1] + best
                direction[i, j] = best_dir
        return cost, direction

except ImportError:
    def _dtw_accumulate(local_cost: np.ndarray, band_width: int) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[misc]
        """Pure-Python fallback when numba is not installed."""
        m, n = local_cost.shape
        cost = np.full((m + 1, n + 1), np.inf, dtype=np.float64)
        direction = np.zeros((m + 1, n + 1), dtype=np.int8)
        cost[0, 0] = 0.0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if band_width >= 0 and abs(i - j) > band_width:
                    continue
                d = cost[i - 1, j - 1]
                u = cost[i - 1, j]
                left = cost[i, j - 1]
                if d <= u and d <= left:
                    best_dir = 0
                    best = d
                elif u <= left:
                    best_dir = 1
                    best = u
                else:
                    best_dir = 2
                    best = left
                cost[i, j] = local_cost[i - 1, j - 1] + best
                direction[i, j] = best_dir
        return cost, direction


class NormalizationStrategy(Enum):
    """DTW cost normalization strategy."""
    PATH_LENGTH = "path_length"    # Divide by alignment path length (default)
    DIAGONAL = "diagonal"          # Divide by max(m, n) — diagonal reference
    BOTH = "both"                  # Divide by (m + n) — symmetric normalization


@dataclass(frozen=True)
class DTWAlignment:
    """Result of a DTW alignment computation."""
    total_cost: float
    normalized_cost: float
    path: list[tuple[int, int]]
    path_costs: list[float]
    gold_length: int
    trainee_length: int
    band_width: int | None = None


# Minimum clips required for meaningful DTW alignment
_MIN_SEQUENCE_LENGTH = 2


def dtw_align(
    gold: np.ndarray,
    trainee: np.ndarray,
    *,
    band_width: int | None = None,
    normalization: NormalizationStrategy = NormalizationStrategy.PATH_LENGTH,
) -> DTWAlignment:
    """Compute DTW alignment between two embedding sequences.

    Args:
        gold:          Gold embedding array, shape (m, D), L2-normalized rows.
        trainee:       Trainee embedding array, shape (n, D), L2-normalized rows.
        band_width:    Sakoe-Chiba band constraint.  If None, unconstrained.
                       Value k means |i - j| <= k is allowed.
        normalization: Strategy for normalizing the total cost.

    Returns:
        DTWAlignment with total cost, normalized cost, alignment path, and
        per-step costs.

    Raises:
        ValueError: If inputs are not 2D or have fewer than 2 clips.
    """
    if gold.ndim != 2 or trainee.ndim != 2:
        raise ValueError("gold and trainee must be 2D arrays")
    if len(gold) < _MIN_SEQUENCE_LENGTH or len(trainee) < _MIN_SEQUENCE_LENGTH:
        raise ValueError(
            f"Both sequences must have at least {_MIN_SEQUENCE_LENGTH} clip embeddings, "
            f"got gold={len(gold)}, trainee={len(trainee)}"
        )

    m = len(gold)
    n = len(trainee)

    # Vectorized local cost: cosine distance via dot product
    # Clip to strict interior to avoid degenerate zero-cost ties
    dot_products = gold @ trainee.T  # (m, n) float32
    dot_products = np.clip(dot_products, -1.0 + 1e-7, 1.0 - 1e-7)
    local_cost = np.asarray(1.0 - dot_products, dtype=np.float32)

    # DP accumulation (numba JIT or pure Python fallback)
    bw = -1 if band_width is None else max(0, band_width)
    cost, direction = _dtw_accumulate(local_cost, bw)

    # Validate that alignment reached the endpoint
    total = float(cost[m, n])
    if not np.isfinite(total):
        if band_width is not None:
            raise ValueError(
                f"DTW alignment failed: band_width={band_width} is too narrow "
                f"for sequences of length {m} and {n}. "
                f"Minimum band_width needed: {abs(m - n)}"
            )
        raise ValueError("DTW alignment failed: accumulated cost is infinite")

    # Traceback — follow direction pointers from (m, n) to (0, 0)
    i, j = m, n
    path: list[tuple[int, int]] = []
    path_costs: list[float] = []

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        path_costs.append(float(local_cost[i - 1, j - 1]))
        move = int(direction[i, j])
        if move == 0:   # diagonal
            i -= 1
            j -= 1
        elif move == 1:  # up (gold advances)
            i -= 1
        else:            # left (trainee advances)
            j -= 1

    # Boundary: if we reached j=0 but i>0, walk down the gold axis
    while i > 0:
        path.append((i - 1, 0))
        path_costs.append(float(local_cost[i - 1, 0]))
        i -= 1

    # Boundary: if we reached i=0 but j>0, walk along the trainee axis
    while j > 0:
        path.append((0, j - 1))
        path_costs.append(float(local_cost[0, j - 1]))
        j -= 1

    path.reverse()
    path_costs.reverse()

    # Normalize cost based on selected strategy
    if normalization == NormalizationStrategy.PATH_LENGTH:
        denominator = max(len(path), 1)
    elif normalization == NormalizationStrategy.DIAGONAL:
        denominator = max(m, n)
    elif normalization == NormalizationStrategy.BOTH:
        denominator = m + n
    else:
        denominator = max(len(path), 1)

    normalized = total / denominator

    return DTWAlignment(
        total_cost=total,
        normalized_cost=normalized,
        path=path,
        path_costs=path_costs,
        gold_length=m,
        trainee_length=n,
        band_width=band_width,
    )
