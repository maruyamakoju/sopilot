"""Sensitivity analysis for fusion weight hyperparameters.

Provides grid search over (audio_weight, motion_weight, proximity_weight) with
BCa bootstrap confidence intervals on accuracy. Results feed directly into the
ablation table in the research paper.

Design notes:
- Weights are constrained to sum to 1.0 and sampled on a simplex grid.
- BCa bootstrap CIs reuse the implementation from evaluation.statistical to
  avoid duplicating statistical machinery.
- The eval_fn interface is intentionally thin so callers can wrap any backend
  (mock, real VLM, batch replay) without modifying this module.

Typical usage::

    from insurance_mvp.evaluation.sensitivity import grid_search_fusion_weights

    def my_eval(aw, mw, pw):
        # configure pipeline with these weights, run on val set
        # return (y_true_list, y_pred_list) as severity strings
        ...

    results = grid_search_fusion_weights(my_eval, n_bootstrap=500)
    for r in results[:5]:
        print(r)
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from insurance_mvp.evaluation.statistical import (
    ConfidenceInterval,
    accuracy_score,
    bootstrap_ci,
)

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH"]
SEVERITY_TO_IDX = {s: i for i, s in enumerate(SEVERITY_LEVELS)}

# Default grid resolution: step of 0.1 on the simplex (→ 66 unique triples)
_DEFAULT_STEP = 0.1


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class WeightResult:
    """Result for a single (audio_w, motion_w, proximity_w) configuration.

    Attributes:
        audio_w: Weight assigned to the audio signal (0 ≤ w ≤ 1).
        motion_w: Weight assigned to the optical-flow motion signal.
        proximity_w: Weight assigned to the YOLO proximity signal.
        accuracy_ci: BCa bootstrap confidence interval on accuracy.
    """

    audio_w: float
    motion_w: float
    proximity_w: float
    accuracy_ci: ConfidenceInterval

    def __repr__(self) -> str:
        return (
            f"WeightResult(audio={self.audio_w:.2f}, motion={self.motion_w:.2f}, "
            f"proximity={self.proximity_w:.2f}, acc={self.accuracy_ci})"
        )


# ---------------------------------------------------------------------------
# Simplex grid generation
# ---------------------------------------------------------------------------


def _simplex_grid(step: float = _DEFAULT_STEP) -> list[tuple[float, float, float]]:
    """Enumerate weight triples (a, m, p) on the 3-simplex with given step size.

    All triples satisfy a + m + p = 1.0 (up to floating-point rounding) and
    a, m, p >= 0.

    Args:
        step: Grid resolution. Smaller values produce finer grids; step=0.1
              yields 66 unique triples (C(12,2) by stars-and-bars).

    Returns:
        List of (audio_w, motion_w, proximity_w) tuples.
    """
    n_steps = round(1.0 / step)
    grid: list[tuple[float, float, float]] = []
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            k = n_steps - i - j
            grid.append((round(i * step, 8), round(j * step, 8), round(k * step, 8)))
    return grid


# ---------------------------------------------------------------------------
# Main grid search
# ---------------------------------------------------------------------------


def grid_search_fusion_weights(
    eval_fn: Callable[[float, float, float], tuple[list[str], list[str]]],
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int = 42,
    step: float = _DEFAULT_STEP,
) -> list[WeightResult]:
    """Grid search over fusion weight space with BCa CIs on accuracy.

    Weights are constrained to sum to 1.0 and sampled from a simplex grid.
    Returns results sorted by mean accuracy descending.

    The function calls *eval_fn(audio_w, motion_w, proximity_w)* once per grid
    point. Each call must return ``(y_true, y_pred)`` as parallel lists of
    severity strings from ``{"NONE", "LOW", "MEDIUM", "HIGH"}``.

    BCa bootstrap confidence intervals are computed on accuracy using the
    implementation in ``insurance_mvp.evaluation.statistical.bootstrap_ci``.
    A single random seed is used; per-grid-point seeds are derived as
    ``seed + grid_index`` to ensure reproducibility without correlation.

    Args:
        eval_fn: Callable ``(audio_w, motion_w, proximity_w) -> (y_true, y_pred)``.
            Must be deterministic for a fixed random seed.
        alpha: Significance level for confidence intervals (default 0.05 → 95% CI).
        n_bootstrap: Number of bootstrap resamples per grid point (default 1000).
            Reduce to 200–500 for rapid iteration; use 5000+ for publication.
        seed: Base random seed for reproducibility.
        step: Grid resolution on the simplex (default 0.1). Finer grids
            (e.g. 0.05) give 231 points but proportionally more eval_fn calls.

    Returns:
        List of WeightResult sorted by ``accuracy_ci.point`` descending.
        The list contains one entry per evaluated grid point.

    Raises:
        ValueError: If eval_fn returns empty predictions or mismatched lengths.

    Example::

        def mock_eval(aw, mw, pw):
            y_true = ["HIGH", "MEDIUM", "LOW", "NONE"]
            y_pred = ["HIGH", "MEDIUM", "LOW", "LOW"]
            return y_true, y_pred

        results = grid_search_fusion_weights(mock_eval, n_bootstrap=200)
        best = results[0]
        print(f"Best weights: audio={best.audio_w}, motion={best.motion_w}, "
              f"proximity={best.proximity_w}")
        print(f"Accuracy: {best.accuracy_ci}")
    """
    grid_points = _simplex_grid(step)
    n_points = len(grid_points)
    logger.info(
        "Starting fusion-weight grid search: %d grid points, "
        "%d bootstrap resamples, alpha=%.3f",
        n_points,
        n_bootstrap,
        alpha,
    )

    results: list[WeightResult] = []

    for idx, (audio_w, motion_w, proximity_w) in enumerate(grid_points):
        logger.debug(
            "Grid point %d/%d: audio=%.2f motion=%.2f proximity=%.2f",
            idx + 1,
            n_points,
            audio_w,
            motion_w,
            proximity_w,
        )

        y_true_str, y_pred_str = eval_fn(audio_w, motion_w, proximity_w)

        if len(y_true_str) == 0:
            raise ValueError(
                f"eval_fn returned empty y_true for weights "
                f"(audio={audio_w}, motion={motion_w}, proximity={proximity_w})."
            )
        if len(y_true_str) != len(y_pred_str):
            raise ValueError(
                f"eval_fn returned mismatched lengths: "
                f"y_true={len(y_true_str)}, y_pred={len(y_pred_str)}."
            )

        # Convert to integer indices for statistical functions
        yt = np.array([SEVERITY_TO_IDX.get(s.upper(), 1) for s in y_true_str])
        yp = np.array([SEVERITY_TO_IDX.get(s.upper(), 1) for s in y_pred_str])

        # BCa bootstrap CI on accuracy; use per-point seed to decorrelate
        point_seed = seed + idx
        ci = bootstrap_ci(yt, yp, accuracy_score, n_bootstrap=n_bootstrap, alpha=alpha, seed=point_seed)

        results.append(
            WeightResult(
                audio_w=audio_w,
                motion_w=motion_w,
                proximity_w=proximity_w,
                accuracy_ci=ci,
            )
        )

    # Sort by mean accuracy descending
    results.sort(key=lambda r: r.accuracy_ci.point, reverse=True)

    logger.info(
        "Grid search complete. Best: audio=%.2f motion=%.2f proximity=%.2f acc=%.4f",
        results[0].audio_w,
        results[0].motion_w,
        results[0].proximity_w,
        results[0].accuracy_ci.point,
    )

    return results


# ---------------------------------------------------------------------------
# Convenience formatting
# ---------------------------------------------------------------------------


def format_sensitivity_table(results: list[WeightResult], top_k: int = 10) -> str:
    """Format top-K weight results as a human-readable table.

    Suitable for inclusion in ablation tables or log output.

    Args:
        results: Sorted list from ``grid_search_fusion_weights``.
        top_k: Number of rows to include (default 10).

    Returns:
        Formatted string table.
    """
    header = (
        f"{'Rank':>4}  {'Audio':>6}  {'Motion':>7}  {'Prox':>6}  "
        f"{'Acc':>6}  {'CI Lower':>9}  {'CI Upper':>9}"
    )
    separator = "-" * len(header)
    lines = [separator, header, separator]
    for rank, r in enumerate(results[:top_k], start=1):
        ci = r.accuracy_ci
        lines.append(
            f"{rank:>4}  {r.audio_w:>6.2f}  {r.motion_w:>7.2f}  {r.proximity_w:>6.2f}  "
            f"{ci.point:>6.4f}  {ci.lower:>9.4f}  {ci.upper:>9.4f}"
        )
    lines.append(separator)
    return "\n".join(lines)
