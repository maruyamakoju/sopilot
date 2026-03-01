"""sopilot.core.uncertainty
=========================
Bootstrap and Bayesian-inspired uncertainty quantification for SOP video scores.

This module replaces the heuristic confidence intervals produced by
:func:`sopilot.core.scoring.compute_score_confidence` with statistically
principled estimates that account for both the quantity and geometry of
clip embeddings.

Statistical background
----------------------
Two orthogonal sources of uncertainty are decomposed:

**Epistemic (model) uncertainty** — arises from limited expressiveness of the
DTW alignment model.  When the normalized DTW cost is high the alignment is
poor and the resulting score is less reliable.  We model this as a scaled
sigmoid of the DTW cost so that it is bounded and monotone.

**Aleatoric (data) uncertainty** — arises from the inherent stochasticity of
the measurement: few clips, highly spread embeddings, or a trainee sequence
that is very different in length from the gold.  This quantity cannot be
reduced by running the same model more carefully; it reflects genuine
variability in the observed data.

**Bootstrap confidence interval** — when raw embedding arrays are available a
non-parametric bootstrap (Efron 1979) is performed.  Both gold and trainee
clip sequences are resampled *with replacement* and a lightweight
cosine-distance DTW is rerun for each replicate.  The 2.5th and 97.5th
percentiles of the resulting score distribution form the 95 % CI.

References
----------
- Efron, B. (1979). Bootstrap methods: Another look at the jackknife.
  The Annals of Statistics, 7(1), 1-26.
- Berndt & Clifford (1994). Using Dynamic Time Warping to Find Patterns in
  Time Series. KDD Workshop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BootstrapCI:
    """A percentile-based bootstrap confidence interval for a score.

    Attributes:
        lower: Lower bound of the confidence interval (clamped to [0, 100]).
        upper: Upper bound of the confidence interval (clamped to [0, 100]).
        width: Total width of the interval (upper - lower).
        stability: Qualitative label derived from width.
            - ``"high"``   — width <= 8
            - ``"medium"`` — width <= 16
            - ``"low"``    — width > 16
        n_bootstrap: Number of bootstrap replicates actually used.
            Zero when the heuristic fallback was invoked.
        alpha: Significance level (default 0.05 -> 95 % CI).
    """

    lower: float
    upper: float
    width: float
    stability: str          # "high" | "medium" | "low"
    n_bootstrap: int        # 0 = heuristic fallback
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.stability not in {"high", "medium", "low"}:
            raise ValueError(f"stability must be 'high', 'medium', or 'low'; got {self.stability!r}")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha}")


@dataclass
class ScoreUncertainty:
    """Full uncertainty decomposition for a single score result.

    Attributes:
        bootstrap_ci: Non-parametric confidence interval (or heuristic
            fallback when embeddings are unavailable).
        epistemic: Model uncertainty component in score-point units.
            High when DTW cost is large (alignment unreliable).
        aleatoric: Data uncertainty component in score-point units.
            High when clip count is small or embedding spread is large.
        total: Combined uncertainty via quadrature (sqrt of sum of squares).
        note: Human-readable explanation of the dominant uncertainty source.
    """

    bootstrap_ci: BootstrapCI
    epistemic: float
    aleatoric: float
    total: float
    note: str = field(default="")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stability_label(width: float) -> str:
    """Map CI width to a qualitative stability label."""
    if width <= 8.0:
        return "high"
    if width <= 16.0:
        return "medium"
    return "low"


def _base_width(n_clips: int) -> float:
    """Data-volume-dependent base CI half-width (score points)."""
    if n_clips < 5:
        return 12.0
    if n_clips < 10:
        return 8.0
    if n_clips < 20:
        return 5.0
    return 3.0


def _dtw_cost_fast(gold: np.ndarray, trainee: np.ndarray) -> float:
    """Compute a normalized DTW cost using pure-numpy vectorized DP.

    Stripped-down version of sopilot.core.dtw.dtw_align with no traceback,
    designed for throughput in bootstrap loops.
    """
    m, n = len(gold), len(trainee)
    if m == 0 or n == 0:
        return 1.0

    dot = np.clip(gold @ trainee.T, -1.0, 1.0).astype(np.float32)
    local = (1.0 - dot).astype(np.float32)

    INF = np.float32(np.inf)
    prev = np.full(n + 1, INF, dtype=np.float32)
    prev[0] = np.float32(0.0)

    for i in range(1, m + 1):
        curr = np.full(n + 1, INF, dtype=np.float32)
        diag = prev[:-1]
        up = prev[1:]
        row_local = local[i - 1]
        min_diag_up = np.minimum(diag, up)

        for j in range(1, n + 1):
            left = curr[j - 1]
            best = min(min_diag_up[j - 1], left)
            if best == INF:
                curr[j] = INF
            else:
                curr[j] = row_local[j - 1] + best

        prev = curr

    total_cost = float(prev[n])
    if not math.isfinite(total_cost):
        return 1.0
    return float(np.clip(total_cost / max(m, n), 0.0, None))


def _score_from_cost(dtw_cost: float, dtw_cost_max: float) -> float:
    """Convert a DTW cost to a 0-100 score."""
    if dtw_cost_max <= 1e-9:
        return 100.0 if dtw_cost < 1e-9 else 0.0
    return float(np.clip(100.0 * (1.0 - dtw_cost / dtw_cost_max), 0.0, 100.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def heuristic_ci(
    base_score: float,
    dtw_cost: float,
    n_clips: int,
) -> BootstrapCI:
    """Compute an improved heuristic confidence interval without re-running DTW.

    Enhanced version of compute_score_confidence that incorporates the DTW
    cost into the CI width::

        half_width = base_width(n_clips) * (1 + 3 * dtw_cost)

    Args:
        base_score: The score value in [0, 100].
        dtw_cost:   Normalized DTW cost from the alignment.
        n_clips:    Total clip count (gold + trainee).

    Returns:
        A BootstrapCI with n_bootstrap=0 (no resampling performed).
    """
    bw = _base_width(n_clips)
    half_width = bw * (1.0 + 3.0 * float(dtw_cost))
    lower = float(np.clip(base_score - half_width, 0.0, 100.0))
    upper = float(np.clip(base_score + half_width, 0.0, 100.0))
    width = upper - lower
    return BootstrapCI(
        lower=round(lower, 2),
        upper=round(upper, 2),
        width=round(width, 2),
        stability=_stability_label(width),
        n_bootstrap=0,
        alpha=0.05,
    )


def bootstrap_score_ci(
    base_score: float,
    clip_embeddings_gold: np.ndarray | None,
    clip_embeddings_trainee: np.ndarray | None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """Compute a non-parametric bootstrap confidence interval for a score.

    Resamples gold and trainee clip sequences with replacement and reruns
    a fast DTW for each replicate. Falls back to heuristic_ci when embeddings
    are unavailable or contain fewer than 3 clips.

    Args:
        base_score:              Observed score (0-100) from the main pipeline.
        clip_embeddings_gold:    Gold embedding matrix of shape (m, D).
        clip_embeddings_trainee: Trainee embedding matrix of shape (n, D).
        n_bootstrap:             Number of bootstrap replicates (default 1000).
        alpha:                   Significance level (default 0.05 -> 95% CI).
        rng:                     Optional RNG for reproducibility.

    Returns:
        BootstrapCI with bootstrap results, or heuristic fallback.

    References:
        Efron, B. (1979). Bootstrap methods: Another look at the jackknife.
    """
    _fallback_n_clips = 0
    _fallback_dtw = 0.0

    if clip_embeddings_gold is None or clip_embeddings_trainee is None:
        n_g, n_t = 0, 0
    else:
        gold = np.asarray(clip_embeddings_gold, dtype=np.float32)
        trainee = np.asarray(clip_embeddings_trainee, dtype=np.float32)
        n_g, n_t = len(gold), len(trainee)
        _fallback_n_clips = n_g + n_t

    if n_g < 3 or n_t < 3:
        return heuristic_ci(base_score=base_score, dtw_cost=_fallback_dtw, n_clips=_fallback_n_clips)

    def _l2_norm(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        return (x / norms).astype(np.float32)

    gold_norm = _l2_norm(gold)
    trainee_norm = _l2_norm(trainee)

    observed_cost = _dtw_cost_fast(gold_norm, trainee_norm)

    if base_score >= 99.9 or observed_cost < 1e-6:
        dtw_cost_max = max(observed_cost * 4.0, 0.1)
    else:
        implied_max = 100.0 * observed_cost / (100.0 - base_score)
        dtw_cost_max = float(np.clip(implied_max, observed_cost * 2.0, observed_cost * 8.0))
        dtw_cost_max = max(dtw_cost_max, 0.05)

    if rng is None:
        rng = np.random.default_rng()

    boot_scores = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx_g = rng.integers(0, n_g, size=n_g)
        idx_t = rng.integers(0, n_t, size=n_t)
        g_sample = _l2_norm(gold_norm[idx_g])
        t_sample = _l2_norm(trainee_norm[idx_t])
        cost_b = _dtw_cost_fast(g_sample, t_sample)
        boot_scores[b] = _score_from_cost(cost_b, dtw_cost_max)

    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    lower = float(np.clip(np.percentile(boot_scores, lo_pct), 0.0, 100.0))
    upper = float(np.clip(np.percentile(boot_scores, hi_pct), 0.0, 100.0))
    width = upper - lower

    return BootstrapCI(
        lower=round(lower, 2),
        upper=round(upper, 2),
        width=round(width, 2),
        stability=_stability_label(width),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )


def compute_score_uncertainty(
    base_score: float,
    dtw_cost: float,
    n_clips_gold: int,
    n_clips_trainee: int,
    clip_embeddings_gold: np.ndarray | None = None,
    clip_embeddings_trainee: np.ndarray | None = None,
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> ScoreUncertainty:
    """Compute a full uncertainty decomposition for a score result.

    Main entry point of the module. Returns a ScoreUncertainty separating
    total uncertainty into epistemic and aleatoric components with a
    bootstrap CI (or heuristic fallback).

    Epistemic uncertainty (model fit quality)::

        epistemic = 15 * sigmoid(10 * (dtw_cost - 0.3))

    Gives ~0.2 pts at dtw_cost=0.1, ~7.5 pts at 0.3, ~14.8 pts at 0.5.

    Aleatoric uncertainty (data volume)::

        n_eff = harmonic_mean(n_clips_gold, n_clips_trainee)
        aleatoric = 10 / sqrt(n_eff) + embedding_spread_term

    Total uncertainty (quadrature)::

        total = sqrt(epistemic^2 + aleatoric^2)

    Args:
        base_score:              Score in [0, 100] from the main pipeline.
        dtw_cost:                Normalized DTW cost from alignment.
        n_clips_gold:            Number of gold clips.
        n_clips_trainee:         Number of trainee clips.
        clip_embeddings_gold:    Optional gold embedding matrix (m, D).
        clip_embeddings_trainee: Optional trainee embedding matrix (n, D).
        n_bootstrap:             Bootstrap replicates (default 1000).
        alpha:                   CI significance level (default 0.05).
        rng:                     Optional RNG for reproducibility.

    Returns:
        ScoreUncertainty with all fields populated.
    """
    dtw_cost = float(dtw_cost)
    base_score = float(np.clip(base_score, 0.0, 100.0))
    n_g = max(int(n_clips_gold), 0)
    n_t = max(int(n_clips_trainee), 0)

    # Epistemic: scaled sigmoid centred at dtw_cost=0.3
    sigmoid_input = float(np.clip(10.0 * (dtw_cost - 0.3), -20.0, 20.0))
    epistemic = 15.0 / (1.0 + math.exp(-sigmoid_input))

    # Aleatoric: harmonic mean of clip counts
    n_total = n_g + n_t
    if n_total > 0 and n_g > 0 and n_t > 0:
        n_eff = 2.0 * n_g * n_t / n_total
    else:
        n_eff = 0.0

    base_aleatoric = 10.0 / math.sqrt(max(n_eff, 1.0))
    spread_term = 0.0
    notes_parts: list[str] = []

    embeddings_available = (
        clip_embeddings_gold is not None
        and clip_embeddings_trainee is not None
        and n_g >= 3
        and n_t >= 3
    )

    if embeddings_available:
        assert clip_embeddings_gold is not None
        assert clip_embeddings_trainee is not None
        g_arr = np.asarray(clip_embeddings_gold, dtype=np.float32)
        t_arr = np.asarray(clip_embeddings_trainee, dtype=np.float32)

        def _safe_norm(x: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            norms = np.where(norms < 1e-9, 1.0, norms)
            return x / norms

        g_norm = _safe_norm(g_arr)
        t_norm = _safe_norm(t_arr)
        gold_centroid = g_norm.mean(axis=0)
        centroid_norm = np.linalg.norm(gold_centroid)
        if centroid_norm > 1e-9:
            gold_centroid = gold_centroid / centroid_norm
        cosine_sims = np.clip(t_norm @ gold_centroid, -1.0, 1.0)
        mean_cos_dist = float(1.0 - np.mean(cosine_sims))
        spread_term = float(np.clip(5.0 * mean_cos_dist, 0.0, 5.0))

    aleatoric = base_aleatoric + spread_term
    total = math.sqrt(epistemic ** 2 + aleatoric ** 2)

    bci = bootstrap_score_ci(
        base_score=base_score,
        clip_embeddings_gold=clip_embeddings_gold,
        clip_embeddings_trainee=clip_embeddings_trainee,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        rng=rng,
    )

    if bci.n_bootstrap == 0:
        notes_parts.append(
            "Bootstrap resampling skipped (fewer than 3 clips); heuristic CI used."
        )

    if epistemic >= aleatoric:
        notes_parts.append(
            f"Epistemic (model) uncertainty dominates ({epistemic:.1f} pts); "
            f"DTW cost {dtw_cost:.3f} indicates {'poor' if dtw_cost > 0.3 else 'moderate'} alignment."
        )
    else:
        notes_parts.append(
            f"Aleatoric (data) uncertainty dominates ({aleatoric:.1f} pts); "
            f"effective clip count {n_eff:.0f} is {'very low' if n_eff < 10 else 'moderate'}."
        )

    if embeddings_available and spread_term > 1.0:
        notes_parts.append(f"Embedding spread adds {spread_term:.1f} pts to aleatoric uncertainty.")

    return ScoreUncertainty(
        bootstrap_ci=bci,
        epistemic=round(epistemic, 3),
        aleatoric=round(aleatoric, 3),
        total=round(total, 3),
        note="  ".join(notes_parts),
    )
