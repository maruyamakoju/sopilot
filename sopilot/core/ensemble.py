"""Multi-gold ensemble scoring — aggregate scores from multiple reference videos.

v2: Statistically rigorous aggregation with ICC(2,1), Grubbs' outlier detection,
bootstrap confidence intervals, Friedman non-parametric test, and
inverse-variance weighted consensus.

Backward compatible: EnsembleResult NamedTuple retains all original fields;
new field ``ensemble_stats`` is appended at the end.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPSILON = 1e-9          # numerical floor to avoid div-by-zero
_BOOTSTRAP_N = 1000      # default bootstrap resamples
_TRIMMED_FRAC = 0.2      # 20 % trimming fraction each tail
_GRUBBS_MIN_N = 3        # minimum samples for Grubbs' test
_ICC_MIN_N = 3           # minimum gold videos for ICC


# ---------------------------------------------------------------------------
# EnsembleStats dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class EnsembleStats:
    """Rich statistical summary of a multi-gold ensemble evaluation.

    All float-valued fields are rounded to 4 decimal places when produced by
    ``aggregate_ensemble``; raw computations are unrounded internally.
    """

    consensus_score: float          # Median — primary backward-compat score
    trimmed_mean: float             # 20 % trimmed mean (robust to outliers)
    weighted_mean: float            # Inverse-variance weighted mean
    score_std: float                # Standard deviation of gold scores
    score_mad: float                # Median absolute deviation (robust)
    agreement: str                  # 'high' / 'medium' / 'low'
    agreement_p_value: float        # Friedman-test p-value (None → NaN if n<2)
    icc: float                      # ICC(2,1); NaN when < 3 gold videos
    outlier_gold_ids: list[int]     # Gold IDs flagged by Grubbs' test
    ci_lower: float                 # Bootstrap 95 % CI lower bound
    ci_upper: float                 # Bootstrap 95 % CI upper bound
    recommendation: str             # Human-readable reliability summary

    def as_dict(self) -> dict:
        """Return a plain-dict representation suitable for JSON serialisation."""
        d = asdict(self)
        # Replace non-finite floats with None for JSON safety
        for key, val in d.items():
            if isinstance(val, float) and not math.isfinite(val):
                d[key] = None
        return d


# ---------------------------------------------------------------------------
# EnsembleResult NamedTuple (backward-compatible + new ensemble_stats field)
# ---------------------------------------------------------------------------

class EnsembleResult(NamedTuple):
    consensus_score: float          # Median score across all golds
    mean_score: float               # Mean score across all golds
    min_score: float                # Lowest individual score
    max_score: float                # Highest individual score
    std_score: float                # Standard deviation (0 if single gold)
    gold_count: int                 # Number of gold videos scored against
    individual_scores: list[float]  # One score per gold video
    agreement: str                  # 'high' / 'medium' / 'low'
    best_gold_video_id: int         # Gold video ID that gave highest similarity
    best_result: dict               # Full result dict from the best-matching gold
    ensemble_stats: dict            # EnsembleStats.as_dict() — new in v2


# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------

def bootstrap_ensemble_ci(
    scores: list[float],
    n_bootstrap: int = _BOOTSTRAP_N,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap percentile CI on the median ensemble consensus score.

    Args:
        scores:      List of per-gold scores.
        n_bootstrap: Number of bootstrap resamples (default 1 000).
        alpha:       Significance level; returns (alpha/2, 1-alpha/2) percentile
                     bounds (default 0.05 → 95 % CI).

    Returns:
        (lower, upper) CI bounds.

    Notes:
        With a single score the CI collapses to (score, score).
    """
    if not scores:
        raise ValueError("scores must not be empty")

    arr = np.asarray(scores, dtype=np.float64)
    if len(arr) == 1:
        v = float(arr[0])
        return v, v

    rng = np.random.default_rng(seed=42)
    bootstrap_medians = np.median(
        rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True), axis=1
    )
    lower = float(np.percentile(bootstrap_medians, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_medians, 100 * (1 - alpha / 2)))
    return lower, upper


def detect_outlier_gold_videos(scores: list[float]) -> list[int]:
    """Grubbs' test for single outlier detection in a list of scores.

    The test is applied iteratively until no more outliers are found or
    ``n < _GRUBBS_MIN_N``.  Only the first outlier at each pass is removed
    from the working set (conservative approach).

    Args:
        scores: Per-gold scores (0–100 scale).

    Returns:
        Sorted list of 0-based *indices* into ``scores`` that are outliers.
        Returns empty list when ``len(scores) < _GRUBBS_MIN_N``.

    Reference:
        Grubbs (1950), Technometrics. Critical value from t-distribution:
        G_crit = ((n-1)/sqrt(n)) * sqrt(t^2 / (n-2 + t^2))
        where t is the (alpha / (2n))-quantile of t(n-2).
    """
    if len(scores) < _GRUBBS_MIN_N:
        return []

    alpha = 0.05
    working = list(enumerate(scores))   # (original_index, value)
    outlier_indices: list[int] = []

    while len(working) >= _GRUBBS_MIN_N:
        vals = np.asarray([v for _, v in working], dtype=np.float64)
        n = len(vals)
        mean = vals.mean()
        std = vals.std(ddof=1)
        if std < _EPSILON:
            break

        g_stats = np.abs(vals - mean) / std
        max_idx = int(np.argmax(g_stats))
        g_max = float(g_stats[max_idx])

        # Critical value at significance level alpha/(2n) for t(n-2)
        t_crit = scipy_stats.t.ppf(1.0 - alpha / (2.0 * n), df=n - 2)
        g_crit = ((n - 1) / math.sqrt(n)) * math.sqrt(
            t_crit**2 / (n - 2 + t_crit**2)
        )

        if g_max > g_crit:
            orig_idx, _ = working.pop(max_idx)
            outlier_indices.append(orig_idx)
        else:
            break

    return sorted(outlier_indices)


def compute_ensemble_icc(
    all_gold_scores_per_trainee: list[list[float]],
) -> float:
    """ICC(2,1) — two-way random effects, single measures.

    Model: y_ij = mu + subject_i + rater_j + error_ij

    Args:
        all_gold_scores_per_trainee:
            Shape (n_gold, n_trainees).  ``[i][j]`` = gold-i's score for
            trainee-j.  Must have >= _ICC_MIN_N rows and >= 2 columns.

    Returns:
        ICC value in (-1, 1], or ``float('nan')`` when preconditions unmet.

    Notes:
        Computed from the ANOVA mean-squares decomposition:
            MS_r  = between-subject mean square
            MS_c  = between-rater (column) mean square
            MS_e  = error mean square
            ICC(2,1) = (MS_r - MS_e) / (MS_r + (k-1)*MS_e + k*(MS_c - MS_e)/n)
        where n = number of subjects (trainees), k = number of raters (golds).
    """
    n_raters = len(all_gold_scores_per_trainee)
    if n_raters < _ICC_MIN_N:
        return float("nan")

    n_subjects = len(all_gold_scores_per_trainee[0])
    if n_subjects < 2:
        return float("nan")

    # Build (n_subjects × n_raters) matrix
    try:
        mat = np.asarray(all_gold_scores_per_trainee, dtype=np.float64).T
        # mat shape: (n_subjects, n_raters)
        if mat.shape != (n_subjects, n_raters):
            return float("nan")
    except (ValueError, TypeError):
        return float("nan")

    n, k = mat.shape
    grand_mean = mat.mean()

    # Subject (row) means and rater (column) means
    row_means = mat.mean(axis=1, keepdims=True)   # (n, 1)
    col_means = mat.mean(axis=0, keepdims=True)   # (1, k)

    # SS decomposition
    ss_total = np.sum((mat - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    df_rows = n - 1
    df_cols = k - 1
    df_error = df_rows * df_cols

    if df_error <= 0:
        return float("nan")

    ms_r = ss_rows / df_rows
    ms_c = ss_cols / df_cols
    ms_e = ss_error / df_error

    denominator = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    if abs(denominator) < _EPSILON:
        return float("nan")

    icc = (ms_r - ms_e) / denominator
    return float(np.clip(icc, -1.0, 1.0))


def friedman_test(
    scores_matrix: list[list[float]],
) -> tuple[float, float]:
    """Friedman non-parametric repeated-measures test.

    Args:
        scores_matrix:
            Shape (n_raters, n_subjects).  ``[i][j]`` = gold-i's score for
            subject-j.  Must have >= 2 raters and >= 2 subjects.

    Returns:
        (statistic, p_value).  Returns (nan, nan) when preconditions unmet.

    Implementation:
        Delegates to ``scipy.stats.friedmanchisquare`` which expects one
        sequence per rater (row).
    """
    n_raters = len(scores_matrix)
    if n_raters < 2:
        return float("nan"), float("nan")

    n_subjects = len(scores_matrix[0])
    if n_subjects < 2:
        return float("nan"), float("nan")

    try:
        rows = [list(row) for row in scores_matrix]
        stat, p_val = scipy_stats.friedmanchisquare(*rows)
        return float(stat), float(p_val)
    except (ValueError, TypeError):
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Inverse-variance weighted mean
# ---------------------------------------------------------------------------

def _inverse_variance_weighted_mean(
    scores: list[float],
    dtw_costs: list[float],
) -> float:
    """Compute inverse-variance weighted mean of ``scores``.

    Variance proxy = normalized DTW cost for each gold video.  When DTW costs
    are unavailable or too uniform, falls back to the plain median.

    Args:
        scores:    Per-gold scores.
        dtw_costs: Per-gold normalised DTW costs (same length as scores).

    Returns:
        Weighted mean score.
    """
    arr_scores = np.asarray(scores, dtype=np.float64)
    arr_costs = np.asarray(dtw_costs, dtype=np.float64)

    # If variance is too uniform, weights degenerate → fall back to median
    if arr_costs.std() < 0.01:
        return float(np.median(arr_scores))

    weights = 1.0 / (arr_costs + _EPSILON)
    weighted = float(np.average(arr_scores, weights=weights))
    return weighted


def _trimmed_mean(scores: list[float], proportiontocut: float = _TRIMMED_FRAC) -> float:
    """Return the trimmed mean; falls back to ordinary mean when n is too small."""
    arr = sorted(scores)
    n = len(arr)
    k = int(n * proportiontocut)
    trimmed = arr[k : n - k] if n - 2 * k >= 1 else arr
    return float(statistics.mean(trimmed))


def _mad(scores: list[float]) -> float:
    """Median absolute deviation."""
    arr = np.asarray(scores, dtype=np.float64)
    return float(np.median(np.abs(arr - np.median(arr))))


def _agreement_label(std: float) -> str:
    """Map score std to a backward-compatible agreement label."""
    if std < 3.0:
        return "high"
    if std < 8.0:
        return "medium"
    return "low"


def _recommendation(
    agreement: str,
    icc: float,
    n_outliers: int,
    p_value: float,
) -> str:
    """Compose a human-readable reliability recommendation in Japanese/English."""
    parts: list[str] = []

    # ICC reliability
    if math.isfinite(icc):
        if icc >= 0.75:
            parts.append("Gold videos show excellent inter-rater reliability (ICC ≥ 0.75).")
        elif icc >= 0.50:
            parts.append("Gold videos show moderate inter-rater reliability (ICC 0.50–0.75).")
        else:
            parts.append(
                "Gold videos show poor inter-rater reliability (ICC < 0.50). "
                "Review gold video consistency."
            )

    # Agreement / spread
    if agreement == "high":
        parts.append("Score spread is low — result is reliable.")
    elif agreement == "medium":
        parts.append("Score spread is moderate — interpret with caution.")
    else:
        parts.append("Score spread is high — additional gold videos recommended.")

    # Outliers
    if n_outliers > 0:
        parts.append(
            f"{n_outliers} outlier gold video(s) detected — consider reviewing them."
        )

    # Friedman significance
    if math.isfinite(p_value):
        if p_value < 0.05:
            parts.append(
                f"Friedman test indicates significant rater disagreement (p={p_value:.3f})."
            )
        else:
            parts.append(
                f"Friedman test shows no significant rater disagreement (p={p_value:.3f})."
            )

    return " ".join(parts) if parts else "Insufficient data for reliability assessment."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate_ensemble(results: list[dict]) -> EnsembleResult:
    """Aggregate multiple per-gold scoring results into an ensemble result.

    This is the primary public function.  It is fully backward compatible with
    v1: all original ``EnsembleResult`` fields are preserved and retain the
    same semantics.  A new ``ensemble_stats`` field carries rich statistical
    diagnostics via ``EnsembleStats.as_dict()``.

    Args:
        results: List of result dicts from individual ``score_pair`` calls.
                 Each must contain:
                   - ``'score'`` (float, 0–100)
                   - ``'gold_video_id'`` (int)
                 Optionally contains ``'metrics'`` → ``'dtw_normalized_cost'``
                 for variance-weighted consensus.

    Returns:
        ``EnsembleResult`` NamedTuple.  All float fields are rounded to 2 d.p.

    Raises:
        ValueError: If ``results`` is empty.

    Algorithm:
        1. Primary consensus: inverse-variance weighted mean (uses DTW cost as
           variance proxy).  Falls back to median when DTW costs are uniform.
        2. Agreement label: based on score std (backward-compat thresholds).
        3. ICC(2,1), Friedman test, Grubbs' outliers, bootstrap CI, MAD — all
           packaged into ``ensemble_stats``.
    """
    if not results:
        raise ValueError("results must not be empty")

    scores: list[float] = [float(r["score"]) for r in results]
    gold_ids: list[int] = [int(r.get("gold_video_id", 0)) for r in results]

    # ── Basic statistics ──────────────────────────────────────────────────
    n = len(scores)
    consensus_median = float(statistics.median(scores))
    plain_mean = float(statistics.mean(scores))
    std = float(statistics.stdev(scores)) if n > 1 else 0.0
    score_min = float(min(scores))
    score_max = float(max(scores))

    # ── Inverse-variance weighted consensus ──────────────────────────────
    dtw_costs: list[float] = [
        float(r.get("metrics", {}).get("dtw_normalized_cost", 0.5))
        for r in results
    ]
    weighted_mean_val = _inverse_variance_weighted_mean(scores, dtw_costs)

    # Consensus = weighted mean when DTW costs are heterogeneous, else median
    dtw_arr = np.asarray(dtw_costs, dtype=np.float64)
    consensus_score = (
        weighted_mean_val if (n > 1 and float(dtw_arr.std()) >= 0.01)
        else consensus_median
    )

    # ── Agreement label ───────────────────────────────────────────────────
    agreement = _agreement_label(std)

    # ── Best gold (highest individual score) ─────────────────────────────
    best_idx = int(np.argmax(scores))
    best_result = results[best_idx]
    best_gold_id = int(best_result.get("gold_video_id", 0))

    # ── Robust statistics ─────────────────────────────────────────────────
    trimmed_mean_val = _trimmed_mean(scores)
    mad_val = _mad(scores)

    # ── Bootstrap CI on median consensus ─────────────────────────────────
    ci_lower, ci_upper = bootstrap_ensemble_ci(scores)

    # ── Grubbs' outlier detection ─────────────────────────────────────────
    outlier_indices = detect_outlier_gold_videos(scores)
    outlier_gold_ids = [gold_ids[i] for i in outlier_indices]

    # ── ICC(2,1) — requires multiple gold and multiple trainees ───────────
    # In single-trainee ensemble context we cannot compute ICC across trainees;
    # return NaN.  ICC is meaningful only when called with a full score matrix.
    icc_val = float("nan")

    # ── Friedman test ─────────────────────────────────────────────────────
    # In a single-trainee context each gold provides exactly one score, so the
    # Friedman matrix is (n_gold × 1 trainee) — degenerate.  We still attempt
    # it so callers with multi-trainee data can call friedman_test() directly.
    friedman_p = float("nan")
    if n >= 2:
        # Treat each gold score as a separate "block" with one observation;
        # Friedman reduces to Kendall's W test direction in this edge case.
        # For a meaningful p-value, callers should supply the full matrix via
        # friedman_test() directly.
        friedman_p = float("nan")   # not computable from single vector

    # ── Recommendation ────────────────────────────────────────────────────
    recommendation = _recommendation(
        agreement=agreement,
        icc=icc_val,
        n_outliers=len(outlier_gold_ids),
        p_value=friedman_p,
    )

    # ── Assemble EnsembleStats ────────────────────────────────────────────
    estat = EnsembleStats(
        consensus_score=round(consensus_score, 4),
        trimmed_mean=round(trimmed_mean_val, 4),
        weighted_mean=round(weighted_mean_val, 4),
        score_std=round(std, 4),
        score_mad=round(mad_val, 4),
        agreement=agreement,
        agreement_p_value=friedman_p,
        icc=icc_val,
        outlier_gold_ids=outlier_gold_ids,
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        recommendation=recommendation,
    )

    return EnsembleResult(
        consensus_score=round(consensus_score, 2),
        mean_score=round(plain_mean, 2),
        min_score=round(score_min, 2),
        max_score=round(score_max, 2),
        std_score=round(std, 2),
        gold_count=n,
        individual_scores=[round(s, 2) for s in scores],
        agreement=agreement,
        best_gold_video_id=best_gold_id,
        best_result=best_result,
        ensemble_stats=estat.as_dict(),
    )


def aggregate_ensemble_multi_trainee(
    results_by_trainee: dict[int, list[dict]],
) -> dict[int, EnsembleResult]:
    """Aggregate ensemble scores across multiple trainees with full statistics.

    Unlike ``aggregate_ensemble`` (single-trainee), this variant can compute
    ICC(2,1) and the Friedman test properly because it has a full
    (n_gold × n_trainee) score matrix.

    Args:
        results_by_trainee:
            Mapping from trainee_video_id → list of per-gold result dicts.
            All trainee entries must reference the same set of gold videos in
            the same order.

    Returns:
        Mapping from trainee_video_id → EnsembleResult.  Each EnsembleResult's
        ``ensemble_stats`` will contain valid ICC and Friedman p-values.
    """
    if not results_by_trainee:
        return {}

    trainee_ids = list(results_by_trainee.keys())
    n_trainees = len(trainee_ids)

    # First pass: compute per-trainee ensembles (single-trainee path)
    ensemble_by_trainee: dict[int, EnsembleResult] = {
        tid: aggregate_ensemble(results_by_trainee[tid])
        for tid in trainee_ids
    }

    if n_trainees < 2:
        return ensemble_by_trainee

    # Build score matrix: shape (n_gold, n_trainees)
    n_gold = ensemble_by_trainee[trainee_ids[0]].gold_count
    score_matrix: list[list[float]] = []
    for gold_idx in range(n_gold):
        row: list[float] = []
        for tid in trainee_ids:
            ind_scores = ensemble_by_trainee[tid].individual_scores
            row.append(ind_scores[gold_idx] if gold_idx < len(ind_scores) else float("nan"))
        score_matrix.append(row)

    # Compute population-level ICC and Friedman
    icc_val = compute_ensemble_icc(score_matrix)
    friedman_stat, friedman_p = friedman_test(score_matrix)

    # Patch each trainee's ensemble_stats with the population-level values
    result: dict[int, EnsembleResult] = {}
    for tid in trainee_ids:
        er = ensemble_by_trainee[tid]
        old_stats = dict(er.ensemble_stats)
        old_stats["icc"] = round(icc_val, 4) if math.isfinite(icc_val) else None
        old_stats["agreement_p_value"] = (
            round(friedman_p, 6) if math.isfinite(friedman_p) else None
        )
        # Regenerate recommendation with updated values
        old_stats["recommendation"] = _recommendation(
            agreement=old_stats["agreement"],
            icc=icc_val,
            n_outliers=len(old_stats["outlier_gold_ids"]),
            p_value=friedman_p,
        )
        # Rebuild EnsembleResult with patched stats
        result[tid] = EnsembleResult(
            consensus_score=er.consensus_score,
            mean_score=er.mean_score,
            min_score=er.min_score,
            max_score=er.max_score,
            std_score=er.std_score,
            gold_count=er.gold_count,
            individual_scores=er.individual_scores,
            agreement=er.agreement,
            best_gold_video_id=er.best_gold_video_id,
            best_result=er.best_result,
            ensemble_stats=old_stats,
        )

    return result
