"""Statistical evaluation framework.

Provides rigorous statistical tools for validating neural component contributions:
- Bootstrap confidence intervals (Efron 1979)
- Permutation tests for significance
- Intraclass Correlation Coefficient (Shrout & Fleiss 1979)
- Ablation study runner with per-component significance
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


def bootstrap_confidence_interval(
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    statistic: Callable[[np.ndarray], float] = np.mean,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a statistic.

    Args:
        scores: (N,) array of observations.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (0.05 = 95% CI).
        statistic: Function to compute on each resample.
        rng: Optional random generator for reproducibility.

    Returns:
        (estimate, ci_lower, ci_upper).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0

    estimate = float(statistic(scores))
    boot_stats = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        boot_stats[i] = statistic(sample)

    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return estimate, ci_lower, ci_upper


def permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_permutations: int = 10000,
    statistic: Callable[[np.ndarray], float] = np.mean,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Two-sample permutation test for difference in statistic.

    Tests H0: the distributions of group_a and group_b are identical.

    Args:
        group_a: (N_a,) observations from condition A.
        group_b: (N_b,) observations from condition B.
        n_permutations: Number of permutation resamples.
        statistic: Function to compute on each group.
        rng: Optional random generator.

    Returns:
        (observed_diff, p_value) where observed_diff = stat(A) - stat(B).
    """
    if rng is None:
        rng = np.random.default_rng()

    observed_diff = float(statistic(group_a) - statistic(group_b))
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = float(statistic(combined[:n_a]) - statistic(combined[n_a:]))
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = float((count_extreme + 1) / (n_permutations + 1))
    return observed_diff, p_value


def _wilson_hilferty_f_ppf(p: float, df1: int, df2: int) -> float:
    """Wilson-Hilferty approximation for F-distribution quantiles.

    Used as fallback when scipy is not available.

    Args:
        p: Probability (quantile) in (0, 1).
        df1: Numerator degrees of freedom.
        df2: Denominator degrees of freedom.

    Returns:
        Approximate F quantile.
    """
    import math

    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float("inf")
    if p < 0.5:
        t = p
        sign = -1.0
    else:
        t = 1.0 - p
        sign = 1.0

    # Rational approximation for normal quantile (Abramowitz & Stegun 26.2.23)
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    s = math.sqrt(-2.0 * math.log(max(t, 1e-300)))
    z_abs = s - (c0 + c1 * s + c2 * s * s) / (1.0 + d1 * s + d2 * s * s + d3 * s * s * s)
    z = sign * z_abs

    # Wilson-Hilferty transformation
    a1 = 2.0 / (9.0 * df1)
    a2 = 2.0 / (9.0 * df2)
    num = (1.0 - a2 + z * math.sqrt(a2)) ** 3
    den = (1.0 - a1 - z * math.sqrt(a1)) ** 3
    if den <= 0:
        return float("inf")
    return max(0.0, num / den)


def intraclass_correlation(
    ratings: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Intraclass Correlation Coefficient ICC(2,1) -- two-way random, single measures.

    Based on Shrout & Fleiss (1979) for inter-rater reliability.
    Confidence intervals from McGraw & Wong (1996) using F-distribution
    bounds on the observed F statistic.

    Args:
        ratings: (N, K) matrix where N=subjects, K=raters.
            Each row is one subject rated by K raters.
        alpha: Significance level for CI (default 0.05 for 95% CI).

    Returns:
        (icc, ci_lower, ci_upper).
    """
    n, k = ratings.shape
    if n < 2 or k < 2:
        return 0.0, 0.0, 0.0

    # Mean squares from two-way ANOVA
    grand_mean = np.mean(ratings)
    row_means = np.mean(ratings, axis=1)
    col_means = np.mean(ratings, axis=0)

    ss_total = np.sum((ratings - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / max(1, n - 1)
    ms_cols = ss_cols / max(1, k - 1)
    ms_error = ss_error / max(1, (n - 1) * (k - 1))

    # ICC(2,1)
    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if abs(denom) < 1e-12:
        return 0.0, 0.0, 0.0
    icc = float((ms_rows - ms_error) / denom)

    # F-distribution based CI (McGraw & Wong 1996, Shrout & Fleiss 1979)
    # F_obs = MS_rows / MS_error
    f_obs = ms_rows / max(ms_error, 1e-12)
    df1 = n - 1
    df2 = (n - 1) * (k - 1)

    # Get F critical values
    try:
        from scipy.stats import f as f_dist

        f_critical_lower = f_dist.ppf(alpha / 2, df1, df2)
        f_critical_upper = f_dist.ppf(1 - alpha / 2, df1, df2)
    except ImportError:
        # Wilson-Hilferty approximation for F quantiles
        f_critical_lower = _wilson_hilferty_f_ppf(alpha / 2, df1, df2)
        f_critical_upper = _wilson_hilferty_f_ppf(1 - alpha / 2, df1, df2)

    # Transform F bounds to ICC bounds
    f_critical_upper = max(f_critical_upper, 1e-12)
    f_critical_lower = max(f_critical_lower, 1e-12)

    f_lower = f_obs / f_critical_upper
    f_upper = f_obs / f_critical_lower

    ci_lower = (f_lower - 1.0) / (f_lower + k - 1.0)
    ci_upper = (f_upper - 1.0) / (f_upper + k - 1.0)

    ci_lower = float(max(-1.0, ci_lower))
    ci_upper = float(min(1.0, ci_upper))

    return float(icc), ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------


@dataclass
class AblationResult:
    """Result from a single ablation condition."""

    name: str
    scores: np.ndarray
    mean: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    diff_from_base: float = 0.0
    p_value: float = 1.0

    def __post_init__(self) -> None:
        if len(self.scores) > 0:
            self.mean, self.ci_lower, self.ci_upper = bootstrap_confidence_interval(self.scores)


@dataclass
class AblationStudy:
    """Systematic ablation study comparing component contributions.

    Usage:
        study = AblationStudy()
        study.add_condition("full_model", full_scores)
        study.add_condition("no_projection", no_proj_scores)
        study.add_condition("no_soft_dtw", no_sdtw_scores)
        results = study.run()
    """

    base_name: str = "full_model"
    _conditions: dict[str, np.ndarray] = field(default_factory=dict)

    def add_condition(self, name: str, scores: np.ndarray) -> None:
        """Add an ablation condition with its scores."""
        self._conditions[name] = np.asarray(scores, dtype=np.float64)

    def run(self, n_bootstrap: int = 10000, n_permutations: int = 10000) -> list[AblationResult]:
        """Run ablation study with significance tests.

        Returns:
            List of AblationResult, one per condition.
        """
        if self.base_name not in self._conditions:
            raise ValueError(f"Base condition {self.base_name!r} not found")

        base_scores = self._conditions[self.base_name]
        results: list[AblationResult] = []

        for name, scores in self._conditions.items():
            result = AblationResult(name=name, scores=scores)

            if name != self.base_name:
                diff, p_value = permutation_test(base_scores, scores, n_permutations=n_permutations)
                result.diff_from_base = diff
                result.p_value = p_value

            results.append(result)

        return results

    def summary(self, results: list[AblationResult] | None = None) -> str:
        """Generate human-readable summary table."""
        if results is None:
            results = self.run()

        lines = [
            f"{'Condition':<25} {'Mean':>8} {'95% CI':>18} {'Δ Base':>8} {'p-value':>10} {'Sig':>5}",
            "-" * 80,
        ]
        for r in results:
            sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
            lines.append(
                f"{r.name:<25} {r.mean:>8.2f} [{r.ci_lower:>7.2f}, {r.ci_upper:>7.2f}] "
                f"{r.diff_from_base:>+8.2f} {r.p_value:>10.4f} {sig:>5}"
            )
        return "\n".join(lines)
