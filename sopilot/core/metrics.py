"""
sopilot.core.metrics
====================
Inter-rater reliability and calibration metrics for research-grade evaluation.

References
----------
- Shrout, P.E. & Fleiss, J.L. (1979). Intraclass correlations: Uses in assessing
  rater reliability. Psychological Bulletin, 86(2), 420-428.
- Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational
  and Psychological Measurement, 20(1), 37-46.
- Krippendorff, K. (2004). Content Analysis: An Introduction to Its Methodology
  (2nd ed.). Thousand Oaks, CA: Sage.
- McNemar, Q. (1947). Note on the sampling error of the difference between
  correlated proportions or percentages. Psychometrika, 12(2), 153-157.
- Guo, C. et al. (2017). On Calibration of Modern Neural Networks. ICML 2017.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import chi2
from scipy.stats import f as f_dist

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _interpret_icc(value: float) -> str:
    """Map ICC value to Koo & Mae (2016) verbal label."""
    if value >= 0.90:
        return "excellent"
    if value >= 0.75:
        return "good"
    if value >= 0.50:
        return "moderate"
    return "poor"


def _interpret_kappa(value: float) -> str:
    """Map Cohen's kappa to Landis & Koch (1977) verbal label."""
    if value >= 0.81:
        return "almost perfect"
    if value >= 0.61:
        return "substantial"
    if value >= 0.41:
        return "moderate"
    if value >= 0.21:
        return "fair"
    if value >= 0.01:
        return "slight"
    return "poor"


def _interpret_agreement(value: float) -> str:
    """Map mean agreement metric to Krippendorff reliability guideline."""
    if value >= 0.80:
        return "excellent"
    if value >= 0.667:
        return "good"
    if value >= 0.50:
        return "tentative"
    return "unreliable"


def _kappa_from_arrays(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's kappa for two 1-D label arrays.

    Returns 1.0 when perfect agreement, 0.0 when arrays are trivially identical
    with a single unique label (no chance disagreement possible in a meaningful
    sense, but by convention kappa = 1).
    """
    labels = np.union1d(a, b)
    n = len(a)
    if n == 0:
        return 0.0

    # Observed agreement
    p_o = np.mean(a == b)

    # Expected agreement
    p_e = 0.0
    for label in labels:
        p_e += (np.mean(a == label)) * (np.mean(b == label))

    if p_e >= 1.0:
        # All raters assigned the same category — perfect agreement by definition
        return 1.0

    return (p_o - p_e) / (1.0 - p_e)


# ---------------------------------------------------------------------------
# 1 & 2 — ICC
# ---------------------------------------------------------------------------

@dataclass
class ICCResult:
    """
    Result of an intraclass correlation computation.

    Attributes
    ----------
    icc : float
        Point estimate of the ICC.
    lower_ci : float
        Lower bound of the 95% confidence interval.
    upper_ci : float
        Upper bound of the 95% confidence interval.
    interpretation : str
        Verbal label: "excellent", "good", "moderate", or "poor".
    n_raters : int
        Number of raters (rows in the input matrix).
    n_subjects : int
        Number of subjects (columns in the input matrix).
    model : str
        Model identifier, e.g. "ICC(2,1)" or "ICC(3,1)".
    """
    icc: float
    lower_ci: float
    upper_ci: float
    interpretation: str
    n_raters: int
    n_subjects: int
    model: str


def compute_icc(
    ratings: list[list[float]],
    model: str = "ICC(2,1)",
) -> ICCResult:
    """
    Compute the Intraclass Correlation Coefficient (ICC).

    Parameters
    ----------
    ratings : list[list[float]]
        ratings[i][j] is rater *i*'s score for subject *j*.
        NaN values are allowed and are handled via pairwise deletion
        (subjects with any NaN are excluded for ICC computation).
    model : str
        "ICC(2,1)"  — two-way random effects, absolute agreement.
        "ICC(3,1)"  — two-way mixed effects, consistency.

    Returns
    -------
    ICCResult

    Notes
    -----
    ANOVA decomposition follows Shrout & Fleiss (1979) Table 1.
    95% CI from the F distribution (McGraw & Wong, 1996).
    """
    if len(ratings) < 2:
        return ICCResult(
            icc=0.0,
            lower_ci=0.0,
            upper_ci=0.0,
            interpretation="poor",
            n_raters=len(ratings),
            n_subjects=0,
            model=model,
        )

    # Build matrix: rows = raters, cols = subjects
    mat = np.array(ratings, dtype=float)  # shape (k, n)
    k_raters, n_subjects = mat.shape

    # Remove subjects (columns) with any NaN
    valid_cols = ~np.any(np.isnan(mat), axis=0)
    mat = mat[:, valid_cols]
    n = mat.shape[1]  # number of usable subjects

    if n < 2 or k_raters < 2:
        return ICCResult(
            icc=0.0,
            lower_ci=0.0,
            upper_ci=0.0,
            interpretation="poor",
            n_raters=k_raters,
            n_subjects=n,
            model=model,
        )

    k = k_raters
    grand_mean = mat.mean()

    # ANOVA sums of squares
    ss_total = np.sum((mat - grand_mean) ** 2)
    subject_means = mat.mean(axis=0)          # mean across raters per subject
    rater_means   = mat.mean(axis=1)          # mean across subjects per rater

    ss_between_subjects = k * np.sum((subject_means - grand_mean) ** 2)
    ss_between_raters   = n * np.sum((rater_means   - grand_mean) ** 2)
    ss_error            = ss_total - ss_between_subjects - ss_between_raters
    ss_within           = ss_total - ss_between_subjects

    # Degrees of freedom
    df_between_subjects = n - 1
    df_between_raters   = k - 1
    df_error            = (n - 1) * (k - 1)
    df_within           = n * (k - 1)          # used for within-subjects MS

    # Mean squares
    ms_between_subjects = ss_between_subjects / df_between_subjects  # MS_S
    ms_between_raters   = ss_between_raters   / df_between_raters    # MS_R  (unused in ICC(2,1) formula directly)
    ms_error            = ss_error            / df_error              # MS_E
    _ms_within          = ss_within           / df_within             # MS_W  (ICC(3,1) denominator, kept for reference)

    # Perfect agreement edge case: zero error variance → ICC = 1.0
    if ms_error < 1e-10:
        return ICCResult(
            icc=1.0, lower_ci=1.0, upper_ci=1.0,
            interpretation=_interpret_icc(1.0),
            n_raters=k, n_subjects=n, model=model,
        )

    # ---- ICC computation ----
    if model == "ICC(3,1)":
        # Two-way mixed, consistency (formula 3 in Shrout & Fleiss 1979)
        icc_val = (ms_between_subjects - ms_error) / (
            ms_between_subjects + (k - 1) * ms_error
        )
        # CI via F distribution (McGraw & Wong 1996, Case 3A)
        F_lower = f_dist.ppf(0.025, df_between_subjects, df_error)
        F_upper = f_dist.ppf(0.975, df_between_subjects, df_error)
        F_obs   = ms_between_subjects / ms_error
        try:
            lower = (F_obs / F_upper - 1) / (F_obs / F_upper + k - 1)
            upper = (F_obs / F_lower - 1) / (F_obs / F_lower + k - 1)
        except ZeroDivisionError:
            lower, upper = 0.0, 1.0

    else:
        # Default: ICC(2,1) — two-way random, absolute agreement
        # Formula 2 in Shrout & Fleiss 1979
        icc_val = (ms_between_subjects - ms_error) / (
            ms_between_subjects
            + (k - 1) * ms_error
            + k * (ms_between_raters - ms_error) / n
        )
        # CI: approximate via Fl & Fu from the 3 MS values (Shrout & Fleiss 1979)
        # Use the simpler McGraw & Wong form for the random-effects case
        F_obs = ms_between_subjects / ms_error
        # Critical F values at 2.5 % and 97.5 %
        F_lower_crit = f_dist.ppf(0.025, df_between_subjects, df_error)
        F_upper_crit = f_dist.ppf(0.975, df_between_subjects, df_error)

        # Bounds from the confidence interval on F ratio
        Fl = F_obs / F_upper_crit
        Fu = F_obs / F_lower_crit

        try:
            lower = (Fl - 1) / (Fl + k - 1)
            upper = (Fu - 1) / (Fu + k - 1)
        except ZeroDivisionError:
            lower, upper = 0.0, 1.0

    # Clamp to [-1, 1]
    icc_val = float(np.clip(icc_val, -1.0, 1.0))
    lower   = float(np.clip(lower,   -1.0, 1.0))
    upper   = float(np.clip(upper,   -1.0, 1.0))

    return ICCResult(
        icc=icc_val,
        lower_ci=lower,
        upper_ci=upper,
        interpretation=_interpret_icc(icc_val),
        n_raters=k,
        n_subjects=n,
        model=model,
    )


# ---------------------------------------------------------------------------
# 3 & 4 — Cohen's Kappa
# ---------------------------------------------------------------------------

@dataclass
class KappaResult:
    """
    Result of a Cohen's kappa computation.

    Attributes
    ----------
    kappa : float
        Point estimate of Cohen's kappa.
    lower_ci : float
        Lower bound of the 95% bootstrap confidence interval.
    upper_ci : float
        Upper bound of the 95% bootstrap confidence interval.
    interpretation : str
        Verbal label per Landis & Koch (1977).
    p_value : float
        Permutation-test p-value (H0: kappa = 0, i.e. chance agreement).
    n_samples : int
        Number of paired observations used.
    """
    kappa: float
    lower_ci: float
    upper_ci: float
    interpretation: str
    p_value: float
    n_samples: int


def compute_cohens_kappa(
    labels_a: list[str],
    labels_b: list[str],
    n_bootstrap: int = 1000,
    n_permutations: int = 500,
) -> KappaResult:
    """
    Compute Cohen's kappa with bootstrap 95 % CI and permutation p-value.

    Parameters
    ----------
    labels_a, labels_b : list[str]
        Paired categorical labels from two raters.
        Expected values: "pass", "fail", "needs_review", "retrain".
    n_bootstrap : int
        Number of bootstrap resamples for CI estimation (default 1000).
    n_permutations : int
        Number of permutations for the p-value (default 500).

    Returns
    -------
    KappaResult

    Notes
    -----
    - Bootstrap uses stratified resampling with replacement (pairs).
    - p-value is the fraction of permuted kappas >= observed kappa.
    - numpy seed fixed at 42 for reproducibility.

    References
    ----------
    Cohen (1960); Efron & Tibshirani (1993) bootstrap methodology.
    """
    if len(labels_a) == 0 or len(labels_b) == 0:
        raise ValueError("labels_a and labels_b must not be empty.")
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"labels_a and labels_b must have equal length "
            f"(got {len(labels_a)} vs {len(labels_b)})."
        )

    a = np.array(labels_a, dtype=str)
    b = np.array(labels_b, dtype=str)
    n = len(a)

    observed_kappa = _kappa_from_arrays(a, b)

    # Perfect agreement shortcut
    if np.all(a == b):
        return KappaResult(
            kappa=1.0,
            lower_ci=1.0,
            upper_ci=1.0,
            interpretation="almost perfect",
            p_value=0.0,
            n_samples=n,
        )

    rng = np.random.default_rng(42)

    # --- Bootstrap CI ---
    boot_kappas: list[float] = []
    indices = np.arange(n)
    for _ in range(n_bootstrap):
        idx = rng.choice(indices, size=n, replace=True)
        boot_kappas.append(_kappa_from_arrays(a[idx], b[idx]))

    boot_arr = np.array(boot_kappas)
    lower_ci = float(np.percentile(boot_arr, 2.5))
    upper_ci = float(np.percentile(boot_arr, 97.5))

    # --- Permutation p-value ---
    perm_count = 0
    b_copy = b.copy()
    for _ in range(n_permutations):
        b_perm = rng.permutation(b_copy)
        perm_k = _kappa_from_arrays(a, b_perm)
        if perm_k >= observed_kappa:
            perm_count += 1

    p_value = float(perm_count / n_permutations)

    return KappaResult(
        kappa=float(observed_kappa),
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        interpretation=_interpret_kappa(observed_kappa),
        p_value=p_value,
        n_samples=n,
    )


# ---------------------------------------------------------------------------
# 5 & 6 — Calibration
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """
    Result of a calibration (reliability) analysis.

    Attributes
    ----------
    expected_calibration_error : float
        ECE — weighted mean absolute difference between predicted probability
        and observed frequency (Guo et al., 2017).
    max_calibration_error : float
        MCE — maximum absolute difference across bins.
    reliability_diagram : list[dict]
        Per-bin data for plotting a reliability diagram. Each entry contains:
        ``{"bin_center": float, "predicted": float, "actual": float, "count": int}``.
    n_bins : int
        Number of bins used.
    """
    expected_calibration_error: float
    max_calibration_error: float
    reliability_diagram: list[dict]
    n_bins: int


def compute_calibration(
    predicted_scores: list[float],
    actual_pass: list[bool],
    n_bins: int = 10,
) -> CalibrationResult:
    """
    Assess how well score magnitudes match empirical pass rates.

    Parameters
    ----------
    predicted_scores : list[float]
        Model scores in [0, 100].  Normalised to [0, 1] as a pseudo-probability.
    actual_pass : list[bool]
        Ground-truth pass/fail outcome for each scored item.
    n_bins : int
        Number of equal-width bins in [0, 1] (default 10).

    Returns
    -------
    CalibrationResult

    Notes
    -----
    ECE = Σ_b (|B_b| / n) * |mean_predicted_b - mean_actual_b|
    MCE = max_b |mean_predicted_b - mean_actual_b|

    Reference
    ---------
    Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    On calibration of modern neural networks. *ICML*.
    """
    if len(predicted_scores) == 0:
        raise ValueError("predicted_scores must not be empty.")
    if len(predicted_scores) != len(actual_pass):
        raise ValueError(
            f"predicted_scores and actual_pass must have equal length "
            f"(got {len(predicted_scores)} vs {len(actual_pass)})."
        )

    scores = np.array(predicted_scores, dtype=float)
    labels = np.array(actual_pass,      dtype=float)  # 1.0 = pass, 0.0 = fail

    # Normalise [0, 100] → [0, 1]
    probs = np.clip(scores / 100.0, 0.0, 1.0)

    n = len(probs)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    reliability_diagram: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bin_center = (lo + hi) / 2.0

        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            # Include right edge for the last bin
            mask = (probs >= lo) & (probs <= hi)

        count = int(mask.sum())
        if count == 0:
            reliability_diagram.append(
                {
                    "bin_center": round(float(bin_center), 6),
                    "predicted":  round(float(bin_center), 6),
                    "actual":     0.0,
                    "count":      0,
                }
            )
            continue

        mean_predicted = float(probs[mask].mean())
        mean_actual    = float(labels[mask].mean())
        abs_diff       = abs(mean_predicted - mean_actual)

        ece += (count / n) * abs_diff
        mce  = max(mce, abs_diff)

        reliability_diagram.append(
            {
                "bin_center": round(float(bin_center), 6),
                "predicted":  round(float(mean_predicted), 6),
                "actual":     round(float(mean_actual), 6),
                "count":      count,
            }
        )

    return CalibrationResult(
        expected_calibration_error=float(ece),
        max_calibration_error=float(mce),
        reliability_diagram=reliability_diagram,
        n_bins=n_bins,
    )


# ---------------------------------------------------------------------------
# 7 & 8 — Multi-rater agreement (ICC mean + Kappa mean + Krippendorff's α)
# ---------------------------------------------------------------------------

@dataclass
class AgreementStats:
    """
    Aggregated multi-rater agreement statistics.

    Attributes
    ----------
    mean_pairwise_icc : float
        Unweighted mean of all pairwise ICC(2,1) estimates.
    mean_pairwise_kappa : float
        Unweighted mean of all pairwise Cohen's kappa estimates after
        thresholding scores into pass/fail categories.
    krippendorffs_alpha : float
        Krippendorff's alpha (continuous / interval metric).
    interpretation : str
        Summary verbal label based on Krippendorff's alpha.
    """
    mean_pairwise_icc: float
    mean_pairwise_kappa: float
    krippendorffs_alpha: float
    interpretation: str


def _krippendorffs_alpha_continuous(data: np.ndarray) -> float:
    """
    Krippendorff's alpha for continuous (interval) data.

    Parameters
    ----------
    data : np.ndarray, shape (k, n)
        Ratings matrix. NaN indicates missing values.

    Returns
    -------
    float
        Alpha in (-inf, 1].  1.0 = perfect agreement.

    Notes
    -----
    Uses the interval/ratio difference function d(v_k, v_l) = (v_k - v_l)^2.

    Reference
    ---------
    Krippendorff, K. (2004). Content Analysis: An Introduction to Its
    Methodology (2nd ed.), Appendix B.
    """
    k, n = data.shape  # k raters, n subjects

    # ---- Observed disagreement D_o ----
    # Sum over all coincident pairs within subjects
    d_o_num = 0.0
    total_coincident = 0
    for j in range(n):
        col = data[:, j]
        valid = col[~np.isnan(col)]
        m = len(valid)
        if m < 2:
            continue
        for vi, vj in combinations(valid, 2):
            d_o_num += (vi - vj) ** 2
            total_coincident += 1

    if total_coincident == 0:
        return 1.0  # no coincident pairs — trivially perfect

    d_o = d_o_num / total_coincident

    # ---- Expected disagreement D_e ----
    # Use all observed values (flattened, ignoring NaN) to form the
    # pairable distribution.
    all_vals = data[~np.isnan(data)]
    N = len(all_vals)
    if N < 2:
        return 1.0

    # D_e for interval data = variance of all values (up to a constant factor
    # that cancels in alpha). Specifically:
    # D_e = (1 / (N*(N-1))) * sum_{u != v} (x_u - x_v)^2
    #      = (2 / (N-1)) * variance(all_vals)    [unbiased]
    # But the standard Krippendorff formula uses:
    # D_e = (1 / (n*(n-1))) * sum_u sum_v (x_u - x_v)^2  over ALL value pairs
    # We compute this efficiently via the identity:
    #   sum_{u,v} (x_u - x_v)^2 = 2*N * sum(x^2) - 2 * (sum(x))^2
    sum_sq   = float(np.sum(all_vals ** 2))
    sum_vals = float(np.sum(all_vals))
    # sum of (v_u - v_v)^2 for ALL ordered pairs (including u==v gives 0)
    sum_diff_sq = 2 * N * sum_sq - 2 * sum_vals ** 2

    d_e = sum_diff_sq / (N * (N - 1))

    if d_e == 0.0:
        # All values identical → perfect agreement
        return 1.0

    alpha = 1.0 - d_o / d_e
    return float(alpha)


def compute_agreement_stats(
    all_rater_scores: list[list[float]],
    pass_threshold: float = 70.0,
) -> AgreementStats:
    """
    Compute comprehensive multi-rater agreement statistics.

    Parameters
    ----------
    all_rater_scores : list[list[float]]
        all_rater_scores[i] = list of numeric scores given by rater *i*,
        one per subject.  All inner lists must have the same length.
    pass_threshold : float
        Score threshold for pass/fail binarisation used in kappa (default 70.0).

    Returns
    -------
    AgreementStats

    Notes
    -----
    - Pairwise ICC uses ICC(2,1).
    - Pairwise kappa uses no bootstrapping / permutations (n_bootstrap=0 path)
      for efficiency; the mean is taken over all rater pairs.
    - Krippendorff's alpha uses the continuous (interval) variant.
    """
    n_raters = len(all_rater_scores)

    if n_raters < 2:
        return AgreementStats(
            mean_pairwise_icc=0.0,
            mean_pairwise_kappa=0.0,
            krippendorffs_alpha=0.0,
            interpretation="unreliable",
        )

    # Validate equal length
    lengths = [len(r) for r in all_rater_scores]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All rater score lists must have the same length. "
            f"Got lengths: {lengths}"
        )

    mat = np.array(all_rater_scores, dtype=float)  # shape (k, n)
    pairs = list(combinations(range(n_raters), 2))

    # ---- Pairwise ICC ----
    icc_vals: list[float] = []
    for i, j in pairs:
        result = compute_icc([list(mat[i]), list(mat[j])], model="ICC(2,1)")
        icc_vals.append(result.icc)

    mean_icc = float(np.mean(icc_vals)) if icc_vals else 0.0

    # ---- Pairwise Kappa (binarise first) ----
    binary_labels = [
        ["pass" if s >= pass_threshold else "fail" for s in scores]
        for scores in all_rater_scores
    ]

    kappa_vals: list[float] = []
    for i, j in pairs:
        a_arr = np.array(binary_labels[i], dtype=str)
        b_arr = np.array(binary_labels[j], dtype=str)
        kappa_vals.append(_kappa_from_arrays(a_arr, b_arr))

    mean_kappa = float(np.mean(kappa_vals)) if kappa_vals else 0.0

    # ---- Krippendorff's alpha ----
    alpha = _krippendorffs_alpha_continuous(mat)

    return AgreementStats(
        mean_pairwise_icc=mean_icc,
        mean_pairwise_kappa=mean_kappa,
        krippendorffs_alpha=alpha,
        interpretation=_interpret_agreement(alpha),
    )


# ---------------------------------------------------------------------------
# 9 — McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(
    system_correct: list[bool],
    baseline_correct: list[bool],
) -> dict[str, Any]:
    """
    McNemar's test for paired binary classifiers.

    Determines whether a new system is significantly better or worse than a
    baseline on the same set of test cases.

    Parameters
    ----------
    system_correct : list[bool]
        Boolean vector: True if the *new* system predicted correctly.
    baseline_correct : list[bool]
        Boolean vector: True if the *baseline* system predicted correctly.

    Returns
    -------
    dict with keys:
        - ``"statistic"`` (float) — chi-squared test statistic (with continuity
          correction).
        - ``"p_value"`` (float) — two-tailed p-value from chi2(df=1).
        - ``"b"`` (int) — count of cases where system correct, baseline wrong.
        - ``"c"`` (int) — count of cases where baseline correct, system wrong.
        - ``"interpretation"`` (str) — one of:
          "system significantly better",
          "baseline significantly better",
          "no significant difference".

    Notes
    -----
    statistic = (|b - c| - 1)^2 / (b + c)  [Yates' continuity correction]

    p-value is set to 1.0 when b + c == 0 (no discordant pairs).

    Reference
    ---------
    McNemar, Q. (1947). Note on the sampling error of the difference between
    correlated proportions or percentages. *Psychometrika*, 12(2), 153-157.
    """
    if len(system_correct) == 0 or len(baseline_correct) == 0:
        raise ValueError("Input lists must not be empty.")
    if len(system_correct) != len(baseline_correct):
        raise ValueError(
            f"system_correct and baseline_correct must have equal length "
            f"(got {len(system_correct)} vs {len(baseline_correct)})."
        )

    sys_arr  = np.array(system_correct,   dtype=bool)
    base_arr = np.array(baseline_correct, dtype=bool)

    # b: system correct AND baseline wrong
    b = int(np.sum( sys_arr & ~base_arr))
    # c: baseline correct AND system wrong
    c = int(np.sum(~sys_arr &  base_arr))

    discordant = b + c

    if discordant == 0:
        # Perfect agreement on correctness — no test possible
        return {
            "statistic":      0.0,
            "p_value":        1.0,
            "b":              b,
            "c":              c,
            "interpretation": "no significant difference",
        }

    # Chi-squared with continuity correction
    statistic = (abs(b - c) - 1) ** 2 / discordant
    p_value   = float(chi2.sf(statistic, df=1))

    if p_value < 0.05:
        interp = "system significantly better" if b > c else "baseline significantly better"
    else:
        interp = "no significant difference"

    return {
        "statistic":      float(statistic),
        "p_value":        p_value,
        "b":              b,
        "c":              c,
        "interpretation": interp,
    }
