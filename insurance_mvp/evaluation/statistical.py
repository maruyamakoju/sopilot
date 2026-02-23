"""Research-grade statistical evaluation for severity classification.

Implements:
- Bootstrap confidence intervals (BCa method)
- McNemar's test for paired model comparison
- Cohen's kappa for inter-rater agreement
- Stratified per-class metrics with CI
- Effect size reporting (Cramér's V)

References:
  Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
  Dietterich (1998) "Approximate Statistical Tests for Comparing Supervised
      Classification Learning Algorithms", Neural Computation 10(7)
  Cohen (1960) "A Coefficient of Agreement for Nominal Scales"
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH"]
SEVERITY_TO_IDX = {s: i for i, s in enumerate(SEVERITY_LEVELS)}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceInterval:
    """Bootstrap confidence interval."""
    point: float
    lower: float
    upper: float
    alpha: float = 0.05
    method: str = "bca"

    def __repr__(self) -> str:
        pct = int((1 - self.alpha) * 100)
        return f"{self.point:.4f} [{self.lower:.4f}, {self.upper:.4f}] ({pct}% CI, {self.method})"


@dataclass
class ClassMetrics:
    """Per-class precision, recall, F1 with confidence intervals."""
    label: str
    support: int
    precision: ConfidenceInterval
    recall: ConfidenceInterval
    f1: ConfidenceInterval


@dataclass
class EvaluationReport:
    """Complete evaluation report with statistical rigor."""
    accuracy: ConfidenceInterval
    macro_f1: ConfidenceInterval
    weighted_f1: ConfidenceInterval
    cohen_kappa: ConfidenceInterval
    cramers_v: float
    per_class: list[ClassMetrics]
    confusion_matrix: np.ndarray
    n_samples: int
    n_classes: int
    labels: list[str]


@dataclass
class ModelComparison:
    """Result of comparing two models via McNemar's test."""
    model_a_accuracy: float
    model_b_accuracy: float
    chi2_statistic: float
    p_value: float
    significant: bool
    effect_size: float  # Odds ratio
    n_discordant: int


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> ConfidenceInterval:
    """Compute bootstrap confidence interval for a metric.

    Uses the bias-corrected and accelerated (BCa) method for
    second-order accurate intervals.

    Args:
        y_true: Ground truth labels (integer indices).
        y_pred: Predicted labels (integer indices).
        metric_fn: Callable(y_true, y_pred) -> float.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (0.05 = 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        ConfidenceInterval with point estimate and bounds.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point_estimate = metric_fn(y_true, y_pred)

    # Bootstrap resamples
    boot_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_stats[b] = metric_fn(y_true[idx], y_pred[idx])

    # BCa correction
    # Bias correction factor z0
    z0 = _norm_ppf(np.mean(boot_stats < point_estimate))

    # Acceleration factor a (jackknife)
    jackknife_stats = np.empty(n)
    for i in range(n):
        mask = np.concatenate([np.arange(0, i), np.arange(i + 1, n)])
        jackknife_stats[i] = metric_fn(y_true[mask], y_pred[mask])
    jack_mean = jackknife_stats.mean()
    num = np.sum((jack_mean - jackknife_stats) ** 3)
    den = 6.0 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    # Adjusted quantiles
    z_alpha_lo = _norm_ppf(alpha / 2)
    z_alpha_hi = _norm_ppf(1 - alpha / 2)

    adj_lo = _norm_cdf(z0 + (z0 + z_alpha_lo) / (1 - a * (z0 + z_alpha_lo)))
    adj_hi = _norm_cdf(z0 + (z0 + z_alpha_hi) / (1 - a * (z0 + z_alpha_hi)))

    # Clamp to valid percentile range
    adj_lo = np.clip(adj_lo, 0.001, 0.999)
    adj_hi = np.clip(adj_hi, 0.001, 0.999)

    lower = float(np.percentile(boot_stats, 100 * adj_lo))
    upper = float(np.percentile(boot_stats, 100 * adj_hi))

    return ConfidenceInterval(
        point=point_estimate,
        lower=lower,
        upper=upper,
        alpha=alpha,
        method="bca",
    )


def _norm_ppf(p: float) -> float:
    """Normal percent point function (inverse CDF) without scipy."""
    # Rational approximation (Abramowitz & Stegun 26.2.23)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    if p < 0.5:
        return -_rational_approx(np.sqrt(-2.0 * np.log(p)))
    else:
        return _rational_approx(np.sqrt(-2.0 * np.log(1 - p)))


def _rational_approx(t: float) -> float:
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)


def _norm_cdf(x: float) -> float:
    """Normal CDF without scipy."""
    return 0.5 * (1 + _erf(x / np.sqrt(2)))


def _erf(x: float) -> float:
    """Error function approximation (Abramowitz & Stegun 7.1.26)."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


# ---------------------------------------------------------------------------
# Metric functions (operate on integer index arrays)
# ---------------------------------------------------------------------------

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy."""
    return float(np.mean(y_true == y_pred))


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Cohen's kappa (chance-corrected agreement).

    kappa = (p_o - p_e) / (1 - p_e)
    where p_o = observed agreement, p_e = expected by chance.
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    n_classes = max(y_true.max(), y_pred.max()) + 1

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    p_o = np.trace(cm) / n
    p_e = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (n * n)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged F1 score."""
    n_classes = max(y_true.max(), y_pred.max()) + 1
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute weighted F1 score (weighted by class support)."""
    n_classes = max(y_true.max(), y_pred.max()) + 1
    n = len(y_true)
    total = 0.0
    for c in range(n_classes):
        support = np.sum(y_true == c)
        if support == 0:
            continue
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        total += f1 * support
    return total / n if n > 0 else 0.0


def cramers_v(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Cramér's V (effect size for categorical association).

    V = sqrt(chi2 / (n * (min(r,c) - 1)))
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    n_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Chi-squared
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    chi2 = 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            expected = row_sums[i] * col_sums[j] / n
            if expected > 0:
                chi2 += (cm[i, j] - expected) ** 2 / expected

    k = min(n_classes, n_classes)
    if k <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * (k - 1))))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# High-level evaluation
# ---------------------------------------------------------------------------

def evaluate(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: list[str] | None = None,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> EvaluationReport:
    """Run complete statistical evaluation.

    Args:
        y_true: Ground truth severity labels.
        y_pred: Predicted severity labels.
        labels: Label names (default: SEVERITY_LEVELS).
        n_bootstrap: Bootstrap resamples.
        alpha: CI significance level.
        seed: Random seed.

    Returns:
        EvaluationReport with CIs on all metrics.
    """
    if labels is None:
        labels = SEVERITY_LEVELS

    label_to_idx = {s: i for i, s in enumerate(labels)}
    yt = np.array([label_to_idx.get(s, 0) for s in y_true])
    yp = np.array([label_to_idx.get(s, 0) for s in y_pred])
    n_classes = len(labels)

    # Global metrics with bootstrap CI
    acc_ci = bootstrap_ci(yt, yp, accuracy_score, n_bootstrap, alpha, seed)
    kappa_ci = bootstrap_ci(yt, yp, cohen_kappa, n_bootstrap, alpha, seed)
    macro_ci = bootstrap_ci(yt, yp, macro_f1, n_bootstrap, alpha, seed)
    weighted_ci = bootstrap_ci(yt, yp, weighted_f1, n_bootstrap, alpha, seed)
    cv = cramers_v(yt, yp)
    cm = confusion_matrix(yt, yp, n_classes)

    # Per-class metrics with CI
    per_class_metrics = []
    for c in range(n_classes):
        support = int(np.sum(yt == c))

        def _precision(yt_, yp_, c_=c):
            tp = np.sum((yt_ == c_) & (yp_ == c_))
            fp = np.sum((yt_ != c_) & (yp_ == c_))
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0

        def _recall(yt_, yp_, c_=c):
            tp = np.sum((yt_ == c_) & (yp_ == c_))
            fn = np.sum((yt_ == c_) & (yp_ != c_))
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        def _f1(yt_, yp_, c_=c):
            p = _precision(yt_, yp_, c_)
            r = _recall(yt_, yp_, c_)
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        p_ci = bootstrap_ci(yt, yp, _precision, n_bootstrap, alpha, seed + c)
        r_ci = bootstrap_ci(yt, yp, _recall, n_bootstrap, alpha, seed + c + n_classes)
        f1_ci = bootstrap_ci(yt, yp, _f1, n_bootstrap, alpha, seed + c + 2 * n_classes)

        per_class_metrics.append(ClassMetrics(
            label=labels[c],
            support=support,
            precision=p_ci,
            recall=r_ci,
            f1=f1_ci,
        ))

    return EvaluationReport(
        accuracy=acc_ci,
        macro_f1=macro_ci,
        weighted_f1=weighted_ci,
        cohen_kappa=kappa_ci,
        cramers_v=cv,
        per_class=per_class_metrics,
        confusion_matrix=cm,
        n_samples=len(yt),
        n_classes=n_classes,
        labels=labels,
    )


def mcnemar_test(
    y_true: Sequence[str],
    y_pred_a: Sequence[str],
    y_pred_b: Sequence[str],
) -> ModelComparison:
    """McNemar's test for comparing two classifiers on paired data.

    Tests whether the disagreement patterns between two models are
    significantly different (i.e., whether one model is better).

    Args:
        y_true: Ground truth labels.
        y_pred_a: Predictions from model A.
        y_pred_b: Predictions from model B.

    Returns:
        ModelComparison with chi2 statistic, p-value, and effect size.
    """
    n = len(y_true)
    correct_a = np.array([t == p for t, p in zip(y_true, y_pred_a)])
    correct_b = np.array([t == p for t, p in zip(y_true, y_pred_b)])

    # Contingency table of correctness
    # b01 = A correct, B wrong; b10 = A wrong, B correct
    b01 = int(np.sum(correct_a & ~correct_b))
    b10 = int(np.sum(~correct_a & correct_b))

    # McNemar's chi-squared with continuity correction
    n_discordant = b01 + b10
    if n_discordant == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(b01 - b10) - 1) ** 2 / n_discordant
        # Approximate p-value from chi2(1) distribution
        p_value = 1 - _chi2_cdf_1df(chi2)

    # Effect size: odds ratio
    effect_size = b01 / b10 if b10 > 0 else float("inf")

    acc_a = float(np.mean(correct_a))
    acc_b = float(np.mean(correct_b))

    return ModelComparison(
        model_a_accuracy=acc_a,
        model_b_accuracy=acc_b,
        chi2_statistic=chi2,
        p_value=p_value,
        significant=p_value < 0.05,
        effect_size=effect_size,
        n_discordant=n_discordant,
    )


def _chi2_cdf_1df(x: float) -> float:
    """CDF of chi-squared distribution with 1 degree of freedom.

    chi2(1) CDF = 2 * Phi(sqrt(x)) - 1 where Phi is normal CDF.
    """
    if x <= 0:
        return 0.0
    return 2 * _norm_cdf(np.sqrt(x)) - 1


def format_report(report: EvaluationReport) -> str:
    """Format evaluation report as human-readable string."""
    lines = [
        "=" * 70,
        "STATISTICAL EVALUATION REPORT",
        "=" * 70,
        f"Samples: {report.n_samples}   Classes: {report.n_classes}",
        "",
        "--- Global Metrics (95% BCa Bootstrap CI) ---",
        f"  Accuracy:     {report.accuracy}",
        f"  Macro F1:     {report.macro_f1}",
        f"  Weighted F1:  {report.weighted_f1}",
        f"  Cohen's κ:    {report.cohen_kappa}",
        f"  Cramér's V:   {report.cramers_v:.4f}",
        "",
        "--- Per-Class Metrics ---",
    ]

    for cm in report.per_class:
        lines.append(f"  {cm.label} (n={cm.support}):")
        lines.append(f"    Precision: {cm.precision}")
        lines.append(f"    Recall:    {cm.recall}")
        lines.append(f"    F1:        {cm.f1}")

    lines.append("")
    lines.append("--- Confusion Matrix (rows=true, cols=pred) ---")
    header = "        " + "  ".join(f"{l:>6s}" for l in report.labels)
    lines.append(header)
    for i, row in enumerate(report.confusion_matrix):
        row_str = "  ".join(f"{v:>6d}" for v in row)
        lines.append(f"  {report.labels[i]:>6s} {row_str}")

    lines.append("=" * 70)
    return "\n".join(lines)
