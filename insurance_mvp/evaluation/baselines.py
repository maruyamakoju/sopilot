"""Baseline classifiers for severity classification benchmarking.

Provides simple baselines that can be compared against the full pipeline
using McNemar's test to assess whether the pipeline provides statistically
significant improvement over naive strategies.

Classes:
    MajorityClassBaseline -- always predicts the most common class
    DangerScoreBaseline   -- predicts severity from danger_score thresholds
    RandomBaseline        -- samples uniformly from class prior
    StratifiedRandomBaseline -- samples proportionally from class distribution

Functions:
    compare_with_baseline -- run McNemar's test between a model and a baseline
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Sequence

from insurance_mvp.evaluation.statistical import ModelComparison, mcnemar_test

SEVERITY_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH"]


# ---------------------------------------------------------------------------
# Base protocol
# ---------------------------------------------------------------------------

class _BaselineClassifier:
    """Common interface for all baseline classifiers."""

    @property
    def name(self) -> str:  # pragma: no cover
        raise NotImplementedError

    def fit(self, y_true: Sequence[str]) -> "_BaselineClassifier":
        """Fit the baseline to observed ground-truth labels.

        Args:
            y_true: Ground-truth severity labels used to compute class prior.

        Returns:
            self (for chaining).
        """
        raise NotImplementedError

    def predict(self, n_samples: int) -> list[str]:
        """Generate predictions for n_samples examples.

        Args:
            n_samples: Number of predictions to produce.

        Returns:
            List of predicted severity labels (strings).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Majority-class baseline
# ---------------------------------------------------------------------------

class MajorityClassBaseline(_BaselineClassifier):
    """Always predicts the majority (most frequent) class.

    In a balanced dataset this is a weak baseline; on skewed data it
    can achieve deceptively high accuracy.  Use McNemar's test to
    determine whether the full pipeline is significantly better.

    Example::

        baseline = MajorityClassBaseline()
        baseline.fit(["NONE", "NONE", "HIGH", "NONE"])
        preds = baseline.predict(4)  # ["NONE", "NONE", "NONE", "NONE"]
    """

    def __init__(self) -> None:
        self._majority_class: str = SEVERITY_LEVELS[0]

    @property
    def name(self) -> str:
        return "MajorityClassBaseline"

    def fit(self, y_true: Sequence[str]) -> "MajorityClassBaseline":
        """Determine the majority class from training labels.

        Args:
            y_true: Ground-truth severity labels.

        Returns:
            self
        """
        if not y_true:
            self._majority_class = SEVERITY_LEVELS[0]
            return self
        counts = Counter(y_true)
        self._majority_class = counts.most_common(1)[0][0]
        return self

    def predict(self, n_samples: int) -> list[str]:
        """Predict the majority class for every sample.

        Args:
            n_samples: Number of predictions to produce.

        Returns:
            List of identical majority-class labels.
        """
        return [self._majority_class] * n_samples

    @property
    def majority_class(self) -> str:
        """The class this baseline will always predict."""
        return self._majority_class


# ---------------------------------------------------------------------------
# Danger-score threshold baseline
# ---------------------------------------------------------------------------

class DangerScoreBaseline(_BaselineClassifier):
    """Predicts severity based on danger_score thresholds alone.

    This isolates the contribution of the mining signal: if the full
    pipeline achieves significantly better accuracy, the VLM adds
    meaningful value beyond raw signal scores.

    Default thresholds (adjustable via constructor):
        score >= high_threshold  -> "HIGH"
        score >= medium_threshold -> "MEDIUM"
        score >= low_threshold    -> "LOW"
        score <  low_threshold    -> "NONE"

    The ``fit`` method is a no-op (thresholds are fixed).
    The ``predict(X)`` method accepts a list of danger scores.

    Example::

        baseline = DangerScoreBaseline()
        preds = baseline.predict([0.1, 0.5, 0.8, 0.95])
        # -> ["NONE", "MEDIUM", "HIGH", "HIGH"]
    """

    def __init__(
        self,
        high_threshold: float = 0.75,
        medium_threshold: float = 0.45,
        low_threshold: float = 0.20,
    ) -> None:
        """Initialise with severity thresholds.

        Args:
            high_threshold:   Danger score >= this -> "HIGH".
            medium_threshold: Danger score >= this -> "MEDIUM".
            low_threshold:    Danger score >= this -> "LOW".
        """
        if not (0.0 <= low_threshold <= medium_threshold <= high_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy 0 <= low <= medium <= high <= 1, "
                f"got low={low_threshold}, medium={medium_threshold}, high={high_threshold}"
            )
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold

    @property
    def name(self) -> str:
        return (
            f"DangerScoreBaseline("
            f"high={self.high_threshold}, "
            f"medium={self.medium_threshold}, "
            f"low={self.low_threshold})"
        )

    def fit(self, y_true: Sequence[str]) -> "DangerScoreBaseline":
        """No-op: thresholds are fixed at construction time.

        Args:
            y_true: Ignored.

        Returns:
            self
        """
        return self

    def predict(self, X: Sequence[float]) -> list[str]:  # type: ignore[override]
        """Map danger scores to severity labels.

        Args:
            X: Sequence of danger_score floats in [0, 1].

        Returns:
            List of severity label strings.
        """
        result: list[str] = []
        for score in X:
            if score >= self.high_threshold:
                result.append("HIGH")
            elif score >= self.medium_threshold:
                result.append("MEDIUM")
            elif score >= self.low_threshold:
                result.append("LOW")
            else:
                result.append("NONE")
        return result


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

class RandomBaseline(_BaselineClassifier):
    """Samples predictions uniformly at random from the class vocabulary.

    This provides the weakest possible baseline: a classifier with
    no information about the input at all.

    Example::

        baseline = RandomBaseline(seed=42)
        baseline.fit(["HIGH", "NONE", "MEDIUM"])  # learns class vocab only
        preds = baseline.predict(5)
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialise with a fixed random seed.

        Args:
            seed: RNG seed for reproducibility.
        """
        self._seed = seed
        self._rng = random.Random(seed)
        self._classes: list[str] = list(SEVERITY_LEVELS)

    @property
    def name(self) -> str:
        return f"RandomBaseline(seed={self._seed})"

    def fit(self, y_true: Sequence[str]) -> "RandomBaseline":
        """Record the unique class labels from training data.

        Args:
            y_true: Ground-truth labels.  Only the set of unique labels
                is retained.

        Returns:
            self
        """
        unique = sorted(set(y_true), key=lambda s: SEVERITY_LEVELS.index(s) if s in SEVERITY_LEVELS else 99)
        self._classes = unique if unique else list(SEVERITY_LEVELS)
        return self

    def predict(self, n_samples: int) -> list[str]:
        """Sample uniformly at random from the class vocabulary.

        Args:
            n_samples: Number of predictions to produce.

        Returns:
            List of randomly selected severity labels.
        """
        return [self._rng.choice(self._classes) for _ in range(n_samples)]


# ---------------------------------------------------------------------------
# Stratified random baseline
# ---------------------------------------------------------------------------

class StratifiedRandomBaseline(_BaselineClassifier):
    """Samples predictions from the empirical class-frequency distribution.

    Unlike RandomBaseline (uniform), this preserves the class imbalance
    of the training set.  A classifier should beat this baseline to
    demonstrate it has learned something beyond memorising class priors.

    Example::

        baseline = StratifiedRandomBaseline(seed=42)
        baseline.fit(["NONE", "NONE", "HIGH", "NONE"])
        preds = baseline.predict(100)
        # ~75% "NONE", ~25% "HIGH"
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialise with a fixed random seed.

        Args:
            seed: RNG seed for reproducibility.
        """
        self._seed = seed
        self._rng = random.Random(seed)
        self._classes: list[str] = list(SEVERITY_LEVELS)
        self._weights: list[float] = [1.0 / len(SEVERITY_LEVELS)] * len(SEVERITY_LEVELS)

    @property
    def name(self) -> str:
        return f"StratifiedRandomBaseline(seed={self._seed})"

    def fit(self, y_true: Sequence[str]) -> "StratifiedRandomBaseline":
        """Compute empirical class frequencies from training labels.

        Args:
            y_true: Ground-truth severity labels.

        Returns:
            self
        """
        if not y_true:
            self._classes = list(SEVERITY_LEVELS)
            self._weights = [1.0 / len(SEVERITY_LEVELS)] * len(SEVERITY_LEVELS)
            return self

        counts = Counter(y_true)
        total = sum(counts.values())
        self._classes = list(counts.keys())
        self._weights = [counts[c] / total for c in self._classes]
        return self

    def predict(self, n_samples: int) -> list[str]:
        """Sample from the empirical class distribution.

        Uses Python's ``random.choices`` (with replacement) weighted by
        the class frequencies observed in ``fit``.

        Args:
            n_samples: Number of predictions to produce.

        Returns:
            List of severity labels drawn from the class prior.
        """
        return self._rng.choices(self._classes, weights=self._weights, k=n_samples)

    @property
    def class_distribution(self) -> dict[str, float]:
        """Class-label -> empirical frequency mapping."""
        return dict(zip(self._classes, self._weights))


# ---------------------------------------------------------------------------
# Model vs. baseline comparison
# ---------------------------------------------------------------------------

def compare_with_baseline(
    y_true: Sequence[str],
    y_pred_model: Sequence[str],
    baseline: _BaselineClassifier,
    alpha: float = 0.05,
) -> ModelComparison:
    """Compare a model against a baseline using McNemar's test.

    The baseline is fitted on the same ground-truth labels and asked to
    produce one prediction per sample (``baseline.predict(n_samples)``).
    For ``DangerScoreBaseline`` you should call ``baseline.predict(scores)``
    directly and pass the result as ``y_pred_model`` instead.

    Args:
        y_true: Ground-truth severity labels.
        y_pred_model: Model predictions (same length as y_true).
        baseline: A fitted baseline classifier instance.  Will be
            fitted on y_true before generating predictions.
        alpha: Significance threshold (default 0.05).

    Returns:
        ModelComparison with chi2 statistic, p-value, significance flag,
        effect size (odds ratio), and n_discordant pairs.

    Raises:
        ValueError: If y_true and y_pred_model have different lengths.
    """
    n = len(y_true)
    if len(y_pred_model) != n:
        raise ValueError(
            f"y_true has {n} samples but y_pred_model has {len(y_pred_model)}."
        )

    # Fit baseline and generate predictions
    baseline.fit(y_true)
    y_pred_baseline = baseline.predict(n)

    comparison = mcnemar_test(y_true, y_pred_model, y_pred_baseline)

    # Override significance using the caller's alpha (mcnemar_test uses 0.05 internally)
    comparison.significant = comparison.p_value < alpha

    return comparison
