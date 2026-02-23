"""Adaptive Conformal Inference (ACI) with coverage monitoring.

Extends split conformal prediction with:
- Adaptive quantile adjustment (Gibbs & Candès 2021)
- Online coverage monitoring with CUSUM change detection
- Mondrian conformal for group-conditional coverage
- Synthetic calibration dataset generation
- Coverage guarantee verification

References:
  Gibbs & Candès (2021) "Adaptive Conformal Inference Under Distribution Shift"
  Vovk et al. (2005) "Algorithmic Learning in a Random World"
  Barber et al. (2023) "Conformal Prediction Beyond Exchangeability"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH"]


# ---------------------------------------------------------------------------
# Calibration data generation
# ---------------------------------------------------------------------------

@dataclass
class CalibrationSample:
    """A single calibration sample with scores and true label."""
    scores: np.ndarray  # (n_classes,) softmax probabilities
    true_label: int      # 0-3 severity index


def generate_calibration_dataset(
    n_samples: int = 500,
    n_classes: int = 4,
    class_priors: np.ndarray | None = None,
    model_accuracy: float = 0.75,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic calibration dataset for conformal prediction.

    Simulates a model with specified accuracy by generating softmax-like
    scores where the true class has elevated probability.

    Args:
        n_samples: Number of calibration samples.
        n_classes: Number of classes.
        class_priors: Prior probabilities per class (uniform if None).
        model_accuracy: Approximate accuracy of the simulated model.
        seed: Random seed.

    Returns:
        Tuple of (scores, y_true) where scores is (n, n_classes) and
        y_true is (n,) integer labels.
    """
    rng = np.random.RandomState(seed)

    if class_priors is None:
        class_priors = np.ones(n_classes) / n_classes

    y_true = rng.choice(n_classes, size=n_samples, p=class_priors)
    scores = np.zeros((n_samples, n_classes))

    for i in range(n_samples):
        # Generate Dirichlet noise
        raw = rng.dirichlet(np.ones(n_classes) * 0.5)
        # Boost true class to achieve target accuracy
        boost = model_accuracy * 2
        raw[y_true[i]] += boost
        # Normalize to valid probabilities
        scores[i] = raw / raw.sum()

    return scores, y_true


# ---------------------------------------------------------------------------
# Adaptive Conformal Inference
# ---------------------------------------------------------------------------

@dataclass
class ACIState:
    """State for online adaptive conformal inference."""
    alpha_target: float = 0.1
    alpha_t: float = 0.1  # Current adaptive alpha
    gamma: float = 0.01   # Learning rate for alpha adjustment
    t: int = 0            # Time step
    coverage_history: list[float] = field(default_factory=list)
    alpha_history: list[float] = field(default_factory=list)
    set_size_history: list[int] = field(default_factory=list)


class AdaptiveConformal:
    """Adaptive Conformal Inference (ACI).

    Unlike static split conformal, ACI adjusts the quantile threshold
    online to maintain target coverage under distribution shift.

    The update rule (Gibbs & Candès 2021):
        alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
    where err_t = 1 if true label not in prediction set, 0 otherwise.

    This gives a long-run coverage guarantee even under covariate shift.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.01):
        """Initialize ACI.

        Args:
            alpha: Target miscoverage rate (0.1 = 90% coverage).
            gamma: Learning rate for adaptive adjustment. Larger gamma
                   reacts faster to distribution shift but is noisier.
        """
        self.state = ACIState(alpha_target=alpha, alpha_t=alpha, gamma=gamma)
        self._calibration_scores: np.ndarray | None = None
        self._n_classes = 4
        self._labels = SEVERITY_LEVELS

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> None:
        """Calibrate on held-out calibration set.

        Args:
            scores: (n_calib, n_classes) softmax probabilities.
            y_true: (n_calib,) true labels (integer indices).
        """
        n = len(scores)
        self._n_classes = scores.shape[1]

        # Compute non-conformity scores: 1 - P(true class)
        nc_scores = np.array([1.0 - scores[i, y_true[i]] for i in range(n)])
        self._calibration_scores = np.sort(nc_scores)
        logger.info("ACI calibrated on %d samples", n)

    def predict_set(self, scores: np.ndarray) -> list[set[str]]:
        """Predict conformal sets using current adaptive alpha.

        Args:
            scores: (n_test, n_classes) softmax probabilities.

        Returns:
            List of prediction sets.
        """
        if self._calibration_scores is None:
            raise RuntimeError("Not calibrated. Call fit() first.")

        quantile = self._compute_quantile(self.state.alpha_t)
        prediction_sets = []

        for score_vec in scores:
            pred_set = set()
            for j in range(self._n_classes):
                if (1.0 - score_vec[j]) <= quantile:
                    pred_set.add(self._labels[j])
            if not pred_set:
                pred_set.add(self._labels[int(np.argmax(score_vec))])
            prediction_sets.append(pred_set)

        return prediction_sets

    def predict_set_single(self, scores: np.ndarray) -> set[str]:
        """Predict conformal set for a single instance."""
        return self.predict_set(scores.reshape(1, -1))[0]

    def update(self, true_label: int, prediction_set: set[str]) -> None:
        """Online update of alpha after observing true label.

        Implements the Gibbs & Candès (2021) update rule.

        Args:
            true_label: True severity index.
            prediction_set: The prediction set that was output.
        """
        covered = self._labels[true_label] in prediction_set
        err_t = 0.0 if covered else 1.0

        # Adaptive update
        self.state.alpha_t = self.state.alpha_t + self.state.gamma * (self.state.alpha_target - err_t)
        # Clamp to valid range
        self.state.alpha_t = np.clip(self.state.alpha_t, 0.001, 0.999)

        self.state.t += 1
        self.state.coverage_history.append(1.0 if covered else 0.0)
        self.state.alpha_history.append(self.state.alpha_t)
        self.state.set_size_history.append(len(prediction_set))

    def _compute_quantile(self, alpha: float) -> float:
        """Compute quantile from calibration scores."""
        n = len(self._calibration_scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = np.clip(q_level, 0, 1)
        return float(np.quantile(self._calibration_scores, q_level))

    @property
    def running_coverage(self) -> float:
        """Compute running empirical coverage."""
        if not self.state.coverage_history:
            return 0.0
        return float(np.mean(self.state.coverage_history))

    @property
    def mean_set_size(self) -> float:
        """Mean prediction set size (indicator of informativeness)."""
        if not self.state.set_size_history:
            return 0.0
        return float(np.mean(self.state.set_size_history))


# ---------------------------------------------------------------------------
# Mondrian Conformal (group-conditional coverage)
# ---------------------------------------------------------------------------

class MondrianConformal:
    """Mondrian conformal prediction for group-conditional coverage.

    Provides separate coverage guarantees per group (e.g., per severity
    class, per video source). This prevents the marginal coverage guarantee
    from hiding poor coverage on minority classes.

    Example: If HIGH severity has only 5% of data, standard conformal
    might achieve 90% overall but only 60% on HIGH. Mondrian guarantees
    90% on EACH group.
    """

    def __init__(self, alpha: float = 0.1, labels: list[str] | None = None):
        self.alpha = alpha
        self.labels = labels or SEVERITY_LEVELS
        self._group_quantiles: dict[int, float] = {}
        self._calibrated = False

    def fit(self, scores: np.ndarray, y_true: np.ndarray, groups: np.ndarray | None = None) -> None:
        """Calibrate per-group quantiles.

        Args:
            scores: (n, n_classes) softmax probabilities.
            y_true: (n,) true labels.
            groups: (n,) group assignments. If None, uses y_true as groups
                    (severity-conditional coverage).
        """
        if groups is None:
            groups = y_true

        unique_groups = np.unique(groups)
        for g in unique_groups:
            mask = groups == g
            group_scores = scores[mask]
            group_labels = y_true[mask]
            n_g = len(group_scores)

            if n_g < 2:
                # Not enough data for calibration; use global quantile
                nc = np.array([1.0 - group_scores[i, group_labels[i]] for i in range(n_g)])
                self._group_quantiles[int(g)] = float(nc.max()) if len(nc) > 0 else 1.0
                continue

            nc_scores = np.array([1.0 - group_scores[i, group_labels[i]] for i in range(n_g)])
            q_level = np.ceil((n_g + 1) * (1 - self.alpha)) / n_g
            q_level = np.clip(q_level, 0, 1)
            self._group_quantiles[int(g)] = float(np.quantile(nc_scores, q_level))

        self._calibrated = True
        logger.info("Mondrian conformal calibrated on %d groups", len(unique_groups))

    def predict_set(self, scores: np.ndarray, groups: np.ndarray | None = None) -> list[set[str]]:
        """Predict with group-conditional quantiles.

        Args:
            scores: (n, n_classes) probabilities.
            groups: (n,) group assignments. If None, uses argmax of scores.

        Returns:
            List of prediction sets with per-group coverage guarantees.
        """
        if not self._calibrated:
            raise RuntimeError("Not calibrated. Call fit() first.")

        n_classes = scores.shape[1]
        if groups is None:
            groups = np.argmax(scores, axis=1)

        prediction_sets = []
        for i, score_vec in enumerate(scores):
            g = int(groups[i])
            quantile = self._group_quantiles.get(g, max(self._group_quantiles.values()))

            pred_set = set()
            for j in range(n_classes):
                if (1.0 - score_vec[j]) <= quantile:
                    pred_set.add(self.labels[j])
            if not pred_set:
                pred_set.add(self.labels[int(np.argmax(score_vec))])
            prediction_sets.append(pred_set)

        return prediction_sets

    def compute_group_coverage(
        self, scores: np.ndarray, y_true: np.ndarray, groups: np.ndarray | None = None
    ) -> dict[str, float]:
        """Compute per-group empirical coverage.

        Returns:
            Dict mapping group label to coverage rate.
        """
        if groups is None:
            groups = y_true

        pred_sets = self.predict_set(scores, groups)
        unique_groups = np.unique(groups)
        coverages = {}

        for g in unique_groups:
            mask = groups == g
            covered = sum(
                self.labels[y_true[i]] in pred_sets[i]
                for i in range(len(y_true)) if mask[i]
            )
            total = int(mask.sum())
            label = self.labels[int(g)] if int(g) < len(self.labels) else str(g)
            coverages[label] = covered / total if total > 0 else 0.0

        return coverages


# ---------------------------------------------------------------------------
# Coverage monitoring (CUSUM change detection)
# ---------------------------------------------------------------------------

@dataclass
class CoverageMonitor:
    """Online coverage monitoring with CUSUM change detection.

    Detects when empirical coverage deviates from the target, signaling
    potential distribution shift or model degradation.

    The CUSUM (Cumulative Sum) statistic accumulates deviations from
    target coverage. An alarm fires when the statistic exceeds threshold h.
    """
    target_coverage: float = 0.9
    h: float = 5.0        # CUSUM alarm threshold
    k: float = 0.02       # CUSUM allowance (sensitivity parameter)
    cusum_pos: float = 0.0  # Upper CUSUM (detects coverage drops)
    cusum_neg: float = 0.0  # Lower CUSUM (detects coverage excess)
    n_observations: int = 0
    n_covered: int = 0
    alarm_triggered: bool = False
    alarm_history: list[int] = field(default_factory=list)

    def update(self, covered: bool) -> bool:
        """Update monitor with new observation.

        Args:
            covered: Whether the true label was in the prediction set.

        Returns:
            True if alarm is triggered (coverage violation detected).
        """
        self.n_observations += 1
        if covered:
            self.n_covered += 1

        # CUSUM update
        x = 1.0 if covered else 0.0
        deviation = self.target_coverage - x

        self.cusum_pos = max(0.0, self.cusum_pos + deviation - self.k)
        self.cusum_neg = max(0.0, self.cusum_neg - deviation - self.k)

        if self.cusum_pos > self.h or self.cusum_neg > self.h:
            self.alarm_triggered = True
            self.alarm_history.append(self.n_observations)
            # Reset after alarm
            self.cusum_pos = 0.0
            self.cusum_neg = 0.0
            logger.warning(
                "Coverage alarm at t=%d: empirical=%.3f target=%.3f",
                self.n_observations, self.empirical_coverage, self.target_coverage,
            )
            return True

        return False

    @property
    def empirical_coverage(self) -> float:
        """Current empirical coverage rate."""
        return self.n_covered / self.n_observations if self.n_observations > 0 else 0.0

    def should_recalibrate(self) -> bool:
        """Whether recalibration is recommended based on monitoring."""
        if self.alarm_triggered:
            return True
        # Also recommend if coverage is significantly off after enough observations
        if self.n_observations >= 50:
            gap = abs(self.empirical_coverage - self.target_coverage)
            return gap > 0.05  # 5% deviation threshold
        return False
