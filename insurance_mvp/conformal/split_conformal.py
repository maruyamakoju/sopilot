"""Split Conformal Prediction for Insurance Risk Assessment

Adapted from SOPilot's nn/conformal.py for insurance video review use case.

Reference: Vovk et al. (2005) "Algorithmic Learning in a Random World"
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalConfig:
    """Configuration for conformal prediction"""

    alpha: float = 0.1  # Miscoverage rate (10% = 90% confidence)
    severity_levels: list[str] = None  # ["NONE", "LOW", "MEDIUM", "HIGH"]

    def __post_init__(self):
        if self.severity_levels is None:
            self.severity_levels = ["NONE", "LOW", "MEDIUM", "HIGH"]


class SplitConformal:
    """
    Split Conformal Prediction for ordinal severity classification.

    Use case: Given AI's point prediction "HIGH", output prediction set
    {"MEDIUM", "HIGH"} with 90% confidence coverage.

    Benefits for insurance review:
    - Quantifies uncertainty (wide set = uncertain = human review priority)
    - Guarantees coverage (90% confidence that true severity is in set)
    - Model-agnostic (works with any AI model)
    """

    def __init__(self, config: ConformalConfig = None):
        self.config = config or ConformalConfig()
        self.quantiles = None  # Calibrated quantiles for each severity level
        self._calibrated = False

    def fit(self, scores: np.ndarray, y_true: np.ndarray):
        """
        Calibrate conformal predictor on calibration set.

        Args:
            scores: (n_calib, n_classes) - Model output scores (e.g., softmax probabilities)
            y_true: (n_calib,) - True severity labels (ordinal: 0=NONE, 1=LOW, 2=MEDIUM, 3=HIGH)
        """
        n_calib = len(scores)

        # Compute non-conformity scores (1 - score_of_true_class)
        non_conformity_scores = []
        for i in range(n_calib):
            true_label = int(y_true[i])
            score_true = scores[i, true_label]
            non_conformity = 1.0 - score_true
            non_conformity_scores.append(non_conformity)

        non_conformity_scores = np.array(non_conformity_scores)

        # Compute quantile for conformal prediction
        q_level = np.ceil((n_calib + 1) * (1 - self.config.alpha)) / n_calib
        self.quantile = np.quantile(non_conformity_scores, q_level)

        self._calibrated = True

    def predict_set(self, scores: np.ndarray) -> list[set[str]]:
        """
        Predict conformal prediction sets.

        Args:
            scores: (n_test, n_classes) - Model output scores

        Returns:
            List of sets, e.g., [{"MEDIUM", "HIGH"}, {"LOW", "MEDIUM"}, ...]
        """
        if not self._calibrated:
            raise RuntimeError("Conformal predictor not calibrated. Call fit() first.")

        prediction_sets = []

        for score_vec in scores:
            # Include label if (1 - score) <= quantile
            pred_set = set()
            for label_idx, score in enumerate(score_vec):
                non_conformity = 1.0 - score
                if non_conformity <= self.quantile:
                    pred_set.add(self.config.severity_levels[label_idx])

            # Ensure non-empty set (add highest score if empty)
            if len(pred_set) == 0:
                max_idx = np.argmax(score_vec)
                pred_set.add(self.config.severity_levels[max_idx])

            prediction_sets.append(pred_set)

        return prediction_sets

    def predict_set_single(self, scores: np.ndarray) -> set[str]:
        """Predict conformal set for a single instance"""
        return self.predict_set(scores.reshape(1, -1))[0]

    def compute_coverage(self, scores: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute empirical coverage rate on test set.

        Ideal: coverage ≈ (1 - alpha). E.g., alpha=0.1 → coverage ≈ 90%
        """
        prediction_sets = self.predict_set(scores)

        n_test = len(y_true)
        n_covered = 0

        for pred_set, true_label_idx in zip(prediction_sets, y_true, strict=False):
            true_label = self.config.severity_levels[int(true_label_idx)]
            if true_label in pred_set:
                n_covered += 1

        coverage = n_covered / n_test
        return coverage

    def compute_set_sizes(self, scores: np.ndarray) -> np.ndarray:
        """Compute prediction set sizes (indicator of uncertainty)"""
        prediction_sets = self.predict_set(scores)
        set_sizes = np.array([len(s) for s in prediction_sets])
        return set_sizes


# Convenience functions for insurance use case


def severity_to_ordinal(severity: str) -> int:
    """Convert severity string to ordinal integer"""
    mapping = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    return mapping.get(severity.upper(), 0)


def ordinal_to_severity(ordinal: int) -> str:
    """Convert ordinal integer to severity string"""
    mapping = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}
    return mapping.get(ordinal, "NONE")


def compute_review_priority(severity: str, prediction_set: set[str]) -> str:
    """
    Compute human review priority based on severity and uncertainty.

    Logic:
    - URGENT: High severity × high uncertainty (wide prediction set)
    - STANDARD: Medium severity or low uncertainty
    - LOW_PRIORITY: Low severity × high certainty
    """
    severity_upper = severity.upper()
    set_size = len(prediction_set)

    # Urgent: HIGH severity with uncertainty (set size >= 2)
    if severity_upper == "HIGH" and set_size >= 2:
        return "URGENT"

    # Urgent: MEDIUM with high uncertainty (set size >= 3)
    if severity_upper == "MEDIUM" and set_size >= 3:
        return "URGENT"

    # Standard: MEDIUM or HIGH with certainty
    if severity_upper in ["MEDIUM", "HIGH"]:
        return "STANDARD"

    # Low priority: LOW or NONE
    return "LOW_PRIORITY"
