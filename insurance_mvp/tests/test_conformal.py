"""Tests for Split Conformal Prediction module.

Covers: SplitConformal calibration/prediction, convenience functions,
and compute_review_priority logic.
"""

import numpy as np
import pytest
from insurance_mvp.conformal.split_conformal import (
    ConformalConfig,
    SplitConformal,
    compute_review_priority,
    ordinal_to_severity,
    severity_to_ordinal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_calibration(n: int = 100, noise: float = 0.1, rng_seed: int = 42):
    """Generate synthetic calibration data.

    Returns (scores, y_true) where scores are softmax-like probabilities
    concentrated around the true label with added noise.
    """
    rng = np.random.RandomState(rng_seed)
    y_true = rng.randint(0, 4, size=n)
    scores = np.full((n, 4), noise / 3)
    for i in range(n):
        scores[i, y_true[i]] = 1.0 - noise
    # Add small noise and re-normalize
    scores += rng.uniform(0, 0.02, size=scores.shape)
    scores = scores / scores.sum(axis=1, keepdims=True)
    return scores, y_true


# ============================================================================
# TestSplitConformal
# ============================================================================


class TestSplitConformal:
    """Core SplitConformal calibration and prediction tests."""

    def test_fit_basic(self):
        """Calibration succeeds on synthetic scores."""
        sc = SplitConformal()
        scores, y_true = _make_synthetic_calibration()
        sc.fit(scores, y_true)
        assert sc._calibrated is True
        assert sc.quantile is not None

    def test_predict_set_uncalibrated(self):
        """RuntimeError raised before fit()."""
        sc = SplitConformal()
        scores = np.array([[0.1, 0.2, 0.3, 0.4]])
        with pytest.raises(RuntimeError, match="not calibrated"):
            sc.predict_set(scores)

    def test_predict_set_confident(self):
        """High-confidence model yields singleton prediction set."""
        sc = SplitConformal(ConformalConfig(alpha=0.1))
        # Near-perfect calibration data
        scores, y_true = _make_synthetic_calibration(n=200, noise=0.02)
        sc.fit(scores, y_true)

        # Test point with very high confidence for HIGH (index 3)
        test_scores = np.array([[0.01, 0.01, 0.01, 0.97]])
        pred_sets = sc.predict_set(test_scores)
        assert len(pred_sets) == 1
        assert "HIGH" in pred_sets[0]
        assert len(pred_sets[0]) <= 2  # Should be singleton or very small

    def test_predict_set_uncertain(self):
        """Low-confidence model yields at least a prediction set."""
        sc = SplitConformal(ConformalConfig(alpha=0.1))
        # Use noisy calibration so quantile is not too tight
        scores, y_true = _make_synthetic_calibration(n=200, noise=0.5)
        sc.fit(scores, y_true)

        # Uniform-ish scores → uncertain
        test_scores = np.array([[0.25, 0.25, 0.25, 0.25]])
        pred_sets = sc.predict_set(test_scores)
        # With noisy calibration and uniform input, set should be wider
        assert len(pred_sets[0]) >= 1  # At minimum, fallback guarantees 1

    def test_predict_set_empty_fallback(self):
        """All scores very low → argmax singleton fallback."""
        sc = SplitConformal(ConformalConfig(alpha=0.01))  # Very strict
        # Calibrate with enough data so quantile computation stays in [0,1]
        scores, y_true = _make_synthetic_calibration(n=200, noise=0.8)
        sc.fit(scores, y_true)

        # Very low scores - none should pass threshold
        test_scores = np.array([[0.01, 0.02, 0.03, 0.04]])
        pred_sets = sc.predict_set(test_scores)
        # Fallback guarantees at least 1 element
        assert len(pred_sets[0]) >= 1
        assert "HIGH" in pred_sets[0]  # argmax of [0.01, 0.02, 0.03, 0.04] = index 3

    def test_predict_set_single(self):
        """Single-instance convenience method returns a set."""
        sc = SplitConformal()
        scores, y_true = _make_synthetic_calibration()
        sc.fit(scores, y_true)

        single_scores = np.array([0.1, 0.2, 0.3, 0.4])
        result = sc.predict_set_single(single_scores)
        assert isinstance(result, set)
        assert len(result) >= 1

    def test_compute_coverage(self):
        """Empirical coverage approximately equals 1-alpha on synthetic data."""
        alpha = 0.1
        sc = SplitConformal(ConformalConfig(alpha=alpha))

        # Calibrate on first half, test on second half
        scores, y_true = _make_synthetic_calibration(n=400, noise=0.15, rng_seed=123)
        sc.fit(scores[:200], y_true[:200])

        coverage = sc.compute_coverage(scores[200:], y_true[200:])
        # Coverage should be approximately 1 - alpha = 0.9
        assert coverage >= 0.80  # Allow some slack for finite sample

    def test_compute_coverage_perfect(self):
        """Perfect model → coverage = 1.0."""
        sc = SplitConformal(ConformalConfig(alpha=0.1))
        # Near-perfect scores
        n = 100
        y_true = np.array([i % 4 for i in range(n)])
        scores = np.full((n, 4), 0.01)
        for i in range(n):
            scores[i, y_true[i]] = 0.97

        sc.fit(scores[:50], y_true[:50])
        coverage = sc.compute_coverage(scores[50:], y_true[50:])
        assert coverage == 1.0

    def test_compute_set_sizes(self):
        """compute_set_sizes returns ndarray of correct length."""
        sc = SplitConformal()
        scores, y_true = _make_synthetic_calibration(n=100)
        sc.fit(scores, y_true)

        test_scores = np.random.dirichlet(np.ones(4), size=20)
        sizes = sc.compute_set_sizes(test_scores)
        assert isinstance(sizes, np.ndarray)
        assert len(sizes) == 20
        assert all(s >= 1 for s in sizes)

    def test_alpha_sensitivity(self):
        """alpha=0.01 yields wider sets than alpha=0.5."""
        scores, y_true = _make_synthetic_calibration(n=200, noise=0.3, rng_seed=99)

        sc_strict = SplitConformal(ConformalConfig(alpha=0.01))
        sc_strict.fit(scores, y_true)

        sc_loose = SplitConformal(ConformalConfig(alpha=0.5))
        sc_loose.fit(scores, y_true)

        test_scores = np.random.dirichlet(np.ones(4), size=50)
        sizes_strict = sc_strict.compute_set_sizes(test_scores)
        sizes_loose = sc_loose.compute_set_sizes(test_scores)

        assert np.mean(sizes_strict) >= np.mean(sizes_loose)

    def test_large_calibration_set(self):
        """n=1000 calibration completes and produces stable quantile."""
        sc = SplitConformal()
        scores, y_true = _make_synthetic_calibration(n=1000, noise=0.2)
        sc.fit(scores, y_true)

        assert sc._calibrated
        assert 0.0 <= sc.quantile <= 1.0


# ============================================================================
# TestConvenienceFunctions
# ============================================================================


class TestConvenienceFunctions:
    """Test severity_to_ordinal and ordinal_to_severity."""

    @pytest.mark.parametrize(
        "severity,expected",
        [
            ("NONE", 0),
            ("LOW", 1),
            ("MEDIUM", 2),
            ("HIGH", 3),
        ],
    )
    def test_severity_to_ordinal(self, severity, expected):
        assert severity_to_ordinal(severity) == expected

    def test_severity_to_ordinal_unknown(self):
        """Unknown severity maps to 0."""
        assert severity_to_ordinal("UNKNOWN") == 0
        assert severity_to_ordinal("CRITICAL") == 0

    @pytest.mark.parametrize(
        "ordinal,expected",
        [
            (0, "NONE"),
            (1, "LOW"),
            (2, "MEDIUM"),
            (3, "HIGH"),
        ],
    )
    def test_ordinal_to_severity(self, ordinal, expected):
        assert ordinal_to_severity(ordinal) == expected

    def test_ordinal_to_severity_unknown(self):
        """Unknown ordinal maps to NONE."""
        assert ordinal_to_severity(5) == "NONE"
        assert ordinal_to_severity(-1) == "NONE"

    @pytest.mark.parametrize("severity", ["NONE", "LOW", "MEDIUM", "HIGH"])
    def test_severity_ordinal_roundtrip(self, severity):
        """Encode then decode yields original value."""
        ordinal = severity_to_ordinal(severity)
        result = ordinal_to_severity(ordinal)
        assert result == severity


# ============================================================================
# TestComputeReviewPriority
# ============================================================================


class TestComputeReviewPriority:
    """Test review priority computation logic."""

    def test_high_uncertain_urgent(self):
        """HIGH + set_size>=2 → URGENT."""
        result = compute_review_priority("HIGH", {"HIGH", "MEDIUM"})
        assert result == "URGENT"

    def test_medium_very_uncertain_urgent(self):
        """MEDIUM + set_size>=3 → URGENT."""
        result = compute_review_priority("MEDIUM", {"LOW", "MEDIUM", "HIGH"})
        assert result == "URGENT"

    def test_high_certain_standard(self):
        """HIGH + set_size=1 → STANDARD."""
        result = compute_review_priority("HIGH", {"HIGH"})
        assert result == "STANDARD"

    def test_medium_standard(self):
        """MEDIUM + set_size<=2 → STANDARD."""
        result = compute_review_priority("MEDIUM", {"MEDIUM"})
        assert result == "STANDARD"

    def test_medium_two_standard(self):
        """MEDIUM + set_size=2 → STANDARD (not URGENT)."""
        result = compute_review_priority("MEDIUM", {"MEDIUM", "HIGH"})
        assert result == "STANDARD"

    def test_low_low_priority(self):
        """LOW → LOW_PRIORITY."""
        result = compute_review_priority("LOW", {"LOW"})
        assert result == "LOW_PRIORITY"

    def test_none_low_priority(self):
        """NONE → LOW_PRIORITY."""
        result = compute_review_priority("NONE", {"NONE"})
        assert result == "LOW_PRIORITY"

    def test_case_insensitive(self):
        """'high' and 'HIGH' yield same result."""
        result_lower = compute_review_priority("high", {"HIGH", "MEDIUM"})
        result_upper = compute_review_priority("HIGH", {"HIGH", "MEDIUM"})
        assert result_lower == result_upper == "URGENT"

    def test_high_wide_set_urgent(self):
        """HIGH + set_size=4 → URGENT."""
        result = compute_review_priority("HIGH", {"NONE", "LOW", "MEDIUM", "HIGH"})
        assert result == "URGENT"
