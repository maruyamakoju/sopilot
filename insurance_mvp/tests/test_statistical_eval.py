"""Comprehensive tests for insurance_mvp.evaluation.statistical module.

Covers bootstrap CI, accuracy, kappa, F1, Cramer's V, confusion matrix,
full evaluate(), McNemar's test, format_report, and edge cases.
"""

import numpy as np
import pytest

from insurance_mvp.evaluation.statistical import (
    ConfidenceInterval,
    EvaluationReport,
    ModelComparison,
    accuracy_score,
    bootstrap_ci,
    cohen_kappa,
    confusion_matrix,
    cramers_v,
    evaluate,
    format_report,
    macro_f1,
    mcnemar_test,
    weighted_f1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_perfect(n: int = 100, n_classes: int = 4, seed: int = 0):
    """Generate perfectly matching y_true, y_pred arrays."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, n_classes, size=n)
    return y.copy(), y.copy()


def _make_random(n: int = 200, n_classes: int = 4, seed: int = 7):
    """Generate random (uncorrelated) y_true, y_pred arrays."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_classes, size=n), rng.randint(0, n_classes, size=n)


# ---------------------------------------------------------------------------
# 1. bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    """Tests for bootstrap_ci."""

    def test_ci_contains_point_estimate(self):
        """Point estimate should lie within [lower, upper]."""
        yt, yp = _make_perfect(50)
        ci = bootstrap_ci(yt, yp, accuracy_score, n_bootstrap=500, seed=99)
        assert ci.lower <= ci.point <= ci.upper

    def test_ci_bounds_ordered(self):
        """Lower bound must be <= upper bound."""
        yt, yp = _make_random(100)
        ci = bootstrap_ci(yt, yp, accuracy_score, n_bootstrap=500, seed=42)
        assert ci.lower <= ci.upper

    def test_ci_alpha_stored(self):
        """alpha parameter should be stored in the result."""
        yt, yp = _make_random(80)
        ci = bootstrap_ci(yt, yp, accuracy_score, n_bootstrap=200, alpha=0.10, seed=1)
        assert ci.alpha == 0.10

    def test_ci_method_is_bca(self):
        """Method should be 'bca'."""
        yt, yp = _make_random(50)
        ci = bootstrap_ci(yt, yp, accuracy_score, n_bootstrap=200, seed=0)
        assert ci.method == "bca"

    def test_perfect_predictions_tight_ci(self):
        """Perfect predictions should yield a tight CI near 1.0."""
        yt, yp = _make_perfect(100)
        ci = bootstrap_ci(yt, yp, accuracy_score, n_bootstrap=1000, seed=42)
        assert ci.point == 1.0
        assert ci.lower >= 0.99

    def test_reproducibility_with_same_seed(self):
        """Same seed should produce identical CI."""
        yt, yp = _make_random(100)
        ci1 = bootstrap_ci(yt, yp, accuracy_score, n_bootstrap=500, seed=123)
        ci2 = bootstrap_ci(yt, yp, accuracy_score, n_bootstrap=500, seed=123)
        assert ci1.point == ci2.point
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper

    def test_wider_ci_with_higher_alpha(self):
        """A 99% CI (alpha=0.01) should be at least as wide as 90% CI (alpha=0.10)."""
        yt, yp = _make_random(150)
        ci_narrow = bootstrap_ci(yt, yp, accuracy_score, n_bootstrap=2000, alpha=0.10, seed=42)
        ci_wide = bootstrap_ci(yt, yp, accuracy_score, n_bootstrap=2000, alpha=0.01, seed=42)
        width_narrow = ci_narrow.upper - ci_narrow.lower
        width_wide = ci_wide.upper - ci_wide.lower
        assert width_wide >= width_narrow - 0.01  # small tolerance for bootstrap variance


# ---------------------------------------------------------------------------
# 2. accuracy_score
# ---------------------------------------------------------------------------

class TestAccuracyScore:
    """Tests for accuracy_score."""

    def test_perfect(self):
        y = np.array([0, 1, 2, 3])
        assert accuracy_score(y, y) == 1.0

    def test_none_correct(self):
        yt = np.array([0, 0, 0, 0])
        yp = np.array([1, 1, 1, 1])
        assert accuracy_score(yt, yp) == 0.0

    def test_half_correct(self):
        yt = np.array([0, 1, 0, 1])
        yp = np.array([0, 1, 1, 0])
        assert accuracy_score(yt, yp) == 0.5

    def test_known_value(self):
        yt = np.array([0, 1, 2, 3, 0])
        yp = np.array([0, 1, 2, 0, 0])
        assert accuracy_score(yt, yp) == pytest.approx(4 / 5)


# ---------------------------------------------------------------------------
# 3. cohen_kappa
# ---------------------------------------------------------------------------

class TestCohenKappa:
    """Tests for cohen_kappa."""

    def test_perfect_agreement(self):
        yt, yp = _make_perfect(100)
        assert cohen_kappa(yt, yp) == pytest.approx(1.0)

    def test_random_agreement_near_zero(self):
        """Random predictions should yield kappa near 0."""
        yt, yp = _make_random(1000, seed=42)
        k = cohen_kappa(yt, yp)
        assert -0.2 < k < 0.2

    def test_systematic_disagreement_negative(self):
        """Systematic swaps should yield negative kappa."""
        # All class 0 predicted as 1, all class 1 predicted as 0
        yt = np.array([0, 0, 0, 1, 1, 1])
        yp = np.array([1, 1, 1, 0, 0, 0])
        k = cohen_kappa(yt, yp)
        assert k < 0

    def test_empty_arrays(self):
        """Empty arrays should return 0.0."""
        yt = np.array([], dtype=int)
        yp = np.array([], dtype=int)
        assert cohen_kappa(yt, yp) == 0.0


# ---------------------------------------------------------------------------
# 4. macro_f1 and weighted_f1
# ---------------------------------------------------------------------------

class TestF1Scores:
    """Tests for macro_f1 and weighted_f1."""

    def test_macro_f1_perfect(self):
        yt, yp = _make_perfect(100)
        assert macro_f1(yt, yp) == pytest.approx(1.0)

    def test_weighted_f1_perfect(self):
        yt, yp = _make_perfect(100)
        assert weighted_f1(yt, yp) == pytest.approx(1.0)

    def test_macro_f1_known_binary(self):
        """Binary classification with known counts."""
        # class 0: TP=3, FP=1, FN=1 -> P=3/4, R=3/4, F1=3/4
        # class 1: TP=2, FP=1, FN=1 -> P=2/3, R=2/3, F1=2/3
        yt = np.array([0, 0, 0, 0, 1, 1, 1])
        yp = np.array([0, 0, 0, 1, 1, 1, 0])
        # class 0: TP=3, FP=1(yp=0 but yt=1), FN=1(yt=0 but yp=1)
        # P0=3/4=0.75, R0=3/4=0.75, F1_0=0.75
        # class 1: TP=2, FP=1, FN=1
        # P1=2/3, R1=2/3, F1_1=2/3
        expected_macro = (0.75 + 2 / 3) / 2
        assert macro_f1(yt, yp) == pytest.approx(expected_macro, abs=1e-6)

    def test_weighted_f1_known_binary(self):
        """Weighted F1 weights by support."""
        yt = np.array([0, 0, 0, 0, 1, 1, 1])
        yp = np.array([0, 0, 0, 1, 1, 1, 0])
        # support: class 0 = 4, class 1 = 3, total = 7
        f1_0 = 0.75
        f1_1 = 2 / 3
        expected_weighted = (f1_0 * 4 + f1_1 * 3) / 7
        assert weighted_f1(yt, yp) == pytest.approx(expected_weighted, abs=1e-6)

    def test_macro_f1_zero_when_all_wrong(self):
        """All-wrong binary gives F1=0 for each class."""
        yt = np.array([0, 0, 1, 1])
        yp = np.array([1, 1, 0, 0])
        assert macro_f1(yt, yp) == pytest.approx(0.0)

    def test_weighted_f1_zero_when_all_wrong(self):
        yt = np.array([0, 0, 1, 1])
        yp = np.array([1, 1, 0, 0])
        assert weighted_f1(yt, yp) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. cramers_v
# ---------------------------------------------------------------------------

class TestCramersV:
    """Tests for cramers_v."""

    def test_perfect_association(self):
        """Perfect prediction should give V=1.0."""
        yt, yp = _make_perfect(200)
        v = cramers_v(yt, yp)
        assert v == pytest.approx(1.0, abs=0.05)

    def test_no_association_near_zero(self):
        """Random predictions should give V near 0."""
        yt, yp = _make_random(2000, seed=99)
        v = cramers_v(yt, yp)
        assert v < 0.15

    def test_empty_arrays(self):
        yt = np.array([], dtype=int)
        yp = np.array([], dtype=int)
        assert cramers_v(yt, yp) == 0.0

    def test_single_class(self):
        """If all same class, V should be 0 (k-1=0)."""
        yt = np.array([0, 0, 0, 0])
        yp = np.array([0, 0, 0, 0])
        assert cramers_v(yt, yp) == 0.0


# ---------------------------------------------------------------------------
# 6. confusion_matrix
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    """Tests for confusion_matrix."""

    def test_shape(self):
        yt = np.array([0, 1, 2, 3])
        yp = np.array([0, 1, 2, 3])
        cm = confusion_matrix(yt, yp, n_classes=4)
        assert cm.shape == (4, 4)

    def test_perfect_diagonal(self):
        yt = np.array([0, 1, 2, 3, 0, 1])
        yp = np.array([0, 1, 2, 3, 0, 1])
        cm = confusion_matrix(yt, yp, n_classes=4)
        assert np.trace(cm) == 6
        assert cm.sum() == 6

    def test_known_values(self):
        yt = np.array([0, 0, 1, 1, 2])
        yp = np.array([0, 1, 1, 2, 2])
        cm = confusion_matrix(yt, yp, n_classes=3)
        expected = np.array([
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ])
        np.testing.assert_array_equal(cm, expected)

    def test_sum_equals_n(self):
        yt, yp = _make_random(50)
        cm = confusion_matrix(yt, yp, n_classes=4)
        assert cm.sum() == 50


# ---------------------------------------------------------------------------
# 7. evaluate (full report)
# ---------------------------------------------------------------------------

class TestEvaluate:
    """Tests for the high-level evaluate function."""

    def test_report_has_all_fields(self):
        y_true = ["NONE", "LOW", "MEDIUM", "HIGH"] * 10
        y_pred = ["NONE", "LOW", "MEDIUM", "HIGH"] * 10
        report = evaluate(y_true, y_pred, n_bootstrap=200, seed=42)
        assert isinstance(report, EvaluationReport)
        assert isinstance(report.accuracy, ConfidenceInterval)
        assert isinstance(report.macro_f1, ConfidenceInterval)
        assert isinstance(report.weighted_f1, ConfidenceInterval)
        assert isinstance(report.cohen_kappa, ConfidenceInterval)
        assert isinstance(report.cramers_v, float)
        assert isinstance(report.confusion_matrix, np.ndarray)
        assert report.n_samples == 40
        assert report.n_classes == 4
        assert report.labels == ["NONE", "LOW", "MEDIUM", "HIGH"]

    def test_report_per_class(self):
        y_true = ["NONE", "LOW", "MEDIUM", "HIGH"] * 5
        y_pred = ["NONE", "LOW", "MEDIUM", "HIGH"] * 5
        report = evaluate(y_true, y_pred, n_bootstrap=200, seed=42)
        assert len(report.per_class) == 4
        for cm in report.per_class:
            assert cm.support == 5
            assert cm.precision.point == pytest.approx(1.0)
            assert cm.recall.point == pytest.approx(1.0)
            assert cm.f1.point == pytest.approx(1.0)

    def test_report_cis_valid(self):
        """All CIs should have lower <= point <= upper."""
        y_true = ["NONE", "LOW", "HIGH", "MEDIUM"] * 10
        y_pred = ["NONE", "LOW", "MEDIUM", "HIGH"] * 10
        report = evaluate(y_true, y_pred, n_bootstrap=500, seed=42)
        for ci in [report.accuracy, report.macro_f1, report.weighted_f1, report.cohen_kappa]:
            assert ci.lower <= ci.point <= ci.upper, f"CI violated for {ci}"

    def test_report_perfect_accuracy(self):
        y_true = ["NONE", "LOW", "MEDIUM", "HIGH"] * 25
        y_pred = ["NONE", "LOW", "MEDIUM", "HIGH"] * 25
        report = evaluate(y_true, y_pred, n_bootstrap=200, seed=42)
        assert report.accuracy.point == 1.0

    def test_report_custom_labels(self):
        labels = ["cat", "dog", "bird"]
        y_true = ["cat", "dog", "bird", "cat", "dog"]
        y_pred = ["cat", "dog", "bird", "cat", "dog"]
        report = evaluate(y_true, y_pred, labels=labels, n_bootstrap=200, seed=42)
        assert report.labels == labels
        assert report.n_classes == 3
        assert len(report.per_class) == 3

    def test_report_confusion_matrix_shape(self):
        y_true = ["NONE", "LOW", "MEDIUM", "HIGH"] * 5
        y_pred = ["NONE", "LOW", "MEDIUM", "HIGH"] * 5
        report = evaluate(y_true, y_pred, n_bootstrap=200, seed=42)
        assert report.confusion_matrix.shape == (4, 4)


# ---------------------------------------------------------------------------
# 8. mcnemar_test
# ---------------------------------------------------------------------------

class TestMcNemarTest:
    """Tests for mcnemar_test."""

    def test_identical_models_not_significant(self):
        """Two identical models should produce p=1.0, not significant."""
        y_true = ["NONE", "LOW", "MEDIUM", "HIGH"] * 10
        y_pred = ["NONE", "LOW", "MEDIUM", "HIGH"] * 10
        result = mcnemar_test(y_true, y_pred, y_pred)
        assert isinstance(result, ModelComparison)
        assert result.p_value == 1.0
        assert result.significant is False
        assert result.n_discordant == 0

    def test_different_models_significant(self):
        """Models with very different error patterns should be significant."""
        rng = np.random.RandomState(42)
        n = 200
        labels = ["NONE", "LOW", "MEDIUM", "HIGH"]
        y_true = [labels[rng.randint(0, 4)] for _ in range(n)]
        # Model A: perfect
        y_pred_a = list(y_true)
        # Model B: random
        y_pred_b = [labels[rng.randint(0, 4)] for _ in range(n)]
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert result.significant == True  # noqa: E712 (numpy bool)
        assert result.p_value < 0.05
        assert result.model_a_accuracy > result.model_b_accuracy

    def test_accuracies_computed(self):
        y_true = ["NONE", "LOW", "MEDIUM"]
        y_pred_a = ["NONE", "LOW", "MEDIUM"]
        y_pred_b = ["LOW", "LOW", "HIGH"]
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert result.model_a_accuracy == pytest.approx(1.0)
        assert result.model_b_accuracy == pytest.approx(1 / 3)

    def test_effect_size_inf_when_b10_zero(self):
        """If model B never corrects model A's mistakes, effect size is inf."""
        y_true = ["NONE", "LOW"]
        y_pred_a = ["NONE", "LOW"]  # perfect
        y_pred_b = ["LOW", "NONE"]  # all wrong
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert result.effect_size == float("inf")


# ---------------------------------------------------------------------------
# 9. format_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    """Tests for format_report."""

    def test_returns_nonempty_string(self):
        y_true = ["NONE", "LOW", "MEDIUM", "HIGH"] * 5
        y_pred = ["NONE", "LOW", "MEDIUM", "HIGH"] * 5
        report = evaluate(y_true, y_pred, n_bootstrap=200, seed=42)
        text = format_report(report)
        assert isinstance(text, str)
        assert len(text) > 100

    def test_contains_key_sections(self):
        y_true = ["NONE", "LOW", "MEDIUM", "HIGH"] * 5
        y_pred = ["NONE", "LOW", "MEDIUM", "HIGH"] * 5
        report = evaluate(y_true, y_pred, n_bootstrap=200, seed=42)
        text = format_report(report)
        assert "STATISTICAL EVALUATION REPORT" in text
        assert "Global Metrics" in text
        assert "Per-Class Metrics" in text
        assert "Confusion Matrix" in text
        assert "Accuracy" in text
        assert "Macro F1" in text
        assert "Weighted F1" in text

    def test_contains_all_labels(self):
        y_true = ["NONE", "LOW", "MEDIUM", "HIGH"] * 5
        y_pred = ["NONE", "LOW", "MEDIUM", "HIGH"] * 5
        report = evaluate(y_true, y_pred, n_bootstrap=200, seed=42)
        text = format_report(report)
        for label in ["NONE", "LOW", "MEDIUM", "HIGH"]:
            assert label in text


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests."""

    def test_single_sample(self):
        """A single sample should not crash."""
        yt = np.array([0])
        yp = np.array([0])
        assert accuracy_score(yt, yp) == 1.0
        cm = confusion_matrix(yt, yp, n_classes=1)
        assert cm.shape == (1, 1)
        assert cm[0, 0] == 1

    def test_all_same_class_accuracy(self):
        yt = np.array([2, 2, 2, 2, 2])
        yp = np.array([2, 2, 2, 2, 2])
        assert accuracy_score(yt, yp) == 1.0

    def test_all_same_class_kappa(self):
        """If all same class predicted and true, kappa should be 1.0 (p_e=1)."""
        yt = np.array([0, 0, 0, 0])
        yp = np.array([0, 0, 0, 0])
        # p_o = 1.0, p_e = 1.0, so kappa = 1.0 (special case)
        assert cohen_kappa(yt, yp) == 1.0

    def test_all_same_class_cramers_v_zero(self):
        """Single class => min(r,c)-1 = 0 => V = 0."""
        yt = np.array([0, 0, 0])
        yp = np.array([0, 0, 0])
        assert cramers_v(yt, yp) == 0.0

    def test_evaluate_single_class(self):
        """evaluate() should handle all-same-class without crashing."""
        report = evaluate(["NONE"] * 10, ["NONE"] * 10, n_bootstrap=200, seed=42)
        assert report.accuracy.point == 1.0
        assert report.n_samples == 10

    def test_bootstrap_ci_custom_metric(self):
        """bootstrap_ci works with any callable metric."""
        def always_half(yt, yp):
            return 0.5

        yt, yp = _make_random(30)
        ci = bootstrap_ci(yt, yp, always_half, n_bootstrap=100, seed=0)
        assert ci.point == 0.5
        assert ci.lower == pytest.approx(0.5)
        assert ci.upper == pytest.approx(0.5)

    def test_confidence_interval_repr(self):
        ci = ConfidenceInterval(point=0.85, lower=0.80, upper=0.90, alpha=0.05, method="bca")
        text = repr(ci)
        assert "0.8500" in text
        assert "0.8000" in text
        assert "0.9000" in text
        assert "95% CI" in text
        assert "bca" in text

    def test_evaluate_unknown_label_maps_to_zero(self):
        """Labels not in the label list should map to index 0."""
        report = evaluate(
            ["NONE", "LOW", "UNKNOWN"],
            ["NONE", "LOW", "NONE"],
            n_bootstrap=200,
            seed=42,
        )
        # "UNKNOWN" maps to 0 (NONE), so the third sample becomes yt=0, yp=0 => correct
        assert report.accuracy.point == 1.0

    def test_confusion_matrix_dtype_int(self):
        yt = np.array([0, 1, 2])
        yp = np.array([0, 1, 2])
        cm = confusion_matrix(yt, yp, n_classes=3)
        assert cm.dtype == int
