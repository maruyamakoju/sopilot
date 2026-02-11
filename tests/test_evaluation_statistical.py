"""Tests for evaluation.statistical — Bootstrap CI, permutation tests, ICC, ablation."""

from __future__ import annotations

import numpy as np
import pytest

from sopilot.evaluation.statistical import (
    bootstrap_confidence_interval,
    permutation_test,
    intraclass_correlation,
    AblationStudy,
    AblationResult,
)


class TestBootstrapCI:
    def test_basic_mean(self) -> None:
        rng = np.random.default_rng(42)
        scores = rng.normal(50, 5, size=100)
        mean, ci_lower, ci_upper = bootstrap_confidence_interval(
            scores, n_bootstrap=1000, rng=rng
        )
        assert 45 < mean < 55
        assert ci_lower < mean
        assert ci_upper > mean
        assert ci_lower < ci_upper

    def test_narrow_ci_with_low_variance(self) -> None:
        scores = np.full(100, 50.0)
        mean, ci_lower, ci_upper = bootstrap_confidence_interval(scores)
        assert mean == 50.0
        assert ci_lower == 50.0
        assert ci_upper == 50.0

    def test_empty_scores(self) -> None:
        mean, ci_lower, ci_upper = bootstrap_confidence_interval(np.array([]))
        assert mean == 0.0

    def test_custom_statistic(self) -> None:
        rng = np.random.default_rng(42)
        scores = rng.normal(0, 1, size=200)
        med, lo, hi = bootstrap_confidence_interval(
            scores, statistic=np.median, rng=rng
        )
        assert -1.0 < med < 1.0

    def test_ci_contains_true_mean(self) -> None:
        """95% CI should contain the true mean most of the time."""
        rng = np.random.default_rng(42)
        true_mean = 75.0
        scores = rng.normal(true_mean, 3, size=200)
        _, ci_lower, ci_upper = bootstrap_confidence_interval(
            scores, n_bootstrap=5000, rng=rng
        )
        assert ci_lower < true_mean < ci_upper


class TestPermutationTest:
    def test_identical_groups_high_p(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(50, 5, size=50)
        group_a = data[:25]
        group_b = data[25:]
        diff, p = permutation_test(group_a, group_b, n_permutations=1000, rng=rng)
        # Same distribution → p should be high
        assert p > 0.05

    def test_different_groups_low_p(self) -> None:
        rng = np.random.default_rng(42)
        group_a = rng.normal(80, 2, size=50)
        group_b = rng.normal(50, 2, size=50)
        diff, p = permutation_test(group_a, group_b, n_permutations=1000, rng=rng)
        assert diff > 20
        assert p < 0.01

    def test_observed_diff_sign(self) -> None:
        rng = np.random.default_rng(42)
        group_a = np.full(10, 100.0)
        group_b = np.full(10, 50.0)
        diff, _ = permutation_test(group_a, group_b, rng=rng)
        assert diff == 50.0


class TestIntraclassCorrelation:
    def test_perfect_agreement(self) -> None:
        # All raters give the same scores
        ratings = np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30]], dtype=np.float64)
        icc, lo, hi = intraclass_correlation(ratings)
        assert icc > 0.95

    def test_no_agreement(self) -> None:
        # Random ratings
        rng = np.random.default_rng(42)
        ratings = rng.standard_normal((20, 3))
        icc, _, _ = intraclass_correlation(ratings)
        assert -0.5 < icc < 0.5

    def test_single_rater(self) -> None:
        ratings = np.array([[10], [20], [30]])
        icc, lo, hi = intraclass_correlation(ratings)
        assert icc == 0.0

    def test_single_subject(self) -> None:
        ratings = np.array([[10, 20, 30]])
        icc, lo, hi = intraclass_correlation(ratings)
        assert icc == 0.0


class TestAblationStudy:
    def test_basic_ablation(self) -> None:
        rng = np.random.default_rng(42)
        study = AblationStudy(base_name="full")
        study.add_condition("full", rng.normal(85, 3, size=50))
        study.add_condition("no_proj", rng.normal(75, 3, size=50))
        study.add_condition("no_soft_dtw", rng.normal(80, 3, size=50))

        results = study.run(n_bootstrap=500, n_permutations=500)
        assert len(results) == 3

        # Base condition should have no diff
        base = [r for r in results if r.name == "full"][0]
        assert base.diff_from_base == 0.0

        # no_proj should have significant diff
        no_proj = [r for r in results if r.name == "no_proj"][0]
        assert no_proj.diff_from_base > 5.0
        assert no_proj.p_value < 0.05

    def test_missing_base_raises(self) -> None:
        study = AblationStudy(base_name="missing")
        study.add_condition("some", np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="Base condition"):
            study.run()

    def test_summary_output(self) -> None:
        rng = np.random.default_rng(42)
        study = AblationStudy(base_name="full")
        study.add_condition("full", rng.normal(80, 3, size=30))
        study.add_condition("ablated", rng.normal(70, 3, size=30))
        results = study.run(n_bootstrap=200, n_permutations=200)
        summary = study.summary(results)
        assert "Condition" in summary
        assert "full" in summary
        assert "ablated" in summary
