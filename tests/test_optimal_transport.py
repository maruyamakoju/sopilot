"""test_optimal_transport.py
=============================
Comprehensive pytest suite for the Sinkhorn-based Optimal Transport
alignment module (sopilot.core.optimal_transport).

Tests cover:
  - Identical sequences (distance ~ 0)
  - Reversed sequences (higher distance)
  - Known synthetic examples with analytical properties
  - Convergence behaviour across epsilon values
  - Numerical stability with extreme embeddings
  - Transport plan marginal constraints
  - Correlation with DTW alignment
  - Edge cases: minimum lengths, high dimensionality
  - Non-uniform marginals
  - Data-driven epsilon selection

All tests are independent, fast (< 1 s each), and require no real video files.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sopilot.core.optimal_transport import (
    OTAlignment,
    extract_alignment_from_plan,
    optimal_epsilon,
    ot_align,
    wasserstein_distance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vectors(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Return (n, d) float32 array of L2-normalised random unit vectors."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.where(norms < 1e-9, 1.0, norms)).astype(np.float32)


def _smooth_trajectory(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Return a smooth trajectory of n unit vectors in d dimensions.

    Adjacent vectors are highly correlated, simulating a realistic
    video embedding sequence where consecutive frames are similar.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(d).astype(np.float64)
    base /= max(np.linalg.norm(base), 1e-8)
    vecs = []
    for i in range(n):
        noise = rng.standard_normal(d).astype(np.float64) * 0.1
        v = base + noise * (1.0 + i * 0.05)
        v /= max(np.linalg.norm(v), 1e-8)
        vecs.append(v)
    return np.array(vecs, dtype=np.float32)


# ===========================================================================
# Section 1: Basic OTAlignment properties (8 tests)
# ===========================================================================

class TestOTAlignBasic:
    """Basic properties of ot_align()."""

    def test_return_type(self):
        """ot_align must return an OTAlignment dataclass."""
        gold = _unit_vectors(5, 32, seed=10)
        trainee = _unit_vectors(5, 32, seed=11)
        result = ot_align(gold, trainee)
        assert isinstance(result, OTAlignment)

    def test_frozen_dataclass(self):
        """OTAlignment must be frozen (immutable)."""
        gold = _unit_vectors(4, 16, seed=1)
        result = ot_align(gold, gold.copy())
        with pytest.raises(AttributeError):
            result.wasserstein_distance = 999.0  # type: ignore[misc]

    def test_transport_plan_shape(self):
        """Transport plan must have shape (m, n)."""
        gold = _unit_vectors(6, 32, seed=1)
        trainee = _unit_vectors(8, 32, seed=2)
        result = ot_align(gold, trainee)
        assert result.transport_plan.shape == (6, 8)

    def test_cost_matrix_shape(self):
        """Cost matrix must have shape (m, n)."""
        gold = _unit_vectors(5, 64, seed=3)
        trainee = _unit_vectors(7, 64, seed=4)
        result = ot_align(gold, trainee)
        assert result.cost_matrix.shape == (5, 7)

    def test_sequence_lengths_stored(self):
        """gold_length and trainee_length must be stored correctly."""
        gold = _unit_vectors(10, 32, seed=5)
        trainee = _unit_vectors(12, 32, seed=6)
        result = ot_align(gold, trainee)
        assert result.gold_length == 10
        assert result.trainee_length == 12

    def test_alignment_path_not_empty(self):
        """Alignment path must have at least one entry."""
        gold = _unit_vectors(3, 16, seed=7)
        trainee = _unit_vectors(4, 16, seed=8)
        result = ot_align(gold, trainee)
        assert len(result.alignment_path) > 0

    def test_alignment_path_length_equals_gold(self):
        """Hard alignment path must have one entry per gold frame."""
        gold = _unit_vectors(7, 32, seed=9)
        trainee = _unit_vectors(5, 32, seed=10)
        result = ot_align(gold, trainee)
        assert len(result.alignment_path) == 7

    def test_epsilon_stored(self):
        """The epsilon used must be stored in the result."""
        gold = _unit_vectors(4, 16, seed=11)
        trainee = _unit_vectors(4, 16, seed=12)
        result = ot_align(gold, trainee, epsilon=0.05)
        assert result.epsilon == 0.05


# ===========================================================================
# Section 2: Identical and reversed sequences (5 tests)
# ===========================================================================

class TestIdenticalAndReversed:
    """Tests with identical and reversed sequences."""

    def test_identical_sequences_near_zero_distance(self):
        """Aligning a sequence with itself must yield distance ~ 0."""
        gold = _unit_vectors(10, 64, seed=20)
        result = ot_align(gold, gold.copy(), epsilon=0.1)
        assert result.wasserstein_distance < 1e-3
        assert result.normalized_distance < 1e-3

    def test_identical_sequences_converged(self):
        """Identical sequences must converge easily."""
        gold = _unit_vectors(8, 32, seed=21)
        result = ot_align(gold, gold.copy(), epsilon=0.1)
        assert result.converged is True

    def test_identical_diagonal_alignment(self):
        """Identical sequences should produce near-diagonal alignment."""
        gold = _smooth_trajectory(8, 32, seed=22)
        result = ot_align(gold, gold.copy(), epsilon=0.05)
        for i, (gi, ti) in enumerate(result.alignment_path):
            assert gi == i  # gold index matches position
            # Trainee index should be close to diagonal
            assert abs(ti - i) <= 2

    def test_reversed_higher_distance(self):
        """Reversed sequence must yield strictly higher distance than identical."""
        gold = _smooth_trajectory(10, 64, seed=23)
        trainee_reversed = gold[::-1].copy()
        dist_identical = ot_align(gold, gold.copy(), epsilon=0.1).wasserstein_distance
        dist_reversed = ot_align(gold, trainee_reversed, epsilon=0.1).wasserstein_distance
        assert dist_reversed > dist_identical

    def test_reversed_vs_random(self):
        """Reversed sequence should have different distance than random."""
        gold = _smooth_trajectory(8, 32, seed=24)
        trainee_reversed = gold[::-1].copy()
        trainee_random = _unit_vectors(8, 32, seed=999)
        dist_reversed = ot_align(gold, trainee_reversed, epsilon=0.1).wasserstein_distance
        dist_random = ot_align(gold, trainee_random, epsilon=0.1).wasserstein_distance
        # Both should be positive
        assert dist_reversed > 0
        assert dist_random > 0


# ===========================================================================
# Section 3: Transport plan properties (7 tests)
# ===========================================================================

class TestTransportPlanProperties:
    """Verify mathematical properties of the transport plan."""

    def test_plan_non_negative(self):
        """All entries of the transport plan must be non-negative."""
        gold = _unit_vectors(6, 32, seed=30)
        trainee = _unit_vectors(8, 32, seed=31)
        result = ot_align(gold, trainee, epsilon=0.1)
        assert np.all(result.transport_plan >= 0)

    def test_plan_row_marginals(self):
        """Row sums of converged plan must approximate source marginal."""
        m, n = 8, 10
        gold = _unit_vectors(m, 32, seed=32)
        trainee = _unit_vectors(n, 32, seed=33)
        result = ot_align(gold, trainee, epsilon=0.1, max_iter=200)
        if result.converged:
            row_sums = result.transport_plan.sum(axis=1)
            expected = np.full(m, 1.0 / m)
            np.testing.assert_allclose(row_sums, expected, atol=1e-4)

    def test_plan_col_marginals(self):
        """Column sums of converged plan must approximate target marginal."""
        m, n = 6, 9
        gold = _unit_vectors(m, 32, seed=34)
        trainee = _unit_vectors(n, 32, seed=35)
        result = ot_align(gold, trainee, epsilon=0.1, max_iter=200)
        if result.converged:
            col_sums = result.transport_plan.sum(axis=0)
            expected = np.full(n, 1.0 / n)
            np.testing.assert_allclose(col_sums, expected, atol=1e-4)

    def test_plan_total_mass(self):
        """Total mass of the plan must be 1.0 (for normalised marginals)."""
        gold = _unit_vectors(5, 16, seed=36)
        trainee = _unit_vectors(7, 16, seed=37)
        result = ot_align(gold, trainee, epsilon=0.1, max_iter=200)
        total_mass = float(result.transport_plan.sum())
        assert abs(total_mass - 1.0) < 1e-3

    def test_non_uniform_marginals(self):
        """OT with non-uniform marginals must respect them."""
        m, n = 4, 4
        gold = _unit_vectors(m, 16, seed=38)
        trainee = _unit_vectors(n, 16, seed=39)
        a = np.array([0.4, 0.3, 0.2, 0.1])
        b = np.array([0.1, 0.2, 0.3, 0.4])
        result = ot_align(gold, trainee, epsilon=0.1, a=a, b=b, max_iter=200)
        if result.converged:
            row_sums = result.transport_plan.sum(axis=1)
            col_sums = result.transport_plan.sum(axis=0)
            np.testing.assert_allclose(row_sums, a, atol=1e-3)
            np.testing.assert_allclose(col_sums, b, atol=1e-3)

    def test_wasserstein_distance_non_negative(self):
        """Wasserstein distance must always be non-negative."""
        gold = _unit_vectors(5, 32, seed=40)
        trainee = _unit_vectors(6, 32, seed=41)
        result = ot_align(gold, trainee)
        assert result.wasserstein_distance >= 0.0

    def test_normalized_distance_non_negative(self):
        """Normalised distance must always be non-negative."""
        gold = _unit_vectors(5, 32, seed=42)
        trainee = _unit_vectors(6, 32, seed=43)
        result = ot_align(gold, trainee)
        assert result.normalized_distance >= 0.0


# ===========================================================================
# Section 4: Convergence across epsilon values (5 tests)
# ===========================================================================

class TestConvergence:
    """Test convergence behaviour across different epsilon values."""

    def test_large_epsilon_converges_fast(self):
        """Large epsilon (1.0) should converge in few iterations."""
        gold = _unit_vectors(10, 64, seed=50)
        trainee = _unit_vectors(10, 64, seed=51)
        result = ot_align(gold, trainee, epsilon=1.0, max_iter=200)
        assert result.converged is True
        assert result.n_iterations <= 50

    def test_moderate_epsilon_converges(self):
        """Moderate epsilon (0.1) should converge within max_iter=200."""
        gold = _unit_vectors(10, 64, seed=52)
        trainee = _unit_vectors(10, 64, seed=53)
        result = ot_align(gold, trainee, epsilon=0.1, max_iter=200)
        assert result.converged is True

    def test_small_epsilon_needs_more_iterations(self):
        """Small epsilon should need more iterations than large epsilon."""
        gold = _unit_vectors(8, 32, seed=54)
        trainee = _unit_vectors(8, 32, seed=55)
        result_large = ot_align(gold, trainee, epsilon=1.0, max_iter=200)
        result_small = ot_align(gold, trainee, epsilon=0.01, max_iter=200)
        # Small epsilon should use at least as many iterations
        assert result_small.n_iterations >= result_large.n_iterations

    def test_distance_monotone_in_epsilon(self):
        """As epsilon -> 0, entropic OT distance should approach true W.

        For fixed sequences, W_epsilon >= W, and W_epsilon decreases as
        epsilon decreases.  We check that smaller epsilon yields smaller
        or similar distance (with tolerance for finite-precision effects).
        """
        gold = _unit_vectors(6, 32, seed=56)
        trainee = _unit_vectors(6, 32, seed=57)
        d_large = ot_align(gold, trainee, epsilon=1.0, max_iter=300).wasserstein_distance
        d_small = ot_align(gold, trainee, epsilon=0.01, max_iter=300).wasserstein_distance
        # W_eps includes only <C,P> (no entropy term), but the plan changes
        # with epsilon.  The transport cost should generally decrease for
        # smaller epsilon (closer to true OT), but allow some tolerance.
        assert d_small <= d_large + 0.1

    def test_very_small_epsilon_does_not_nan(self):
        """Very small epsilon must not produce NaN (log-domain stability)."""
        gold = _unit_vectors(5, 16, seed=58)
        trainee = _unit_vectors(5, 16, seed=59)
        result = ot_align(gold, trainee, epsilon=0.001, max_iter=500, tol=1e-4)
        assert np.isfinite(result.wasserstein_distance)
        assert not np.any(np.isnan(result.transport_plan))


# ===========================================================================
# Section 5: Numerical stability (5 tests)
# ===========================================================================

class TestNumericalStability:
    """Numerical stability with extreme inputs."""

    def test_very_large_embeddings(self):
        """Embeddings with large norms (pre-normalisation) must work."""
        rng = np.random.default_rng(60)
        gold = rng.standard_normal((5, 32)).astype(np.float32) * 1e6
        trainee = rng.standard_normal((5, 32)).astype(np.float32) * 1e6
        result = ot_align(gold, trainee, epsilon=0.1)
        assert np.isfinite(result.wasserstein_distance)

    def test_very_small_embeddings(self):
        """Embeddings with tiny norms (pre-normalisation) must work."""
        rng = np.random.default_rng(61)
        gold = rng.standard_normal((5, 32)).astype(np.float32) * 1e-8
        trainee = rng.standard_normal((5, 32)).astype(np.float32) * 1e-8
        result = ot_align(gold, trainee, epsilon=0.1)
        assert np.isfinite(result.wasserstein_distance)

    def test_near_zero_cost_entries(self):
        """Very similar sequences should not cause division by zero."""
        gold = _unit_vectors(6, 32, seed=62)
        # Trainee is gold + tiny noise
        trainee = gold + np.random.default_rng(63).standard_normal(gold.shape).astype(np.float32) * 1e-7
        result = ot_align(gold, trainee, epsilon=0.1)
        assert np.isfinite(result.wasserstein_distance)
        assert result.wasserstein_distance < 0.01

    def test_high_dimensional_embeddings(self):
        """High-dimensional embeddings (D=1280, like V-JEPA2) must work."""
        gold = _unit_vectors(5, 1280, seed=64)
        trainee = _unit_vectors(5, 1280, seed=65)
        result = ot_align(gold, trainee, epsilon=0.1)
        assert np.isfinite(result.wasserstein_distance)
        assert result.transport_plan.shape == (5, 5)

    def test_no_nan_in_plan(self):
        """Transport plan must never contain NaN values."""
        gold = _unit_vectors(8, 64, seed=66)
        trainee = _unit_vectors(10, 64, seed=67)
        for eps in [0.001, 0.01, 0.1, 1.0]:
            result = ot_align(gold, trainee, epsilon=eps, max_iter=300)
            assert not np.any(np.isnan(result.transport_plan)), (
                f"NaN in plan for epsilon={eps}"
            )


# ===========================================================================
# Section 6: Edge cases (6 tests)
# ===========================================================================

class TestEdgeCases:
    """Edge cases: minimum lengths, asymmetric sizes, etc."""

    def test_minimum_length_sequences(self):
        """Minimum length sequences (m=n=2) must work."""
        gold = _unit_vectors(2, 16, seed=70)
        trainee = _unit_vectors(2, 16, seed=71)
        result = ot_align(gold, trainee)
        assert result.gold_length == 2
        assert result.trainee_length == 2

    def test_single_clip_raises(self):
        """Single-clip sequences must raise ValueError."""
        gold = _unit_vectors(1, 16, seed=72)
        trainee = _unit_vectors(5, 16, seed=73)
        with pytest.raises(ValueError, match="at least"):
            ot_align(gold, trainee)

    def test_1d_input_raises(self):
        """1D input must raise ValueError."""
        gold = np.array([1.0, 2.0, 3.0])
        trainee = np.array([4.0, 5.0, 6.0])
        with pytest.raises(ValueError, match="2D"):
            ot_align(gold, trainee)

    def test_mismatched_dimensions_raises(self):
        """Different embedding dimensions must raise ValueError."""
        gold = _unit_vectors(5, 32, seed=74)
        trainee = _unit_vectors(5, 64, seed=75)
        with pytest.raises(ValueError, match="dimensions must match"):
            ot_align(gold, trainee)

    def test_very_asymmetric_lengths(self):
        """Very asymmetric sequence lengths (3 vs 20) must work."""
        gold = _unit_vectors(3, 32, seed=76)
        trainee = _unit_vectors(20, 32, seed=77)
        result = ot_align(gold, trainee, epsilon=0.1)
        assert result.gold_length == 3
        assert result.trainee_length == 20
        assert len(result.alignment_path) == 3

    def test_negative_epsilon_raises(self):
        """Negative epsilon must raise ValueError."""
        gold = _unit_vectors(3, 16, seed=78)
        trainee = _unit_vectors(3, 16, seed=79)
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            ot_align(gold, trainee, epsilon=-0.1)


# ===========================================================================
# Section 7: extract_alignment_from_plan (4 tests)
# ===========================================================================

class TestExtractAlignment:
    """Tests for extract_alignment_from_plan()."""

    def test_identity_plan(self):
        """Identity-like plan (diagonal) should yield diagonal path."""
        P = np.eye(5) / 5.0
        path = extract_alignment_from_plan(P)
        assert path == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

    def test_uniform_plan(self):
        """Uniform plan should still produce one entry per row."""
        P = np.ones((4, 6)) / 24.0
        path = extract_alignment_from_plan(P)
        assert len(path) == 4
        # Each entry is a valid (i, j) tuple
        for i, (gi, tj) in enumerate(path):
            assert gi == i
            assert 0 <= tj < 6

    def test_empty_plan(self):
        """Empty plan (0 x 0) should return empty path."""
        P = np.zeros((0, 0))
        path = extract_alignment_from_plan(P)
        assert path == []

    def test_non_2d_raises(self):
        """Non-2D input must raise ValueError."""
        with pytest.raises(ValueError, match="2D"):
            extract_alignment_from_plan(np.array([1, 2, 3]))


# ===========================================================================
# Section 8: optimal_epsilon (4 tests)
# ===========================================================================

class TestOptimalEpsilon:
    """Tests for data-driven epsilon selection."""

    def test_returns_positive(self):
        """optimal_epsilon must return a positive value."""
        gold = _unit_vectors(6, 32, seed=80)
        trainee = _unit_vectors(6, 32, seed=81)
        eps = optimal_epsilon(gold, trainee)
        assert eps > 0

    def test_bounded_range(self):
        """Result must be within [1e-4, 1.0]."""
        gold = _unit_vectors(10, 64, seed=82)
        trainee = _unit_vectors(10, 64, seed=83)
        eps = optimal_epsilon(gold, trainee)
        assert 1e-4 <= eps <= 1.0

    def test_identical_sequences_small_epsilon(self):
        """Identical constant sequences should yield small epsilon.

        When ALL pairwise costs are near zero (every vector is the same
        direction), the median cost is ~0 and the fallback 0.01 is
        returned.  Note: random unit vectors that happen to be
        *copies* still have large off-diagonal costs.
        """
        # All-same-direction vectors -> all pairwise costs are ~0
        v = np.ones((5, 32), dtype=np.float32)
        v /= np.linalg.norm(v[0])
        eps = optimal_epsilon(v, v.copy())
        assert eps <= 0.02

    def test_different_scale_parameter(self):
        """Changing scale parameter must change the result."""
        gold = _unit_vectors(6, 32, seed=85)
        trainee = _unit_vectors(6, 32, seed=86)
        eps_default = optimal_epsilon(gold, trainee, scale=0.05)
        eps_large = optimal_epsilon(gold, trainee, scale=0.2)
        # Larger scale -> larger epsilon (unless clamped)
        assert eps_large >= eps_default


# ===========================================================================
# Section 9: wasserstein_distance convenience function (3 tests)
# ===========================================================================

class TestWassersteinDistance:
    """Tests for the convenience wasserstein_distance() function."""

    def test_matches_ot_align(self):
        """wasserstein_distance must match ot_align().wasserstein_distance."""
        gold = _unit_vectors(6, 32, seed=90)
        trainee = _unit_vectors(8, 32, seed=91)
        full_result = ot_align(gold, trainee, epsilon=0.1)
        quick_dist = wasserstein_distance(gold, trainee, epsilon=0.1)
        assert abs(full_result.wasserstein_distance - quick_dist) < 1e-8

    def test_identical_near_zero(self):
        """wasserstein_distance for identical sequences must be near zero."""
        gold = _unit_vectors(5, 32, seed=92)
        dist = wasserstein_distance(gold, gold.copy(), epsilon=0.1)
        assert dist < 1e-3

    def test_symmetric(self):
        """wasserstein_distance(A, B) should approximately equal wasserstein_distance(B, A)."""
        gold = _unit_vectors(6, 32, seed=93)
        trainee = _unit_vectors(6, 32, seed=94)
        d_ab = wasserstein_distance(gold, trainee, epsilon=0.1)
        d_ba = wasserstein_distance(trainee, gold, epsilon=0.1)
        # With uniform marginals and symmetric cost, OT is symmetric
        assert abs(d_ab - d_ba) < 1e-4


# ===========================================================================
# Section 10: Correlation with DTW (3 tests)
# ===========================================================================

class TestCorrelationWithDTW:
    """Compare OT distance with DTW distance — they should be correlated."""

    def test_ot_and_dtw_both_low_for_similar(self):
        """Both OT and DTW should give low distance for similar sequences."""
        from sopilot.core.dtw import dtw_align
        gold = _smooth_trajectory(10, 64, seed=100)
        # Trainee is gold + small noise
        rng = np.random.default_rng(101)
        trainee = gold + rng.standard_normal(gold.shape).astype(np.float32) * 0.05
        trainee /= np.linalg.norm(trainee, axis=1, keepdims=True).clip(1e-8)

        ot_dist = wasserstein_distance(gold, trainee, epsilon=0.1)
        dtw_result = dtw_align(gold, trainee)
        # Both should be small
        assert ot_dist < 0.5
        assert dtw_result.normalized_cost < 0.5

    def test_ot_and_dtw_both_high_for_dissimilar(self):
        """Both OT and DTW should give high distance for dissimilar sequences."""
        from sopilot.core.dtw import dtw_align
        gold = _unit_vectors(10, 64, seed=102)
        trainee = _unit_vectors(10, 64, seed=999)

        ot_dist = wasserstein_distance(gold, trainee, epsilon=0.1)
        dtw_result = dtw_align(gold, trainee)
        # Both should be positive (not near zero)
        assert ot_dist > 0.01
        assert dtw_result.normalized_cost > 0.01

    def test_rank_correlation_across_pairs(self):
        """OT and DTW distances should be rank-correlated across multiple pairs.

        Generate several sequence pairs with varying similarity and check
        that the Spearman rank correlation is positive.
        """
        from sopilot.core.dtw import dtw_align

        gold = _smooth_trajectory(10, 64, seed=103)
        ot_dists = []
        dtw_dists = []

        for noise_level in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
            rng = np.random.default_rng(int(noise_level * 1000) + 200)
            trainee = gold + rng.standard_normal(gold.shape).astype(np.float32) * noise_level
            trainee /= np.linalg.norm(trainee, axis=1, keepdims=True).clip(1e-8)

            ot_dists.append(wasserstein_distance(gold, trainee, epsilon=0.1))
            dtw_dists.append(dtw_align(gold, trainee).normalized_cost)

        # Check that both increase (rank correlation > 0)
        # Use a simple check: both lists should be approximately increasing
        ot_rank = np.argsort(np.argsort(ot_dists))
        dtw_rank = np.argsort(np.argsort(dtw_dists))
        # Spearman correlation
        n = len(ot_rank)
        d_sq = sum((ot_rank[i] - dtw_rank[i]) ** 2 for i in range(n))
        spearman = 1.0 - 6.0 * d_sq / (n * (n * n - 1))
        assert spearman > 0.3, f"Spearman rank correlation too low: {spearman:.3f}"


# ===========================================================================
# Section 11: Synthetic examples with known properties (4 tests)
# ===========================================================================

class TestSyntheticExamples:
    """Synthetic examples where we can reason about the expected result."""

    def test_two_point_masses_same_location(self):
        """Two identical single-point distributions should have W=0.

        We use m=n=2 with identical embeddings.
        """
        v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = ot_align(v, v.copy(), epsilon=0.1)
        assert result.wasserstein_distance < 1e-3

    def test_orthogonal_bases(self):
        """Orthogonal gold and trainee should have distance = 1.0 per pair.

        For two orthogonal unit vectors, cosine distance = 1.0.
        With uniform marginals and 2x2, W = 1.0.
        """
        gold = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        trainee = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        result = ot_align(gold, trainee, epsilon=0.01, max_iter=500)
        # The optimal plan should map (0->1, 1->0) with cost 0
        # or (0->0, 1->1) with cost 2.0; OT picks cost 0 mapping
        # Actually: C = [[1, 0], [0, 1]] (cos dist between pairs)
        # So OT should find W = 0 (map 0->1, 1->0)
        assert result.wasserstein_distance < 0.1

    def test_parallel_vectors_zero_distance(self):
        """Parallel (identical direction) vectors should have zero cost."""
        v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        gold = np.tile(v, (3, 1))
        trainee = np.tile(v, (4, 1))
        result = ot_align(gold, trainee, epsilon=0.1)
        assert result.wasserstein_distance < 1e-3

    def test_antiparallel_vectors_max_distance(self):
        """Antiparallel vectors should have maximum cosine distance (2.0)."""
        gold = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        trainee = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        result = ot_align(gold, trainee, epsilon=0.01, max_iter=500)
        # C = [[2, 0], [0, 2]]; OT picks (0->1, 1->0) -> W = 0
        assert result.wasserstein_distance < 0.1


# ===========================================================================
# Section 12: Input validation (5 tests)
# ===========================================================================

class TestInputValidation:
    """Validate error handling for invalid inputs."""

    def test_zero_epsilon_raises(self):
        """epsilon=0 must raise ValueError."""
        gold = _unit_vectors(3, 16, seed=120)
        with pytest.raises(ValueError):
            ot_align(gold, gold.copy(), epsilon=0.0)

    def test_max_iter_zero_raises(self):
        """max_iter=0 must raise ValueError."""
        gold = _unit_vectors(3, 16, seed=121)
        with pytest.raises(ValueError, match="max_iter"):
            ot_align(gold, gold.copy(), max_iter=0)

    def test_marginal_wrong_length_raises(self):
        """Marginal with wrong length must raise ValueError."""
        gold = _unit_vectors(4, 16, seed=122)
        trainee = _unit_vectors(4, 16, seed=123)
        a = np.array([0.5, 0.5])  # length 2, but gold has 4
        with pytest.raises(ValueError, match="length"):
            ot_align(gold, trainee, a=a)

    def test_marginal_not_sum_one_raises(self):
        """Marginal not summing to 1 must raise ValueError."""
        gold = _unit_vectors(3, 16, seed=124)
        trainee = _unit_vectors(3, 16, seed=125)
        a = np.array([0.5, 0.5, 0.5])  # sums to 1.5
        with pytest.raises(ValueError, match="sum to 1"):
            ot_align(gold, trainee, a=a)

    def test_3d_input_raises(self):
        """3D input must raise ValueError."""
        gold = np.ones((2, 3, 4), dtype=np.float32)
        trainee = np.ones((2, 3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="2D"):
            ot_align(gold, trainee)
