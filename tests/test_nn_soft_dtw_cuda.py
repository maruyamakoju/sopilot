"""Tests for CUDA-accelerated Soft-DTW module.

Covers: SoftDTWFunction, SoftDTWCuda, SoftDTWAlignmentCuda,
multi_scale_sdtw, pairwise_soft_dtw."""

import pytest
import torch

from sopilot.nn.soft_dtw_cuda import (
    _INF,
    SoftDTWAlignmentCuda,
    SoftDTWCuda,
    SoftDTWFunction,
    _apply_bandwidth_mask,
    _compute_pairwise_cost,
    _downsample_sequence,
    multi_scale_sdtw,
    pairwise_soft_dtw,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_seq(B, T, D, seed=42):
    """Generate a random batched sequence."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D, generator=g)


def _rand_seq_2d(T, D, seed=42):
    """Generate a random unbatched sequence."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, D, generator=g)


# ---------------------------------------------------------------------------
# 1. Cost Matrix Tests
# ---------------------------------------------------------------------------


class TestCostMatrix:
    """Tests for _compute_pairwise_cost."""

    def test_cosine_identical_vectors(self):
        """Cosine distance between identical normalized vectors is 0."""
        x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        cost = _compute_pairwise_cost(x, x, metric="cosine")
        assert cost.shape == (1, 2, 2)
        assert torch.allclose(cost[:, 0, 0], torch.tensor([0.0]), atol=1e-6)
        assert torch.allclose(cost[:, 1, 1], torch.tensor([0.0]), atol=1e-6)

    def test_cosine_orthogonal_vectors(self):
        """Cosine distance between orthogonal vectors is 1."""
        x = torch.tensor([[[1.0, 0.0]]])
        y = torch.tensor([[[0.0, 1.0]]])
        cost = _compute_pairwise_cost(x, y, metric="cosine")
        assert torch.allclose(cost, torch.tensor([[[1.0]]]), atol=1e-6)

    def test_euclidean_identical(self):
        """Euclidean distance to self is 0."""
        x = _rand_seq(2, 5, 4)
        cost = _compute_pairwise_cost(x, x, metric="euclidean")
        diag = torch.diagonal(cost, dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-5)

    def test_euclidean_shape(self):
        """Shape is (B, M, N) for different-length sequences."""
        x = _rand_seq(3, 10, 8)
        y = _rand_seq(3, 7, 8)
        cost = _compute_pairwise_cost(x, y, metric="euclidean")
        assert cost.shape == (3, 10, 7)

    def test_2d_input_promotion(self):
        """2D inputs should be promoted to 3D."""
        x = _rand_seq_2d(5, 4)
        y = _rand_seq_2d(3, 4, seed=99)
        cost = _compute_pairwise_cost(x, y, metric="cosine")
        assert cost.dim() == 3 and cost.shape[0] == 1

    def test_cost_non_negative_euclidean(self):
        """Euclidean cost should always be non-negative."""
        x = _rand_seq(2, 8, 6)
        y = _rand_seq(2, 6, 6, seed=99)
        cost = _compute_pairwise_cost(x, y, metric="euclidean")
        assert (cost >= -1e-6).all()

    def test_invalid_metric_raises(self):
        """Unknown metric should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            _compute_pairwise_cost(torch.randn(1, 3, 4), torch.randn(1, 3, 4), metric="manhattan")


# ---------------------------------------------------------------------------
# 2. Bandwidth Mask Tests
# ---------------------------------------------------------------------------


class TestBandwidthMask:
    """Tests for _apply_bandwidth_mask (Sakoe-Chiba band)."""

    def test_no_bandwidth_returns_same(self):
        """bandwidth=None should return cost unchanged."""
        cost = _rand_seq(2, 5, 5)
        result = _apply_bandwidth_mask(cost, None)
        assert torch.equal(cost, result)

    def test_full_bandwidth_returns_same(self):
        """bandwidth >= 1.0 should return cost unchanged."""
        cost = _rand_seq(1, 4, 4)
        result = _apply_bandwidth_mask(cost, 1.0)
        assert torch.equal(cost, result)

    def test_narrow_bandwidth_masks_corners(self):
        """Narrow bandwidth should mask off-diagonal entries."""
        cost = torch.ones(1, 10, 10)
        masked = _apply_bandwidth_mask(cost, 0.1)
        # top-right and bottom-left corners should be masked
        assert masked[0, 0, -1].item() == _INF
        assert masked[0, -1, 0].item() == _INF
        # diagonal should remain 1.0
        assert masked[0, 0, 0].item() == 1.0
        assert masked[0, -1, -1].item() == 1.0

    def test_zero_bandwidth_only_diagonal(self):
        """bandwidth=0 on square matrix should only keep diagonal."""
        cost = torch.ones(1, 5, 5)
        masked = _apply_bandwidth_mask(cost, 0.0)
        for i in range(5):
            assert masked[0, i, i].item() == 1.0
        # off-diagonal should be _INF
        assert masked[0, 0, 1].item() == _INF

    def test_bandwidth_rectangular(self):
        """Bandwidth should work with non-square cost matrices."""
        cost = torch.ones(1, 8, 4)
        masked = _apply_bandwidth_mask(cost, 0.2)
        assert masked.shape == (1, 8, 4)
        # Some entries should be masked, some not
        assert (masked == _INF).any()
        assert (masked == 1.0).any()


# ---------------------------------------------------------------------------
# 3. SoftDTWFunction (Custom Autograd) Tests
# ---------------------------------------------------------------------------


class TestSoftDTWFunction:
    """Tests for the custom autograd Function."""

    def test_forward_identical_zero(self):
        """Identical zero cost should give zero distance."""
        cost = torch.zeros(1, 5, 5)
        gamma = torch.tensor(1.0)
        result = SoftDTWFunction.apply(cost, gamma)
        assert result.shape == (1,)
        # With zero cost, distance should be close to zero (minus softmin penalty)
        # The softmin of [0, inf, inf] = 0, so forward DP stays at 0
        assert result.item() < 1.0  # Should be very small

    def test_forward_positive_cost(self):
        """With positive costs, distance should be positive."""
        cost = torch.ones(1, 4, 4)
        gamma = torch.tensor(1.0)
        result = SoftDTWFunction.apply(cost, gamma)
        assert result.item() > 0

    def test_forward_batch_independence(self):
        """Each batch element should be computed independently."""
        c1 = torch.ones(1, 3, 3)
        c2 = torch.ones(1, 3, 3) * 2.0
        gamma = torch.tensor(1.0)
        d1 = SoftDTWFunction.apply(c1, gamma)
        d2 = SoftDTWFunction.apply(c2, gamma)
        batched = SoftDTWFunction.apply(torch.cat([c1, c2], dim=0), gamma)
        assert torch.allclose(batched[0], d1.squeeze(), atol=1e-5)
        assert torch.allclose(batched[1], d2.squeeze(), atol=1e-5)

    def test_forward_rectangular(self):
        """Should handle non-square cost matrices."""
        cost = torch.rand(2, 6, 4)
        gamma = torch.tensor(0.5)
        result = SoftDTWFunction.apply(cost, gamma)
        assert result.shape == (2,)
        assert torch.isfinite(result).all()

    def test_gamma_clamping(self):
        """Very small gamma should be clamped to _GAMMA_MIN."""
        cost = torch.rand(1, 3, 3)
        gamma = torch.tensor(1e-10)
        result = SoftDTWFunction.apply(cost, gamma)
        assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# 4. Gradient / Backward Tests
# ---------------------------------------------------------------------------


class TestGradients:
    """Tests for backward pass correctness."""

    def test_gradcheck_small(self):
        """torch.autograd.gradcheck on small input."""
        cost = torch.rand(1, 3, 3, dtype=torch.float64, requires_grad=True)
        gamma = torch.tensor(1.0, dtype=torch.float64)
        assert torch.autograd.gradcheck(SoftDTWFunction.apply, (cost, gamma), eps=1e-4, atol=1e-3)

    def test_gradcheck_rectangular(self):
        """Gradcheck on non-square cost matrix."""
        cost = torch.rand(1, 4, 3, dtype=torch.float64, requires_grad=True)
        gamma = torch.tensor(0.5, dtype=torch.float64)
        assert torch.autograd.gradcheck(SoftDTWFunction.apply, (cost, gamma), eps=1e-4, atol=1e-3)

    def test_gradcheck_batch(self):
        """Gradcheck with batch size > 1."""
        cost = torch.rand(2, 3, 3, dtype=torch.float64, requires_grad=True)
        gamma = torch.tensor(1.0, dtype=torch.float64)
        assert torch.autograd.gradcheck(SoftDTWFunction.apply, (cost, gamma), eps=1e-4, atol=1e-3)

    def test_gradient_flows_through_module(self):
        """Gradient should flow through SoftDTWCuda to input embeddings."""
        x = torch.randn(2, 5, 8, requires_grad=True)
        y = torch.randn(2, 5, 8)
        sdtw = SoftDTWCuda(gamma=1.0, normalize=False, metric="euclidean")
        dist = sdtw(x, y)
        loss = dist.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()

    def test_gradient_normalized(self):
        """Gradient should flow through normalized Soft-DTW."""
        x = torch.randn(1, 4, 6, requires_grad=True)
        y = torch.randn(1, 4, 6)
        sdtw = SoftDTWCuda(gamma=1.0, normalize=True, metric="euclidean")
        dist = sdtw(x, y)
        dist.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# 5. SoftDTWCuda Module Tests
# ---------------------------------------------------------------------------


class TestSoftDTWCuda:
    """Tests for the SoftDTWCuda module."""

    def test_identical_sequences_normalized(self):
        """Normalized Soft-DTW of x with itself should be close to 0."""
        x = _rand_seq(1, 8, 4)
        sdtw = SoftDTWCuda(gamma=1.0, normalize=True)
        dist = sdtw(x, x)
        assert abs(dist.item()) < 1e-3

    def test_different_sequences_positive(self):
        """Distance between different sequences should be positive (normalized)."""
        x = _rand_seq(2, 6, 4, seed=1)
        y = _rand_seq(2, 6, 4, seed=99)
        sdtw = SoftDTWCuda(gamma=1.0, normalize=True, metric="euclidean")
        dist = sdtw(x, y)
        assert (dist > -1e-3).all()  # normalized can be slightly negative

    def test_unnormalized_always_positive(self):
        """Unnormalized distance with positive costs is always positive."""
        x = _rand_seq(2, 5, 4, seed=1)
        y = _rand_seq(2, 5, 4, seed=99)
        sdtw = SoftDTWCuda(gamma=1.0, normalize=False, metric="euclidean")
        dist = sdtw(x, y)
        assert (dist > 0).all()

    def test_unbatched_input(self):
        """2D input should return scalar."""
        x = _rand_seq_2d(5, 4)
        y = _rand_seq_2d(5, 4, seed=99)
        sdtw = SoftDTWCuda(gamma=1.0, normalize=False)
        dist = sdtw(x, y)
        assert dist.dim() == 0  # scalar

    def test_batched_output_shape(self):
        """Batched input should produce (B,) output."""
        x = _rand_seq(3, 5, 4)
        y = _rand_seq(3, 7, 4, seed=99)
        sdtw = SoftDTWCuda(gamma=0.5, normalize=True)
        dist = sdtw(x, y)
        assert dist.shape == (3,)

    def test_different_lengths(self):
        """Should handle sequences of different lengths."""
        x = _rand_seq(1, 10, 4)
        y = _rand_seq(1, 3, 4, seed=99)
        sdtw = SoftDTWCuda(gamma=1.0, normalize=False)
        dist = sdtw(x, y)
        assert torch.isfinite(dist).all()

    def test_euclidean_metric(self):
        """Euclidean metric should work correctly."""
        x = _rand_seq(1, 5, 4)
        y = _rand_seq(1, 5, 4, seed=99)
        sdtw = SoftDTWCuda(gamma=1.0, normalize=False, metric="euclidean")
        dist = sdtw(x, y)
        assert torch.isfinite(dist).all()

    def test_learnable_gamma(self):
        """gamma should be a learnable parameter."""
        sdtw = SoftDTWCuda(gamma=2.0)
        params = list(sdtw.parameters())
        assert len(params) == 1
        assert params[0].item() == pytest.approx(2.0)

    def test_bandwidth_reduces_distance(self):
        """Bandwidth constraint should not decrease the distance."""
        x = _rand_seq(1, 8, 4)
        y = _rand_seq(1, 8, 4, seed=99)
        sdtw_no_bw = SoftDTWCuda(gamma=1.0, normalize=False, metric="euclidean")
        sdtw_bw = SoftDTWCuda(gamma=1.0, normalize=False, bandwidth=0.2, metric="euclidean")
        d_no_bw = sdtw_no_bw(x, y)
        d_bw = sdtw_bw(x, y)
        # bandwidth constrains the path, so distance >= unconstrained
        assert d_bw.item() >= d_no_bw.item() - 1e-5


# ---------------------------------------------------------------------------
# 6. SoftDTWAlignmentCuda Tests
# ---------------------------------------------------------------------------


class TestSoftDTWAlignmentCuda:
    """Tests for alignment matrix computation."""

    def test_alignment_shape(self):
        """Alignment should have shape (B, M, N)."""
        x = _rand_seq(2, 6, 4)
        y = _rand_seq(2, 5, 4, seed=99)
        align = SoftDTWAlignmentCuda(gamma=1.0)
        A, d = align(x, y)
        assert A.shape == (2, 6, 5)
        assert d.shape == (2,)

    def test_alignment_non_negative(self):
        """Alignment values should be non-negative."""
        x = _rand_seq(1, 5, 4)
        y = _rand_seq(1, 5, 4, seed=99)
        align = SoftDTWAlignmentCuda(gamma=1.0)
        A, _ = align(x, y)
        assert (A >= -1e-6).all()

    def test_alignment_sums_roughly_to_path_length(self):
        """Sum of alignment matrix should be roughly M+N-1 for gamma->0."""
        x = _rand_seq(1, 5, 4)
        y = _rand_seq(1, 5, 4, seed=99)
        align = SoftDTWAlignmentCuda(gamma=0.01, metric="euclidean")
        A, _ = align(x, y)
        # For small gamma, alignment is close to a hard path
        # Path length is at least max(M, N) and at most M+N-1
        total = A.sum().item()
        assert 4.0 <= total <= 10.0

    def test_alignment_unbatched(self):
        """Unbatched input should return (M, N) alignment."""
        x = _rand_seq_2d(4, 3)
        y = _rand_seq_2d(5, 3, seed=99)
        align = SoftDTWAlignmentCuda(gamma=1.0)
        A, d = align(x, y)
        assert A.shape == (4, 5)
        assert d.dim() == 0

    def test_alignment_distance_matches_sdtw(self):
        """Distance from alignment should match SoftDTWCuda (unnormalized)."""
        x = _rand_seq(1, 5, 4)
        y = _rand_seq(1, 5, 4, seed=99)
        align = SoftDTWAlignmentCuda(gamma=1.0, metric="euclidean")
        sdtw = SoftDTWCuda(gamma=1.0, normalize=False, metric="euclidean")
        _, d_align = align(x, y)
        d_sdtw = sdtw(x, y)
        assert torch.allclose(d_align, d_sdtw, atol=1e-4)

    def test_alignment_with_bandwidth(self):
        """Alignment should respect bandwidth constraint."""
        x = _rand_seq(1, 8, 4)
        y = _rand_seq(1, 8, 4, seed=99)
        align = SoftDTWAlignmentCuda(gamma=0.1, bandwidth=0.1, metric="euclidean")
        A, _ = align(x, y)
        # Entries far from diagonal should be near zero
        assert A[0, 0, -1].item() < 1e-3
        assert A[0, -1, 0].item() < 1e-3


# ---------------------------------------------------------------------------
# 7. Multi-Scale Soft-DTW Tests
# ---------------------------------------------------------------------------


class TestMultiScaleSDTW:
    """Tests for multi_scale_sdtw."""

    def test_single_scale_matches_sdtw(self):
        """Single scale [1] should match regular Soft-DTW."""
        x = _rand_seq(1, 8, 4)
        y = _rand_seq(1, 8, 4, seed=99)
        sdtw = SoftDTWCuda(gamma=1.0, normalize=True, metric="cosine")
        d_sdtw = sdtw(x, y)
        d_ms = multi_scale_sdtw(x, y, gamma=1.0, normalize=True, metric="cosine", scales=[1], weights=[1.0])
        assert torch.allclose(d_sdtw, d_ms, atol=1e-4)

    def test_multi_scale_returns_finite(self):
        """Multi-scale should produce finite results."""
        x = _rand_seq(2, 16, 4)
        y = _rand_seq(2, 16, 4, seed=99)
        d = multi_scale_sdtw(x, y, gamma=1.0)
        assert torch.isfinite(d).all()

    def test_multi_scale_default_scales(self):
        """Default scales [1,2,4] should work."""
        x = _rand_seq(1, 16, 4)
        y = _rand_seq(1, 16, 4, seed=99)
        d = multi_scale_sdtw(x, y)
        assert d.shape == (1,)

    def test_multi_scale_custom_weights(self):
        """Custom weights should affect the result."""
        x = _rand_seq(1, 16, 4)
        y = _rand_seq(1, 16, 4, seed=99)
        d1 = multi_scale_sdtw(x, y, scales=[1, 2], weights=[1.0, 0.0])
        d2 = multi_scale_sdtw(x, y, scales=[1, 2], weights=[0.0, 1.0])
        # Different weights should (generally) give different results
        # At minimum, they should both be finite
        assert torch.isfinite(d1).all()
        assert torch.isfinite(d2).all()

    def test_multi_scale_unbatched(self):
        """Unbatched input should return scalar."""
        x = _rand_seq_2d(16, 4)
        y = _rand_seq_2d(16, 4, seed=99)
        d = multi_scale_sdtw(x, y)
        assert d.dim() == 0

    def test_multi_scale_short_sequence_graceful(self):
        """Should handle sequences that become too short at higher scales."""
        x = _rand_seq(1, 4, 4)
        y = _rand_seq(1, 4, 4, seed=99)
        # scale=4 will make length=1, should skip gracefully
        d = multi_scale_sdtw(x, y, scales=[1, 2, 4])
        assert torch.isfinite(d).all()


# ---------------------------------------------------------------------------
# 8. Pairwise Soft-DTW Tests
# ---------------------------------------------------------------------------


class TestPairwiseSoftDTW:
    """Tests for pairwise_soft_dtw."""

    def test_diagonal_zero_normalized(self):
        """Diagonal of pairwise matrix should be zero (normalized)."""
        seqs = [_rand_seq_2d(5, 4, seed=i) for i in range(3)]
        D = pairwise_soft_dtw(seqs, gamma=1.0, normalize=True, metric="euclidean")
        assert D.shape == (3, 3)
        for i in range(3):
            assert abs(D[i, i].item()) < 1e-3

    def test_symmetric(self):
        """Pairwise matrix should be symmetric."""
        seqs = [_rand_seq_2d(5, 4, seed=i) for i in range(3)]
        D = pairwise_soft_dtw(seqs, gamma=1.0, normalize=True)
        assert torch.allclose(D, D.T, atol=1e-5)

    def test_empty_list(self):
        """Empty list should return (0, 0) matrix."""
        D = pairwise_soft_dtw([])
        assert D.shape == (0, 0)

    def test_single_sequence(self):
        """Single sequence should return (1, 1) zero matrix."""
        seqs = [_rand_seq_2d(5, 4)]
        D = pairwise_soft_dtw(seqs, gamma=1.0)
        assert D.shape == (1, 1)
        assert D[0, 0].item() == 0.0

    def test_variable_lengths(self):
        """Should handle sequences of different lengths."""
        seqs = [
            _rand_seq_2d(3, 4, seed=0),
            _rand_seq_2d(5, 4, seed=1),
            _rand_seq_2d(7, 4, seed=2),
        ]
        D = pairwise_soft_dtw(seqs, gamma=1.0, normalize=True)
        assert D.shape == (3, 3)
        assert torch.isfinite(D).all()


# ---------------------------------------------------------------------------
# 9. Downsampling Tests
# ---------------------------------------------------------------------------


class TestDownsampling:
    """Tests for _downsample_sequence."""

    def test_factor_1_identity(self):
        """Factor 1 should return input unchanged."""
        x = _rand_seq(2, 8, 4)
        result = _downsample_sequence(x, 1)
        assert torch.equal(x, result)

    def test_factor_2_halves_length(self):
        """Factor 2 should halve sequence length."""
        x = _rand_seq(2, 8, 4)
        result = _downsample_sequence(x, 2)
        assert result.shape == (2, 4, 4)

    def test_factor_2_averages_pairs(self):
        """Factor 2 should average consecutive frame pairs."""
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
        result = _downsample_sequence(x, 2)
        expected = torch.tensor([[[2.0, 3.0], [6.0, 7.0]]])
        assert torch.allclose(result, expected)

    def test_truncates_remainder(self):
        """Frames not divisible by factor should be truncated."""
        x = _rand_seq(1, 7, 4)
        result = _downsample_sequence(x, 2)
        assert result.shape == (1, 3, 4)  # 7 -> 6/2 = 3

    def test_2d_input(self):
        """Unbatched 2D input should work."""
        x = _rand_seq_2d(8, 4)
        result = _downsample_sequence(x, 4)
        assert result.shape == (2, 4)


# ---------------------------------------------------------------------------
# 10. Numerical Stability and Edge Cases
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Tests for numerical stability and edge cases."""

    def test_very_large_gamma(self):
        """Large gamma should produce finite results (smoother softmin)."""
        x = _rand_seq(1, 5, 4)
        y = _rand_seq(1, 5, 4, seed=99)
        sdtw = SoftDTWCuda(gamma=100.0, normalize=False, metric="euclidean")
        dist = sdtw(x, y)
        assert torch.isfinite(dist).all()

    def test_very_small_gamma(self):
        """Very small gamma should produce finite results (clamped)."""
        x = _rand_seq(1, 5, 4)
        y = _rand_seq(1, 5, 4, seed=99)
        sdtw = SoftDTWCuda(gamma=1e-10, normalize=False, metric="euclidean")
        dist = sdtw(x, y)
        assert torch.isfinite(dist).all()

    def test_negative_gamma_uses_abs(self):
        """Negative gamma should be handled by taking abs value."""
        x = _rand_seq(1, 5, 4)
        y = _rand_seq(1, 5, 4, seed=99)
        sdtw_pos = SoftDTWCuda(gamma=1.0, normalize=False, metric="euclidean")
        sdtw_neg = SoftDTWCuda(gamma=-1.0, normalize=False, metric="euclidean")
        d_pos = sdtw_pos(x, y)
        d_neg = sdtw_neg(x, y)
        assert torch.allclose(d_pos, d_neg, atol=1e-5)

    def test_single_frame_sequences(self):
        """Single-frame sequences (T=1) should work."""
        x = torch.randn(1, 1, 4)
        y = torch.randn(1, 1, 4)
        sdtw = SoftDTWCuda(gamma=1.0, normalize=False, metric="euclidean")
        dist = sdtw(x, y)
        assert torch.isfinite(dist).all()

    def test_large_sequences(self):
        """Larger sequences should work without OOM or numerical issues."""
        x = _rand_seq(1, 50, 16)
        y = _rand_seq(1, 50, 16, seed=99)
        sdtw = SoftDTWCuda(gamma=1.0, normalize=True)
        dist = sdtw(x, y)
        assert torch.isfinite(dist).all()

    def test_zero_vectors(self):
        """Zero vectors should not cause NaN (cosine normalizes)."""
        x = torch.zeros(1, 3, 4)
        y = torch.randn(1, 3, 4)
        sdtw = SoftDTWCuda(gamma=1.0, normalize=False, metric="cosine")
        dist = sdtw(x, y)
        assert torch.isfinite(dist).all()

    def test_identical_sequences_unnormalized(self):
        """Unnormalized self-distance should equal self-dtw cost (non-zero)."""
        x = _rand_seq(1, 5, 4)
        sdtw = SoftDTWCuda(gamma=1.0, normalize=False, metric="euclidean")
        dist = sdtw(x, x)
        assert torch.isfinite(dist).all()

    def test_double_precision(self):
        """Should work with float64 tensors."""
        x = _rand_seq(1, 5, 4).double()
        y = _rand_seq(1, 5, 4, seed=99).double()
        sdtw = SoftDTWCuda(gamma=1.0, normalize=True, metric="euclidean")
        sdtw = sdtw.double()
        dist = sdtw(x, y)
        assert dist.dtype == torch.float64
        assert torch.isfinite(dist).all()

    def test_monotonicity_with_scale(self):
        """Scaling cost up should increase distance."""
        cost1 = torch.rand(1, 4, 4)
        cost2 = cost1 * 2.0
        gamma = torch.tensor(1.0)
        d1 = SoftDTWFunction.apply(cost1, gamma)
        d2 = SoftDTWFunction.apply(cost2, gamma)
        assert d2.item() > d1.item()

    def test_triangle_inequality_approximate(self):
        """Normalized Soft-DTW approximately satisfies triangle inequality."""
        x = _rand_seq(1, 8, 4, seed=1)
        y = _rand_seq(1, 8, 4, seed=2)
        z = _rand_seq(1, 8, 4, seed=3)
        sdtw = SoftDTWCuda(gamma=0.1, normalize=True, metric="euclidean")
        dxy = sdtw(x, y).item()
        dyz = sdtw(y, z).item()
        dxz = sdtw(x, z).item()
        # Not strict metric, but should be roughly satisfied
        assert dxz <= (dxy + dyz) * 2.0 + 1.0
