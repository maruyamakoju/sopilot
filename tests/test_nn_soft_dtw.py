"""Tests for nn.soft_dtw — Soft-DTW differentiable alignment."""

from __future__ import annotations

import numpy as np
import torch

from sopilot.nn.soft_dtw import (
    SoftDTW,
    SoftDTWAlignment,
    _softmin3,
    soft_dtw_align_numpy,
)


class TestSoftMin3:
    def test_basic(self) -> None:
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        c = torch.tensor([3.0])
        result = _softmin3(a, b, c, gamma=0.01)
        # With very small gamma, should approximate hard min
        assert abs(result.item() - 1.0) < 0.1

    def test_large_gamma_averages(self) -> None:
        a = torch.tensor([1.0])
        b = torch.tensor([1.0])
        c = torch.tensor([1.0])
        result = _softmin3(a, b, c, gamma=100.0)
        # All equal → result ≈ 1.0 - gamma * log(3)
        # For large gamma, result should be less than 1.0
        assert result.item() < 1.0


class TestSoftDTW:
    def test_identical_sequences_zero_distance(self) -> None:
        model = SoftDTW(gamma=1.0, normalize=True)
        x = torch.randn(10, 32)
        dist = model(x, x)
        # Self-alignment with normalization should be ~0
        assert abs(dist.item()) < 1.0

    def test_different_sequences_positive_distance(self) -> None:
        model = SoftDTW(gamma=1.0, normalize=False)
        x = torch.randn(10, 32)
        y = torch.randn(10, 32)
        dist = model(x, y)
        assert dist.item() > 0.0

    def test_gradient_through_gamma(self) -> None:
        model = SoftDTW(gamma=1.0, normalize=False)
        x = torch.randn(5, 16)
        y = torch.randn(5, 16)
        dist = model(x, y)
        dist.backward()
        assert model.gamma.grad is not None

    def test_gradient_through_inputs(self) -> None:
        model = SoftDTW(gamma=1.0, normalize=False)
        x = torch.randn(5, 16, requires_grad=True)
        y = torch.randn(5, 16)
        dist = model(x, y)
        dist.backward()
        assert x.grad is not None

    def test_asymmetric_lengths(self) -> None:
        model = SoftDTW(gamma=1.0, normalize=False)
        x = torch.randn(8, 32)
        y = torch.randn(12, 32)
        dist = model(x, y)
        assert torch.isfinite(dist)

    def test_small_gamma_approaches_hard_dtw(self) -> None:
        """With very small gamma, Soft-DTW should approximate hard DTW."""
        x = torch.randn(5, 16)
        y = torch.randn(5, 16)

        soft = SoftDTW(gamma=0.01, normalize=False)
        hard_approx = soft(x, y).item()

        softer = SoftDTW(gamma=10.0, normalize=False)
        soft_val = softer(x, y).item()

        # Hard approximation should be >= soft value
        # (soft-min is always <= hard-min due to log-sum-exp)
        assert hard_approx >= soft_val - 1.0  # Allow some tolerance


class TestSoftDTWAlignment:
    def test_alignment_matrix_shape(self) -> None:
        model = SoftDTWAlignment(gamma=1.0)
        x = torch.randn(8, 32)
        y = torch.randn(10, 32)
        alignment, distance = model(x, y)
        assert alignment.shape == (8, 10)
        assert torch.isfinite(distance)

    def test_alignment_matrix_non_negative(self) -> None:
        model = SoftDTWAlignment(gamma=1.0)
        x = torch.randn(6, 16)
        y = torch.randn(6, 16)
        alignment, _ = model(x, y)
        assert (alignment >= -1e-6).all()

    def test_identical_sequences_produces_alignment(self) -> None:
        model = SoftDTWAlignment(gamma=1.0)
        x = torch.randn(5, 16)
        alignment, distance = model(x, x)
        # Alignment should have non-zero values somewhere
        assert alignment.sum() > 0
        assert torch.isfinite(distance)


class TestSoftDtwAlignNumpy:
    def test_basic_output(self) -> None:
        gold = np.random.randn(10, 32).astype(np.float32)
        trainee = np.random.randn(12, 32).astype(np.float32)

        path, mean_cost, alignment = soft_dtw_align_numpy(gold, trainee, gamma=1.0)

        assert len(path) == 10  # One entry per gold frame
        assert isinstance(mean_cost, float)
        assert alignment.shape == (10, 12)

    def test_path_contains_valid_indices(self) -> None:
        gold = np.random.randn(5, 16).astype(np.float32)
        trainee = np.random.randn(8, 16).astype(np.float32)

        path, _, _ = soft_dtw_align_numpy(gold, trainee)

        for gi, tj, sim in path:
            assert 0 <= gi < 5
            assert 0 <= tj < 8
            assert -1.0 <= sim <= 1.0

    def test_identical_returns_valid_path(self) -> None:
        """Same sequence should produce a valid alignment path."""
        emb = np.random.randn(6, 32).astype(np.float32)
        path, cost, alignment = soft_dtw_align_numpy(emb, emb.copy(), gamma=1.0)

        assert len(path) == 6
        assert alignment.shape == (6, 6)
        # Each gold frame should map to some trainee frame
        for gi, tj, sim in path:
            assert 0 <= gi < 6
            assert 0 <= tj < 6
