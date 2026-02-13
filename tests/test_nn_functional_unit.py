"""Unit tests for nn/functional.py."""

import pytest
import torch

from sopilot.nn.functional import pairwise_cosine_dist, pairwise_euclidean_sq, softmin3


class TestSoftmin3:
    """Tests for soft-minimum of three values."""

    def test_equal_inputs(self):
        a = torch.tensor([1.0])
        result = softmin3(a, a, a, gamma=1.0)
        # softmin_γ(v,v,v) = v - γ*ln(3)
        expected = 1.0 - 1.0 * torch.log(torch.tensor(3.0))
        assert torch.allclose(result, expected, atol=1e-5)

    def test_one_dominant(self):
        a = torch.tensor([0.0])
        b = torch.tensor([100.0])
        c = torch.tensor([100.0])
        result = softmin3(a, b, c, gamma=1.0)
        # With large gap, softmin ≈ hard min = 0.0
        assert result.item() == pytest.approx(0.0, abs=0.1)

    def test_gamma_as_tensor(self):
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        c = torch.tensor([3.0])
        gamma_t = torch.tensor(1.0)
        gamma_f = 1.0
        result_t = softmin3(a, b, c, gamma_t)
        result_f = softmin3(a, b, c, gamma_f)
        assert torch.allclose(result_t, result_f, atol=1e-6)

    def test_small_gamma_approaches_hard_min(self):
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        c = torch.tensor([3.0])
        result = softmin3(a, b, c, gamma=0.001)
        assert result.item() == pytest.approx(1.0, abs=0.01)

    def test_batched(self):
        a = torch.tensor([1.0, 5.0])
        b = torch.tensor([2.0, 3.0])
        c = torch.tensor([3.0, 1.0])
        result = softmin3(a, b, c, gamma=0.001)
        assert result.shape == (2,)
        assert result[0].item() == pytest.approx(1.0, abs=0.01)
        assert result[1].item() == pytest.approx(1.0, abs=0.01)


class TestPairwiseEuclideanSq:
    """Tests for pairwise squared Euclidean distance."""

    def test_identity_zero_diagonal(self):
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        D = pairwise_euclidean_sq(x, x)
        assert D.shape == (2, 2)
        assert D[0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert D[1, 1].item() == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        x = torch.tensor([[0.0, 0.0]])
        y = torch.tensor([[3.0, 4.0]])
        D = pairwise_euclidean_sq(x, y)
        assert D[0, 0].item() == pytest.approx(25.0, abs=1e-5)

    def test_symmetric(self):
        x = torch.randn(3, 4)
        y = torch.randn(5, 4)
        D_xy = pairwise_euclidean_sq(x, y)
        D_yx = pairwise_euclidean_sq(y, x)
        assert torch.allclose(D_xy, D_yx.T, atol=1e-5)

    def test_2d_shape(self):
        x = torch.randn(3, 8)
        y = torch.randn(5, 8)
        D = pairwise_euclidean_sq(x, y)
        assert D.shape == (3, 5)

    def test_3d_batched(self):
        x = torch.randn(2, 3, 8)
        y = torch.randn(2, 5, 8)
        D = pairwise_euclidean_sq(x, y)
        assert D.shape == (2, 3, 5)

    def test_non_negative(self):
        x = torch.randn(4, 6)
        y = torch.randn(7, 6)
        D = pairwise_euclidean_sq(x, y)
        assert (D >= 0).all()

    def test_2d_promotes_to_batch1(self):
        """2D input promotes to (1, M, D) for batched path."""
        x = torch.randn(3, 4)
        y = torch.randn(1, 5, 4)
        D = pairwise_euclidean_sq(x, y)
        assert D.shape == (1, 3, 5)


class TestPairwisCosineDist:
    """Tests for pairwise cosine distance."""

    def test_identical_vectors(self):
        x = torch.tensor([[1.0, 0.0, 0.0]])
        D = pairwise_cosine_dist(x, x)
        assert D[0, 0].item() == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        x = torch.tensor([[1.0, 0.0]])
        y = torch.tensor([[0.0, 1.0]])
        D = pairwise_cosine_dist(x, y)
        assert D[0, 0].item() == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors(self):
        x = torch.tensor([[1.0, 0.0]])
        y = torch.tensor([[-1.0, 0.0]])
        D = pairwise_cosine_dist(x, y)
        assert D[0, 0].item() == pytest.approx(2.0, abs=1e-6)

    def test_range(self):
        x = torch.randn(5, 8)
        y = torch.randn(3, 8)
        D = pairwise_cosine_dist(x, y)
        assert (D >= -1e-6).all()
        assert (D <= 2.0 + 1e-6).all()

    def test_2d_shape(self):
        x = torch.randn(3, 8)
        y = torch.randn(5, 8)
        D = pairwise_cosine_dist(x, y)
        assert D.shape == (3, 5)

    def test_3d_batched(self):
        x = torch.randn(2, 3, 8)
        y = torch.randn(2, 5, 8)
        D = pairwise_cosine_dist(x, y)
        assert D.shape == (2, 3, 5)

    def test_gradients_flow(self):
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(5, 4, requires_grad=True)
        D = pairwise_cosine_dist(x, y)
        loss = D.sum()
        loss.backward()
        assert x.grad is not None
        assert y.grad is not None
