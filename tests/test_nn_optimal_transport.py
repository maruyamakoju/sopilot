"""Tests for nn.optimal_transport -- Optimal Transport for procedure alignment."""

from __future__ import annotations

import pytest
import torch

from sopilot.nn.optimal_transport import (
    FusedGromovWasserstein,
    GromovWassersteinDistance,
    HierarchicalOTAlignment,
    SinkhornDistance,
    WassersteinBarycenter,
    _cosine_cost_matrix,
    _euclidean_cost_matrix,
)

# ---------------------------------------------------------------------------
# Sinkhorn tests
# ---------------------------------------------------------------------------


class TestSinkhornMarginals:
    """Test 1: Transport plan marginals match input distributions."""

    def test_uniform_marginals(self) -> None:
        torch.manual_seed(42)
        C = torch.rand(5, 8)
        solver = SinkhornDistance(epsilon=0.1, max_iter=200)
        dist, P = solver(C)
        row_sums = P.sum(dim=1)
        col_sums = P.sum(dim=0)
        expected_row = torch.full((5,), 1.0 / 5)
        expected_col = torch.full((8,), 1.0 / 8)
        assert torch.allclose(row_sums, expected_row, atol=1e-4)
        assert torch.allclose(col_sums, expected_col, atol=1e-4)

    def test_custom_marginals(self) -> None:
        torch.manual_seed(42)
        C = torch.rand(4, 6)
        a = torch.tensor([0.1, 0.2, 0.3, 0.4])
        b = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
        solver = SinkhornDistance(epsilon=0.1, max_iter=200)
        dist, P = solver(C, a, b)
        assert torch.allclose(P.sum(dim=1), a, atol=1e-4)
        assert torch.allclose(P.sum(dim=0), b, atol=1e-4)


class TestSinkhornIdentical:
    """Test 2: Identical distributions -> 0 distance."""

    def test_zero_cost_zero_distance(self) -> None:
        C = torch.zeros(5, 5)
        solver = SinkhornDistance(epsilon=0.1, max_iter=100)
        dist, P = solver(C)
        assert abs(dist.item()) < 1e-6

    def test_identity_cost_symmetric(self) -> None:
        """Transporting identical distributions over symmetric cost."""
        torch.manual_seed(42)
        n = 6
        x = torch.randn(n, 16)
        C = _euclidean_cost_matrix(x, x).squeeze(0)
        solver = SinkhornDistance(epsilon=0.1, max_iter=200)
        dist, P = solver(C)
        # Self-transport should be cheap (close to diagonal)
        diag_mass = P.diag().sum().item()
        assert diag_mass > 0.5 / n * n  # Most mass on diagonal


class TestSinkhornGradient:
    """Test 3: Gradient flow through distance."""

    def test_gradient_through_cost(self) -> None:
        torch.manual_seed(42)
        C = torch.rand(5, 8, requires_grad=True)
        solver = SinkhornDistance(epsilon=0.1, max_iter=100)
        dist, P = solver(C)
        dist.backward()
        assert C.grad is not None
        assert torch.isfinite(C.grad).all()

    def test_gradient_through_marginals(self) -> None:
        torch.manual_seed(42)
        C = torch.rand(4, 4)
        a = torch.tensor([0.25, 0.25, 0.25, 0.25], requires_grad=True)
        solver = SinkhornDistance(epsilon=0.1, max_iter=100)
        dist, P = solver(C, a)
        dist.backward()
        assert a.grad is not None


class TestSinkhornLogDomain:
    """Test 4: Log-domain matches linear-domain for well-conditioned problems."""

    def test_log_domain_produces_valid_plan(self) -> None:
        torch.manual_seed(42)
        C = torch.rand(6, 6) + 0.1  # Ensure strictly positive costs
        solver = SinkhornDistance(epsilon=0.5, max_iter=200)
        dist, P = solver(C)
        assert (P >= -1e-8).all(), "Transport plan has negative entries"
        assert torch.isfinite(P).all()
        assert torch.isfinite(dist)

    def test_epsilon_scaling_matches_standard(self) -> None:
        """Epsilon-scaling should converge to similar result as standard."""
        torch.manual_seed(42)
        C = torch.rand(8, 8) + 0.1
        solver_std = SinkhornDistance(epsilon=0.05, max_iter=500)
        solver_scale = SinkhornDistance(epsilon=0.05, max_iter=500, epsilon_scaling=True, scaling_steps=5)
        dist_std, P_std = solver_std(C)
        dist_scale, P_scale = solver_scale(C)
        # Results should be close (not exact due to different convergence paths)
        assert abs(dist_std.item() - dist_scale.item()) < 0.05


class TestSinkhornConvergence:
    """Test 5: Convergence within max_iter for typical problems."""

    def test_converges_for_moderate_epsilon(self) -> None:
        torch.manual_seed(42)
        C = torch.rand(10, 10)
        solver = SinkhornDistance(epsilon=0.1, max_iter=100, tol=1e-6)
        dist, P = solver(C)
        # Verify marginals are close to uniform
        row_sums = P.sum(dim=1)
        expected = torch.full((10,), 0.1)
        assert torch.allclose(row_sums, expected, atol=1e-4)

    def test_unbalanced_ot(self) -> None:
        """Unbalanced OT should produce non-negative plan with relaxed marginals."""
        torch.manual_seed(42)
        C = torch.rand(5, 8)
        solver = SinkhornDistance(epsilon=0.1, max_iter=200, unbalanced_tau=1.0)
        dist, P = solver(C)
        assert (P >= -1e-8).all()
        assert torch.isfinite(dist)


# ---------------------------------------------------------------------------
# GW tests
# ---------------------------------------------------------------------------


class TestGWIdenticalStructures:
    """Test 6: Identical structures -> low distance vs different structures."""

    def test_same_structure_lower_than_different(self) -> None:
        """GW(D, D) should be much less than GW(D1, D2) for random D1, D2."""
        torch.manual_seed(42)
        # Use small normalized distance matrices for numerical stability
        D = torch.rand(5, 5)
        D = (D + D.T) / 2
        D.fill_diagonal_(0.0)
        D = D / D.max()  # Normalize to [0, 1]
        D2 = torch.rand(5, 5)
        D2 = (D2 + D2.T) / 2
        D2.fill_diagonal_(0.0)
        D2 = D2 / D2.max()
        gw = GromovWassersteinDistance(epsilon=0.1, max_outer_iter=80)
        cost_same, _ = gw(D, D)
        cost_diff, _ = gw(D, D2)
        # Self-comparison should be cheaper than cross-comparison
        assert cost_same.item() <= cost_diff.item() + 0.01, (
            f"Same-structure cost {cost_same.item()} should be <= different-structure cost {cost_diff.item()}"
        )

    def test_identity_distance_matrix(self) -> None:
        """Identity matrix as distance should give near-zero GW with itself."""
        D = torch.eye(4) * 0.0  # Zero distance matrix (all points identical)
        gw = GromovWassersteinDistance(epsilon=0.1, max_outer_iter=50)
        cost, P = gw(D, D)
        assert cost.item() < 1e-6, f"Expected near-zero cost, got {cost.item()}"


class TestGWGradient:
    """Test 7: Gradient flow through GW distance."""

    def test_gradient_through_distance_matrices(self) -> None:
        torch.manual_seed(42)
        D1 = torch.rand(4, 4, requires_grad=True)
        D2 = torch.rand(4, 4)
        gw = GromovWassersteinDistance(epsilon=0.05, max_outer_iter=20)
        cost, P = gw(D1, D2)
        cost.backward()
        assert D1.grad is not None
        assert torch.isfinite(D1.grad).all()


class TestGWEfficientGradient:
    """Test 8: Efficient gradient matches naive for small problems."""

    def test_gradient_matches_naive(self) -> None:
        torch.manual_seed(42)
        M, N = 3, 4
        D1 = torch.rand(1, M, M)
        D1 = D1 + D1.transpose(1, 2)  # Symmetric
        D2 = torch.rand(1, N, N)
        D2 = D2 + D2.transpose(1, 2)
        # Uniform marginals -> outer product
        a = torch.full((1, M), 1.0 / M)
        b = torch.full((1, N), 1.0 / N)
        P = a.unsqueeze(2) * b.unsqueeze(1)
        G_efficient = GromovWassersteinDistance._compute_gw_gradient(D1, D2, P)
        G_naive = GromovWassersteinDistance._compute_gw_gradient_naive(D1, D2, P)
        assert torch.allclose(G_efficient, G_naive, atol=1e-5), (
            f"Max diff: {(G_efficient - G_naive).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Fused GW tests
# ---------------------------------------------------------------------------


class TestFusedGWAlpha:
    """Test 9: alpha=1 matches pure Sinkhorn, alpha=0 matches pure GW."""

    def test_alpha_one_is_sinkhorn(self) -> None:
        """With alpha=1, only feature cost matters (pure Sinkhorn)."""
        torch.manual_seed(42)
        M, N = 5, 5
        C = torch.rand(M, N)
        D1 = torch.rand(M, M)
        D2 = torch.rand(N, N)
        eps = 0.1
        fgw = FusedGromovWasserstein(alpha=1.0, epsilon=eps, max_outer_iter=50, max_inner_iter=100, tol=1e-6)
        cost_fgw, P_fgw = fgw(C, D1, D2)
        # Compare with pure Sinkhorn
        sink = SinkhornDistance(epsilon=eps, max_iter=100, tol=1e-6)
        cost_sink, P_sink = sink(C)
        # Transport plans should be very similar
        assert torch.allclose(P_fgw, P_sink, atol=0.05), f"Max diff: {(P_fgw - P_sink).abs().max().item()}"

    def test_alpha_zero_ignores_features(self) -> None:
        """With alpha=0, only structure (GW) matters."""
        torch.manual_seed(42)
        M, N = 5, 5
        # Two different feature costs but same structure
        C1 = torch.rand(M, N)
        C2 = torch.rand(M, N) * 10
        D1 = torch.rand(M, M)
        D1 = D1 + D1.T
        D2 = torch.rand(N, N)
        D2 = D2 + D2.T
        eps = 0.05
        fgw1 = FusedGromovWasserstein(alpha=0.0, epsilon=eps, max_outer_iter=50)
        fgw2 = FusedGromovWasserstein(alpha=0.0, epsilon=eps, max_outer_iter=50)
        _, P1 = fgw1(C1, D1, D2)
        _, P2 = fgw2(C2, D1, D2)
        # Plans should be identical since feature cost is ignored
        assert torch.allclose(P1, P2, atol=0.05)


# ---------------------------------------------------------------------------
# Hierarchical OT tests
# ---------------------------------------------------------------------------


class TestHierarchicalConstraint:
    """Test 10: Coarse alignment constrains fine alignment."""

    def test_constrained_plan_respects_phases(self) -> None:
        torch.manual_seed(42)
        T, D = 20, 16
        # Create two sequences with clear phase structure
        x = torch.zeros(T, D)
        y = torch.zeros(T, D)
        # Phase 1: frames 0-9 similar, Phase 2: frames 10-19 similar
        x[:10] = torch.randn(1, D).expand(10, D)
        x[10:] = torch.randn(1, D).expand(10, D)
        y[:10] = x[:10] + 0.01 * torch.randn(10, D)
        y[10:] = x[10:] + 0.01 * torch.randn(10, D)

        model = HierarchicalOTAlignment(n_phases=2, epsilon_coarse=0.1, epsilon_fine=0.05, max_iter=50)
        dist, P_fine, P_coarse = model(x, y)

        # Fine plan should have most mass in diagonal blocks
        block1_mass = P_fine[:10, :10].sum().item()
        block2_mass = P_fine[10:, 10:].sum().item()
        off_diag_mass = P_fine[:10, 10:].sum().item() + P_fine[10:, :10].sum().item()
        assert block1_mass + block2_mass > off_diag_mass, (
            f"Diagonal blocks ({block1_mass + block2_mass:.3f}) should dominate off-diagonal ({off_diag_mass:.3f})"
        )

    def test_hierarchical_output_shapes(self) -> None:
        torch.manual_seed(42)
        model = HierarchicalOTAlignment(n_phases=3, max_iter=10)
        x = torch.randn(12, 16)
        y = torch.randn(15, 16)
        dist, P_fine, P_coarse = model(x, y)
        assert P_fine.shape == (12, 15)
        assert P_coarse.shape == (3, 3)
        assert torch.isfinite(dist)


# ---------------------------------------------------------------------------
# Barycenter tests
# ---------------------------------------------------------------------------


class TestBarycenterIdentical:
    """Test 11: Barycenter of identical distributions is that distribution."""

    def test_identical_distributions(self) -> None:
        torch.manual_seed(42)
        n = 5
        p = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.2])
        # Cost matrix: squared Euclidean between 1-D support points
        support = torch.arange(n, dtype=torch.float32)
        C = (support.unsqueeze(1) - support.unsqueeze(0)) ** 2
        bary = WassersteinBarycenter(epsilon=0.05, max_iter=100, max_outer_iter=30)
        result = bary([p, p, p], [C, C, C])
        assert torch.allclose(result, p, atol=0.05), f"Barycenter {result} should match input {p}"

    def test_uniform_barycenter(self) -> None:
        torch.manual_seed(42)
        n = 4
        p = torch.full((n,), 1.0 / n)
        support = torch.arange(n, dtype=torch.float32)
        C = (support.unsqueeze(1) - support.unsqueeze(0)) ** 2
        bary = WassersteinBarycenter(epsilon=0.05, max_iter=100, max_outer_iter=30)
        result = bary([p, p], [C, C])
        assert torch.allclose(result, p, atol=0.05)


# ---------------------------------------------------------------------------
# Batched computation tests
# ---------------------------------------------------------------------------


class TestBatchedComputation:
    """Test 12: Batched computation matches single."""

    def test_sinkhorn_batched(self) -> None:
        torch.manual_seed(42)
        C1 = torch.rand(5, 8)
        C2 = torch.rand(5, 8)
        C_batch = torch.stack([C1, C2], dim=0)
        solver = SinkhornDistance(epsilon=0.1, max_iter=100)
        # Batched
        dist_batch, P_batch = solver(C_batch)
        # Single
        dist1, P1 = solver(C1)
        dist2, P2 = solver(C2)
        assert torch.allclose(dist_batch[0], dist1, atol=1e-5)
        assert torch.allclose(dist_batch[1], dist2, atol=1e-5)
        assert torch.allclose(P_batch[0], P1, atol=1e-5)
        assert torch.allclose(P_batch[1], P2, atol=1e-5)

    def test_gw_batched(self) -> None:
        torch.manual_seed(42)
        D1_a = torch.rand(4, 4)
        D1_a = D1_a + D1_a.T
        D2_a = torch.rand(4, 4)
        D2_a = D2_a + D2_a.T
        D1_b = torch.rand(4, 4)
        D1_b = D1_b + D1_b.T
        D2_b = torch.rand(4, 4)
        D2_b = D2_b + D2_b.T
        D1_batch = torch.stack([D1_a, D1_b])
        D2_batch = torch.stack([D2_a, D2_b])
        gw = GromovWassersteinDistance(epsilon=0.05, max_outer_iter=30)
        cost_batch, P_batch = gw(D1_batch, D2_batch)
        cost_a, P_a = gw(D1_a, D2_a)
        cost_b, P_b = gw(D1_b, D2_b)
        assert torch.allclose(cost_batch[0], cost_a, atol=1e-4)
        assert torch.allclose(cost_batch[1], cost_b, atol=1e-4)


# ---------------------------------------------------------------------------
# GPU / CPU consistency tests
# ---------------------------------------------------------------------------


class TestGPUCPU:
    """Test 13: GPU matches CPU (skipped if no GPU available)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
    def test_sinkhorn_gpu_cpu(self) -> None:
        torch.manual_seed(42)
        C_cpu = torch.rand(6, 8)
        C_gpu = C_cpu.cuda()
        solver = SinkhornDistance(epsilon=0.1, max_iter=100)
        dist_cpu, P_cpu = solver(C_cpu)
        dist_gpu, P_gpu = solver(C_gpu)
        assert torch.allclose(dist_cpu, dist_gpu.cpu(), atol=1e-4)
        assert torch.allclose(P_cpu, P_gpu.cpu(), atol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
    def test_gw_gpu_cpu(self) -> None:
        torch.manual_seed(42)
        D1_cpu = torch.rand(4, 4)
        D1_cpu = D1_cpu + D1_cpu.T
        D2_cpu = torch.rand(4, 4)
        D2_cpu = D2_cpu + D2_cpu.T
        gw = GromovWassersteinDistance(epsilon=0.05, max_outer_iter=30)
        cost_cpu, _ = gw(D1_cpu, D2_cpu)
        cost_gpu, _ = gw(D1_cpu.cuda(), D2_cpu.cuda())
        assert torch.allclose(cost_cpu, cost_gpu.cpu(), atol=1e-3)


# ---------------------------------------------------------------------------
# Numerical stability tests
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Test 14: Numerical stability with extreme values."""

    def test_large_cost_values(self) -> None:
        """Sinkhorn should handle large cost values without NaN."""
        torch.manual_seed(42)
        C = torch.rand(5, 5) * 100
        solver = SinkhornDistance(epsilon=1.0, max_iter=200)
        dist, P = solver(C)
        assert torch.isfinite(dist)
        assert torch.isfinite(P).all()
        assert (P >= -1e-8).all()

    def test_very_small_epsilon(self) -> None:
        """Small epsilon should still produce finite results."""
        torch.manual_seed(42)
        C = torch.rand(5, 5) + 0.1
        solver = SinkhornDistance(epsilon=0.001, max_iter=500)
        dist, P = solver(C)
        assert torch.isfinite(dist)
        assert torch.isfinite(P).all()

    def test_near_zero_marginals(self) -> None:
        """Very small marginal entries should not cause NaN."""
        torch.manual_seed(42)
        C = torch.rand(4, 4)
        a = torch.tensor([0.001, 0.001, 0.499, 0.499])
        solver = SinkhornDistance(epsilon=0.1, max_iter=200)
        dist, P = solver(C, a)
        assert torch.isfinite(dist)
        assert torch.isfinite(P).all()

    def test_asymmetric_sizes(self) -> None:
        """Highly asymmetric sizes should work."""
        torch.manual_seed(42)
        C = torch.rand(3, 20)
        solver = SinkhornDistance(epsilon=0.1, max_iter=200)
        dist, P = solver(C)
        assert torch.isfinite(dist)
        assert P.shape == (3, 20)
        row_sums = P.sum(dim=1)
        expected = torch.full((3,), 1.0 / 3)
        assert torch.allclose(row_sums, expected, atol=1e-4)

    def test_gw_with_zero_diagonal(self) -> None:
        """GW should handle proper distance matrices (zero diagonal)."""
        torch.manual_seed(42)
        D = torch.rand(5, 5)
        D = D + D.T
        D.fill_diagonal_(0.0)
        gw = GromovWassersteinDistance(epsilon=0.05, max_outer_iter=30)
        cost, P = gw(D, D)
        assert torch.isfinite(cost)
        assert torch.isfinite(P).all()


# ---------------------------------------------------------------------------
# Additional integration tests
# ---------------------------------------------------------------------------


class TestCostMatrices:
    """Verify helper cost matrix functions."""

    def test_cosine_cost_self_zero_diagonal(self) -> None:
        x = torch.randn(1, 5, 16)
        C = _cosine_cost_matrix(x, x)
        diag = C[0].diag()
        assert torch.allclose(diag, torch.zeros(5), atol=1e-6)

    def test_euclidean_cost_self_zero_diagonal(self) -> None:
        x = torch.randn(1, 5, 16)
        C = _euclidean_cost_matrix(x, x)
        diag = C[0].diag()
        assert torch.allclose(diag, torch.zeros(5), atol=1e-5)

    def test_cost_non_negative(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(2, 5, 16)
        y = torch.randn(2, 8, 16)
        assert (_cosine_cost_matrix(x, y) >= -1e-6).all()
        assert (_euclidean_cost_matrix(x, y) >= -1e-6).all()
