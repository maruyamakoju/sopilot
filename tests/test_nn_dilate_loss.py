"""Tests for DILATE loss module."""

import torch


class TestShapeDTWLoss:
    def test_zero_for_identical(self):
        from sopilot.nn.dilate_loss import ShapeDTWLoss

        loss_fn = ShapeDTWLoss(gamma=0.1)
        x = torch.randn(10, 4)
        loss = loss_fn(x, x)
        assert loss.item() < 1e-3

    def test_gradient_flow(self):
        from sopilot.nn.dilate_loss import ShapeDTWLoss

        loss_fn = ShapeDTWLoss(gamma=1.0)
        x = torch.randn(8, 4, requires_grad=True)
        y = torch.randn(6, 4)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_positive_for_different(self):
        from sopilot.nn.dilate_loss import ShapeDTWLoss

        loss_fn = ShapeDTWLoss(gamma=0.1)
        x = torch.ones(5, 4)
        y = torch.ones(5, 4) * 10
        loss = loss_fn(x, y)
        assert loss.item() > 0


class TestTemporalDistortionLoss:
    def test_zero_for_diagonal(self):
        from sopilot.nn.dilate_loss import TemporalDistortionLoss

        loss_fn = TemporalDistortionLoss(gamma=0.1)
        # Identical sequences should have near-diagonal alignment
        x = torch.randn(10, 4)
        loss = loss_fn(x, x)
        assert loss.item() < 0.5  # Should be small for identical

    def test_gradient_flow(self):
        from sopilot.nn.dilate_loss import TemporalDistortionLoss

        loss_fn = TemporalDistortionLoss(gamma=1.0)
        x = torch.randn(6, 4, requires_grad=True)
        y = torch.randn(8, 4)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None


class TestDILATELoss:
    def test_alpha_one_gives_pure_shape(self):
        from sopilot.nn.dilate_loss import DILATELoss, ShapeDTWLoss

        dilate = DILATELoss(alpha=1.0, gamma=1.0)
        shape_only = ShapeDTWLoss(gamma=1.0)

        x = torch.randn(8, 4)
        y = torch.randn(6, 4)

        total, comp = dilate(x, y)
        shape_loss = shape_only(x, y)

        assert abs(total.item() - shape_loss.item()) < 1e-4

    def test_components_returned(self):
        from sopilot.nn.dilate_loss import DILATELoss

        dilate = DILATELoss(alpha=0.5, gamma=1.0)
        x = torch.randn(8, 4)
        y = torch.randn(6, 4)

        total, comp = dilate(x, y)
        assert "shape" in comp
        assert "temporal" in comp
        assert "alignment" in comp
        assert comp["alignment"].shape == (8, 6)

    def test_gradient_flow_both_components(self):
        from sopilot.nn.dilate_loss import DILATELoss

        dilate = DILATELoss(alpha=0.5, gamma=1.0)
        x = torch.randn(6, 4, requires_grad=True)
        y = torch.randn(8, 4)

        total, _ = dilate(x, y)
        total.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_batched(self):
        from sopilot.nn.dilate_loss import DILATELoss

        dilate = DILATELoss(alpha=0.5, gamma=1.0)
        x = torch.randn(2, 8, 4)
        y = torch.randn(2, 6, 4)

        total, comp = dilate(x, y)
        assert total.dim() == 0  # Scalar


class TestSOPDilateLoss:
    def test_boundary_alignment(self):
        from sopilot.nn.dilate_loss import SOPDilateLoss

        loss_fn = SOPDilateLoss(gamma=1.0)
        gold = torch.randn(20, 4)
        trainee = torch.randn(18, 4)
        boundaries = [0, 5, 10, 15, 20]

        total, comp = loss_fn(trainee, gold, boundaries)
        assert "boundary" in comp
        assert "order" in comp
        assert "coverage" in comp
        assert total.dim() == 0

    def test_gradient_flow(self):
        from sopilot.nn.dilate_loss import SOPDilateLoss

        loss_fn = SOPDilateLoss(gamma=1.0)
        gold = torch.randn(15, 4)
        trainee = torch.randn(12, 4, requires_grad=True)
        boundaries = [0, 5, 10, 15]

        total, _ = loss_fn(trainee, gold, boundaries)
        total.backward()
        assert trainee.grad is not None
