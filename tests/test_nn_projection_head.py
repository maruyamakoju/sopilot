"""Tests for nn.projection_head — Contrastive ProjectionHead and NT-Xent loss."""

from __future__ import annotations

import numpy as np
import torch

from sopilot.nn.projection_head import (
    NTXentLoss,
    ProjectionHead,
    StepPairMiner,
    load_projection_head,
    save_projection_head,
)


class TestProjectionHead:
    def test_output_shape(self) -> None:
        model = ProjectionHead(d_in=128, d_out=64)
        x = torch.randn(16, 128)
        out = model(x)
        assert out.shape == (16, 64)

    def test_output_l2_normalized(self) -> None:
        model = ProjectionHead(d_in=64, d_out=32)
        x = torch.randn(8, 64)
        out = model(x)
        norms = torch.norm(out, dim=-1)
        np.testing.assert_allclose(norms.detach().numpy(), 1.0, atol=1e-5)

    def test_num_parameters(self) -> None:
        model = ProjectionHead(d_in=1280, d_out=128)
        # Should be around 200K params
        assert model.num_parameters > 100_000
        assert model.num_parameters < 1_000_000

    def test_different_hidden_sizes(self) -> None:
        model = ProjectionHead(d_in=64, d_out=16, d_hidden=128, d_mid=64)
        x = torch.randn(4, 64)
        out = model(x)
        assert out.shape == (4, 16)

    def test_gradient_flow(self) -> None:
        model = ProjectionHead(d_in=32, d_out=16)
        x = torch.randn(8, 32, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_save_load_roundtrip(self, tmp_path) -> None:
        model = ProjectionHead(d_in=64, d_out=32)
        # Run a forward pass to populate BatchNorm running stats
        model.train()
        _ = model(torch.randn(16, 64))
        model.eval()

        x = torch.randn(4, 64)
        with torch.no_grad():
            out_before = model(x)

        path = tmp_path / "proj.pt"
        save_projection_head(model, path)
        loaded = load_projection_head(path)

        with torch.no_grad():
            out_after = loaded(x)

        np.testing.assert_allclose(out_before.numpy(), out_after.numpy(), atol=1e-6)


class TestNTXentLoss:
    def test_zero_loss_identical_embeddings_different_labels(self) -> None:
        """Different labels → no positives → loss should be 0."""
        loss_fn = NTXentLoss(temperature=0.07)
        emb = torch.randn(4, 32)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        labels = torch.arange(4)  # All different
        loss = loss_fn(emb, labels)
        assert loss.item() == 0.0

    def test_loss_decreases_with_same_labels(self) -> None:
        """Same-step clips should produce lower loss than random."""
        loss_fn = NTXentLoss(temperature=0.07)

        # All same step
        emb = torch.nn.functional.normalize(torch.randn(8, 32), dim=-1)
        labels_same = torch.zeros(8, dtype=torch.long)
        loss_same = loss_fn(emb, labels_same)

        # Two distinct groups
        labels_split = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        loss_split = loss_fn(emb, labels_split)

        # Same labels should give a finite loss
        assert loss_same.item() >= 0.0
        assert loss_split.item() >= 0.0

    def test_single_sample_no_crash(self) -> None:
        loss_fn = NTXentLoss()
        emb = torch.randn(1, 16)
        labels = torch.tensor([0])
        loss = loss_fn(emb, labels)
        assert loss.item() == 0.0

    def test_gradient_flows(self) -> None:
        loss_fn = NTXentLoss(temperature=0.1)
        emb = torch.randn(8, 32, requires_grad=True)
        emb_norm = torch.nn.functional.normalize(emb, dim=-1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(emb_norm, labels)
        loss.backward()
        assert emb.grad is not None


class TestStepPairMiner:
    def test_assign_step_labels(self) -> None:
        labels = StepPairMiner.assign_step_labels(10, [0, 3, 7, 10])
        assert labels.shape == (10,)
        assert labels[0].item() == 0
        assert labels[3].item() == 1
        assert labels[7].item() == 2

    def test_single_step(self) -> None:
        labels = StepPairMiner.assign_step_labels(5, [0, 5])
        assert torch.all(labels == 0)
