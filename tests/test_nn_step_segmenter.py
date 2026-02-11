"""Tests for nn.step_segmenter — Neural temporal action segmentation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sopilot.nn.step_segmenter import (
    NeuralStepSegmenter,
    DilatedResidualBlock,
    SegmentationLoss,
    generate_pseudo_labels,
    save_segmenter,
    load_segmenter,
)


class TestDilatedResidualBlock:
    def test_output_shape(self) -> None:
        block = DilatedResidualBlock(channels=64, dilation=2)
        x = torch.randn(2, 64, 20)
        out = block(x)
        assert out.shape == (2, 64, 20)

    def test_residual_connection(self) -> None:
        """Output should differ from input (not identity)."""
        block = DilatedResidualBlock(channels=32, dilation=1)
        x = torch.randn(1, 32, 10)
        out = block(x)
        assert not torch.allclose(x, out, atol=1e-6)


class TestNeuralStepSegmenter:
    def test_output_shapes(self) -> None:
        model = NeuralStepSegmenter(d_in=128)
        x = torch.randn(2, 128, 50)
        logits1, logits2 = model(x)
        assert logits1.shape == (2, 2, 50)
        assert logits2.shape == (2, 2, 50)

    def test_num_parameters(self) -> None:
        model = NeuralStepSegmenter(d_in=128)
        assert model.num_parameters > 10_000
        assert model.num_parameters < 500_000

    def test_predict_boundaries_basic(self) -> None:
        model = NeuralStepSegmenter(d_in=32)
        model.eval()
        embeddings = np.random.randn(20, 32).astype(np.float32)
        boundaries = model.predict_boundaries(embeddings, min_step_clips=2)
        assert boundaries[0] == 0
        assert boundaries[-1] == 20
        assert len(boundaries) >= 2

    def test_predict_boundaries_single_clip(self) -> None:
        model = NeuralStepSegmenter(d_in=16)
        model.eval()
        embeddings = np.random.randn(1, 16).astype(np.float32)
        boundaries = model.predict_boundaries(embeddings)
        assert boundaries == [0, 1]

    def test_gradient_flow(self) -> None:
        model = NeuralStepSegmenter(d_in=32)
        x = torch.randn(1, 32, 10, requires_grad=True)
        logits1, logits2 = model(x)
        loss = logits2.sum()
        loss.backward()
        assert x.grad is not None

    def test_save_load_roundtrip(self, tmp_path) -> None:
        model = NeuralStepSegmenter(d_in=32)
        x = torch.randn(1, 32, 10)
        model.eval()
        with torch.no_grad():
            _, out_before = model(x)

        path = tmp_path / "seg.pt"
        save_segmenter(model, path)
        loaded = load_segmenter(path)

        with torch.no_grad():
            _, out_after = loaded(x)

        np.testing.assert_allclose(
            out_before.numpy(), out_after.numpy(), atol=1e-6
        )


class TestSegmentationLoss:
    def test_loss_is_finite(self) -> None:
        loss_fn = SegmentationLoss()
        logits1 = torch.randn(2, 2, 20)
        logits2 = torch.randn(2, 2, 20)
        targets = torch.randint(0, 2, (2, 20))
        loss = loss_fn(logits1, logits2, targets)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_smoothing_effect(self) -> None:
        """Higher smoothing weight should penalize noisy predictions more."""
        logits1 = torch.randn(1, 2, 20)
        logits2 = torch.randn(1, 2, 20)
        targets = torch.randint(0, 2, (1, 20))

        loss_low = SegmentationLoss(smoothing_weight=0.01)(logits1, logits2, targets)
        loss_high = SegmentationLoss(smoothing_weight=10.0)(logits1, logits2, targets)

        # Higher smoothing weight should increase total loss
        assert loss_high.item() > loss_low.item()


class TestGeneratePseudoLabels:
    def test_basic(self) -> None:
        gold_boundaries = [0, 5, 10]
        alignment_path = [(i, i, 0.9) for i in range(10)]
        labels = generate_pseudo_labels(gold_boundaries, 10, alignment_path)
        assert labels.shape == (10,)
        assert labels[5] == 1  # Gold boundary at 5 should map to trainee 5

    def test_no_alignment(self) -> None:
        labels = generate_pseudo_labels([0, 10], 10, [])
        assert labels.sum() == 0

    def test_no_interior_boundaries(self) -> None:
        labels = generate_pseudo_labels([0, 10], 10, [(i, i, 0.9) for i in range(10)])
        assert labels.sum() == 0
