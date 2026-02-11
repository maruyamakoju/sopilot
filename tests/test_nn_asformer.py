"""Tests for ASFormer temporal action segmentation module."""

import numpy as np
import pytest
import torch


class TestASFormerForward:
    def test_output_shapes(self):
        from sopilot.nn.asformer import ASFormer

        model = ASFormer(d_in=32, d_model=16, n_classes=2, n_heads=2,
                         n_encoder_layers=3, n_decoder_layers=3, n_decoders=2)
        x = torch.randn(2, 32, 50)  # (B, D, T)
        logits = model(x)

        assert len(logits) == 3  # 1 encoder + 2 decoders
        for stage_logits in logits:
            assert stage_logits.shape == (2, 2, 50)

    def test_gradient_flow(self):
        from sopilot.nn.asformer import ASFormer

        model = ASFormer(d_in=16, d_model=16, n_heads=2,
                         n_encoder_layers=2, n_decoder_layers=2, n_decoders=1)
        x = torch.randn(1, 16, 20, requires_grad=True)
        logits = model(x)
        loss = logits[-1].sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_mask_handling(self):
        from sopilot.nn.asformer import ASFormer

        model = ASFormer(d_in=16, d_model=16, n_heads=2,
                         n_encoder_layers=2, n_decoder_layers=2, n_decoders=1)
        x = torch.randn(1, 16, 30)
        mask = torch.ones(1, 30, dtype=torch.bool)
        mask[0, 20:] = False

        logits = model(x, mask)
        assert len(logits) >= 2

    def test_num_parameters(self):
        from sopilot.nn.asformer import ASFormer

        model = ASFormer(d_in=64, d_model=64, n_heads=4,
                         n_encoder_layers=5, n_decoder_layers=5, n_decoders=2)
        assert model.num_parameters > 0


class TestASFormerLoss:
    def test_loss_computes(self):
        from sopilot.nn.asformer import ASFormer, ASFormerLoss

        model = ASFormer(d_in=16, d_model=16, n_heads=2,
                         n_encoder_layers=2, n_decoder_layers=2, n_decoders=1)
        loss_fn = ASFormerLoss(n_classes=2)

        x = torch.randn(2, 16, 30)
        targets = torch.zeros(2, 30, dtype=torch.long)
        targets[:, 10] = 1
        targets[:, 20] = 1

        logits = model(x)
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_loss_decreases(self):
        from sopilot.nn.asformer import ASFormer, ASFormerLoss

        torch.manual_seed(42)
        model = ASFormer(d_in=8, d_model=8, n_heads=2,
                         n_encoder_layers=2, n_decoder_layers=2, n_decoders=1)
        loss_fn = ASFormerLoss(n_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(1, 8, 20)
        targets = torch.zeros(1, 20, dtype=torch.long)
        targets[0, 5] = 1
        targets[0, 15] = 1

        initial_loss = None
        for epoch in range(20):
            logits = model(x)
            loss = loss_fn(logits, targets)
            if initial_loss is None:
                initial_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert loss.item() < initial_loss


class TestPredictBoundaries:
    def test_returns_valid_boundaries(self):
        from sopilot.nn.asformer import ASFormer, predict_boundaries_asformer

        model = ASFormer(d_in=16, d_model=16, n_heads=2,
                         n_encoder_layers=2, n_decoder_layers=2, n_decoders=1)
        embeddings = np.random.randn(50, 16).astype(np.float32)

        boundaries, probs = predict_boundaries_asformer(model, embeddings)

        assert boundaries[0] == 0
        assert boundaries[-1] == 50
        assert len(boundaries) >= 2
        assert probs.shape == (50,)
        assert all(0 <= p <= 1 for p in probs)


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        from sopilot.nn.asformer import ASFormer, save_asformer, load_asformer

        model = ASFormer(d_in=16, d_model=16, n_heads=2,
                         n_encoder_layers=2, n_decoder_layers=2, n_decoders=1)
        model.eval()
        x = torch.randn(1, 16, 20)

        with torch.no_grad():
            original_out = model(x)[-1]

        path = tmp_path / "asformer.pt"
        save_asformer(model, path)
        loaded = load_asformer(path)

        with torch.no_grad():
            loaded_out = loaded(x)[-1]

        assert torch.allclose(original_out, loaded_out, atol=1e-5)
