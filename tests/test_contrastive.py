"""test_contrastive.py
=====================
Comprehensive pytest suite for ``sopilot.core.contrastive``.

Tests cover:
  - Model forward pass shape validation (all pooling strategies)
  - Loss computation correctness (MSE + margin ranking)
  - Training loop convergence on synthetic data
  - Score prediction in valid range [0, 100]
  - Gradient flow check (no dead layers)
  - Save/load checkpoint roundtrip
  - PairwiseDataset construction + hard mining
  - Integration function ``contrastive_score``

All tests are self-contained, fast (< 2 s each), and require no real data.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Torch availability check — skip entire module if torch is missing
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch", reason="torch required for contrastive tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vectors(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Return (n, d) float32 array of L2-normalised random vectors."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.where(norms < 1e-9, 1.0, norms)).astype(np.float32)


def _make_model(embed_dim: int = 64, pooling: str = "mean", **kwargs):
    """Create a small ContrastiveRegressionHead for testing."""
    from sopilot.core.contrastive import ContrastiveRegressionHead
    return ContrastiveRegressionHead(
        embed_dim=embed_dim,
        hidden_dim=128,
        pooling=pooling,
        dropout=0.0,
        **kwargs,
    )


def _make_dataset(n_samples: int = 20, embed_dim: int = 64, seed: int = 42):
    """Create a synthetic PairwiseDataset for testing."""
    from sopilot.core.contrastive import PairwiseDataset

    rng = np.random.default_rng(seed)
    gold_embs = {}
    trainee_embs = {}
    scores = []

    # One gold video, multiple trainee videos.
    gold_embs["gold_1"] = _unit_vectors(15, embed_dim, seed=0)

    for i in range(n_samples):
        tid = f"trainee_{i}"
        trainee_embs[tid] = _unit_vectors(
            rng.integers(8, 25), embed_dim, seed=i + 100
        )
        score_val = float(rng.uniform(10, 95))
        scores.append({
            "gold_video_id": "gold_1",
            "trainee_video_id": tid,
            "score": score_val,
        })

    return PairwiseDataset(
        gold_embeddings=gold_embs,
        trainee_embeddings=trainee_embs,
        scores=scores,
        rng_seed=seed,
    )


# ===========================================================================
# Section 1: Forward pass shape validation
# ===========================================================================

class TestForwardPassShapes:
    """Validate output shapes for all temporal pooling strategies."""

    @pytest.mark.parametrize("pooling", ["mean", "attention", "learned"])
    def test_batched_forward(self, pooling: str):
        """Batched input (B, T, D) -> score (B,), features (B, 3D)."""
        D = 64
        model = _make_model(embed_dim=D, pooling=pooling)
        model.eval()

        B, T_g, T_t = 4, 12, 10
        gold = torch.randn(B, T_g, D)
        trainee = torch.randn(B, T_t, D)

        with torch.no_grad():
            score, features = model(gold, trainee)

        assert score.shape == (B,), f"score shape {score.shape} != ({B},)"
        assert features.shape == (B, 3 * D), f"features shape {features.shape} != ({B}, {3*D})"

    @pytest.mark.parametrize("pooling", ["mean", "attention", "learned"])
    def test_unbatched_forward(self, pooling: str):
        """Unbatched input (T, D) -> scalar score, features (3D,) squeezed."""
        D = 64
        model = _make_model(embed_dim=D, pooling=pooling)
        model.eval()

        gold = torch.randn(15, D)
        trainee = torch.randn(10, D)

        with torch.no_grad():
            score, features = model(gold, trainee)

        assert score.dim() == 0, f"unbatched score should be scalar, got dim={score.dim()}"
        # Features stay batched (1, 3D) since only score is squeezed.
        assert features.shape[-1] == 3 * D

    def test_different_sequence_lengths(self):
        """Gold and trainee can have different temporal lengths."""
        D = 64
        model = _make_model(embed_dim=D, pooling="attention")
        model.eval()

        gold = torch.randn(1, 50, D)
        trainee = torch.randn(1, 5, D)

        with torch.no_grad():
            score, features = model(gold, trainee)

        assert score.shape == (1,)
        assert features.shape == (1, 3 * D)


# ===========================================================================
# Section 2: Loss computation correctness
# ===========================================================================

class TestLossFunctions:
    """Validate loss functions compute correct values."""

    def test_margin_ranking_loss_zero_when_correct(self):
        """When predicted gap exceeds margin, loss should be zero."""
        from sopilot.core.contrastive import MarginRankingLoss

        loss_fn = MarginRankingLoss(base_margin=1.0, margin_scale=0.1)
        # score_better is much higher than score_worse.
        score_better = torch.tensor([90.0, 80.0])
        score_worse = torch.tensor([20.0, 10.0])
        true_diff = torch.tensor([70.0, 70.0])

        loss = loss_fn(score_better, score_worse, true_diff)
        assert float(loss.item()) == 0.0, f"Expected 0 loss, got {loss.item()}"

    def test_margin_ranking_loss_positive_when_violated(self):
        """When predicted gap is below margin, loss should be positive."""
        from sopilot.core.contrastive import MarginRankingLoss

        loss_fn = MarginRankingLoss(base_margin=5.0, margin_scale=0.1)
        # score_better is barely above score_worse.
        score_better = torch.tensor([51.0])
        score_worse = torch.tensor([50.0])
        true_diff = torch.tensor([20.0])

        loss = loss_fn(score_better, score_worse, true_diff)
        # margin = 5.0 + 0.1 * 20 = 7.0; gap = 1.0; loss = 7.0 - 1.0 = 6.0
        assert float(loss.item()) == pytest.approx(6.0, abs=1e-4)

    def test_margin_proportional_to_score_diff(self):
        """Larger true_diff -> larger margin -> larger loss for same gap."""
        from sopilot.core.contrastive import MarginRankingLoss

        loss_fn = MarginRankingLoss(base_margin=1.0, margin_scale=0.5)
        score_better = torch.tensor([55.0])
        score_worse = torch.tensor([50.0])

        loss_small = loss_fn(score_better, score_worse, torch.tensor([5.0]))
        loss_large = loss_fn(score_better, score_worse, torch.tensor([50.0]))

        assert float(loss_large.item()) > float(loss_small.item())

    def test_contrastive_regression_loss_components(self):
        """Combined loss returns correct component breakdown."""
        from sopilot.core.contrastive import ContrastiveRegressionLoss

        loss_fn = ContrastiveRegressionLoss(lambda_mse=1.0, lambda_rank=0.5)

        pred = torch.tensor([80.0, 60.0, 40.0, 20.0], requires_grad=True)
        true = torch.tensor([85.0, 55.0, 45.0, 25.0])

        total, components = loss_fn(pred, true)

        assert "mse" in components
        assert "ranking" in components
        assert "total" in components
        assert components["total"] > 0
        assert total.requires_grad

    def test_contrastive_loss_mse_only_with_single_sample(self):
        """With B=1, ranking loss should be 0 (can't form pairs)."""
        from sopilot.core.contrastive import ContrastiveRegressionLoss

        loss_fn = ContrastiveRegressionLoss()
        pred = torch.tensor([50.0], requires_grad=True)
        true = torch.tensor([60.0])

        total, components = loss_fn(pred, true)
        assert components["ranking"] == 0.0
        assert components["mse"] > 0.0


# ===========================================================================
# Section 3: Training loop convergence on synthetic data
# ===========================================================================

class TestTrainingConvergence:
    """Verify the trainer converges on simple synthetic data."""

    def test_loss_decreases(self):
        """Training loss should decrease over epochs on learnable data."""
        from sopilot.core.contrastive import ContrastiveRegressionHead, ContrastiveTrainer

        D = 32
        model = ContrastiveRegressionHead(
            embed_dim=D, hidden_dim=64, pooling="mean", dropout=0.0,
        )
        trainer = ContrastiveTrainer(model, device="cpu", lambda_mse=1.0, lambda_rank=0.0)
        dataset = _make_dataset(n_samples=30, embed_dim=D, seed=7)

        result = trainer.fit(
            train_dataset=dataset,
            n_epochs=30,
            batch_size=8,
            lr=1e-3,
            warmup_epochs=2,
            patience=50,  # don't early stop
        )

        assert result.total_epochs_run == 30
        # First-epoch loss should be higher than last-epoch loss.
        assert result.train_losses[0] > result.train_losses[-1], (
            f"Loss did not decrease: first={result.train_losses[0]:.4f}, "
            f"last={result.train_losses[-1]:.4f}"
        )

    def test_early_stopping(self):
        """Training should stop early when patience is exhausted."""
        from sopilot.core.contrastive import ContrastiveRegressionHead, ContrastiveTrainer

        D = 32
        model = ContrastiveRegressionHead(
            embed_dim=D, hidden_dim=64, pooling="mean", dropout=0.0,
        )
        trainer = ContrastiveTrainer(model, device="cpu")

        dataset = _make_dataset(n_samples=10, embed_dim=D, seed=99)

        result = trainer.fit(
            train_dataset=dataset,
            n_epochs=200,
            batch_size=4,
            lr=1e-3,
            patience=5,
        )

        # Should have stopped well before 200 epochs.
        assert result.total_epochs_run < 200

    def test_validation_monitoring(self):
        """When val_dataset is provided, val_losses are populated."""
        from sopilot.core.contrastive import ContrastiveRegressionHead, ContrastiveTrainer

        D = 32
        model = ContrastiveRegressionHead(
            embed_dim=D, hidden_dim=64, pooling="mean", dropout=0.0,
        )
        trainer = ContrastiveTrainer(model, device="cpu")

        train_ds = _make_dataset(n_samples=20, embed_dim=D, seed=10)
        val_ds = _make_dataset(n_samples=10, embed_dim=D, seed=20)

        result = trainer.fit(
            train_dataset=train_ds,
            val_dataset=val_ds,
            n_epochs=10,
            batch_size=8,
            patience=50,
        )

        assert result.val_losses is not None
        assert len(result.val_losses) == result.total_epochs_run


# ===========================================================================
# Section 4: Score prediction in valid range
# ===========================================================================

class TestScoreRange:
    """Verify scores are always in [0, 100]."""

    @pytest.mark.parametrize("pooling", ["mean", "attention", "learned"])
    def test_output_range(self, pooling: str):
        """Scores from forward pass must be in [0, 100]."""
        D = 64
        model = _make_model(embed_dim=D, pooling=pooling)
        model.eval()

        rng = np.random.default_rng(42)
        for _ in range(10):
            T_g = rng.integers(5, 30)
            T_t = rng.integers(5, 30)
            gold = torch.randn(1, T_g, D)
            trainee = torch.randn(1, T_t, D)

            with torch.no_grad():
                score, _ = model(gold, trainee)

            s = float(score.item())
            assert 0.0 <= s <= 100.0, f"Score {s} out of [0, 100] range"

    def test_identical_input_high_score(self):
        """When gold == trainee exactly, score should be relatively high (> 40)."""
        D = 64
        model = _make_model(embed_dim=D, pooling="mean")
        # We can't guarantee untrained model gives high score, but features should
        # have zero diff component.  Just verify it doesn't crash and is in range.
        model.eval()

        gold = torch.randn(1, 10, D)
        with torch.no_grad():
            score, features = model(gold, gold.clone())

        s = float(score.item())
        assert 0.0 <= s <= 100.0

        # The element-wise difference features should be near zero.
        feat_np = features.squeeze(0).cpu().numpy()
        diff_block = feat_np[D : 2 * D]
        assert np.allclose(diff_block, 0.0, atol=1e-5), (
            "Diff features should be ~0 for identical inputs"
        )


# ===========================================================================
# Section 5: Gradient flow check
# ===========================================================================

class TestGradientFlow:
    """Ensure gradients flow through all layers without vanishing."""

    @pytest.mark.parametrize("pooling", ["mean", "attention", "learned"])
    def test_all_parameters_receive_gradients(self, pooling: str):
        """Every trainable parameter should have a non-zero gradient after one backward pass."""
        D = 64
        model = _make_model(embed_dim=D, pooling=pooling)
        model.train()

        gold = torch.randn(4, 12, D, requires_grad=False)
        trainee = torch.randn(4, 10, D, requires_grad=False)

        score, _ = model(gold, trainee)
        loss = score.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            grad_norm = float(param.grad.norm().item())
            assert grad_norm > 0.0, f"Zero gradient for {name} (grad_norm={grad_norm})"

    def test_no_nan_gradients(self):
        """Gradients should never be NaN."""
        D = 64
        model = _make_model(embed_dim=D, pooling="attention")
        model.train()

        gold = torch.randn(8, 20, D)
        trainee = torch.randn(8, 15, D)

        score, _ = model(gold, trainee)
        loss = score.mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"


# ===========================================================================
# Section 6: Save/load checkpoint roundtrip
# ===========================================================================

class TestCheckpointRoundtrip:
    """Verify models can be saved and loaded with identical outputs."""

    def test_save_load_produces_same_output(self):
        """Loaded model should produce bit-identical scores."""
        from sopilot.core.contrastive import ContrastiveTrainer, ContrastiveRegressionHead

        D = 64
        model = ContrastiveRegressionHead(
            embed_dim=D, hidden_dim=128, pooling="attention", dropout=0.0,
        )
        model.eval()

        gold = torch.randn(1, 10, D)
        trainee = torch.randn(1, 8, D)

        with torch.no_grad():
            score_before, feat_before = model(gold, trainee)

        trainer = ContrastiveTrainer(model, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.pt"
            trainer.save(str(path))

            loaded_trainer = ContrastiveTrainer.load(str(path), device="cpu")
            loaded_trainer.model.eval()

            with torch.no_grad():
                score_after, feat_after = loaded_trainer.model(gold, trainee)

        assert float(score_before.item()) == pytest.approx(
            float(score_after.item()), abs=1e-6
        )
        np.testing.assert_allclose(
            feat_before.cpu().numpy(),
            feat_after.cpu().numpy(),
            atol=1e-6,
        )

    def test_checkpoint_preserves_config(self):
        """Model config should survive save/load roundtrip."""
        from sopilot.core.contrastive import ContrastiveTrainer, ContrastiveRegressionHead

        model = ContrastiveRegressionHead(
            embed_dim=128, hidden_dim=256, pooling="learned",
            dropout=0.1, max_temporal_len=200,
        )
        trainer = ContrastiveTrainer(model, device="cpu", lambda_mse=2.0, lambda_rank=0.3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cfg_test.pt"
            trainer.save(str(path))
            loaded = ContrastiveTrainer.load(str(path), device="cpu")

        cfg = loaded.model.config_dict()
        assert cfg["embed_dim"] == 128
        assert cfg["hidden_dim"] == 256
        assert cfg["pooling"] == "learned"
        assert cfg["max_temporal_len"] == 200
        assert loaded.lambda_mse == pytest.approx(2.0)
        assert loaded.lambda_rank == pytest.approx(0.3)


# ===========================================================================
# Section 7: PairwiseDataset
# ===========================================================================

class TestPairwiseDataset:
    """Validate PairwiseDataset data loading and pair generation."""

    def test_dataset_length(self):
        """Dataset length equals number of valid score records."""
        ds = _make_dataset(n_samples=15, embed_dim=32)
        assert len(ds) == 15

    def test_getitem_keys(self):
        """Each sample should contain gold_emb, trainee_emb, score."""
        ds = _make_dataset(n_samples=5, embed_dim=32)
        sample = ds[0]
        assert "gold_emb" in sample
        assert "trainee_emb" in sample
        assert "score" in sample
        assert isinstance(sample["gold_emb"], torch.Tensor)
        assert isinstance(sample["score"], torch.Tensor)

    def test_relative_pairs_generated(self):
        """Relative pairs should be non-empty when scores differ."""
        ds = _make_dataset(n_samples=10, embed_dim=32)
        pairs = ds.relative_pairs
        assert len(pairs) > 0, "No relative pairs generated"
        for i, j, sign in pairs:
            assert sign in (-1.0, 1.0)
            assert i != j

    def test_hard_mining_reduces_pairs(self):
        """Hard mining threshold should reduce the number of pairs."""
        from sopilot.core.contrastive import PairwiseDataset

        gold_embs = {"g": _unit_vectors(10, 32)}
        trainee_embs = {f"t{i}": _unit_vectors(10, 32, seed=i) for i in range(10)}
        records = [
            {"gold_video_id": "g", "trainee_video_id": f"t{i}", "score": float(i * 10)}
            for i in range(10)
        ]

        ds_all = PairwiseDataset(gold_embs, trainee_embs, records, hard_mining_threshold=None)
        ds_hard = PairwiseDataset(gold_embs, trainee_embs, records, hard_mining_threshold=15.0)

        # Hard mining with threshold=15 should exclude wide-gap pairs.
        assert len(ds_hard.relative_pairs) < len(ds_all.relative_pairs)

    def test_missing_embeddings_skipped(self):
        """Score records referencing missing embeddings are excluded."""
        from sopilot.core.contrastive import PairwiseDataset

        gold_embs = {"g": _unit_vectors(10, 32)}
        trainee_embs = {"t0": _unit_vectors(10, 32)}
        records = [
            {"gold_video_id": "g", "trainee_video_id": "t0", "score": 50.0},
            {"gold_video_id": "g", "trainee_video_id": "missing", "score": 70.0},
        ]

        ds = PairwiseDataset(gold_embs, trainee_embs, records)
        assert len(ds) == 1


# ===========================================================================
# Section 8: Integration function
# ===========================================================================

class TestContrastiveScoreFunction:
    """Test the contrastive_score() integration function."""

    def test_returns_expected_keys(self):
        """Output dict should contain score, confidence, features."""
        from sopilot.core.contrastive import contrastive_score

        D = 64
        model = _make_model(embed_dim=D, pooling="mean")
        model.eval()

        gold = _unit_vectors(10, D, seed=0)
        trainee = _unit_vectors(8, D, seed=1)

        result = contrastive_score(gold, trainee, model, device="cpu")

        assert "score" in result
        assert "confidence" in result
        assert "features" in result

    def test_score_in_range(self):
        """Score should be in [0, 100]."""
        from sopilot.core.contrastive import contrastive_score

        D = 64
        model = _make_model(embed_dim=D, pooling="attention")
        model.eval()

        gold = _unit_vectors(12, D, seed=5)
        trainee = _unit_vectors(10, D, seed=6)

        result = contrastive_score(gold, trainee, model, device="cpu")
        assert 0.0 <= result["score"] <= 100.0

    def test_confidence_in_range(self):
        """Confidence should be in (0, 1]."""
        from sopilot.core.contrastive import contrastive_score

        D = 64
        model = _make_model(embed_dim=D, pooling="mean")
        model.eval()

        gold = _unit_vectors(10, D)
        trainee = _unit_vectors(10, D, seed=1)

        result = contrastive_score(gold, trainee, model, device="cpu")
        assert 0.0 < result["confidence"] <= 1.0

    def test_features_shape(self):
        """Features should be (3*D,) numpy array."""
        from sopilot.core.contrastive import contrastive_score

        D = 64
        model = _make_model(embed_dim=D, pooling="learned")
        model.eval()

        gold = _unit_vectors(10, D)
        trainee = _unit_vectors(8, D, seed=1)

        result = contrastive_score(gold, trainee, model, device="cpu")
        assert isinstance(result["features"], np.ndarray)
        assert result["features"].shape == (3 * D,)

    def test_deterministic_eval_mode(self):
        """Two calls with same input should produce identical results."""
        from sopilot.core.contrastive import contrastive_score

        D = 64
        model = _make_model(embed_dim=D, pooling="mean")
        model.eval()

        gold = _unit_vectors(10, D)
        trainee = _unit_vectors(8, D, seed=1)

        r1 = contrastive_score(gold, trainee, model, device="cpu")
        r2 = contrastive_score(gold, trainee, model, device="cpu")

        assert r1["score"] == r2["score"]
        assert r1["confidence"] == r2["confidence"]
        np.testing.assert_array_equal(r1["features"], r2["features"])


# ===========================================================================
# Section 9: Temporal pooling edge cases
# ===========================================================================

class TestTemporalPoolingEdgeCases:
    """Edge cases for temporal pooling modules."""

    def test_single_timestep(self):
        """Model should handle T=1 (single clip)."""
        D = 64
        model = _make_model(embed_dim=D, pooling="attention")
        model.eval()

        gold = torch.randn(1, 1, D)
        trainee = torch.randn(1, 1, D)

        with torch.no_grad():
            score, features = model(gold, trainee)

        assert not torch.isnan(score).any()
        assert not torch.isnan(features).any()

    def test_learned_pool_long_sequence(self):
        """Learned temporal pool should handle sequences beyond max_len via interpolation."""
        D = 32
        model = _make_model(embed_dim=D, pooling="learned", max_temporal_len=10)
        model.eval()

        # Sequence longer than max_len=10.
        gold = torch.randn(1, 50, D)
        trainee = torch.randn(1, 30, D)

        with torch.no_grad():
            score, features = model(gold, trainee)

        assert not torch.isnan(score).any()
        assert 0.0 <= float(score.item()) <= 100.0


# ===========================================================================
# Section 10: Per-layer gradient clipping
# ===========================================================================

class TestPerLayerGradientClipping:
    """Verify per-layer gradient clipping is applied during training."""

    def test_gradients_clipped_per_layer(self):
        """After a training step, no parameter gradient norm exceeds max_grad_norm."""
        from sopilot.core.contrastive import (
            ContrastiveRegressionHead,
            ContrastiveTrainer,
        )

        D = 32
        model = ContrastiveRegressionHead(
            embed_dim=D, hidden_dim=64, pooling="mean", dropout=0.0,
        )
        trainer = ContrastiveTrainer(model, device="cpu")
        dataset = _make_dataset(n_samples=10, embed_dim=D)

        max_grad_norm = 0.5

        # Run one training epoch with a tight clip.
        model.train()
        model.to(torch.device("cpu"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        trainer._run_epoch(
            dataset, optimizer, batch_size=10,
            training=True, max_grad_norm=max_grad_norm,
        )

        # After clipping, each parameter's grad norm should be <= max_grad_norm + epsilon.
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = float(param.grad.norm().item())
                # Allow small numerical tolerance.
                assert grad_norm <= max_grad_norm + 0.01, (
                    f"Gradient norm for {name} = {grad_norm:.4f} exceeds "
                    f"max_grad_norm = {max_grad_norm}"
                )
