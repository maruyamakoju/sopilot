"""Tests for nn.trainer — SOPilotTrainer multi-phase pipeline."""

from __future__ import annotations

import numpy as np

from sopilot.nn.trainer import SOPilotTrainer, TrainingConfig


class TestTrainingConfig:
    def test_defaults(self) -> None:
        cfg = TrainingConfig()
        assert cfg.device == "cpu"
        assert cfg.proj_d_in == 1280
        assert cfg.proj_d_out == 128
        assert cfg.gamma_init == 1.0

    def test_custom(self) -> None:
        cfg = TrainingConfig(proj_d_in=768, device="cpu")
        assert cfg.proj_d_in == 768


class TestSOPilotTrainer:
    def _make_synthetic_data(
        self, n_videos: int = 3, n_clips: int = 20, d: int = 64
    ) -> tuple[list[np.ndarray], list[list[int]]]:
        """Create synthetic embeddings with step boundaries."""
        rng = np.random.default_rng(42)
        embeddings_list = []
        boundaries_list = []
        for _ in range(n_videos):
            emb = rng.standard_normal((n_clips, d)).astype(np.float32)
            # Create 3 steps
            boundaries = [0, n_clips // 3, 2 * n_clips // 3, n_clips]
            embeddings_list.append(emb)
            boundaries_list.append(boundaries)
        return embeddings_list, boundaries_list

    def test_train_projection_head(self) -> None:
        cfg = TrainingConfig(proj_d_in=64, proj_d_out=32, proj_epochs=3, proj_batch_size=16)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = self._make_synthetic_data(d=64)

        log = trainer.train_projection_head(embeddings, boundaries)

        assert log.phase == "projection_head"
        assert log.epochs_completed == 3
        assert log.num_parameters > 0
        assert len(log.epoch_losses) == 3
        assert trainer.projection_head is not None

    def test_train_step_segmenter(self) -> None:
        cfg = TrainingConfig(proj_d_in=64, seg_epochs=3)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = self._make_synthetic_data(d=64)

        log = trainer.train_step_segmenter(embeddings, boundaries)

        assert log.phase == "step_segmenter"
        assert log.epochs_completed == 3
        assert trainer.segmenter is not None

    def test_train_scoring_head(self) -> None:
        cfg = TrainingConfig(score_epochs=5, score_batch_size=4)
        trainer = SOPilotTrainer(cfg)

        rng = np.random.default_rng(42)
        metrics = rng.standard_normal((20, 15)).astype(np.float32)
        scores = rng.uniform(0, 100, size=20).astype(np.float32)

        log = trainer.train_scoring_head(metrics, scores)

        assert log.phase == "scoring_head"
        assert log.epochs_completed == 5
        assert trainer.scoring_head is not None

    def test_save_all(self, tmp_path) -> None:
        cfg = TrainingConfig(
            proj_d_in=32,
            proj_d_out=16,
            proj_epochs=2,
            proj_batch_size=16,
            seg_epochs=2,
            score_epochs=2,
            output_dir=tmp_path,
        )
        trainer = SOPilotTrainer(cfg)

        embeddings, boundaries = self._make_synthetic_data(d=32, n_clips=16)
        trainer.train_projection_head(embeddings, boundaries)
        trainer.train_step_segmenter(embeddings, boundaries)

        rng = np.random.default_rng(42)
        trainer.train_scoring_head(
            rng.standard_normal((10, 15)).astype(np.float32),
            rng.uniform(0, 100, size=10).astype(np.float32),
        )

        paths = trainer.save_all(tmp_path)
        assert "projection_head" in paths
        assert "step_segmenter" in paths
        assert "scoring_head" in paths

    def test_training_summary(self) -> None:
        cfg = TrainingConfig(proj_d_in=32, proj_d_out=16, proj_epochs=2, proj_batch_size=16)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = self._make_synthetic_data(d=32, n_clips=16)
        trainer.train_projection_head(embeddings, boundaries)

        summary = trainer.training_summary()
        assert "phases" in summary
        assert len(summary["phases"]) == 1
        assert summary["phases"][0]["phase"] == "projection_head"
        assert summary["total_trainable_parameters"] > 0

    def test_calibrate(self) -> None:
        cfg = TrainingConfig()
        trainer = SOPilotTrainer(cfg)
        predicted = np.array([20, 40, 60, 80], dtype=np.float64)
        actual = np.array([25, 45, 55, 85], dtype=np.float64)
        trainer.calibrate(predicted, actual)
        assert trainer.calibrator is not None
