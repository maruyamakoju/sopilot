"""End-to-end integration tests for the full neural training pipeline.

Exercises every phase of SOPilotTrainer with synthetic data:
    Phase 1a: ProjectionHead (contrastive)
    Phase 1b: StepSegmenter (MS-TCN++)
    Phase 1c: ASFormer (transformer segmenter)
    Phase 2:  ScoringHead (MLP)
    Phase 3:  Joint fine-tune (end-to-end)
    Phase 4:  Calibration (isotonic + conformal)

Then verifies save/load, evaluate_sop in neural_mode, and conformal roundtrip.
All models are tiny to keep total runtime under 30 seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sopilot.nn.trainer import SOPilotTrainer, TrainingConfig, TrainingLog
from sopilot.step_engine import detect_step_boundaries, evaluate_sop, invalidate_neural_caches

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

DIM = 16
N_VIDEOS = 4
CLIPS_PER_VIDEO = 30


def _seed() -> None:
    """Set deterministic seeds."""
    torch.manual_seed(42)
    np.random.seed(42)


def _make_embeddings(
    n_videos: int = N_VIDEOS,
    n_clips: int = CLIPS_PER_VIDEO,
    d: int = DIM,
) -> tuple[list[np.ndarray], list[list[int]]]:
    """Create synthetic embeddings and boundaries for *n_videos* videos.

    Each video has 3 steps with slightly different embedding distributions
    per step so that contrastive learning has a meaningful signal.
    """
    rng = np.random.default_rng(42)
    embeddings_list: list[np.ndarray] = []
    boundaries_list: list[list[int]] = []

    for v in range(n_videos):
        step_size = n_clips // 3
        parts = []
        # Each step gets a distinct mean offset so clips within a step are
        # more similar to each other than to clips in other steps.
        for s in range(3):
            center = rng.standard_normal(d).astype(np.float32) * 3.0
            noise = rng.standard_normal((step_size, d)).astype(np.float32) * 0.3
            parts.append(noise + center)
        emb = np.concatenate(parts, axis=0)  # (step_size*3, d)
        boundaries = [0, step_size, 2 * step_size, emb.shape[0]]
        embeddings_list.append(emb)
        boundaries_list.append(boundaries)

    return embeddings_list, boundaries_list


def _make_scoring_data(
    n_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic (metrics, scores) pairs for ScoringHead training."""
    rng = np.random.default_rng(123)
    metrics = rng.standard_normal((n_samples, 15)).astype(np.float32)
    # Scores loosely correlated with metrics for a learnable signal
    scores = (50.0 + 10.0 * metrics[:, 0] - 5.0 * metrics[:, 1]).astype(np.float32)
    scores = np.clip(scores, 0.0, 100.0)
    return metrics, scores


def _make_tiny_config(output_dir: Path) -> TrainingConfig:
    """Build a TrainingConfig with minimal sizes for fast tests."""
    return TrainingConfig(
        device="cpu",
        # Projection head
        proj_d_in=DIM,
        proj_d_out=8,
        proj_lr=5e-3,
        proj_epochs=2,
        proj_batch_size=16,
        proj_temperature=0.1,
        # Step segmenter
        seg_lr=1e-3,
        seg_epochs=2,
        seg_batch_size=4,
        # Scoring head
        score_lr=5e-3,
        score_epochs=2,
        score_batch_size=16,
        # Joint fine-tune
        joint_lr=1e-3,
        joint_epochs=2,
        # ASFormer (very small)
        asformer_d_model=8,
        asformer_n_heads=2,
        asformer_n_encoder_layers=2,
        asformer_n_decoder_layers=2,
        asformer_n_decoders=1,
        asformer_lr=5e-3,
        asformer_epochs=2,
        # Soft-DTW
        gamma_init=1.0,
        # DILATE
        dilate_alpha=0.5,
        # Conformal
        conformal_alpha=0.1,
        # Output
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Full pipeline test class
# ---------------------------------------------------------------------------


class TestFullNeuralPipeline:
    """End-to-end tests that run every phase in sequence."""

    # ---- Phase 1a ----

    def test_phase1a_projection_head(self, tmp_path: Path) -> None:
        _seed()
        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = _make_embeddings()

        log = trainer.train_projection_head(embeddings, boundaries)

        assert log.phase == "projection_head"
        assert log.epochs_completed == cfg.proj_epochs
        assert log.num_parameters > 0
        assert len(log.epoch_losses) == cfg.proj_epochs
        assert log.final_loss != 0.0, "final_loss should be non-zero"
        assert trainer.projection_head is not None

    # ---- Phase 1b ----

    def test_phase1b_step_segmenter(self, tmp_path: Path) -> None:
        _seed()
        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = _make_embeddings()

        log = trainer.train_step_segmenter(embeddings, boundaries)

        assert log.phase == "step_segmenter"
        assert log.epochs_completed == cfg.seg_epochs
        assert log.num_parameters > 0
        assert len(log.epoch_losses) == cfg.seg_epochs
        assert log.final_loss != 0.0
        assert trainer.segmenter is not None

    # ---- Phase 1c ----

    def test_phase1c_asformer(self, tmp_path: Path) -> None:
        _seed()
        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = _make_embeddings()

        log = trainer.train_asformer(embeddings, boundaries)

        assert log.phase == "asformer"
        assert log.epochs_completed == cfg.asformer_epochs
        assert log.num_parameters > 0
        assert len(log.epoch_losses) == cfg.asformer_epochs
        assert log.final_loss != 0.0
        assert trainer.asformer is not None

    # ---- Phase 2 ----

    def test_phase2_scoring_head(self, tmp_path: Path) -> None:
        _seed()
        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        metrics, scores = _make_scoring_data()

        log = trainer.train_scoring_head(metrics, scores)

        assert log.phase == "scoring_head"
        assert log.epochs_completed == cfg.score_epochs
        assert log.num_parameters > 0
        assert len(log.epoch_losses) == cfg.score_epochs
        assert log.final_loss != 0.0
        assert trainer.scoring_head is not None

    # ---- Phase 3 ----

    def test_phase3_joint_finetune(self, tmp_path: Path) -> None:
        _seed()
        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = _make_embeddings()

        # Must train prerequisites first
        trainer.train_projection_head(embeddings, boundaries)
        metrics, scores = _make_scoring_data()
        trainer.train_scoring_head(metrics, scores)

        # Build paired data for joint fine-tuning
        rng = np.random.default_rng(99)
        gold_list = [embeddings[0], embeddings[1], embeddings[2]]
        trainee_list = [
            embeddings[0] + rng.standard_normal(embeddings[0].shape).astype(np.float32) * 0.5,
            embeddings[1] + rng.standard_normal(embeddings[1].shape).astype(np.float32) * 0.5,
            embeddings[2] + rng.standard_normal(embeddings[2].shape).astype(np.float32) * 0.5,
        ]
        target_scores = np.array([85.0, 70.0, 55.0], dtype=np.float32)

        log = trainer.joint_finetune(gold_list, trainee_list, target_scores)

        assert log.phase == "joint_finetune"
        assert log.epochs_completed == cfg.joint_epochs
        assert log.num_parameters > 0
        assert len(log.epoch_losses) == cfg.joint_epochs
        assert log.final_loss != 0.0
        assert trainer.soft_dtw is not None
        assert trainer.alignment_bridge is not None

    # ---- Phase 4 ----

    def test_phase4_calibration(self, tmp_path: Path) -> None:
        _seed()
        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)

        predicted = np.array([30.0, 50.0, 65.0, 80.0, 92.0], dtype=np.float64)
        actual = np.array([35.0, 48.0, 60.0, 82.0, 88.0], dtype=np.float64)

        trainer.calibrate(predicted, actual)

        assert trainer.calibrator is not None
        assert trainer.conformal is not None
        assert trainer.conformal._quantile is not None

    # ---- All phases sequentially ----

    def test_full_pipeline_sequential(self, tmp_path: Path) -> None:
        """Run all 6 phases in order and check every log has non-zero loss."""
        _seed()
        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = _make_embeddings()
        metrics, scores = _make_scoring_data()

        # Phase 1a
        log_proj = trainer.train_projection_head(embeddings, boundaries)
        assert log_proj.final_loss != 0.0

        # Phase 1b
        log_seg = trainer.train_step_segmenter(embeddings, boundaries)
        assert log_seg.final_loss != 0.0

        # Phase 1c
        log_asf = trainer.train_asformer(embeddings, boundaries)
        assert log_asf.final_loss != 0.0

        # Phase 2
        log_score = trainer.train_scoring_head(metrics, scores)
        assert log_score.final_loss != 0.0

        # Phase 3
        rng = np.random.default_rng(77)
        gold_list = embeddings[:3]
        trainee_list = [e + rng.standard_normal(e.shape).astype(np.float32) * 0.4 for e in gold_list]
        target_scores = np.array([90.0, 60.0, 40.0], dtype=np.float32)
        log_joint = trainer.joint_finetune(gold_list, trainee_list, target_scores)
        assert log_joint.final_loss != 0.0

        # Phase 4
        cal_pred = np.array([20.0, 40.0, 60.0, 80.0], dtype=np.float64)
        cal_actual = np.array([25.0, 38.0, 62.0, 75.0], dtype=np.float64)
        trainer.calibrate(cal_pred, cal_actual)

        # All logs should be present
        assert len(trainer.logs) == 5  # proj, seg, asf, score, joint
        for log_entry in trainer.logs:
            assert isinstance(log_entry, TrainingLog)
            assert log_entry.final_loss != 0.0
            assert log_entry.num_parameters > 0
            assert log_entry.epochs_completed > 0

    def test_training_summary(self, tmp_path: Path) -> None:
        _seed()
        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = _make_embeddings()

        trainer.train_projection_head(embeddings, boundaries)
        trainer.train_step_segmenter(embeddings, boundaries)
        trainer.train_asformer(embeddings, boundaries)

        summary = trainer.training_summary()
        assert "phases" in summary
        assert len(summary["phases"]) == 3
        assert summary["total_trainable_parameters"] > 0
        phases = [p["phase"] for p in summary["phases"]]
        assert phases == ["projection_head", "step_segmenter", "asformer"]


# ---------------------------------------------------------------------------
# Save / load tests
# ---------------------------------------------------------------------------


class TestSaveAndLoad:
    """Verify save_all produces expected files and models can be reloaded."""

    def test_save_all_produces_expected_files(self, tmp_path: Path) -> None:
        """After a full pipeline run, save_all should produce all model files."""
        _seed()
        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = _make_embeddings()
        metrics, scores = _make_scoring_data()

        # Train everything
        trainer.train_projection_head(embeddings, boundaries)
        trainer.train_step_segmenter(embeddings, boundaries)
        trainer.train_asformer(embeddings, boundaries)
        trainer.train_scoring_head(metrics, scores)

        # Joint finetune (creates soft_dtw + alignment_bridge)
        rng = np.random.default_rng(55)
        gold_list = embeddings[:3]
        trainee_list = [e + rng.standard_normal(e.shape).astype(np.float32) * 0.4 for e in gold_list]
        target_scores = np.array([85.0, 60.0, 40.0], dtype=np.float32)
        trainer.joint_finetune(gold_list, trainee_list, target_scores)

        # Calibrate
        cal_pred = np.array([20.0, 40.0, 60.0, 80.0], dtype=np.float64)
        cal_actual = np.array([25.0, 38.0, 62.0, 75.0], dtype=np.float64)
        trainer.calibrate(cal_pred, cal_actual)

        # Save
        out_dir = tmp_path / "saved_models"
        paths = trainer.save_all(out_dir)

        # Check every component was saved
        expected_keys = [
            "projection_head",
            "step_segmenter",
            "asformer",
            "scoring_head",
            "soft_dtw",
            "alignment_bridge",
            "isotonic_calibrator",
            "conformal_predictor",
        ]
        for key in expected_keys:
            assert key in paths, f"Missing saved component: {key}"
            assert Path(paths[key]).exists(), f"File not found: {paths[key]}"

    def test_projection_head_roundtrip(self, tmp_path: Path) -> None:
        """Save and reload ProjectionHead, verify output shape."""
        _seed()
        from sopilot.nn.projection_head import load_projection_head

        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = _make_embeddings()
        trainer.train_projection_head(embeddings, boundaries)

        save_path = tmp_path / "projection_head.pt"
        from sopilot.nn.projection_head import save_projection_head

        save_projection_head(trainer.projection_head, save_path)

        loaded = load_projection_head(save_path, device="cpu")
        test_input = torch.randn(5, DIM)
        with torch.no_grad():
            output = loaded(test_input)
        assert output.shape == (5, cfg.proj_d_out)

    def test_segmenter_roundtrip(self, tmp_path: Path) -> None:
        """Save and reload NeuralStepSegmenter, verify predict_boundaries."""
        _seed()
        from sopilot.nn.step_segmenter import load_segmenter, save_segmenter

        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = _make_embeddings()
        trainer.train_step_segmenter(embeddings, boundaries)

        save_path = tmp_path / "step_segmenter.pt"
        save_segmenter(trainer.segmenter, save_path)

        loaded = load_segmenter(save_path, device="cpu")
        test_emb = np.random.randn(20, DIM).astype(np.float32)
        result_boundaries = loaded.predict_boundaries(test_emb, min_step_clips=2)
        assert result_boundaries[0] == 0
        assert result_boundaries[-1] == 20

    def test_asformer_roundtrip(self, tmp_path: Path) -> None:
        """Save and reload ASFormer, verify predict_boundaries_asformer."""
        _seed()
        from sopilot.nn.asformer import load_asformer, predict_boundaries_asformer, save_asformer

        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = _make_embeddings()
        trainer.train_asformer(embeddings, boundaries)

        save_path = tmp_path / "asformer.pt"
        save_asformer(trainer.asformer, save_path)

        loaded = load_asformer(save_path, device="cpu")
        test_emb = np.random.randn(25, DIM).astype(np.float32)
        result_boundaries, probs = predict_boundaries_asformer(loaded, test_emb, min_step_clips=2, device="cpu")
        assert result_boundaries[0] == 0
        assert result_boundaries[-1] == 25
        assert probs.shape == (25,)

    def test_scoring_head_roundtrip(self, tmp_path: Path) -> None:
        """Save and reload ScoringHead, verify predict_with_uncertainty."""
        _seed()
        from sopilot.nn.scoring_head import load_scoring_head, save_scoring_head

        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        metrics, scores = _make_scoring_data()
        trainer.train_scoring_head(metrics, scores)

        save_path = tmp_path / "scoring_head.pt"
        save_scoring_head(trainer.scoring_head, save_path)

        loaded = load_scoring_head(save_path, device="cpu")
        test_input = torch.randn(1, 15)
        result = loaded.predict_with_uncertainty(test_input, n_samples=10)
        assert "score" in result
        assert "uncertainty" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert 0.0 <= result["score"] <= 100.0

    def test_isotonic_calibrator_roundtrip(self, tmp_path: Path) -> None:
        """Save and reload IsotonicCalibrator."""
        _seed()
        from sopilot.nn.scoring_head import IsotonicCalibrator

        cal = IsotonicCalibrator()
        predicted = np.array([10.0, 30.0, 50.0, 70.0, 90.0], dtype=np.float64)
        actual = np.array([15.0, 32.0, 48.0, 68.0, 88.0], dtype=np.float64)
        cal.fit(predicted, actual)

        save_path = tmp_path / "isotonic_calibrator.npz"
        cal.save(save_path)

        loaded = IsotonicCalibrator.load(save_path)
        # Should give similar calibrated value for known input
        calibrated_before = cal.calibrate(50.0)
        calibrated_after = loaded.calibrate(50.0)
        assert abs(calibrated_before - calibrated_after) < 1e-10


# ---------------------------------------------------------------------------
# Conformal predictor save/load roundtrip via trainer
# ---------------------------------------------------------------------------


class TestConformalRoundtrip:
    """Verify conformal predictor survives save → load → predict cycle."""

    def test_conformal_save_load_predict(self, tmp_path: Path) -> None:
        _seed()
        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)

        cal_pred = np.array([20.0, 40.0, 55.0, 70.0, 85.0, 95.0], dtype=np.float64)
        cal_actual = np.array([22.0, 38.0, 58.0, 68.0, 80.0, 93.0], dtype=np.float64)
        trainer.calibrate(cal_pred, cal_actual)

        original_quantile = trainer.conformal._quantile
        assert original_quantile is not None

        # Save through trainer
        out_dir = tmp_path / "conformal_models"
        paths = trainer.save_all(out_dir)
        assert "conformal_predictor" in paths

        # Manually reload the conformal predictor (same path as step_engine uses)
        from sopilot.nn.conformal import SplitConformalPredictor

        cp = SplitConformalPredictor()
        data = np.load(out_dir / "conformal_predictor.npz")
        cp._quantile = float(data["quantile"])
        cp._calibrated = True

        # Should produce valid intervals
        point_pred = 60.0
        pred_val, lo, hi = cp.predict(point_pred)
        assert pred_val == point_pred
        assert lo < point_pred
        assert hi > point_pred
        assert abs(cp._quantile - original_quantile) < 1e-10

    def test_conformal_coverage_direction(self, tmp_path: Path) -> None:
        """Wider alpha => narrower intervals (less coverage demanded)."""
        _seed()
        from sopilot.nn.conformal import SplitConformalPredictor

        pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
        actual = np.array([12.0, 18.0, 33.0, 37.0, 52.0], dtype=np.float64)

        cp_tight = SplitConformalPredictor(alpha=0.05)
        cp_tight.calibrate(pred, actual)

        cp_wide = SplitConformalPredictor(alpha=0.5)
        cp_wide.calibrate(pred, actual)

        assert cp_tight.interval_width >= cp_wide.interval_width


# ---------------------------------------------------------------------------
# evaluate_sop with neural_mode tests
# ---------------------------------------------------------------------------


class TestEvaluateSopNeuralMode:
    """Verify that evaluate_sop respects neural_mode flag and uses neural models."""

    def _build_meta(self, n_clips: int) -> list[dict]:
        """Build minimal clip metadata for evaluate_sop."""
        return [{"start_sec": float(i), "end_sec": float(i + 1)} for i in range(n_clips)]

    def test_neural_mode_flag_in_result(self, tmp_path: Path) -> None:
        """When neural_mode=True, result should contain neural_mode=True."""
        _seed()
        n_clips = 20
        gold = np.random.randn(n_clips, DIM).astype(np.float32)
        trainee = np.random.randn(n_clips, DIM).astype(np.float32)
        gold_meta = self._build_meta(n_clips)
        trainee_meta = self._build_meta(n_clips)

        # Without a model dir it will fall back but still set the flag
        result = evaluate_sop(
            gold_embeddings=gold,
            trainee_embeddings=trainee,
            gold_meta=gold_meta,
            trainee_meta=trainee_meta,
            threshold_factor=1.0,
            min_step_clips=2,
            low_similarity_threshold=0.3,
            w_miss=10.0,
            w_swap=8.0,
            w_dev=5.0,
            w_time=3.0,
            w_warp=12.0,
            neural_mode=True,
            neural_model_dir=tmp_path,  # no models saved yet
            neural_device="cpu",
        )
        assert result["neural_mode"] is True
        assert "score" in result
        assert 0.0 <= result["score"] <= 100.0

    def test_neural_mode_false(self, tmp_path: Path) -> None:
        """When neural_mode=False, result should contain neural_mode=False."""
        _seed()
        n_clips = 20
        gold = np.random.randn(n_clips, DIM).astype(np.float32)
        trainee = np.random.randn(n_clips, DIM).astype(np.float32)

        result = evaluate_sop(
            gold_embeddings=gold,
            trainee_embeddings=trainee,
            gold_meta=self._build_meta(n_clips),
            trainee_meta=self._build_meta(n_clips),
            threshold_factor=1.0,
            min_step_clips=2,
            low_similarity_threshold=0.3,
            w_miss=10.0,
            w_swap=8.0,
            w_dev=5.0,
            w_time=3.0,
            neural_mode=False,
        )
        assert result["neural_mode"] is False
        assert "neural_score" not in result

    def test_neural_mode_with_trained_models(self, tmp_path: Path) -> None:
        """Train full pipeline, save, then run evaluate_sop in neural mode.

        This is the most realistic integration test: it trains all components,
        saves them, clears caches, and verifies evaluate_sop loads and uses them.
        """
        _seed()
        invalidate_neural_caches()

        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = _make_embeddings()
        metrics, scores = _make_scoring_data()

        # Train all phases
        trainer.train_projection_head(embeddings, boundaries)
        trainer.train_step_segmenter(embeddings, boundaries)
        trainer.train_asformer(embeddings, boundaries)
        trainer.train_scoring_head(metrics, scores)

        # Joint finetune
        rng = np.random.default_rng(33)
        gold_list = embeddings[:3]
        trainee_list = [e + rng.standard_normal(e.shape).astype(np.float32) * 0.3 for e in gold_list]
        target_scores = np.array([80.0, 60.0, 45.0], dtype=np.float32)
        trainer.joint_finetune(gold_list, trainee_list, target_scores)

        # Calibrate
        cal_pred = np.array([30.0, 50.0, 70.0, 90.0], dtype=np.float64)
        cal_actual = np.array([35.0, 48.0, 72.0, 85.0], dtype=np.float64)
        trainer.calibrate(cal_pred, cal_actual)

        # Save everything
        model_dir = tmp_path / "neural_models"
        trainer.save_all(model_dir)

        # Clear caches so evaluate_sop must reload from disk
        invalidate_neural_caches()

        # Run evaluate_sop in neural mode
        n_clips = CLIPS_PER_VIDEO
        gold_emb = embeddings[0]
        trainee_emb = embeddings[1]

        result = evaluate_sop(
            gold_embeddings=gold_emb,
            trainee_embeddings=trainee_emb,
            gold_meta=self._build_meta(n_clips),
            trainee_meta=self._build_meta(n_clips),
            threshold_factor=1.0,
            min_step_clips=2,
            low_similarity_threshold=0.3,
            w_miss=10.0,
            w_swap=8.0,
            w_dev=5.0,
            w_time=3.0,
            w_warp=12.0,
            neural_mode=True,
            neural_model_dir=model_dir,
            neural_device="cpu",
            neural_soft_dtw_gamma=1.0,
            neural_uncertainty_samples=10,
            neural_calibration_enabled=True,
            neural_cuda_dtw=False,
            neural_ot_alignment=False,
            neural_conformal_alpha=0.1,
        )

        assert result["neural_mode"] is True
        assert "score" in result
        assert 0.0 <= result["score"] <= 100.0
        assert "step_boundaries" in result
        assert "metrics" in result

        # Neural scoring should be present since we saved a scoring head
        assert "neural_score" in result, "evaluate_sop should produce neural_score when models are available"
        ns = result["neural_score"]
        assert "score" in ns
        assert "uncertainty" in ns
        assert "ci_lower" in ns
        assert "ci_upper" in ns
        assert ns["n_samples"] == 10

        # Calibrated score should be present
        assert ns["calibrated_score"] is not None

        # Conformal intervals should be present
        assert ns["conformal_ci_lower"] is not None
        assert ns["conformal_ci_upper"] is not None
        assert ns["conformal_ci_lower"] <= ns["conformal_ci_upper"]

        # Clean up caches after test
        invalidate_neural_caches()


# ---------------------------------------------------------------------------
# detect_step_boundaries with neural models
# ---------------------------------------------------------------------------


class TestDetectBoundariesNeural:
    """Verify that detect_step_boundaries dispatches to neural segmenters."""

    def test_fallback_when_no_models(self, tmp_path: Path) -> None:
        """Without saved models, should fall back to threshold heuristic."""
        _seed()
        invalidate_neural_caches()
        emb = np.random.randn(25, DIM).astype(np.float32)
        boundaries = detect_step_boundaries(
            emb,
            threshold_factor=1.0,
            min_step_clips=2,
            neural_model_dir=tmp_path,
            neural_device="cpu",
        )
        assert boundaries[0] == 0
        assert boundaries[-1] == 25
        invalidate_neural_caches()

    def test_with_segmenter_model(self, tmp_path: Path) -> None:
        """With a saved MS-TCN segmenter, should use it for boundaries."""
        _seed()
        invalidate_neural_caches()
        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = _make_embeddings()
        trainer.train_step_segmenter(embeddings, boundaries)

        model_dir = tmp_path / "seg_models"
        trainer.save_all(model_dir)
        invalidate_neural_caches()

        emb = np.random.randn(25, DIM).astype(np.float32)
        result_boundaries = detect_step_boundaries(
            emb,
            threshold_factor=1.0,
            min_step_clips=2,
            neural_model_dir=model_dir,
            neural_device="cpu",
        )
        assert result_boundaries[0] == 0
        assert result_boundaries[-1] == 25
        invalidate_neural_caches()

    def test_with_asformer_model(self, tmp_path: Path) -> None:
        """With a saved ASFormer, should prefer it over MS-TCN."""
        _seed()
        invalidate_neural_caches()
        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = _make_embeddings()
        trainer.train_asformer(embeddings, boundaries)

        model_dir = tmp_path / "asf_models"
        trainer.save_all(model_dir)
        invalidate_neural_caches()

        emb = np.random.randn(25, DIM).astype(np.float32)
        result_boundaries = detect_step_boundaries(
            emb,
            threshold_factor=1.0,
            min_step_clips=2,
            neural_model_dir=model_dir,
            neural_device="cpu",
        )
        assert result_boundaries[0] == 0
        assert result_boundaries[-1] == 25
        invalidate_neural_caches()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error paths."""

    def test_joint_finetune_requires_prerequisites(self, tmp_path: Path) -> None:
        """joint_finetune should raise if projection_head or scoring_head missing."""
        _seed()
        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)

        with pytest.raises(RuntimeError, match="Must train"):
            trainer.joint_finetune(
                gold_embeddings_list=[np.zeros((10, DIM), dtype=np.float32)],
                trainee_embeddings_list=[np.zeros((10, DIM), dtype=np.float32)],
                target_scores=np.array([50.0], dtype=np.float32),
            )

    def test_save_all_with_no_models(self, tmp_path: Path) -> None:
        """save_all on a fresh trainer produces empty dict."""
        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        paths = trainer.save_all(tmp_path / "empty")
        assert paths == {}

    def test_single_video_training(self, tmp_path: Path) -> None:
        """Pipeline should work with just one video."""
        _seed()
        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)
        embeddings, boundaries = _make_embeddings(n_videos=1, n_clips=30)

        log_proj = trainer.train_projection_head(embeddings, boundaries)
        assert log_proj.epochs_completed == cfg.proj_epochs

        log_seg = trainer.train_step_segmenter(embeddings, boundaries)
        assert log_seg.epochs_completed == cfg.seg_epochs

    def test_very_short_video(self, tmp_path: Path) -> None:
        """Segmenter should handle very short sequences gracefully."""
        _seed()
        cfg = _make_tiny_config(tmp_path)
        trainer = SOPilotTrainer(cfg)

        # 6 clips only (minimum for 3 steps of 2 clips each)
        embeddings, boundaries = _make_embeddings(n_videos=3, n_clips=6)

        log_seg = trainer.train_step_segmenter(embeddings, boundaries)
        assert log_seg.epochs_completed == cfg.seg_epochs

    def test_invalidate_neural_caches(self) -> None:
        """invalidate_neural_caches should clear all global caches."""
        # Just verify it runs without error
        invalidate_neural_caches()
        # Call again to verify idempotent
        invalidate_neural_caches()

    def test_reproducibility(self, tmp_path: Path) -> None:
        """Same seeds should produce same training losses."""
        losses_run1 = []
        losses_run2 = []

        for losses in (losses_run1, losses_run2):
            _seed()
            cfg = _make_tiny_config(tmp_path / "repro")
            trainer = SOPilotTrainer(cfg)
            embeddings, boundaries = _make_embeddings()
            log = trainer.train_projection_head(embeddings, boundaries)
            losses.extend(log.epoch_losses)

        assert len(losses_run1) == len(losses_run2)
        for l1, l2 in zip(losses_run1, losses_run2, strict=False):
            assert abs(l1 - l2) < 1e-6, f"Losses differ: {l1} vs {l2}"
