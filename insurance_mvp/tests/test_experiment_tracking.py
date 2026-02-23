"""Tests for experiment tracking and ablation infrastructure.

Validates ExperimentTracker, AblationRunner, and reproducibility guarantees.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from insurance_mvp.evaluation.experiment import (
    AblationReport,
    AblationRunner,
    ExperimentConfig,
    ExperimentResult,
    ExperimentTracker,
    compute_standard_metrics,
    format_ablation_report,
)


class TestExperimentConfig:
    def test_config_creation(self):
        cfg = ExperimentConfig(
            name="test_exp",
            seed=42,
            description="Test experiment",
            hyperparams={"lr": 0.01, "batch_size": 32},
        )
        assert cfg.name == "test_exp"
        assert cfg.seed == 42

    def test_config_with_ablation_flags(self):
        cfg = ExperimentConfig(
            name="ablation",
            seed=0,
            hyperparams={},
            ablation_flags={"use_recalibration": True, "use_prompt_v2": True},
        )
        assert cfg.ablation_flags["use_recalibration"] is True

    def test_derive_creates_copy(self):
        base = ExperimentConfig(
            name="base", seed=42,
            ablation_flags={"a": True, "b": True},
        )
        derived = base.derive(name="derived", ablation_flags={"a": False, "b": True})
        assert derived.name == "derived"
        assert derived.seed == 42  # Inherited
        assert derived.ablation_flags["a"] is False
        assert base.ablation_flags["a"] is True  # Original unchanged

    def test_fingerprint_deterministic(self):
        cfg1 = ExperimentConfig(name="x", seed=1, hyperparams={"a": 1, "b": 2})
        cfg2 = ExperimentConfig(name="x", seed=1, hyperparams={"b": 2, "a": 1})
        assert cfg1.fingerprint == cfg2.fingerprint

    def test_fingerprint_changes_with_content(self):
        cfg1 = ExperimentConfig(name="x", seed=1)
        cfg2 = ExperimentConfig(name="x", seed=2)
        assert cfg1.fingerprint != cfg2.fingerprint


class TestExperimentTracker:
    def test_register_and_retrieve(self):
        tracker = ExperimentTracker()
        cfg = ExperimentConfig(name="exp1", seed=42, hyperparams={"alpha": 0.1})
        tracker.register(cfg)
        assert "exp1" in tracker.configs

    def test_record_metrics(self):
        tracker = ExperimentTracker()
        cfg = ExperimentConfig(name="exp1", seed=42, hyperparams={})
        tracker.register(cfg)
        result = tracker.record("exp1", {"accuracy": 0.85, "f1": 0.82})
        assert result.metrics["accuracy"] == 0.85
        # Also check via results_for
        results = tracker.results_for("exp1")
        assert len(results) == 1
        assert results[0].metrics["accuracy"] == 0.85

    def test_record_unregistered_raises(self):
        tracker = ExperimentTracker()
        with pytest.raises(KeyError):
            tracker.record("nonexistent", {"accuracy": 0.5})

    def test_compare_experiments(self):
        tracker = ExperimentTracker()
        cfg1 = ExperimentConfig(name="baseline", seed=42, hyperparams={})
        cfg2 = ExperimentConfig(name="improved", seed=42, hyperparams={"lr": 0.01})
        tracker.register(cfg1)
        tracker.register(cfg2)
        tracker.record("baseline", {"accuracy": 0.70})
        tracker.record("improved", {"accuracy": 0.85})
        comparison = tracker.compare("baseline", "improved")
        # compare returns delta = a - b, so baseline(0.70) - improved(0.85) = -0.15
        assert comparison["accuracy"]["delta"] == pytest.approx(-0.15, abs=0.001)
        assert comparison["accuracy"]["a"] == pytest.approx(0.70, abs=0.001)
        assert comparison["accuracy"]["b"] == pytest.approx(0.85, abs=0.001)

    def test_compare_no_results_raises(self):
        tracker = ExperimentTracker()
        cfg = ExperimentConfig(name="exp", seed=42)
        tracker.register(cfg)
        with pytest.raises(KeyError):
            tracker.compare("exp", "nonexistent")

    def test_save_and_load(self):
        tracker = ExperimentTracker()
        cfg = ExperimentConfig(name="exp1", seed=42, hyperparams={"x": 1})
        tracker.register(cfg)
        tracker.record("exp1", {"acc": 0.9})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        tracker.save(path)

        # Verify JSON structure
        with open(path) as f:
            data = json.load(f)
        assert data["schema_version"] == 1
        assert "exp1" in data["configs"]
        assert len(data["results"]) == 1

        # Load into a new tracker
        tracker2 = ExperimentTracker()
        tracker2.load(path)
        assert "exp1" in tracker2.configs
        results = tracker2.results_for("exp1")
        assert len(results) == 1
        assert results[0].metrics["acc"] == 0.9

        Path(path).unlink()

    def test_seed_reproducibility(self):
        """Same seed should produce same results in deterministic code."""
        tracker = ExperimentTracker()
        cfg1 = ExperimentConfig(name="run1", seed=42, hyperparams={})
        cfg2 = ExperimentConfig(name="run2", seed=42, hyperparams={})
        tracker.register(cfg1)
        tracker.register(cfg2)

        rng1 = np.random.RandomState(cfg1.seed)
        rng2 = np.random.RandomState(cfg2.seed)
        assert rng1.rand() == rng2.rand()

    def test_duplicate_identical_is_idempotent(self):
        """Re-registering the same config is a no-op."""
        tracker = ExperimentTracker()
        cfg = ExperimentConfig(name="dup", seed=0, hyperparams={})
        tracker.register(cfg)
        tracker.register(cfg)  # Should not raise
        assert len(tracker.configs) == 1

    def test_duplicate_different_fingerprint_raises(self):
        """Registering a different config with same name raises."""
        tracker = ExperimentTracker()
        cfg1 = ExperimentConfig(name="dup", seed=0, hyperparams={})
        cfg2 = ExperimentConfig(name="dup", seed=999, hyperparams={"different": True})
        tracker.register(cfg1)
        with pytest.raises(ValueError, match="already registered"):
            tracker.register(cfg2)

    def test_best_result(self):
        tracker = ExperimentTracker()
        cfg1 = ExperimentConfig(name="low", seed=1)
        cfg2 = ExperimentConfig(name="high", seed=2)
        tracker.register(cfg1)
        tracker.register(cfg2)
        tracker.record("low", {"accuracy": 0.60})
        tracker.record("high", {"accuracy": 0.95})
        best = tracker.best_result(metric="accuracy")
        assert best is not None
        assert best.config.name == "high"

    def test_best_result_empty(self):
        tracker = ExperimentTracker()
        assert tracker.best_result() is None


class TestComputeStandardMetrics:
    def test_perfect_predictions(self):
        y_true = [0, 1, 2, 3, 0, 1, 2, 3]
        y_pred = [0, 1, 2, 3, 0, 1, 2, 3]
        metrics = compute_standard_metrics(y_true, y_pred, n_classes=4)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["cohen_kappa"] == 1.0

    def test_random_predictions(self):
        y_true = [0, 0, 1, 1, 2, 2, 3, 3]
        y_pred = [3, 2, 0, 1, 1, 0, 2, 3]
        metrics = compute_standard_metrics(y_true, y_pred, n_classes=4)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0

    def test_empty_input(self):
        metrics = compute_standard_metrics([], [], n_classes=4)
        assert metrics["accuracy"] == 0.0


class TestAblationRunner:
    def test_ablation_basic(self):
        base = ExperimentConfig(
            name="base",
            seed=42,
            ablation_flags={"feature_a": True, "feature_b": True, "feature_c": True},
        )

        def eval_fn(config: ExperimentConfig):
            acc = 0.5
            if config.ablation_flags.get("feature_a"):
                acc += 0.15
            if config.ablation_flags.get("feature_b"):
                acc += 0.10
            if config.ablation_flags.get("feature_c"):
                acc += 0.05
            return {"accuracy": acc}, 0.1

        runner = AblationRunner(base_config=base, eval_fn=eval_fn)
        report = runner.run()

        assert isinstance(report, AblationReport)
        assert report.base_metrics["accuracy"] == pytest.approx(0.80, abs=0.01)
        assert len(report.ablations) == 3

        # Find the ablation for feature_a
        abl_a = next(a for a in report.ablations if a.flag_name == "feature_a")
        assert abl_a.metrics["accuracy"] == pytest.approx(0.65, abs=0.01)
        assert abl_a.delta["accuracy"] == pytest.approx(0.15, abs=0.01)

    def test_ablation_identifies_most_impactful(self):
        base = ExperimentConfig(
            name="base", seed=42,
            ablation_flags={"a": True, "b": True},
        )

        def eval_fn(config: ExperimentConfig):
            acc = 0.5
            if config.ablation_flags.get("a"):
                acc += 0.30
            if config.ablation_flags.get("b"):
                acc += 0.05
            return {"accuracy": acc}, 0.1

        runner = AblationRunner(base_config=base, eval_fn=eval_fn)
        report = runner.run()

        # Feature importance is sorted by |delta|, most impactful first
        assert report.feature_importance[0][0] == "a"
        assert report.feature_importance[0][1] > report.feature_importance[1][1]

    def test_ablation_no_flags(self):
        base = ExperimentConfig(name="bare", seed=0, ablation_flags={})

        def eval_fn(config: ExperimentConfig):
            return {"accuracy": 1.0}, 0.01

        runner = AblationRunner(base_config=base, eval_fn=eval_fn)
        report = runner.run()
        assert len(report.ablations) == 0
        assert report.base_metrics["accuracy"] == 1.0

    def test_ablation_with_tracker(self):
        tracker = ExperimentTracker()
        base = ExperimentConfig(
            name="tracked", seed=42,
            ablation_flags={"x": True},
        )

        def eval_fn(config: ExperimentConfig):
            acc = 0.9 if config.ablation_flags.get("x") else 0.7
            return {"accuracy": acc}, 0.05

        runner = AblationRunner(base_config=base, eval_fn=eval_fn, tracker=tracker)
        report = runner.run()

        # Tracker should have recorded baseline + 1 ablation
        assert len(tracker.results) == 2


class TestExperimentResult:
    def test_result_creation(self):
        cfg = ExperimentConfig(name="test", seed=0, hyperparams={})
        result = ExperimentResult(
            config=cfg,
            metrics={"accuracy": 0.9, "f1": 0.85},
            timing_sec=12.5,
        )
        assert result.metrics["accuracy"] == 0.9
        assert result.timing_sec == 12.5

    def test_result_serialization_roundtrip(self):
        cfg = ExperimentConfig(name="rt", seed=42, hyperparams={"lr": 0.01})
        result = ExperimentResult(
            config=cfg,
            metrics={"accuracy": 0.9},
            timing_sec=5.0,
        )
        data = result.to_dict()
        restored = ExperimentResult.from_dict(data)
        assert restored.config.name == "rt"
        assert restored.metrics["accuracy"] == 0.9
        assert restored.timing_sec == 5.0

    def test_result_has_timestamp(self):
        cfg = ExperimentConfig(name="ts", seed=0)
        result = ExperimentResult(config=cfg, metrics={}, timing_sec=0)
        assert result.timestamp != ""


class TestFormatAblationReport:
    def test_format_produces_string(self):
        base = ExperimentConfig(
            name="fmt", seed=42,
            ablation_flags={"x": True},
        )

        def eval_fn(config: ExperimentConfig):
            acc = 0.8 if config.ablation_flags.get("x") else 0.6
            return {"accuracy": acc}, 0.05

        runner = AblationRunner(base_config=base, eval_fn=eval_fn)
        report = runner.run()
        text = format_ablation_report(report)
        assert "ABLATION STUDY REPORT" in text
        assert "BASELINE METRICS" in text
        assert "FEATURE IMPORTANCE" in text
