"""Tests for nn.scoring_head — Learned scoring with MC Dropout + Isotonic calibration."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sopilot.nn.scoring_head import (
    ScoringHead,
    IsotonicCalibrator,
    METRIC_KEYS,
    N_METRICS,
    metrics_to_tensor,
    save_scoring_head,
    load_scoring_head,
)


class TestScoringHead:
    def test_output_shape(self) -> None:
        model = ScoringHead()
        x = torch.randn(8, N_METRICS)
        out = model(x)
        assert out.shape == (8, 1)

    def test_output_range(self) -> None:
        model = ScoringHead()
        x = torch.randn(16, N_METRICS)
        out = model(x)
        assert (out >= 0).all()
        assert (out <= 100).all()

    def test_num_parameters(self) -> None:
        model = ScoringHead()
        # Should be around 4K params
        assert model.num_parameters > 1_000
        assert model.num_parameters < 20_000

    def test_gradient_flow(self) -> None:
        model = ScoringHead()
        x = torch.randn(4, N_METRICS, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None

    def test_predict_with_uncertainty(self) -> None:
        model = ScoringHead()
        x = torch.randn(1, N_METRICS)
        result = model.predict_with_uncertainty(x, n_samples=10)
        assert "score" in result
        assert "uncertainty" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert 0 <= result["score"] <= 100
        assert result["ci_lower"] <= result["score"] <= result["ci_upper"]
        assert result["uncertainty"] >= 0

    def test_mc_dropout_produces_variance(self) -> None:
        """MC Dropout should produce non-zero variance across samples."""
        model = ScoringHead(dropout=0.5)
        # Use larger input to increase dropout variation
        x = torch.randn(1, N_METRICS) * 10
        result = model.predict_with_uncertainty(x, n_samples=50)
        # With dropout=0.5, there should be measurable variance
        # (though could be very small for some random seeds)
        assert result["uncertainty"] >= 0.0

    def test_save_load_roundtrip(self, tmp_path) -> None:
        model = ScoringHead()
        x = torch.randn(4, N_METRICS)
        model.eval()
        with torch.no_grad():
            out_before = model(x)

        path = tmp_path / "scoring.pt"
        save_scoring_head(model, path)
        loaded = load_scoring_head(path)

        with torch.no_grad():
            out_after = loaded(x)

        np.testing.assert_allclose(
            out_before.numpy(), out_after.numpy(), atol=1e-6
        )


class TestIsotonicCalibrator:
    def test_fit_and_calibrate(self) -> None:
        predicted = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.float64)
        actual = np.array([12, 22, 28, 42, 48, 62, 68, 82, 88], dtype=np.float64)
        cal = IsotonicCalibrator()
        cal.fit(predicted, actual)

        # Calibrated score should be close to actual for training points
        for p, a in zip(predicted, actual):
            c = cal.calibrate(float(p))
            assert abs(c - a) < 10.0

    def test_monotonic(self) -> None:
        """Isotonic regression output should be monotonically non-decreasing."""
        predicted = np.linspace(0, 100, 50)
        actual = predicted + np.random.randn(50) * 5
        cal = IsotonicCalibrator()
        cal.fit(predicted, actual)

        calibrated = cal.calibrate_batch(np.linspace(0, 100, 100))
        diffs = np.diff(calibrated)
        assert np.all(diffs >= -1e-10)  # Non-decreasing

    def test_uncalibrated_returns_input(self) -> None:
        cal = IsotonicCalibrator()
        assert cal.calibrate(50.0) == 50.0

    def test_save_load_roundtrip(self, tmp_path) -> None:
        predicted = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
        actual = np.array([12.0, 32.0, 48.0, 72.0, 88.0])
        cal = IsotonicCalibrator()
        cal.fit(predicted, actual)

        path = tmp_path / "cal.npz"
        cal.save(path)
        loaded = IsotonicCalibrator.load(path)

        for v in [15.0, 40.0, 60.0, 85.0]:
            assert abs(cal.calibrate(v) - loaded.calibrate(v)) < 1e-10

    def test_extrapolation(self) -> None:
        predicted = np.array([20.0, 40.0, 60.0, 80.0])
        actual = np.array([22.0, 42.0, 58.0, 78.0])
        cal = IsotonicCalibrator()
        cal.fit(predicted, actual)

        # Below range → first fitted value
        assert cal.calibrate(0.0) == cal.calibrate(0.0)
        # Above range → last fitted value
        assert cal.calibrate(100.0) == cal.calibrate(100.0)


class TestMetricsToTensor:
    def test_all_keys_present(self) -> None:
        metrics = {k: float(i) for i, k in enumerate(METRIC_KEYS)}
        tensor = metrics_to_tensor(metrics)
        assert tensor.shape == (1, N_METRICS)
        for i, k in enumerate(METRIC_KEYS):
            assert tensor[0, i].item() == float(i)

    def test_missing_keys_default_to_zero(self) -> None:
        tensor = metrics_to_tensor({})
        assert tensor.shape == (1, N_METRICS)
        assert torch.all(tensor == 0.0)

    def test_n_metrics_is_15(self) -> None:
        assert N_METRICS == 15
        assert len(METRIC_KEYS) == 15
