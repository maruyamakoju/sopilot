"""Tests for Phase 42: Edge Deployment Optimizer.

Tests cover:
    - EdgeProfile: for_device (all 4 devices + unknown fallback), to_dict
    - ModelProfile: fields, to_dict (input_shape serialisation)
    - EdgeOptimizer: init, profile_model, recommend_quantization,
      export_onnx, generate_quantization_config, get_optimization_report,
      get_stats
    - All three quantization paths: int8 / fp16 / fp32
    - ONNX export success + failure (bad model)
    - Latency recommendations vs device constraints
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sopilot.perception.edge_optimizer import (
    EdgeOptimizer,
    EdgeProfile,
    ModelProfile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_model():
    """Return a tiny torch nn.Linear for profiling tests."""
    torch = pytest.importorskip("torch")
    return torch.nn.Linear(4, 2)


def _make_conv_model():
    torch = pytest.importorskip("torch")
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 4),
    )


# ---------------------------------------------------------------------------
# EdgeProfile
# ---------------------------------------------------------------------------


class TestEdgeProfileForDevice:
    def test_jetson_orin_profile(self):
        p = EdgeProfile.for_device("jetson_orin")
        assert p.device == "jetson_orin"
        assert p.max_vram_mb == 12288
        assert p.max_ram_mb == 32768
        assert p.supports_cuda is True
        assert p.supports_tensorrt is True
        assert p.target_fps == 30.0
        assert p.max_latency_ms == 33.0

    def test_jetson_nano_profile(self):
        p = EdgeProfile.for_device("jetson_nano")
        assert p.device == "jetson_nano"
        assert p.max_vram_mb == 4096
        assert p.supports_cuda is True
        assert p.supports_tensorrt is True
        assert p.target_fps == 10.0
        assert p.max_latency_ms == 100.0

    def test_rpi5_coral_profile(self):
        p = EdgeProfile.for_device("rpi5_coral")
        assert p.device == "rpi5_coral"
        assert p.max_vram_mb == 0
        assert p.supports_cuda is False
        assert p.supports_tensorrt is False
        assert p.target_fps == 15.0
        assert p.max_latency_ms == pytest.approx(67.0)

    def test_x86_cpu_profile(self):
        p = EdgeProfile.for_device("x86_cpu")
        assert p.device == "x86_cpu"
        assert p.max_vram_mb == 0
        assert p.supports_cuda is False
        assert p.target_fps == 5.0
        assert p.max_latency_ms == 200.0

    def test_unknown_device_falls_back_to_x86(self):
        p = EdgeProfile.for_device("quantum_computer")
        assert p.device == "x86_cpu"

    def test_profile_is_frozen(self):
        p = EdgeProfile.for_device("jetson_orin")
        with pytest.raises((AttributeError, TypeError)):
            p.max_vram_mb = 99999  # type: ignore[misc]


class TestEdgeProfileToDict:
    def test_to_dict_keys(self):
        p = EdgeProfile.for_device("jetson_orin")
        d = p.to_dict()
        assert "device" in d
        assert "max_vram_mb" in d
        assert "max_ram_mb" in d
        assert "supports_cuda" in d
        assert "supports_tensorrt" in d
        assert "target_fps" in d
        assert "max_latency_ms" in d

    def test_to_dict_values_match(self):
        p = EdgeProfile.for_device("jetson_nano")
        d = p.to_dict()
        assert d["max_vram_mb"] == 4096
        assert d["supports_cuda"] is True


# ---------------------------------------------------------------------------
# ModelProfile
# ---------------------------------------------------------------------------


class TestModelProfile:
    def test_defaults(self):
        mp = ModelProfile(model_name="TestNet", input_shape=(1, 3, 224, 224))
        assert mp.model_name == "TestNet"
        assert mp.avg_latency_ms == 0.0
        assert mp.throughput_fps == 0.0
        assert mp.param_count == 0
        assert mp.n_runs == 0

    def test_to_dict_converts_input_shape(self):
        mp = ModelProfile(model_name="Net", input_shape=(2, 3, 64, 64))
        d = mp.to_dict()
        assert isinstance(d["input_shape"], list)
        assert d["input_shape"] == [2, 3, 64, 64]

    def test_to_dict_all_keys(self):
        mp = ModelProfile(
            model_name="MyModel",
            input_shape=(1, 3, 32, 32),
            param_count=1000,
            model_size_mb=4.0,
            avg_latency_ms=12.5,
            p50_latency_ms=12.0,
            p95_latency_ms=15.0,
            p99_latency_ms=20.0,
            throughput_fps=80.0,
            peak_memory_mb=32.0,
            n_runs=50,
            device="cpu",
        )
        d = mp.to_dict()
        for key in ("model_name", "input_shape", "param_count", "model_size_mb",
                    "avg_latency_ms", "p50_latency_ms", "p95_latency_ms",
                    "p99_latency_ms", "throughput_fps", "peak_memory_mb",
                    "n_runs", "device"):
            assert key in d


# ---------------------------------------------------------------------------
# EdgeOptimizer init + stats
# ---------------------------------------------------------------------------


class TestEdgeOptimizerInit:
    def test_init_no_error(self):
        opt = EdgeOptimizer(EdgeProfile.for_device("jetson_orin"))
        assert opt is not None

    def test_get_stats_initial(self):
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        stats = opt.get_stats()
        assert stats["device"] == "x86_cpu"
        assert stats["profiles_run"] == 0
        assert stats["exports_done"] == 0
        assert stats["target_fps"] == 5.0
        assert stats["max_latency_ms"] == 200.0

    def test_get_stats_all_devices(self):
        for dev in ("jetson_orin", "jetson_nano", "rpi5_coral", "x86_cpu"):
            opt = EdgeOptimizer(EdgeProfile.for_device(dev))
            assert opt.get_stats()["device"] == dev


# ---------------------------------------------------------------------------
# Profile model
# ---------------------------------------------------------------------------


class TestProfileModel:
    def test_returns_model_profile(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        result = opt.profile_model(model, (1, 4), n_runs=5, warmup_runs=2)
        assert isinstance(result, ModelProfile)

    def test_profile_model_name(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        result = opt.profile_model(model, (1, 4), n_runs=3)
        assert result.model_name == "Linear"

    def test_profile_latency_positive(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        result = opt.profile_model(model, (1, 4), n_runs=5)
        assert result.avg_latency_ms > 0
        assert result.throughput_fps > 0

    def test_profile_param_count(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()  # Linear(4,2) → 4*2 + 2 = 10 params
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        result = opt.profile_model(model, (1, 4), n_runs=3)
        assert result.param_count == 10

    def test_profile_p99_geq_p50(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        result = opt.profile_model(model, (1, 4), n_runs=20)
        assert result.p99_latency_ms >= result.p50_latency_ms

    def test_profile_n_runs(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        result = opt.profile_model(model, (1, 4), n_runs=10)
        assert result.n_runs == 10

    def test_profile_input_shape_stored(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        shape = (1, 4)
        result = opt.profile_model(model, shape, n_runs=3)
        assert result.input_shape == shape

    def test_profile_increments_count(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        opt.profile_model(model, (1, 4), n_runs=3)
        opt.profile_model(model, (1, 4), n_runs=3)
        assert opt.get_stats()["profiles_run"] == 2

    def test_profile_conv_model(self):
        torch = pytest.importorskip("torch")
        model = _make_conv_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        result = opt.profile_model(model, (1, 3, 32, 32), n_runs=5)
        assert result.avg_latency_ms >= 0


# ---------------------------------------------------------------------------
# Recommend quantization
# ---------------------------------------------------------------------------


class TestRecommendQuantization:
    def _opt(self, device: str) -> EdgeOptimizer:
        return EdgeOptimizer(EdgeProfile.for_device(device))

    def test_returns_dict_with_keys(self):
        opt = self._opt("jetson_orin")
        mp = ModelProfile(model_name="Net", input_shape=(1,), avg_latency_ms=50.0)
        rec = opt.recommend_quantization(mp)
        for key in ("recommended_precision", "estimated_speedup",
                    "estimated_size_reduction_pct", "estimated_latency_ms",
                    "target_latency_ms", "meets_latency_target",
                    "fits_on_device", "warnings", "device"):
            assert key in rec

    def test_int8_for_slow_model_with_tensorrt(self):
        """Jetson Orin, latency >> 33ms → int8."""
        opt = self._opt("jetson_orin")
        mp = ModelProfile(model_name="Net", input_shape=(1,), avg_latency_ms=200.0)
        rec = opt.recommend_quantization(mp)
        assert rec["recommended_precision"] == "int8"
        assert rec["estimated_speedup"] == 3.0

    def test_fp16_for_moderately_slow_cuda_model(self):
        """CUDA device, latency between 0.5×max and max → fp16."""
        opt = self._opt("jetson_orin")
        # max_latency = 33ms → trigger fp16 when avg > 0.5*33 = 16.5ms
        mp = ModelProfile(model_name="Net", input_shape=(1,), avg_latency_ms=20.0)
        rec = opt.recommend_quantization(mp)
        assert rec["recommended_precision"] == "fp16"
        assert rec["estimated_speedup"] == 1.8

    def test_fp32_for_fast_model(self):
        """Latency already within target → fp32."""
        opt = self._opt("jetson_orin")
        mp = ModelProfile(model_name="Net", input_shape=(1,), avg_latency_ms=5.0)
        rec = opt.recommend_quantization(mp)
        assert rec["recommended_precision"] == "fp32"
        assert rec["estimated_speedup"] == 1.0

    def test_no_cuda_cpu_model_fp32(self):
        """x86_cpu has no CUDA → fp32 regardless of latency (150ms < 200ms)."""
        opt = self._opt("x86_cpu")
        mp = ModelProfile(model_name="Net", input_shape=(1,), avg_latency_ms=150.0)
        rec = opt.recommend_quantization(mp)
        # max_latency=200ms, 0.5*200=100ms, avg=150 > 100 BUT no cuda → fp32 still
        # Actually: supports_cuda=False → elif supports_cuda fails → fp32
        assert rec["recommended_precision"] == "fp32"

    def test_fits_on_device_true(self):
        opt = self._opt("x86_cpu")  # max_ram=16384MB
        mp = ModelProfile(model_name="Net", input_shape=(1,), model_size_mb=100.0)
        rec = opt.recommend_quantization(mp)
        assert rec["fits_on_device"] is True

    def test_fits_on_device_false_generates_warning(self):
        opt = self._opt("jetson_nano")  # max_ram=4096MB
        mp = ModelProfile(model_name="Net", input_shape=(1,), model_size_mb=8000.0)
        rec = opt.recommend_quantization(mp)
        assert rec["fits_on_device"] is False
        assert len(rec["warnings"]) > 0

    def test_device_name_in_result(self):
        opt = self._opt("rpi5_coral")
        mp = ModelProfile(model_name="Net", input_shape=(1,))
        rec = opt.recommend_quantization(mp)
        assert rec["device"] == "rpi5_coral"

    def test_target_latency_matches_profile(self):
        opt = self._opt("jetson_nano")
        mp = ModelProfile(model_name="Net", input_shape=(1,))
        rec = opt.recommend_quantization(mp)
        assert rec["target_latency_ms"] == 100.0


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------


class TestExportOnnx:
    def test_successful_export(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "model.onnx"
            result = opt.export_onnx(model, (1, 4), out)
            assert result["success"] is True
            assert result["model_size_mb"] >= 0  # tiny model rounds to 0.0 MB
            assert result["opset"] == 17
            assert out.exists()
            assert out.stat().st_size > 0  # file must have non-zero bytes

    def test_export_increments_count(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            opt.export_onnx(model, (1, 4), Path(tmpdir) / "m.onnx")
            assert opt.get_stats()["exports_done"] == 1

    def test_export_creates_parent_dirs(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "nested" / "dir" / "model.onnx"
            result = opt.export_onnx(model, (1, 4), out)
            assert result["success"] is True
            assert out.exists()

    def test_export_custom_opset(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            result = opt.export_onnx(model, (1, 4), Path(tmpdir) / "m.onnx", opset=14)
            assert result["opset"] == 14

    def test_export_failure_returns_error(self):
        """A non-module object should fail export gracefully."""
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            result = opt.export_onnx("not_a_model", (1, 4), Path(tmpdir) / "fail.onnx")
            assert result["success"] is False
            assert result["error"] != ""

    def test_export_path_returned_as_string(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "m.onnx"
            result = opt.export_onnx(model, (1, 4), out)
            assert isinstance(result["output_path"], str)


# ---------------------------------------------------------------------------
# Quantization config
# ---------------------------------------------------------------------------


class TestGenerateQuantizationConfig:
    def test_int8_config_keys(self):
        opt = EdgeOptimizer(EdgeProfile.for_device("jetson_orin"))
        cfg = opt.generate_quantization_config("int8")
        assert cfg["precision"] == "int8"
        assert "calibration_algorithm" in cfg
        assert "tensorrt" in cfg
        assert "onnxruntime" in cfg
        assert cfg["tensorrt"]["int8_mode"] is True

    def test_fp16_config(self):
        opt = EdgeOptimizer(EdgeProfile.for_device("jetson_nano"))
        cfg = opt.generate_quantization_config("fp16")
        assert cfg["precision"] == "fp16"
        assert "tensorrt" in cfg
        assert cfg["tensorrt"]["fp16_mode"] is True

    def test_fp32_config(self):
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        cfg = opt.generate_quantization_config("fp32")
        assert cfg["precision"] == "fp32"
        assert cfg["tensorrt"]["fp16_mode"] is False
        assert cfg["tensorrt"]["int8_mode"] is False

    def test_int8_workspace_capped(self):
        """workspace_size_mb should not exceed 2048 MB."""
        opt = EdgeOptimizer(EdgeProfile.for_device("jetson_orin"))
        cfg = opt.generate_quantization_config("int8")
        ws = cfg["tensorrt"]["workspace_size_mb"]
        assert ws <= 2048

    def test_cpu_device_onnxruntime_cpu_provider(self):
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        cfg = opt.generate_quantization_config("int8")
        assert cfg["onnxruntime"]["execution_provider"] == "CPUExecutionProvider"

    def test_cuda_device_onnxruntime_cuda_provider(self):
        opt = EdgeOptimizer(EdgeProfile.for_device("jetson_orin"))
        cfg = opt.generate_quantization_config("int8")
        assert cfg["onnxruntime"]["execution_provider"] == "CUDAExecutionProvider"

    def test_device_name_in_config(self):
        for dev in ("jetson_orin", "x86_cpu"):
            opt = EdgeOptimizer(EdgeProfile.for_device(dev))
            cfg = opt.generate_quantization_config("int8")
            assert cfg["device"] == dev


# ---------------------------------------------------------------------------
# Full optimization report
# ---------------------------------------------------------------------------


class TestGetOptimizationReport:
    def test_returns_dict(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        report = opt.get_optimization_report(model, (1, 4), n_runs=5)
        assert isinstance(report, dict)

    def test_report_keys(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        report = opt.get_optimization_report(model, (1, 4), n_runs=5)
        for key in ("device_profile", "model_profile", "recommendation",
                    "quantization_config", "deployment_checklist"):
            assert key in report

    def test_checklist_is_list(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        report = opt.get_optimization_report(model, (1, 4), n_runs=5)
        assert isinstance(report["deployment_checklist"], list)
        assert len(report["deployment_checklist"]) >= 3

    def test_checklist_includes_tensorrt_for_jetson(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("jetson_orin"))
        report = opt.get_optimization_report(model, (1, 4), n_runs=5)
        checklist_text = " ".join(report["deployment_checklist"])
        assert "TensorRT" in checklist_text

    def test_model_profile_in_report_is_dict(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        report = opt.get_optimization_report(model, (1, 4), n_runs=5)
        assert isinstance(report["model_profile"], dict)
        assert "avg_latency_ms" in report["model_profile"]

    def test_recommendation_in_report(self):
        torch = pytest.importorskip("torch")
        model = _make_simple_model()
        opt = EdgeOptimizer(EdgeProfile.for_device("x86_cpu"))
        report = opt.get_optimization_report(model, (1, 4), n_runs=5)
        rec = report["recommendation"]
        assert "recommended_precision" in rec
        assert rec["recommended_precision"] in ("int8", "fp16", "fp32")
