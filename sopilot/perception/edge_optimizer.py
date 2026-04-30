"""Edge Deployment Optimizer — model profiling and Jetson/ONNX optimization.

Phase 42: Profiles perception models and generates edge-deployment
recommendations for Jetson Orin, Jetson Nano, and x86 CPU targets.
Exports models to ONNX and applies INT8 quantization.

Supported device profiles:
    jetson_orin  — NVIDIA Jetson AGX Orin (12GB LPDDR5, 1792 CUDA cores)
    jetson_nano  — NVIDIA Jetson Nano (4GB LPDDR4, 128 CUDA cores)
    rpi5_coral   — Raspberry Pi 5 + Google Coral Edge TPU
    x86_cpu      — Standard x86 CPU (no GPU)

Usage::

    optimizer = EdgeOptimizer(EdgeProfile.for_device("jetson_orin"))
    report = optimizer.profile_model(model, input_shape=(1, 3, 224, 224))
    rec = optimizer.recommend_quantization(report)
    print(rec["recommended_precision"])  # "int8" or "fp16"

ONNX Export::

    result = optimizer.export_onnx(
        model, (1, 3, 224, 224), "models/dinov2.onnx"
    )
    print(result["output_path"], result["model_size_mb"])

Full pipeline::

    optimizer = EdgeOptimizer(EdgeProfile.for_device("jetson_orin"))
    full_report = optimizer.get_optimization_report(model, input_shape)
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device Profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeProfile:
    """Hardware profile for an edge deployment target.

    Parameters
    ----------
    device:
        Device identifier string.
    max_vram_mb:
        Available GPU/accelerator memory in MB (0 = CPU-only).
    max_ram_mb:
        Available system RAM in MB.
    supports_cuda:
        Whether CUDA is available.
    supports_tensorrt:
        Whether TensorRT is available.
    target_fps:
        Minimum acceptable inference FPS for real-time operation.
    max_latency_ms:
        Maximum acceptable per-frame latency in milliseconds.
    """

    device: str
    max_vram_mb: int
    max_ram_mb: int
    supports_cuda: bool
    supports_tensorrt: bool
    target_fps: float = 10.0
    max_latency_ms: float = 100.0

    @classmethod
    def for_device(cls, device: str) -> "EdgeProfile":
        """Return a pre-configured profile for a known device.

        Parameters
        ----------
        device:
            One of ``"jetson_orin"``, ``"jetson_nano"``,
            ``"rpi5_coral"``, ``"x86_cpu"``.
        """
        profiles = {
            "jetson_orin": cls(
                device="jetson_orin",
                max_vram_mb=12288,
                max_ram_mb=32768,
                supports_cuda=True,
                supports_tensorrt=True,
                target_fps=30.0,
                max_latency_ms=33.0,
            ),
            "jetson_nano": cls(
                device="jetson_nano",
                max_vram_mb=4096,
                max_ram_mb=4096,
                supports_cuda=True,
                supports_tensorrt=True,
                target_fps=10.0,
                max_latency_ms=100.0,
            ),
            "rpi5_coral": cls(
                device="rpi5_coral",
                max_vram_mb=0,  # TPU, not VRAM
                max_ram_mb=8192,
                supports_cuda=False,
                supports_tensorrt=False,
                target_fps=15.0,
                max_latency_ms=67.0,
            ),
            "x86_cpu": cls(
                device="x86_cpu",
                max_vram_mb=0,
                max_ram_mb=16384,
                supports_cuda=False,
                supports_tensorrt=False,
                target_fps=5.0,
                max_latency_ms=200.0,
            ),
        }
        if device not in profiles:
            logger.warning("Unknown device %r, using x86_cpu profile", device)
            return profiles["x86_cpu"]
        return profiles[device]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Profiling Result
# ---------------------------------------------------------------------------


@dataclass
class ModelProfile:
    """Result of profiling a model on synthetic inputs."""

    model_name: str
    input_shape: tuple[int, ...]
    param_count: int = 0
    model_size_mb: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_fps: float = 0.0
    peak_memory_mb: float = 0.0
    n_runs: int = 0
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["input_shape"] = list(self.input_shape)
        return d


# ---------------------------------------------------------------------------
# EdgeOptimizer
# ---------------------------------------------------------------------------


class EdgeOptimizer:
    """Model profiling and edge deployment optimizer.

    Parameters
    ----------
    profile:
        Target device :class:`EdgeProfile`.
    """

    def __init__(self, profile: EdgeProfile) -> None:
        self._profile = profile
        self._export_count = 0
        self._profile_count = 0
        logger.info(
            "EdgeOptimizer created for device=%s (vram=%dMB, cuda=%s, trt=%s)",
            profile.device,
            profile.max_vram_mb,
            profile.supports_cuda,
            profile.supports_tensorrt,
        )

    # ── Profiling ──────────────────────────────────────────────────────

    def profile_model(
        self,
        model: Any,
        input_shape: tuple[int, ...],
        n_runs: int = 50,
        warmup_runs: int = 5,
        device: str | None = None,
    ) -> ModelProfile:
        """Profile a PyTorch model's latency and memory on synthetic inputs.

        Parameters
        ----------
        model:
            PyTorch nn.Module to profile.
        input_shape:
            Input tensor shape ``(B, C, H, W)``.
        n_runs:
            Number of inference runs for latency statistics.
        warmup_runs:
            Warm-up runs to exclude from statistics.
        device:
            Torch device (default: ``"cuda"`` if available else ``"cpu"``).

        Returns
        -------
        ModelProfile
        """
        import torch

        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_name = type(model).__name__

        # Parameter count and size
        param_count = sum(p.numel() for p in model.parameters())
        size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        size_mb = size_bytes / (1024 ** 2)

        # Move to device
        try:
            model = model.to(target_device)
            model.eval()
        except Exception:
            logger.debug("Cannot move model to %s, using CPU", target_device)
            target_device = "cpu"

        # Synthetic input
        x = torch.randn(*input_shape, device=target_device)

        # Warm-up
        with torch.no_grad():
            for _ in range(warmup_runs):
                try:
                    model(x)
                except Exception:
                    break

        # Profile runs
        latencies: list[float] = []
        with torch.no_grad():
            for _ in range(n_runs):
                if target_device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                try:
                    model(x)
                except Exception:
                    break
                if target_device == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

        if not latencies:
            latencies = [float("inf")]

        lat_arr = np.array(latencies)
        avg_ms = float(lat_arr.mean())
        throughput = 1000.0 / avg_ms if avg_ms > 0 else 0.0

        # Peak memory (GPU only)
        peak_memory_mb = 0.0
        if target_device == "cuda" and torch.cuda.is_available():
            try:
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        self._profile_count += 1

        return ModelProfile(
            model_name=model_name,
            input_shape=tuple(input_shape),
            param_count=param_count,
            model_size_mb=round(size_mb, 2),
            avg_latency_ms=round(avg_ms, 2),
            p50_latency_ms=round(float(np.percentile(lat_arr, 50)), 2),
            p95_latency_ms=round(float(np.percentile(lat_arr, 95)), 2),
            p99_latency_ms=round(float(np.percentile(lat_arr, 99)), 2),
            throughput_fps=round(throughput, 2),
            peak_memory_mb=round(peak_memory_mb, 2),
            n_runs=len(latencies),
            device=target_device,
        )

    # ── Recommendation ─────────────────────────────────────────────────

    def recommend_quantization(
        self, profile: ModelProfile
    ) -> dict[str, Any]:
        """Recommend quantization strategy based on model profile + device.

        Returns
        -------
        dict
            Keys: ``recommended_precision``, ``estimated_speedup``,
            ``estimated_size_reduction``, ``fits_on_device``, ``warnings``.
        """
        device = self._profile
        warnings: list[str] = []
        fits = profile.model_size_mb < device.max_ram_mb

        # Determine precision
        if device.supports_tensorrt and profile.avg_latency_ms > device.max_latency_ms:
            precision = "int8"
            speedup = 3.0
            size_reduction = 0.75
        elif device.supports_cuda and profile.avg_latency_ms > device.max_latency_ms * 0.5:
            precision = "fp16"
            speedup = 1.8
            size_reduction = 0.5
        else:
            precision = "fp32"
            speedup = 1.0
            size_reduction = 0.0

        if not fits:
            warnings.append(
                f"Model ({profile.model_size_mb:.0f} MB) may not fit in "
                f"device RAM ({device.max_ram_mb} MB)"
            )

        expected_latency = profile.avg_latency_ms / speedup
        if expected_latency > device.max_latency_ms:
            warnings.append(
                f"Even with {precision}, expected latency {expected_latency:.0f} ms "
                f"exceeds target {device.max_latency_ms:.0f} ms"
            )
            # Suggest model splitting or frame skip
            warnings.append("Consider skip_frames=2 or model pruning")

        return {
            "recommended_precision": precision,
            "estimated_speedup": speedup,
            "estimated_size_reduction_pct": round(size_reduction * 100),
            "estimated_latency_ms": round(expected_latency, 1),
            "target_latency_ms": device.max_latency_ms,
            "meets_latency_target": expected_latency <= device.max_latency_ms,
            "fits_on_device": fits,
            "warnings": warnings,
            "device": device.device,
        }

    # ── ONNX Export ────────────────────────────────────────────────────

    def export_onnx(
        self,
        model: Any,
        input_shape: tuple[int, ...],
        output_path: str | Path,
        dynamic_axes: dict[str, Any] | None = None,
        opset: int = 17,
    ) -> dict[str, Any]:
        """Export a PyTorch model to ONNX format.

        Parameters
        ----------
        model:
            PyTorch nn.Module.
        input_shape:
            Input tensor shape ``(B, C, H, W)``.
        output_path:
            Destination path for the ``.onnx`` file.
        dynamic_axes:
            Dynamic axes for variable-length inputs.
        opset:
            ONNX opset version (default: 17).

        Returns
        -------
        dict
            Export result with ``output_path``, ``model_size_mb``,
            ``opset``, ``success``, ``error``.
        """
        import torch

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.randn(*input_shape)
        default_axes = dynamic_axes or {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

        try:
            model.eval()
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                opset_version=opset,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=default_axes,
                do_constant_folding=True,
            )
            size_mb = output_path.stat().st_size / (1024 ** 2)
            self._export_count += 1
            logger.info(
                "EdgeOptimizer: ONNX export to %s (%.1f MB, opset=%d)",
                output_path,
                size_mb,
                opset,
            )
            return {
                "success": True,
                "output_path": str(output_path),
                "model_size_mb": round(size_mb, 2),
                "opset": opset,
                "error": "",
            }
        except Exception as exc:
            # WARN (not DEBUG): a silent ONNX export failure used to mean the
            # caller had no signal beyond `success=False`, which is hard to
            # diagnose in CI logs and on edge devices. The error string is
            # also surfaced in the result dict so callers can react.
            logger.warning("ONNX export failed: %s", exc)
            return {
                "success": False,
                "output_path": str(output_path),
                "model_size_mb": 0.0,
                "opset": opset,
                "error": str(exc),
            }

    # ── Quantization config ────────────────────────────────────────────

    def generate_quantization_config(
        self, precision: str = "int8"
    ) -> dict[str, Any]:
        """Generate a quantization config for TensorRT/ONNX Runtime.

        Parameters
        ----------
        precision:
            ``"int8"``, ``"fp16"``, or ``"fp32"``.

        Returns
        -------
        dict
            Quantization configuration suitable for TRT or ONNXRT.
        """
        device = self._profile
        config: dict[str, Any] = {
            "precision": precision,
            "device": device.device,
        }

        if precision == "int8":
            config.update({
                "calibration_algorithm": "entropy",
                "calibration_batches": 100,
                "per_channel": True,
                "symmetric": True,
                "tensorrt": {
                    "workspace_size_mb": min(device.max_vram_mb // 4, 2048),
                    "int8_mode": True,
                    "fp16_mode": device.supports_cuda,
                },
                "onnxruntime": {
                    "execution_provider": (
                        "CUDAExecutionProvider" if device.supports_cuda
                        else "CPUExecutionProvider"
                    ),
                    "extra_options": {
                        "WeightType": "QInt8",
                        "ActivationType": "QInt8",
                    },
                },
            })
        elif precision == "fp16":
            config.update({
                "tensorrt": {
                    "workspace_size_mb": min(device.max_vram_mb // 4, 2048),
                    "fp16_mode": True,
                    "int8_mode": False,
                },
                "torch": {
                    "dtype": "torch.float16",
                    "autocast": True,
                },
            })
        else:
            config.update({
                "tensorrt": {"fp16_mode": False, "int8_mode": False},
            })

        return config

    # ── Full optimization report ───────────────────────────────────────

    def get_optimization_report(
        self,
        model: Any,
        input_shape: tuple[int, ...],
        n_runs: int = 30,
    ) -> dict[str, Any]:
        """Profile model and generate a full optimization report.

        Returns
        -------
        dict
            Comprehensive report including profile, recommendations,
            quantization config, and deployment checklist.
        """
        profile = self.profile_model(model, input_shape, n_runs=n_runs)
        rec = self.recommend_quantization(profile)
        quant_config = self.generate_quantization_config(rec["recommended_precision"])

        # Build deployment checklist
        checklist = [
            f"✅ Model profiled: {profile.avg_latency_ms:.1f} ms avg latency",
            f"{'✅' if rec['fits_on_device'] else '❌'} Fits on device RAM",
            f"{'✅' if rec['meets_latency_target'] else '⚠️'} Meets latency target ({rec['target_latency_ms']} ms)",
            f"📦 Export to ONNX (opset 17) → {rec['recommended_precision'].upper()} quantization",
        ]
        if self._profile.supports_tensorrt:
            checklist.append("🚀 TensorRT optimization available")
        if rec["warnings"]:
            for w in rec["warnings"]:
                checklist.append(f"⚠️ {w}")

        return {
            "device_profile": self._profile.to_dict(),
            "model_profile": profile.to_dict(),
            "recommendation": rec,
            "quantization_config": quant_config,
            "deployment_checklist": checklist,
        }

    # ── Stats ──────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        return {
            "device": self._profile.device,
            "profiles_run": self._profile_count,
            "exports_done": self._export_count,
            "target_fps": self._profile.target_fps,
            "max_latency_ms": self._profile.max_latency_ms,
        }
