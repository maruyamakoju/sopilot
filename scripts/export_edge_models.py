#!/usr/bin/env python3
"""Export SOPilot perception models to ONNX for edge deployment.

Phase 42: Exports DINOv2 backbone, NeuralReID embedder, and YOLO stub
to ONNX format using the EdgeOptimizer, with INT8/FP16 config generation.

Usage::

    python scripts/export_edge_models.py --device jetson_orin --output models/edge/
    python scripts/export_edge_models.py --device jetson_nano --precision fp16
    python scripts/export_edge_models.py --list-devices

Supported models:
    - dinov2_cls     : DINOv2 ViT-B/14 CLS token extractor (768-dim)
    - neural_reid    : NeuralReID person re-identification embedder
    - yolo_stub      : YOLO detection head stub (for profiling)

Output structure::

    models/edge/
    ├── dinov2_cls_jetson_orin_int8.onnx
    ├── neural_reid_jetson_orin_int8.onnx
    ├── yolo_stub_jetson_orin_int8.onnx
    ├── quantization_config_int8.json
    └── export_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("export_edge_models")

# Repository root: scripts/ is one level below project root
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from sopilot.perception.edge_optimizer import EdgeOptimizer, EdgeProfile  # noqa: E402

_SUPPORTED_DEVICES = ("jetson_orin", "jetson_nano", "rpi5_coral", "x86_cpu")


# ---------------------------------------------------------------------------
# Stub model builders (no real weights required for shape/latency profiling)
# ---------------------------------------------------------------------------


def _build_dinov2_stub():
    """Return a tiny stub that mimics DINOv2 CLS token output (768-dim)."""
    import torch
    import torch.nn as nn

    class DINOv2Stub(nn.Module):
        """Mimics DINOv2 ViT-B/14 CLS token extraction."""

        def __init__(self) -> None:
            super().__init__()
            # Patch embed: 14×14 patches, 768-dim → simplified for export
            self.patch_embed = nn.Conv2d(3, 768, kernel_size=14, stride=14)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
            self.head = nn.LayerNorm(768)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            patches = self.patch_embed(x)  # (B, 768, H/14, W/14)
            patches = patches.flatten(2).transpose(1, 2)  # (B, N, 768)
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, patches], dim=1)
            # Return CLS token → (B, 768)
            return self.head(tokens[:, 0])

    return DINOv2Stub()


def _build_neural_reid_stub():
    """Return a stub that mimics NeuralReID's 256-dim L2-normalised embedding."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class NeuralReIDStub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(8),
            )
            self.head = nn.Linear(32 * 8 * 8, 256)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.backbone(x).flatten(1)
            emb = self.head(feat)
            return F.normalize(emb, dim=-1)

    return NeuralReIDStub()


def _build_yolo_stub():
    """Return a minimal stub mimicking YOLO backbone + detection head."""
    import torch
    import torch.nn as nn

    class YOLOStub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(32, 85 * 3)  # 85 = 4 bbox + 1 obj + 80 cls

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.backbone(x).squeeze(-1).squeeze(-1)
            return self.head(feat)

    return YOLOStub()


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    "dinov2_cls": {
        "builder": _build_dinov2_stub,
        "input_shape": (1, 3, 224, 224),
        "dynamic_axes": {"input": {0: "batch"}, "output": {0: "batch"}},
    },
    "neural_reid": {
        "builder": _build_neural_reid_stub,
        "input_shape": (1, 3, 128, 64),  # standard Re-ID resolution
        "dynamic_axes": {"input": {0: "batch"}, "output": {0: "batch"}},
    },
    "yolo_stub": {
        "builder": _build_yolo_stub,
        "input_shape": (1, 3, 640, 640),
        "dynamic_axes": {"input": {0: "batch"}, "output": {0: "batch"}},
    },
}


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------


def export_all(
    device: str,
    output_dir: Path,
    precision: str | None,
    models: list[str],
    dry_run: bool,
    n_profile_runs: int,
) -> dict:
    """Export all requested models and return a summary report."""
    profile = EdgeProfile.for_device(device)
    optimizer = EdgeOptimizer(profile)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Device profile: %s (VRAM=%dMB, TRT=%s)", device, profile.max_vram_mb, profile.supports_tensorrt)

    results = []
    start_total = time.time()

    for model_name in models:
        if model_name not in _MODEL_REGISTRY:
            logger.warning("Unknown model %r — skipping", model_name)
            continue

        entry = _MODEL_REGISTRY[model_name]
        input_shape = entry["input_shape"]
        logger.info("── Profiling: %s  input=%s", model_name, input_shape)

        try:
            model = entry["builder"]()
        except ImportError as e:
            logger.error("Cannot build %s: %s", model_name, e)
            results.append({"model": model_name, "success": False, "error": str(e)})
            continue

        # Profile
        mp = optimizer.profile_model(model, input_shape, n_runs=n_profile_runs)
        rec = optimizer.recommend_quantization(mp)

        chosen_precision = precision or rec["recommended_precision"]
        logger.info(
            "  Recommended: %s (speedup=%.1fx, latency=%.1fms)",
            chosen_precision,
            rec["estimated_speedup"],
            rec["estimated_latency_ms"],
        )

        # Quantization config
        quant_cfg = optimizer.generate_quantization_config(chosen_precision)
        quant_cfg_path = output_dir / f"quantization_config_{chosen_precision}.json"
        if not dry_run:
            quant_cfg_path.write_text(json.dumps(quant_cfg, indent=2))
            logger.info("  Saved quant config → %s", quant_cfg_path)

        # ONNX export
        onnx_fname = f"{model_name}_{device}_{chosen_precision}.onnx"
        onnx_path = output_dir / onnx_fname

        if dry_run:
            logger.info("  [dry-run] Would export → %s", onnx_path)
            result_entry = {
                "model": model_name,
                "device": device,
                "precision": chosen_precision,
                "output_path": str(onnx_path),
                "success": True,
                "dry_run": True,
                "profile": mp.to_dict(),
                "recommendation": rec,
            }
        else:
            export_result = optimizer.export_onnx(
                model,
                input_shape,
                onnx_path,
                dynamic_axes=entry["dynamic_axes"],
            )
            if export_result["success"]:
                logger.info(
                    "  Exported → %s (%.2f MB)", onnx_path, export_result["model_size_mb"]
                )
            else:
                logger.error("  Export FAILED: %s", export_result["error"])

            result_entry = {
                "model": model_name,
                "device": device,
                "precision": chosen_precision,
                "output_path": str(onnx_path),
                "success": export_result["success"],
                "model_size_mb": export_result["model_size_mb"],
                "error": export_result.get("error", ""),
                "profile": mp.to_dict(),
                "recommendation": rec,
            }

        results.append(result_entry)

    total_time = round(time.time() - start_total, 2)
    report = {
        "device": device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_time_seconds": total_time,
        "models_requested": models,
        "results": results,
        "optimizer_stats": optimizer.get_stats(),
    }

    report_path = output_dir / "export_report.json"
    if not dry_run:
        report_path.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Export report saved → %s", report_path)

    successful = sum(1 for r in results if r.get("success"))
    logger.info(
        "Done: %d/%d models exported in %.2fs",
        successful, len(results), total_time,
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export SOPilot perception models to ONNX for edge deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--device",
        choices=_SUPPORTED_DEVICES,
        default="jetson_orin",
        help="Target edge device profile (default: jetson_orin)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/edge"),
        help="Output directory for ONNX files (default: models/edge/)",
    )
    parser.add_argument(
        "--precision",
        choices=("int8", "fp16", "fp32"),
        default=None,
        help="Override quantization precision (default: auto-recommended)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(_MODEL_REGISTRY),
        default=list(_MODEL_REGISTRY),
        metavar="MODEL",
        help=(
            "Models to export: %(choices)s  (default: all). "
            "Example: --models dinov2_cls neural_reid"
        ),
    )
    parser.add_argument(
        "--n-profile-runs",
        type=int,
        default=20,
        help="Number of profiling runs per model (default: 20)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Profile and recommend but do not write ONNX files",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print supported device profiles and exit",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_devices:
        print("\nSupported edge device profiles:")
        for dev in _SUPPORTED_DEVICES:
            p = EdgeProfile.for_device(dev)
            print(
                f"  {dev:<16} VRAM={p.max_vram_mb:>5}MB  "
                f"CUDA={'✓' if p.supports_cuda else '✗'}  "
                f"TRT={'✓' if p.supports_tensorrt else '✗'}  "
                f"target={p.target_fps:.0f}fps  "
                f"max_latency={p.max_latency_ms:.0f}ms"
            )
        return

    report = export_all(
        device=args.device,
        output_dir=args.output,
        precision=args.precision,
        models=args.models,
        dry_run=args.dry_run,
        n_profile_runs=args.n_profile_runs,
    )

    # Print summary
    successes = [r for r in report["results"] if r.get("success")]
    print(f"\n{'='*60}")
    print(f"Export complete: {len(successes)}/{len(report['results'])} models")
    for r in report["results"]:
        status = "✅" if r.get("success") else "❌"
        mb = r.get("model_size_mb", 0)
        size_str = f"({mb:.1f} MB)" if mb else ""
        print(f"  {status} {r['model']:<16} → {r.get('precision','?')} {size_str}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
