#!/usr/bin/env python3
"""Export YOLO-World and YOLOv8-pose models to ONNX format.

Usage::

    python scripts/export_onnx.py
    python scripts/export_onnx.py --dry-run   # CI smoke test (no download)
    python scripts/export_onnx.py --output-dir data/models

Requirements: ``pip install sopilot[yolo_world]`` (ultralytics>=8.0.0)
"""

import argparse
import sys
from pathlib import Path


def export_model(model_name: str, output_dir: Path, *, dry_run: bool = False) -> Path:
    """Export a single ultralytics model to ONNX.

    Parameters
    ----------
    model_name:
        Ultralytics model identifier (e.g. ``"yolov8s-worldv2.pt"``).
    output_dir:
        Directory to write the ``.onnx`` file into.
    dry_run:
        When True, skip the actual export (for CI smoke tests).

    Returns the path where the ONNX file would be written.
    """
    onnx_name = Path(model_name).stem + ".onnx"
    onnx_path = output_dir / onnx_name

    if dry_run:
        print(f"[dry-run] Would export {model_name} → {onnx_path}")
        return onnx_path

    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError as exc:
        print(
            "ultralytics not installed. Install with: pip install sopilot[yolo_world]",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {model_name} ...")
    model = YOLO(model_name)
    print(f"Exporting to ONNX (dynamic=True) → {onnx_path} ...")
    model.export(format="onnx", dynamic=True, simplify=True, imgsz=640)

    # ultralytics writes to the same directory as the .pt file; move to output_dir
    default_onnx = Path(model_name).with_suffix(".onnx")
    if default_onnx.exists() and default_onnx != onnx_path:
        default_onnx.rename(onnx_path)

    print(f"Exported: {onnx_path}")
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO models to ONNX")
    parser.add_argument("--output-dir", default="data/models", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Skip actual export (CI mode)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    models = [
        "yolov8s-worldv2.pt",
        "yolov8s-pose.pt",
    ]

    for model_name in models:
        export_model(model_name, output_dir, dry_run=args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
