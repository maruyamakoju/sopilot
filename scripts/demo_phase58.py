"""Phase 58: Real Inference Stack Demo — RTX 4090

End-to-end demonstration of the full real inference stack:
    1. PerceptionConfig.for_rtx4090()  — production GPU preset
    2. PerceptionEngine with real models
    3. Cascade stage breakdown (YOLO / GDINO / LLaVA latency)
    4. DINOv2 CLS feature vector (768-dim, cosine similarity)
    5. LLaVA scene description (Japanese + English)
    6. VRAM usage report

Usage::

    # Real inference (requires GPU + downloaded models)
    python scripts/demo_phase58.py

    # Dry run with synthetic frame (no real model weights needed)
    python scripts/demo_phase58.py --mock

    # Use a specific video frame
    python scripts/demo_phase58.py --image path/to/frame.jpg
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo_phase58")


# ── Frame loading ──────────────────────────────────────────────────────────


def load_frame(image_path: str | None) -> np.ndarray:
    """Load frame from file or generate a synthetic test frame."""
    if image_path and Path(image_path).exists():
        import cv2
        frame = cv2.imread(image_path)
        if frame is not None:
            logger.info("Loaded frame from %s (%dx%d)", image_path, frame.shape[1], frame.shape[0])
            return frame
        logger.warning("Could not read %s — using synthetic frame", image_path)

    # Try to extract first frame from factory demo video
    demo_video = Path("factory_demo.mp4")
    if demo_video.exists():
        try:
            import cv2
            cap = cv2.VideoCapture(str(demo_video))
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                logger.info("Loaded first frame from %s", demo_video)
                return frame
        except Exception:
            pass

    # Synthetic: gradient frame with colored rectangles simulating people+equipment
    logger.info("Using synthetic test frame (640x480)")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Background gradient (blue-ish factory floor)
    for y in range(480):
        frame[y, :] = [int(20 + y * 0.1), int(25 + y * 0.05), int(40 + y * 0.08)]
    # Simulate person 1 (yellow safety vest)
    frame[150:350, 100:200] = [0, 200, 255]  # yellow
    frame[100:160, 120:180] = [50, 50, 200]  # head (dark)
    # Simulate person 2 (orange vest, no helmet)
    frame[200:400, 350:450] = [0, 140, 255]  # orange
    frame[150:210, 370:430] = [60, 60, 60]   # head
    # Simulate machinery (grey box)
    frame[50:300, 480:620] = [120, 120, 130]
    # Simulate restricted zone marker (red line)
    frame[380:390, :] = [0, 0, 200]
    return frame


# ── VRAM report ───────────────────────────────────────────────────────────


def print_vram_report() -> None:
    """Print current VRAM usage via torch.cuda."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n[VRAM] CUDA not available")
            return
        alloc = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n┌─ VRAM Usage ───────────────────────────────────┐")
        print(f"│  Allocated : {alloc:>6.2f} GB")
        print(f"│  Reserved  : {reserved:>6.2f} GB")
        print(f"│  Total     : {total:>6.2f} GB")
        print(f"│  Free      : {total - reserved:>6.2f} GB")
        print(f"└────────────────────────────────────────────────┘")
    except ImportError:
        print("\n[VRAM] torch not available")


# ── DINOv2 feature demo ───────────────────────────────────────────────────


def demo_dinov2_features(engine: object, frame: np.ndarray) -> None:
    """Extract DINOv2 CLS features and compute self-similarity."""
    backbone = getattr(engine, "_dinov2_backbone", None)
    if backbone is None:
        print("\n[DINOv2] Not initialized (run with real GPU stack)")
        return
    try:
        t0 = time.monotonic()
        features = backbone.extract_cls_features(frame)
        elapsed = (time.monotonic() - t0) * 1000
        norm = float(np.linalg.norm(features))
        # cosine similarity with itself should be 1.0
        cos_sim = float(np.dot(features, features) / (norm * norm + 1e-8))
        print(f"\n┌─ DINOv2 ViT-B/14 Features ─────────────────────┐")
        print(f"│  Shape     : {features.shape}")
        print(f"│  Norm      : {norm:.4f}")
        print(f"│  Self-cos  : {cos_sim:.6f} (should be ≈1.0)")
        print(f"│  Latency   : {elapsed:.1f} ms")
        print(f"│  Preview   : [{', '.join(f'{v:.3f}' for v in features[:6])} ...]")
        print(f"└────────────────────────────────────────────────┘")
    except Exception as e:
        print(f"\n[DINOv2] Feature extraction failed: {e}")


# ── LLaVA scene description ───────────────────────────────────────────────


def demo_llava_description(engine: object, frame: np.ndarray) -> None:
    """Run LLaVA scene description (JP + EN)."""
    llava = getattr(engine, "_llava_client", None)
    if llava is None:
        print("\n[LLaVA] Not initialized")
        return
    try:
        t0 = time.monotonic()
        result = llava.describe_scene(frame, language="both")
        elapsed = (time.monotonic() - t0) * 1000
        mock = result.get("mock", False)
        label = "(mock)" if mock else "(real inference)"
        print(f"\n┌─ LLaVA-NeXT Scene Description {label} ───────────")
        print(f"│  [JP] {result.get('description_ja', '')[:120]}")
        print(f"│  [EN] {result.get('description_en', '')[:120]}")
        print(f"│  Latency : {elapsed:.1f} ms")
        stats = llava.get_stats()
        print(f"│  4-bit   : {stats.get('load_in_4bit', False)}")
        print(f"└────────────────────────────────────────────────┘")
    except Exception as e:
        print(f"\n[LLaVA] describe_scene failed: {e}")


# ── Cascade stats ─────────────────────────────────────────────────────────


def demo_cascade_stats(engine: object) -> None:
    """Print cascade detector stage statistics."""
    det = getattr(engine, "_detector", None)
    if det is None or not hasattr(det, "get_stats"):
        print("\n[Cascade] Detector not available or no stats method")
        return
    try:
        stats = det.get_stats()
        print(f"\n┌─ Cascade Detector Stats ───────────────────────┐")
        for k, v in stats.items():
            print(f"│  {k:<30} : {v}")
        print(f"└────────────────────────────────────────────────┘")
    except Exception as e:
        print(f"\n[Cascade] get_stats failed: {e}")


# ── Full frame inference ──────────────────────────────────────────────────


def run_inference(engine: object, frame: np.ndarray) -> None:
    """Run a single frame through the full perception pipeline."""
    process_fn = getattr(engine, "process_frame", None)
    if process_fn is None:
        print("\n[Pipeline] engine.process_frame not found")
        return
    try:
        t0 = time.monotonic()
        result = process_fn(frame, frame_number=1, camera_id="demo_cam")
        elapsed = (time.monotonic() - t0) * 1000
        n_det = getattr(result, "detections_count", "?")
        n_trk = getattr(result, "tracks_count", "?")
        n_vio = len(getattr(result, "violations", []))
        vlm_called = getattr(result, "vlm_called", False)
        print(f"\n┌─ Frame Inference Result ────────────────────────┐")
        print(f"│  Total latency : {elapsed:.1f} ms")
        print(f"│  Detections    : {n_det}")
        print(f"│  Tracks        : {n_trk}")
        print(f"│  Violations    : {n_vio}")
        print(f"│  VLM called    : {vlm_called}")
        if n_vio:
            for v in result.violations[:3]:
                print(f"│    [{v.severity.value.upper()}] {v.description_ja[:80]}")
        print(f"└────────────────────────────────────────────────┘")
    except Exception as e:
        logger.exception("Frame inference failed: %s", e)


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 58 Real Inference Stack Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock models (no real weights needed)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to input image/frame",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  SOPilot Phase 58 — Real Inference Stack Demo")
    print("  v6.1.0 | RTX 4090 | Camera OS")
    print("=" * 60)

    # Build config
    try:
        from sopilot.perception.types import PerceptionConfig

        if args.mock:
            config = PerceptionConfig(device="cpu")
            logger.info("Using mock config (CPU mode)")
        else:
            config = PerceptionConfig.for_rtx4090()
            logger.info("Using for_rtx4090() production config")

    except ImportError as e:
        logger.error("Cannot import sopilot: %s", e)
        return 1

    # Print config summary
    print(f"\n[Config] device={config.device}")
    print(f"[Config] orchestrator={config.orchestrator_enabled}, cascade={config.detector_backend}")
    print(f"[Config] dinov2={config.dinov2_enabled}, llava_4bit={config.llava_load_in_4bit}")
    print(f"[Config] sam={config.sam_enabled}, temporal={config.temporal_fusion_enabled}")
    print(f"[Config] vjepa={config.video_foundation_enabled} (backend={config.video_foundation_backend})")

    # Build engine
    try:
        from sopilot.perception.engine import build_perception_engine

        logger.info("Building PerceptionEngine ...")
        t0 = time.monotonic()
        engine = build_perception_engine(config)
        elapsed = (time.monotonic() - t0) * 1000
        logger.info("PerceptionEngine ready in %.0f ms", elapsed)
    except Exception as e:
        logger.exception("Failed to build PerceptionEngine: %s", e)
        return 1

    # Load frame
    frame = load_frame(args.image)

    # Run demos
    run_inference(engine, frame)
    demo_dinov2_features(engine, frame)
    demo_llava_description(engine, frame)
    demo_cascade_stats(engine)
    print_vram_report()

    print("\n✓ Phase 58 demo complete.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
