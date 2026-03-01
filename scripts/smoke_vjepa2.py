"""V-JEPA2 GPU smoke test.

Quick end-to-end validation that the V-JEPA2 embedder loads correctly,
produces a unit-norm embedding of the expected shape, and reports
inference latency + VRAM usage.

Usage
-----
# Fastest: synthetic video (no real file needed)
python scripts/smoke_vjepa2.py

# With a real video file
python scripts/smoke_vjepa2.py --video path/to/clip.mp4

# Override variant / device
python scripts/smoke_vjepa2.py --variant vjepa2_vit_large --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def _ensure_repo_on_path() -> None:
    """Make the repository root importable when running this script directly."""
    root = Path(__file__).resolve().parent.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_synthetic_frames(
    n_frames: int = 16,
    height: int = 256,
    width: int = 256,
    seed: int = 42,
) -> list[np.ndarray]:
    """Return a list of (H, W, 3) uint8 frames with random content."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        for _ in range(n_frames)
    ]


def _make_frames_from_video(path: str, n_frames: int = 16) -> list[np.ndarray]:
    """Extract n_frames uniformly from a video file using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), n_frames, dtype=int)
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames extracted from {path}")
    return frames


def _vram_free_gb() -> float | None:
    """Return free VRAM in GB, or None if not on CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _ = torch.cuda.mem_get_info()
        return free / 1024 ** 3
    except Exception:
        return None


def _print_section(title: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="V-JEPA2 GPU smoke test")
    parser.add_argument("--video", default=None, help="Path to a real video file (optional)")
    parser.add_argument(
        "--variant",
        default=os.getenv("SOPILOT_VJEPA2_VARIANT", "vjepa2_vit_large"),
        help="torch.hub entry point (default: vjepa2_vit_large)",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("SOPILOT_VJEPA2_DEVICE", "auto"),
        help="Device string: 'auto', 'cuda', 'cuda:0', 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=int(os.getenv("SOPILOT_VJEPA2_CROP_SIZE", "256")),
        help="Preprocessor crop size (default: 256)",
    )
    parser.add_argument(
        "--pooling",
        default=os.getenv("SOPILOT_VJEPA2_POOLING", "mean_tokens"),
        choices=["mean_tokens", "first_token", "flatten"],
        help="Token pooling mode (default: mean_tokens)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=16,
        help="Number of frames to embed (default: 16)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warm-up passes before timing (default: 1)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed passes (default: 3)",
    )
    args = parser.parse_args()

    _ensure_repo_on_path()
    from sopilot.services.embedder import VJEPA2Config, VJEPA2HubEmbedder

    # ── configuration ────────────────────────────────────────────────────────
    _print_section("V-JEPA2 Smoke Test — Configuration")
    config = VJEPA2Config(
        variant=args.variant,
        pretrained=True,
        crop_size=args.crop_size,
        device=args.device,
        use_amp=True,
        pooling=args.pooling,
    )
    print(f"  variant    : {config.variant}")
    print(f"  device     : {config.device}")
    print(f"  crop_size  : {config.crop_size}")
    print(f"  pooling    : {config.pooling}")
    print(f"  use_amp    : {config.use_amp}")
    print(f"  n_frames   : {args.frames}")

    # ── build frames ─────────────────────────────────────────────────────────
    _print_section("Frame Source")
    if args.video:
        print(f"  source     : {args.video}")
        frames = _make_frames_from_video(args.video, n_frames=args.frames)
    else:
        print("  source     : synthetic (random noise)")
        frames = _make_synthetic_frames(n_frames=args.frames)
    print(f"  shape      : {len(frames)} × {frames[0].shape}")

    # ── model load ───────────────────────────────────────────────────────────
    _print_section("Model Load")
    vram_before_load = _vram_free_gb()
    embedder = VJEPA2HubEmbedder(config)
    t0_load = time.perf_counter()
    embedder._ensure_ready()
    t_load = time.perf_counter() - t0_load
    vram_after_load = _vram_free_gb()
    print(f"  load time  : {t_load:.1f}s")
    if vram_before_load is not None and vram_after_load is not None:
        used = vram_before_load - vram_after_load
        print(f"  VRAM used  : {used:.2f} GB  (free before={vram_before_load:.2f} GB, after={vram_after_load:.2f} GB)")
    else:
        print("  VRAM       : N/A (not on CUDA)")

    # ── warm-up ──────────────────────────────────────────────────────────────
    if args.warmup > 0:
        _print_section("Warm-up")
        for i in range(args.warmup):
            _ = embedder.embed(frames)
            print(f"  warm-up {i + 1}/{args.warmup} done")

    # ── timed inference ──────────────────────────────────────────────────────
    _print_section("Timed Inference")
    latencies: list[float] = []
    for i in range(args.repeats):
        t0 = time.perf_counter()
        embedding = embedder.embed(frames)
        latencies.append(time.perf_counter() - t0)
        print(f"  pass {i + 1}/{args.repeats}: {latencies[-1] * 1000:.1f} ms")

    # ── results ──────────────────────────────────────────────────────────────
    _print_section("Results")
    arr = np.array(latencies)
    print(f"  latency    : mean={arr.mean() * 1000:.1f} ms  min={arr.min() * 1000:.1f} ms  max={arr.max() * 1000:.1f} ms")
    print(f"  emb shape  : {embedding.shape}")
    print(f"  emb dtype  : {embedding.dtype}")
    l2 = float(np.linalg.norm(embedding))
    print(f"  L2 norm    : {l2:.6f}  (should be ≈1.000 — unit-normalized)")

    # ── assertions ───────────────────────────────────────────────────────────
    _print_section("Assertions")
    ok = True

    if embedding.ndim != 1:
        print(f"  FAIL  embedding must be 1-D, got shape {embedding.shape}")
        ok = False
    else:
        print(f"  PASS  embedding is 1-D, shape={embedding.shape}")

    if embedding.dtype != np.float32:
        print(f"  FAIL  embedding dtype must be float32, got {embedding.dtype}")
        ok = False
    else:
        print("  PASS  dtype=float32")

    if abs(l2 - 1.0) > 1e-3:
        print(f"  FAIL  L2 norm {l2:.6f} deviates from 1.0 by > 0.001 (not unit-normalized)")
        ok = False
    else:
        print(f"  PASS  unit-normalized (|L2 - 1| = {abs(l2 - 1.0):.2e})")

    vram_final = _vram_free_gb()
    if vram_after_load is not None and vram_final is not None:
        leaked = vram_after_load - vram_final
        if leaked > 0.1:
            print(f"  WARN  potential VRAM leak: {leaked:.3f} GB accumulated across {args.repeats} passes")
        else:
            print(f"  PASS  VRAM stable (delta={leaked:.3f} GB after {args.repeats} passes)")

    print()
    if ok:
        print("✓  All checks passed.")
    else:
        print("✗  Some checks failed — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
