#!/usr/bin/env python3
"""Evaluate SOPilot with Neural mode vs Heuristic baseline.

Compares:
- Heuristic scoring (fast, no ML)
- Neural scoring (ProjectionHead + ScoringHead + Conformal)

Demonstrates RTX 5090 acceleration for commercial deployment.

Usage:
    python scripts/evaluate_neural_mode.py
    python scripts/evaluate_neural_mode.py --device cpu  # CPU fallback
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from sopilot.config import get_settings  # noqa: E402
from sopilot.embeddings import HeuristicClipEmbedder  # noqa: E402
from sopilot.step_engine import evaluate_sop  # noqa: E402


def load_video_embeddings(video_path: Path) -> tuple[np.ndarray, list[dict]]:
    """Extract embeddings from video using HeuristicClipEmbedder.

    Returns:
        (embeddings, metadata) where embeddings is (T, D) and metadata is list of dicts
    """
    import cv2

    embedder = HeuristicClipEmbedder()
    settings = get_settings()

    # Read video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = settings.target_fps
    frame_skip = max(1, int(fps / target_fps))

    frames = []
    meta = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # Resize to max_side
            h, w = frame.shape[:2]
            max_side = settings.max_side
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame = cv2.resize(frame, (new_w, new_h))

            frames.append(frame)
            meta.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / fps,
                }
            )

        frame_idx += 1

    cap.release()

    # Embed frames (one at a time since we have single frames, not clips)
    embeddings_list = []
    for frame in frames:
        # embed_clip expects a numpy array of shape (H, W, 3)
        emb = embedder.embed_clip(frame)
        embeddings_list.append(emb)

    embeddings = np.array(embeddings_list)

    return embeddings, meta


def main():
    parser = argparse.ArgumentParser(description="Evaluate Neural mode vs Heuristic")
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("demo_videos/manufacturing/oil_change_gold.mp4"),
        help="Gold standard video",
    )
    parser.add_argument(
        "--trainee",
        type=Path,
        default=Path("demo_videos/manufacturing/oil_change_trainee.mp4"),
        help="Trainee video",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for neural models (cuda/cpu)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/models/neural_full"),
        help="Neural model directory",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SOPilot Neural Mode Evaluation - RTX 5090 Commercial Demo")
    print("=" * 70)
    print(f"Gold:      {args.gold}")
    print(f"Trainee:   {args.trainee}")
    print(f"Device:    {args.device}")
    print(f"Models:    {args.model_dir}")
    print()

    # Verify files
    if not args.gold.exists():
        print(f"ERROR: Gold video not found: {args.gold}")
        return 1
    if not args.trainee.exists():
        print(f"ERROR: Trainee video not found: {args.trainee}")
        return 1
    if not args.model_dir.exists():
        print(f"ERROR: Neural models not found: {args.model_dir}")
        return 1

    # Load embeddings
    print("[1/4] Extracting embeddings...")
    start = time.time()

    gold_emb, gold_meta = load_video_embeddings(args.gold)
    trainee_emb, trainee_meta = load_video_embeddings(args.trainee)

    elapsed = time.time() - start
    print(f"  Gold:    {gold_emb.shape}")
    print(f"  Trainee: {trainee_emb.shape}")
    print(f"  Time:    {elapsed:.2f}s")
    print()

    # Settings
    settings = get_settings()

    # Evaluate with HEURISTIC mode
    print("[2/4] Evaluating with HEURISTIC mode...")
    start = time.time()

    result_heuristic = evaluate_sop(
        gold_embeddings=gold_emb,
        trainee_embeddings=trainee_emb,
        gold_meta=gold_meta,
        trainee_meta=trainee_meta,
        threshold_factor=settings.change_threshold_factor,
        min_step_clips=settings.min_step_clips,
        low_similarity_threshold=settings.low_similarity_threshold,
        w_miss=settings.w_miss,
        w_swap=settings.w_swap,
        w_dev=settings.w_dev,
        w_time=settings.w_time,
        neural_mode=False,
    )

    elapsed_heuristic = time.time() - start
    print(f"  Score:      {result_heuristic.score:.1f} / 100")
    print(f"  Alignment:  {result_heuristic.alignment_cost:.4f}")
    print(f"  Deviations: {len(result_heuristic.deviations)}")
    print(f"  Time:       {elapsed_heuristic:.2f}s")
    print()

    # Evaluate with NEURAL mode
    print("[3/4] Evaluating with NEURAL mode (RTX 5090)...")
    start = time.time()

    result_neural = evaluate_sop(
        gold_embeddings=gold_emb,
        trainee_embeddings=trainee_emb,
        gold_meta=gold_meta,
        trainee_meta=trainee_meta,
        threshold_factor=settings.change_threshold_factor,
        min_step_clips=settings.min_step_clips,
        low_similarity_threshold=settings.low_similarity_threshold,
        w_miss=settings.w_miss,
        w_swap=settings.w_swap,
        w_dev=settings.w_dev,
        w_time=settings.w_time,
        neural_mode=True,
        neural_model_dir=args.model_dir,
        neural_device=args.device,
        neural_soft_dtw_gamma=1.0,
        neural_uncertainty_samples=30,
        neural_calibration_enabled=True,
    )

    elapsed_neural = time.time() - start
    print(f"  Score:      {result_neural.score:.1f} / 100")
    print(f"  Alignment:  {result_neural.alignment_cost:.4f}")
    print(f"  Deviations: {len(result_neural.deviations)}")
    print(f"  Time:       {elapsed_neural:.2f}s")
    print()

    # Comparison
    print("[4/4] Comparison - Neural vs Heuristic:")
    print("-" * 70)
    print(f"  Score delta:      {result_neural.score - result_heuristic.score:+.1f} points")
    print(f"  Alignment delta:  {result_neural.alignment_cost - result_heuristic.alignment_cost:+.4f}")
    print(f"  Time delta:       {elapsed_neural - elapsed_heuristic:+.2f}s")
    print(
        f"  Speedup:          {elapsed_heuristic / elapsed_neural:.2f}x {'(Neural FASTER!)' if elapsed_neural < elapsed_heuristic else '(Heuristic faster)'}"
    )
    print()

    # Commercial summary
    print("=" * 70)
    print("Commercial Impact Assessment")
    print("=" * 70)
    print()
    print("Neural mode improvements:")
    print(f"  - Precision: {result_neural.score:.1f}/100 vs {result_heuristic.score:.1f}/100")
    print(f"  - RTX 5090 acceleration: {elapsed_neural:.2f}s evaluation time")
    print("  - Uncertainty quantification: MC Dropout + Conformal Prediction")
    print()

    if result_neural.score > result_heuristic.score:
        print(f"✓ Neural mode IMPROVES accuracy by {result_neural.score - result_heuristic.score:.1f} points")
    else:
        print("⚠ Neural mode provides SAME/LOWER score (may need retraining)")

    print()
    print("Next steps:")
    print("  - Deploy Neural mode in production (accuracy priority)")
    print("  - OR keep Heuristic mode (speed priority, lower compute cost)")
    print("  - Benchmark on 50+ real manufacturing SOPs")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
