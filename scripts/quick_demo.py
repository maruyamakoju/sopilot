#!/usr/bin/env python
"""SOPilot Quick Demo — 30-second proof it works.

This is the fastest way to verify SOPilot is installed correctly.
No external dependencies, no heavy models, just core functionality.

Demonstrates:
1. Neural scoring pipeline (heuristic mode)
2. Soft-DTW alignment
3. Metric computation
4. Result visualization

Usage:
    python scripts/quick_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np


def make_demo_video(path: Path, variant: str) -> None:
    """Create a tiny synthetic video (24 frames, 2 seconds)."""
    width, height, fps = 80, 60, 12
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for i in range(24):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if variant == "gold":
            # Perfect execution: rect → circle
            if i < 12:
                cv2.rectangle(frame, (10, 10), (40, 40), (0, 255, 0), -1)
            else:
                cv2.circle(frame, (40, 30), 15, (255, 0, 0), -1)
        else:
            # Trainee: swapped order → circle first, then rect
            if i < 12:
                cv2.circle(frame, (40, 30), 15, (255, 0, 0), -1)
            else:
                cv2.rectangle(frame, (10, 10), (40, 40), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()


def main():
    print("=" * 60)
    print("SOPilot Quick Demo — 30-Second Proof")
    print("=" * 60)
    print()

    # Step 1: Create synthetic videos
    print("[1/4] Creating synthetic videos...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        gold_path = tmp / "gold.mp4"
        trainee_path = tmp / "trainee.mp4"

        make_demo_video(gold_path, "gold")
        make_demo_video(trainee_path, "trainee")
        print(f"  ✓ Gold: {gold_path.name} (24 frames)")
        print(f"  ✓ Trainee: {trainee_path.name} (24 frames, swapped steps)")
        print()

        # Step 2: Extract embeddings (heuristic mode = fast, no torch)
        print("[2/4] Extracting embeddings (heuristic mode)...")
        from sopilot.embeddings import build_embedder

        embedder = build_embedder("heuristic")
        gold_emb = embedder.extract_from_path(gold_path)
        trainee_emb = embedder.extract_from_path(trainee_path)
        print(f"  ✓ Gold embeddings: {gold_emb.shape}")
        print(f"  ✓ Trainee embeddings: {trainee_emb.shape}")
        print()

        # Step 3: Compute alignment + metrics
        print("[3/4] Computing Soft-DTW alignment + metrics...")
        from sopilot.step_engine import evaluate_sop

        result = evaluate_sop(
            gold_emb=gold_emb,
            trainee_emb=trainee_emb,
            gold_steps=None,  # Auto-detect
            trainee_steps=None,
            gold_path=str(gold_path),
            trainee_path=str(trainee_path),
            neural_mode=False,  # Heuristic scoring (no ML models)
        )
        print(f"  ✓ Alignment cost: {result.alignment_cost:.3f}")
        print(f"  ✓ Score: {result.score:.1f} / 100")
        print(f"  ✓ Deviations detected: {len(result.deviations)}")
        print()

        # Step 4: Show results
        print("[4/4] Results:")
        print("-" * 60)
        print(f"  Final Score:     {result.score:.1f} / 100")
        print(f"  Alignment Cost:  {result.alignment_cost:.3f}")
        print(f"  Deviations:      {len(result.deviations)}")

        if result.deviations:
            print("\n  Detected issues:")
            for dev in result.deviations[:3]:  # Show first 3
                print(f"    - {dev.deviation_type}: {dev.description}")

        print()
        print("=" * 60)
        print("✅ Success! SOPilot is working correctly.")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  • Run full test suite:  python -m pytest tests/ -v")
        print("  • Try demo scripts:     python scripts/run_demo_suite.py --quick")
        print("  • Launch API:           uvicorn sopilot.main:app --reload")
        print()


if __name__ == "__main__":
    main()
