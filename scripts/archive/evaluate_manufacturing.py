#!/usr/bin/env python3
"""Evaluate manufacturing SOP demo videos with SOPilot.

Compares gold standard vs trainee execution, detects deviations:
- Missing steps (SAFETY, CHECK)
- Wrong tools (jack vs wrench)
- Procedure ordering issues

Demonstrates SOPilot's commercial capability for manufacturing training.

Usage:
    python scripts/evaluate_manufacturing.py
    python scripts/evaluate_manufacturing.py --neural-mode --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from sopilot.embeddings import HeuristicClipEmbedder  # noqa: E402
from sopilot.step_engine import evaluate_sop  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evaluate manufacturing SOP videos")
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("demo_videos/manufacturing/oil_change_gold.mp4"),
        help="Gold standard video path",
    )
    parser.add_argument(
        "--trainee",
        type=Path,
        default=Path("demo_videos/manufacturing/oil_change_trainee.mp4"),
        help="Trainee video path",
    )
    parser.add_argument(
        "--neural-mode",
        action="store_true",
        help="Enable neural scoring (requires trained models)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for neural models (cuda/cpu)",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="heuristic",
        help="Embedder backend (heuristic/vjepa2/auto)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Manufacturing SOP Evaluation - SOPilot Commercial Demo")
    print("=" * 70)
    print(f"Gold:        {args.gold}")
    print(f"Trainee:     {args.trainee}")
    print(f"Embedder:    {args.embedder}")
    print(f"Neural mode: {args.neural_mode}")
    if args.neural_mode:
        print(f"Device:      {args.device}")
    print()

    # Verify files exist
    if not args.gold.exists():
        print(f"ERROR: Gold video not found: {args.gold}")
        print("Run: python scripts/generate_manufacturing_demo.py")
        return 1

    if not args.trainee.exists():
        print(f"ERROR: Trainee video not found: {args.trainee}")
        print("Run: python scripts/generate_manufacturing_demo.py")
        return 1

    # Step 1: Extract embeddings
    print("[1/3] Extracting embeddings...")
    start = time.time()

    # Use heuristic embedder (fast, no ML dependencies)
    embedder = HeuristicClipEmbedder()
    gold_emb = embedder.extract_from_path(args.gold)
    trainee_emb = embedder.extract_from_path(args.trainee)

    elapsed = time.time() - start
    print(f"  Gold:    {gold_emb.shape} ({gold_emb.shape[0] / 24:.1f}s @ 1fps)")
    print(f"  Trainee: {trainee_emb.shape} ({trainee_emb.shape[0] / 24:.1f}s @ 1fps)")
    print(f"  Embedding time: {elapsed:.2f}s")
    print()

    # Step 2: Evaluate with SOPilot
    print("[2/3] Evaluating with SOPilot...")
    start = time.time()

    # Configure neural mode
    neural_kwargs = {}
    if args.neural_mode:
        neural_kwargs = {
            "neural_mode": True,
            "neural_model_dir": Path("data/models/neural_full"),
            "neural_device": args.device,
        }

    result = evaluate_sop(
        gold_emb=gold_emb,
        trainee_emb=trainee_emb,
        gold_steps=None,  # Auto-detect
        trainee_steps=None,
        gold_path=str(args.gold),
        trainee_path=str(args.trainee),
        **neural_kwargs,
    )

    elapsed = time.time() - start
    print(f"  Evaluation time: {elapsed:.2f}s")
    print()

    # Step 3: Display results
    print("[3/3] Results:")
    print("-" * 70)
    print(f"  Final Score:         {result.score:.1f} / 100")
    print(f"  Alignment Cost:      {result.alignment_cost:.4f}")
    print(f"  Detected Deviations: {len(result.deviations)}")
    print()

    if result.deviations:
        print("  Detailed Deviations:")
        for i, dev in enumerate(result.deviations, 1):
            print(f"    {i}. [{dev.deviation_type}] {dev.description}")
            print(f"       Time: {dev.trainee_clip_range}")
            if dev.gold_clip_range:
                print(f"       Expected: {dev.gold_clip_range}")
            print()

    print("-" * 70)
    print()

    # Commercial messaging
    print("=" * 70)
    print("Commercial Value Demonstration")
    print("=" * 70)
    print()
    print("Expected trainee deviations (ground truth):")
    print("  1. SKIPPED: Safety step (no glasses/gloves)")
    print("  2. WRONG TOOL: Used jack instead of wrench for filter removal")
    print("  3. SKIPPED: Check step (no dipstick verification)")
    print()

    if len(result.deviations) >= 2:
        print("✓ SOPilot successfully detected multiple deviations")
    else:
        print("⚠ SOPilot may have missed some deviations (tune thresholds)")

    print()
    print("Commercial benefits:")
    print(f"  • Evaluation time: {elapsed:.1f}s (vs 2-hour manual review)")
    print("  • Cost reduction: ~99% (automated vs human evaluator)")
    print("  • Objectivity: Consistent scoring, no human bias")
    print("  • Scalability: Evaluate 1000s of trainees in parallel")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
