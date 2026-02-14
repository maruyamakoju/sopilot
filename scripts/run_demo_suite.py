#!/usr/bin/env python
"""Run all demo scripts in sequence.

Generates the complete demo suite (12 figures) with a single command.
Output saved to demo_outputs/.

Usage:
    python scripts/run_demo_suite.py
    python scripts/run_demo_suite.py --quick  # Fast mode (0.1x epochs for convergence)
    python scripts/run_demo_suite.py --device cuda  # Use GPU
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo_suite")

_REPO_ROOT = Path(__file__).resolve().parent.parent


def run_demo(script: str, args: list[str]) -> bool:
    """Run a demo script and return success status."""
    script_path = _REPO_ROOT / "scripts" / script
    if not script_path.exists():
        logger.error("Script not found: %s", script_path)
        return False

    cmd = [sys.executable, str(script_path)] + args
    logger.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error("Script failed with code %d: %s", e.returncode, script)
        return False
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete SOPilot demo suite (12 figures)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/run_demo_suite.py
  python scripts/run_demo_suite.py --quick --device cpu
  python scripts/run_demo_suite.py --device cuda --convergence-epochs 1.0
""",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="demo_outputs",
        help="Output directory for all figures (default: demo_outputs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device for convergence demo (default: auto)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: use 0.1x epochs for convergence (faster, less accurate)",
    )
    parser.add_argument(
        "--convergence-epochs",
        type=float,
        default=None,
        help="Epochs multiplier for convergence demo (default: 1.0, or 0.1 if --quick)",
    )
    parser.add_argument(
        "--skip-convergence",
        action="store_true",
        help="Skip training convergence demo (saves time, but loses proof)",
    )

    args = parser.parse_args()

    out_dir = args.out_dir
    device = args.device
    epochs_mult = args.convergence_epochs
    if epochs_mult is None:
        epochs_mult = 0.1 if args.quick else 1.0

    logger.info("=" * 70)
    logger.info("SOPilot Demo Suite Runner")
    logger.info("=" * 70)
    logger.info("Output directory: %s", out_dir)
    logger.info("Device: %s", device)
    logger.info("Convergence epochs multiplier: %.2f", epochs_mult)
    logger.info("=" * 70)

    demos = []
    t_start = time.perf_counter()

    # Demo 1: Neural pipeline visualization (6 figures, ~10s)
    logger.info("\n[1/4] Neural Pipeline Visualization (6 figures)...")
    success = run_demo("demo_neural_pipeline.py", [f"--out-dir={out_dir}"])
    demos.append(("Neural Pipeline (6 fig)", success, time.perf_counter() - t_start))

    # Demo 2: Ablation study (5 figures + JSON, ~30s)
    logger.info("\n[2/4] Ablation Study (5 experiments + JSON)...")
    t_abl = time.perf_counter()
    success = run_demo("demo_ablation_study.py", [f"--out-dir={out_dir}"])
    demos.append(("Ablation Study (5 fig)", success, time.perf_counter() - t_abl))

    # Demo 3: End-to-end pipeline (10-panel figure, ~5s)
    logger.info("\n[3/4] End-to-End Pipeline (10-panel figure)...")
    t_e2e = time.perf_counter()
    success = run_demo("demo_e2e_pipeline.py", [f"--out-dir={out_dir}"])
    demos.append(("E2E Pipeline (10-panel)", success, time.perf_counter() - t_e2e))

    # Demo 4: Training convergence (8-panel figure + JSON, ~5-30min depending on epochs)
    if not args.skip_convergence:
        logger.info("\n[4/4] Training Convergence (8-panel + JSON, epochs=%.2f)...", epochs_mult)
        logger.info("  This may take 5-30 minutes depending on epochs and device...")
        t_conv = time.perf_counter()
        conv_args = [
            f"--out-dir={out_dir}",
            f"--device={device}",
            f"--epochs-multiplier={epochs_mult}",
        ]
        success = run_demo("demo_training_convergence.py", conv_args)
        demos.append(("Training Convergence (8-panel)", success, time.perf_counter() - t_conv))
    else:
        logger.info("\n[4/4] Training Convergence skipped (--skip-convergence)")
        demos.append(("Training Convergence", None, 0))

    total_time = time.perf_counter() - t_start

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Demo Suite Summary")
    logger.info("=" * 70)
    for name, success, duration in demos:
        if success is None:
            status = "SKIPPED"
        elif success:
            status = "SUCCESS"
        else:
            status = "FAILED"
        logger.info("  %-35s %s  (%.1fs)", name, status, duration)
    logger.info("-" * 70)
    logger.info("  Total time: %.1fs", total_time)
    logger.info("=" * 70)

    # Check if all succeeded
    failed = [name for name, success, _ in demos if success is False]
    if failed:
        logger.error("Failed demos: %s", ", ".join(failed))
        sys.exit(1)

    logger.info("\n✅ All demos completed successfully!")
    logger.info("Output: %s/", out_dir)
    logger.info("\nKey figures:")
    logger.info("  - training_convergence.png  (proof: 1.7 → 81.5)")
    logger.info("  - ablation_01_alignment.png (Soft-DTW 43000× discrimination)")
    logger.info("  - e2e_pipeline.png          (10-panel architecture)")


if __name__ == "__main__":
    main()
