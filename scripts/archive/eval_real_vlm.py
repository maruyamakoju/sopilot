#!/usr/bin/env python3
"""Run real VLM evaluation on dashcam videos with clip extraction.

Convenience wrapper around real_data_benchmark.py for GPU evaluation.

Usage:
    # JP dashcam (20 videos)
    python scripts/eval_real_vlm.py --source jp --max-videos 5

    # Nexar collision dataset (10 videos)
    python scripts/eval_real_vlm.py --source nexar --max-videos 10

    # Custom directory
    python scripts/eval_real_vlm.py --input path/to/videos --max-videos 3

Requirements:
    - GPU with >= 16GB VRAM (RTX 3090/4090/5090)
    - transformers >= 4.51.3
    - qwen-vl-utils >= 0.0.10
    - pip install -e ".[insurance]"
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# Predefined sources
SOURCES = {
    "jp": {
        "input": "data/jp_dashcam",
        "output": "reports/jp_dashcam_real_vlm.json",
        "desc": "Japanese dashcam compilations (yt-dlp)",
    },
    "nexar": {
        "input": "data/real_dashcam/nexar",
        "output": "reports/nexar_real_vlm.json",
        "desc": "Nexar collision prediction dataset",
    },
    "demo": {
        "input": "data/dashcam_demo",
        "output": "reports/demo_real_vlm.json",
        "desc": "Synthetic demo videos (10 scenarios)",
    },
}


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            return True
        else:
            print("WARNING: No CUDA GPU detected. Real VLM requires GPU.")
            return False
    except ImportError:
        print("WARNING: PyTorch not installed. Cannot check GPU.")
        return False


def check_vlm_deps():
    """Check VLM dependencies."""
    missing = []
    try:
        import transformers
        if hasattr(transformers, '__version__'):
            ver = transformers.__version__
            print(f"transformers: {ver}")
    except ImportError:
        missing.append("transformers>=4.51.3")

    try:
        import qwen_vl_utils  # noqa: F401
        print("qwen-vl-utils: OK")
    except ImportError:
        missing.append("qwen-vl-utils>=0.0.10")

    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print(f"Install: pip install {' '.join(missing)}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Real VLM evaluation on dashcam videos")
    parser.add_argument("--source", choices=list(SOURCES.keys()), help="Predefined video source")
    parser.add_argument("--input", type=str, help="Custom input directory")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--max-videos", type=int, default=5, help="Max videos to evaluate (default: 5)")
    parser.add_argument("--extract-clips", action="store_true", default=True, help="Extract danger clips")
    parser.add_argument("--backend", choices=["mock", "real"], default="real", help="VLM backend")
    parser.add_argument("--skip-checks", action="store_true", help="Skip GPU/dependency checks")
    args = parser.parse_args()

    # Resolve source
    if args.source:
        src = SOURCES[args.source]
        input_dir = args.input or src["input"]
        output_path = args.output or src["output"]
        print(f"Source: {args.source} — {src['desc']}")
    elif args.input:
        input_dir = args.input
        output_path = args.output or f"reports/{Path(input_dir).name}_real_vlm.json"
    else:
        parser.error("Specify --source or --input")
        return

    print(f"Input:  {input_dir}")
    print(f"Output: {output_path}")
    print(f"Max:    {args.max_videos} videos")
    print(f"Backend: {args.backend}")
    print()

    # Pre-flight checks
    if not args.skip_checks and args.backend == "real":
        print("=== Pre-flight Checks ===")
        gpu_ok = check_gpu()
        deps_ok = check_vlm_deps()
        print()

        if not gpu_ok or not deps_ok:
            print("Pre-flight checks failed. Use --skip-checks to bypass.")
            print("Or use --backend mock for testing without GPU.")
            sys.exit(1)

    # Build command
    cmd = [
        sys.executable,
        "scripts/real_data_benchmark.py",
        "--input", input_dir,
        "--output", output_path,
        "--backend", args.backend,
        "--max-videos", str(args.max_videos),
    ]
    if args.extract_clips:
        cmd.append("--extract-clips")

    print(f"=== Running Benchmark ===")
    print(f"Command: {' '.join(cmd)}")
    print()

    start = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Results: {output_path}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
