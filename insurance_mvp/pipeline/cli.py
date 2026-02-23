"""CLI entry point for the Insurance MVP pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from insurance_mvp.config import load_config
from insurance_mvp.pipeline.orchestrator import InsurancePipeline


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Insurance MVP - End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video
  python -m insurance_mvp.pipeline --video-path data/dashcam001.mp4 --output-dir results/

  # Batch processing
  python -m insurance_mvp.pipeline --video-dir data/dashcam/ --parallel 4

  # With config file
  python -m insurance_mvp.pipeline --config config.yaml --video-path data/dashcam001.mp4

  # Override config via CLI
  python -m insurance_mvp.pipeline --video-path data/dashcam001.mp4 --cosmos-backend mock
""",
    )

    parser.add_argument("--video-path", type=str, help="Path to single video file")
    parser.add_argument("--video-dir", type=str, help="Directory containing video files")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    parser.add_argument("--parallel", type=int, help="Number of parallel workers")
    parser.add_argument("--cosmos-backend", choices=["qwen2.5-vl-7b", "mock"], help="Cosmos backend")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--no-conformal", action="store_true", help="Disable conformal prediction")
    parser.add_argument("--no-transcription", action="store_true", help="Disable transcription")

    args = parser.parse_args()

    if not args.video_path and not args.video_dir:
        parser.error("Either --video-path or --video-dir must be specified")

    override_dict: dict = {"output_dir": args.output_dir}
    if args.parallel:
        override_dict["parallel_workers"] = args.parallel
    if args.cosmos_backend:
        override_dict["cosmos"] = {"backend": args.cosmos_backend}
    if args.log_level:
        override_dict["log_level"] = args.log_level
    if args.no_conformal:
        override_dict["enable_conformal"] = False
    if args.no_transcription:
        override_dict["enable_transcription"] = False

    config = load_config(yaml_path=args.config, override_dict=override_dict)
    pipeline = InsurancePipeline(config)

    if args.video_path:
        video_paths = [args.video_path]
    else:
        video_dir = Path(args.video_dir)
        video_paths = [str(p) for p in list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))]

    if not video_paths:
        print("No video files found!")
        return

    results = pipeline.process_batch(video_paths)

    failed_count = sum(1 for r in results if not r.success)
    if failed_count > 0:
        print(f"\nWarning: {failed_count} videos failed to process")
        exit(1)


if __name__ == "__main__":
    main()
