#!/usr/bin/env python
"""Demo script for multimodal hazard mining pipeline

Usage:
    python -m insurance_mvp.scripts.demo_mining_pipeline --video-path data/dashcam001.mp4

Features:
- Full multimodal analysis (audio + motion + proximity)
- Top-K hazard clip extraction
- Signal visualization (optional)
- JSON report output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from insurance_mvp.mining import (
    AudioAnalyzer,
    AudioConfig,
    FusionConfig,
    MotionAnalyzer,
    MotionConfig,
    ProximityAnalyzer,
    ProximityConfig,
    SignalFuser,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Multimodal Hazard Mining Pipeline Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--video-path",
        type=Path,
        required=True,
        help="Path to dashcam video file",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for results",
    )

    parser.add_argument(
        "--audio-weight",
        type=float,
        default=0.3,
        help="Audio signal weight",
    )

    parser.add_argument(
        "--motion-weight",
        type=float,
        default=0.4,
        help="Motion signal weight",
    )

    parser.add_argument(
        "--proximity-weight",
        type=float,
        default=0.3,
        help="Proximity signal weight",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top hazard clips to extract",
    )

    parser.add_argument(
        "--min-peak-score",
        type=float,
        default=0.3,
        help="Minimum peak score threshold",
    )

    parser.add_argument(
        "--skip-proximity",
        action="store_true",
        help="Skip proximity analysis (faster, no YOLOv8)",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate signal visualization plots",
    )

    parser.add_argument(
        "--extract-clips",
        action="store_true",
        help="Extract hazard clips as separate video files",
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="Frame skip for motion/proximity (higher = faster)",
    )

    return parser.parse_args()


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps > 0:
        return frame_count / fps
    else:
        return 0.0


def extract_clip(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path,
) -> bool:
    """Extract video clip to separate file"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate frame range
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        # Create writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        return True

    except Exception as e:
        logger.error(f"Failed to extract clip: {e}")
        return False


def main():
    """Main entry point"""
    args = parse_args()

    # Validate inputs
    if not args.video_path.exists():
        logger.error(f"Video file not found: {args.video_path}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Multimodal Hazard Mining Pipeline")
    logger.info("=" * 80)
    logger.info(f"Video: {args.video_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("")

    # Get video duration
    video_duration = get_video_duration(args.video_path)
    logger.info(f"Video duration: {video_duration:.1f} seconds")
    logger.info("")

    # Initialize analyzers
    audio_config = AudioConfig()
    motion_config = MotionConfig(frame_skip=args.frame_skip)
    proximity_config = ProximityConfig(frame_skip=args.frame_skip)
    fusion_config = FusionConfig(
        audio_weight=args.audio_weight,
        motion_weight=args.motion_weight,
        proximity_weight=args.proximity_weight,
        top_k_peaks=args.top_k,
        min_peak_score=args.min_peak_score,
    )

    audio_analyzer = AudioAnalyzer(audio_config)
    motion_analyzer = MotionAnalyzer(motion_config)
    proximity_analyzer = ProximityAnalyzer(proximity_config) if not args.skip_proximity else None
    fuser = SignalFuser(fusion_config)

    # Step 1: Audio analysis
    logger.info("Step 1/4: Analyzing audio signals...")
    t0 = time.time()
    audio_scores = audio_analyzer.analyze(args.video_path)
    t1 = time.time()
    logger.info(f"  ✓ Audio analysis complete ({t1 - t0:.1f}s)")
    logger.info(f"    Mean: {audio_scores.mean():.3f}, Max: {audio_scores.max():.3f}")
    logger.info("")

    # Step 2: Motion analysis
    logger.info("Step 2/4: Analyzing motion (optical flow)...")
    t0 = time.time()
    motion_scores = motion_analyzer.analyze(args.video_path)
    t1 = time.time()
    logger.info(f"  ✓ Motion analysis complete ({t1 - t0:.1f}s)")
    logger.info(f"    Mean: {motion_scores.mean():.3f}, Max: {motion_scores.max():.3f}")
    logger.info("")

    # Step 3: Proximity analysis (optional)
    if proximity_analyzer is not None:
        logger.info("Step 3/4: Analyzing proximity (YOLOv8 object detection)...")
        t0 = time.time()
        proximity_scores = proximity_analyzer.analyze(args.video_path)
        t1 = time.time()
        logger.info(f"  ✓ Proximity analysis complete ({t1 - t0:.1f}s)")
        logger.info(f"    Mean: {proximity_scores.mean():.3f}, Max: {proximity_scores.max():.3f}")
    else:
        logger.info("Step 3/4: Skipping proximity analysis (--skip-proximity)")
        proximity_scores = np.zeros_like(motion_scores)

    logger.info("")

    # Step 4: Signal fusion and clip extraction
    logger.info("Step 4/4: Fusing signals and extracting hazard clips...")
    t0 = time.time()
    clips = fuser.fuse_and_extract(
        audio_scores,
        motion_scores,
        proximity_scores,
        video_duration_sec=video_duration,
    )
    t1 = time.time()
    logger.info(f"  ✓ Fusion complete ({t1 - t0:.1f}s)")
    logger.info(f"    Extracted {len(clips)} hazard clips")
    logger.info("")

    # Save results
    logger.info("Saving results...")

    # Save signal arrays
    np.save(args.output_dir / "audio_scores.npy", audio_scores)
    np.save(args.output_dir / "motion_scores.npy", motion_scores)
    np.save(args.output_dir / "proximity_scores.npy", proximity_scores)
    logger.info(f"  ✓ Saved signal arrays to {args.output_dir}")

    # Save clips JSON
    clips_json = {
        "video_path": str(args.video_path),
        "video_duration_sec": video_duration,
        "n_clips": len(clips),
        "config": {
            "audio_weight": fusion_config.audio_weight,
            "motion_weight": fusion_config.motion_weight,
            "proximity_weight": fusion_config.proximity_weight,
            "top_k_peaks": fusion_config.top_k_peaks,
            "min_peak_score": fusion_config.min_peak_score,
        },
        "clips": [
            {
                "clip_id": i + 1,
                "start_sec": float(clip.start_sec),
                "end_sec": float(clip.end_sec),
                "peak_sec": float(clip.peak_sec),
                "duration_sec": float(clip.duration_sec),
                "score": float(clip.score),
                "breakdown": {
                    "audio": float(clip.audio_score),
                    "motion": float(clip.motion_score),
                    "proximity": float(clip.proximity_score),
                },
            }
            for i, clip in enumerate(clips)
        ],
    }

    clips_json_path = args.output_dir / "hazard_clips.json"
    with open(clips_json_path, "w") as f:
        json.dump(clips_json, f, indent=2)

    logger.info(f"  ✓ Saved clip metadata to {clips_json_path}")

    # Generate visualization
    if args.visualize:
        logger.info("Generating signal visualization...")
        viz_path = args.output_dir / "signal_fusion_viz.png"
        fuser.visualize_signals(
            audio_scores,
            motion_scores,
            proximity_scores,
            clips,
            output_path=viz_path,
        )
        logger.info(f"  ✓ Saved visualization to {viz_path}")

    # Extract clips as separate videos
    if args.extract_clips and len(clips) > 0:
        logger.info("Extracting hazard clips as video files...")
        clips_dir = args.output_dir / "clips"
        clips_dir.mkdir(exist_ok=True)

        for i, clip in enumerate(clips[:10]):  # Extract top 10
            clip_path = clips_dir / f"clip_{i + 1:02d}_score_{clip.score:.3f}.mp4"
            success = extract_clip(
                args.video_path,
                clip.start_sec,
                clip.end_sec,
                clip_path,
            )
            if success:
                logger.info(f"  ✓ Extracted clip {i + 1}: {clip_path.name}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Video: {args.video_path.name} ({video_duration:.1f}s)")
    logger.info(f"Hazard clips: {len(clips)}")

    if len(clips) > 0:
        logger.info("")
        logger.info("Top 5 hazard clips:")
        for i, clip in enumerate(clips[:5]):
            logger.info(
                f"  {i + 1}. [{clip.start_sec:6.1f}s - {clip.end_sec:6.1f}s] "
                f"score={clip.score:.3f} "
                f"(audio={clip.audio_score:.2f}, motion={clip.motion_score:.2f}, "
                f"proximity={clip.proximity_score:.2f})"
            )

    logger.info("")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
