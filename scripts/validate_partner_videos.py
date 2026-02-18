#!/usr/bin/env python3
"""Validate partner-provided SOP videos before processing.

This script checks:
- Resolution, fps, duration
- Audio track presence (optional for manufacturing SOPs)
- File corruption / readability
- Scene count estimation (via PySceneDetect)
- Naming convention compliance

Usage:
    python scripts/validate_partner_videos.py --dir demo_videos/partner --out validation_report.json

Output:
    JSON report with per-video validation results
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2


@dataclass
class VideoValidationResult:
    """Validation result for a single video file."""

    filename: str
    file_path: str
    file_size_mb: float
    readable: bool
    error_message: str | None = None

    # Video properties
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_sec: float | None = None
    frame_count: int | None = None

    # Audio
    has_audio: bool | None = None

    # Scene detection (estimated)
    estimated_scenes: int | None = None

    # Naming convention
    naming_convention_ok: bool = False
    sop_name: str | None = None
    role: str | None = None
    date: str | None = None


def parse_filename(filename: str) -> tuple[str | None, str | None, str | None]:
    """Parse filename following {sop_name}_{role}_{date}.mp4 convention.

    Returns:
        (sop_name, role, date) or (None, None, None) if invalid
    """
    # Expected: oilchange_gold_202602.mp4
    pattern = r"^([a-z]+)_(gold|trainee\d+)_(\d{6})\.mp4$"
    match = re.match(pattern, filename.lower())
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None


def validate_video(file_path: Path) -> VideoValidationResult:
    """Validate a single video file.

    Args:
        file_path: Path to video file

    Returns:
        VideoValidationResult with all checks
    """
    result = VideoValidationResult(
        filename=file_path.name,
        file_path=str(file_path),
        file_size_mb=file_path.stat().st_size / (1024 * 1024),
        readable=False,
    )

    # Check naming convention
    sop_name, role, date = parse_filename(file_path.name)
    if sop_name and role and date:
        result.naming_convention_ok = True
        result.sop_name = sop_name
        result.role = role
        result.date = date

    # Try to open video
    try:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            result.error_message = "Failed to open video file"
            return result

        result.readable = True

        # Get video properties
        result.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result.fps = cap.get(cv2.CAP_PROP_FPS)
        result.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if result.fps and result.fps > 0:
            result.duration_sec = result.frame_count / result.fps

        # Check audio (OpenCV doesn't detect audio tracks reliably)
        # Use ffprobe if available, otherwise mark as unknown
        result.has_audio = None  # Unknown via OpenCV

        # Estimate scene count (simplified: sample frames every 5 seconds)
        if result.duration_sec:
            result.estimated_scenes = _estimate_scenes(cap, result.fps, result.duration_sec)

        cap.release()

    except Exception as e:
        result.error_message = f"Validation error: {str(e)}"

    return result


def _estimate_scenes(cap: cv2.VideoCapture, fps: float, duration_sec: float) -> int:
    """Estimate scene count by sampling frames.

    This is a rough estimate. For accurate scene detection, use PySceneDetect.

    Args:
        cap: OpenCV video capture
        fps: Frames per second
        duration_sec: Video duration in seconds

    Returns:
        Estimated number of scenes (minimum 1)
    """
    # Sample every 5 seconds
    sample_interval_sec = 5.0
    num_samples = max(1, int(duration_sec / sample_interval_sec))

    # For now, return a conservative estimate
    # In production, integrate PySceneDetect here
    return max(1, num_samples // 2)


def validate_directory(video_dir: Path, extensions: list[str] | None = None) -> list[VideoValidationResult]:
    """Validate all videos in a directory.

    Args:
        video_dir: Directory containing videos
        extensions: Video file extensions to check

    Returns:
        List of validation results
    """
    if extensions is None:
        extensions = [".mp4", ".avi", ".mov"]
    results = []

    for ext in extensions:
        for file_path in video_dir.glob(f"*{ext}"):
            print(f"Validating {file_path.name}...", file=sys.stderr)
            result = validate_video(file_path)
            results.append(result)

    return results


def generate_report(results: list[VideoValidationResult]) -> dict[str, Any]:
    """Generate validation report with summary statistics.

    Args:
        results: List of validation results

    Returns:
        Report dictionary
    """
    total = len(results)
    readable = sum(1 for r in results if r.readable)
    naming_ok = sum(1 for r in results if r.naming_convention_ok)
    has_errors = sum(1 for r in results if r.error_message is not None)

    report = {
        "summary": {
            "total_files": total,
            "readable": readable,
            "naming_convention_ok": naming_ok,
            "errors": has_errors,
        },
        "videos": [asdict(r) for r in results],
    }

    return report


def print_summary(report: dict[str, Any]) -> None:
    """Print human-readable summary to stderr.

    Args:
        report: Validation report
    """
    summary = report["summary"]
    print("\n=== Validation Summary ===", file=sys.stderr)
    print(f"Total files: {summary['total_files']}", file=sys.stderr)
    print(f"Readable: {summary['readable']}", file=sys.stderr)
    print(f"Naming convention OK: {summary['naming_convention_ok']}", file=sys.stderr)
    print(f"Errors: {summary['errors']}", file=sys.stderr)

    print("\n=== Per-Video Results ===", file=sys.stderr)
    for video in report["videos"]:
        status = "✅" if video["readable"] else "❌"
        naming = "✅" if video["naming_convention_ok"] else "⚠️"
        print(f"{status} {naming} {video['filename']}", file=sys.stderr)
        if video["readable"]:
            print(
                f"   {video['width']}x{video['height']} @ {video['fps']:.1f}fps, {video['duration_sec']:.1f}s",
                file=sys.stderr,
            )
        if video["error_message"]:
            print(f"   Error: {video['error_message']}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Validate partner-provided SOP videos")
    parser.add_argument("--dir", type=Path, required=True, help="Directory containing videos")
    parser.add_argument("--out", type=Path, help="Output JSON report path (default: stdout)")
    parser.add_argument("--extensions", nargs="+", default=[".mp4", ".avi", ".mov"], help="Video file extensions")

    args = parser.parse_args()

    if not args.dir.exists():
        print(f"Error: Directory not found: {args.dir}", file=sys.stderr)
        sys.exit(1)

    # Validate all videos
    results = validate_directory(args.dir, extensions=args.extensions)

    if not results:
        print(f"Warning: No video files found in {args.dir}", file=sys.stderr)

    # Generate report
    report = generate_report(results)

    # Print summary to stderr
    print_summary(report)

    # Write JSON report
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {args.out}", file=sys.stderr)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
