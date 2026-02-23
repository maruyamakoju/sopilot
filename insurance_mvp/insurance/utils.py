"""Utility Functions for Insurance Module.

Helper functions for video processing, metadata extraction, and evidence handling.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

from .schema import Evidence

logger = logging.getLogger(__name__)


class VideoMetadata:
    """Video file metadata."""

    def __init__(
        self,
        duration_sec: float,
        fps: float,
        width: int,
        height: int,
        num_frames: int,
        file_size_mb: float,
        codec: str = "unknown",
    ):
        """Initialize video metadata.

        Args:
            duration_sec: Total video duration in seconds.
            fps: Frames per second.
            width: Video width in pixels.
            height: Video height in pixels.
            num_frames: Total number of frames.
            file_size_mb: File size in megabytes.
            codec: Video codec (e.g., 'h264', 'hevc').
        """
        self.duration_sec = duration_sec
        self.fps = fps
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.file_size_mb = file_size_mb
        self.codec = codec

    def __repr__(self) -> str:
        return (
            f"VideoMetadata(duration={self.duration_sec:.1f}s, "
            f"fps={self.fps:.1f}, "
            f"resolution={self.width}x{self.height}, "
            f"frames={self.num_frames}, "
            f"size={self.file_size_mb:.1f}MB)"
        )


def extract_video_metadata(video_path: str) -> VideoMetadata:
    """Extract metadata from video file using OpenCV.

    Args:
        video_path: Path to video file.

    Returns:
        VideoMetadata object with video properties.

    Raises:
        ValueError: If video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    try:
        # Extract properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

        # Calculate duration
        duration_sec = num_frames / fps if fps > 0 else 0.0

        # Get file size
        file_path = Path(video_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0.0

        # Decode fourcc to codec string
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).strip()

        metadata = VideoMetadata(
            duration_sec=duration_sec,
            fps=fps,
            width=width,
            height=height,
            num_frames=num_frames,
            file_size_mb=file_size_mb,
            codec=codec,
        )

        logger.debug(
            "Video metadata: %s duration=%.2fs fps=%.2f res=%dx%d frames=%d",
            video_path, duration_sec, fps, width, height, num_frames,
        )

        return metadata

    finally:
        cap.release()


def extract_keyframes(
    video_path: str,
    timestamps_sec: list[float],
    output_dir: str | None = None,
) -> list[Evidence]:
    """Extract keyframes at specified timestamps as evidence.

    Args:
        video_path: Path to video file.
        timestamps_sec: List of timestamps (in seconds) to extract frames.
        output_dir: Directory to save keyframe images. If None, frames not saved.

    Returns:
        List of Evidence objects with keyframe paths and descriptions.

    Raises:
        ValueError: If video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        evidence_list = []

        # Create output directory if needed
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        for timestamp_sec in sorted(timestamps_sec):
            frame_number = int(timestamp_sec * fps)

            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if not ret:
                logger.warning(
                    "Failed to extract keyframe at %.2fs (frame %d)", timestamp_sec, frame_number,
                )
                continue

            # Save frame if output directory provided
            frame_path = None
            if output_dir:
                frame_filename = f"keyframe_{timestamp_sec:.2f}s.jpg"
                frame_path = str(output_path / frame_filename)
                cv2.imwrite(frame_path, frame)

            # Create evidence entry
            evidence = Evidence(
                timestamp_sec=timestamp_sec,
                description=f"Keyframe at {format_timestamp(timestamp_sec)}",
                frame_path=frame_path,
            )
            evidence_list.append(evidence)

        logger.info(
            "Extracted %d keyframes from %s to %s", len(evidence_list), video_path, output_dir,
        )

        return evidence_list

    finally:
        cap.release()


def format_timestamp(seconds: float, include_hours: bool = True) -> str:
    """Format seconds as human-readable timestamp.

    Args:
        seconds: Time in seconds.
        include_hours: Whether to include hours in format.

    Returns:
        Formatted timestamp string (e.g., "00:01:23.45" or "01:23.45").
    """
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = int((seconds - total_seconds) * 100)

    if include_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}.{milliseconds:02d}"


def parse_timestamp(timestamp_str: str) -> float:
    """Parse timestamp string to seconds.

    Supports formats:
    - "MM:SS.mm" (minutes:seconds.centiseconds)
    - "HH:MM:SS.mm" (hours:minutes:seconds.centiseconds)
    - "SS.mm" (seconds.centiseconds)

    Args:
        timestamp_str: Timestamp string.

    Returns:
        Time in seconds.

    Raises:
        ValueError: If timestamp format is invalid.
    """
    parts = timestamp_str.strip().split(":")
    try:
        if len(parts) == 1:
            # "SS.mm" format
            return float(parts[0])
        elif len(parts) == 2:
            # "MM:SS.mm" format
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        elif len(parts) == 3:
            # "HH:MM:SS.mm" format
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            raise ValueError(f"Invalid timestamp format: {timestamp_str}")
    except ValueError as e:
        raise ValueError(f"Cannot parse timestamp '{timestamp_str}': {e}") from e


def calculate_frame_difference(
    video_path: str,
    timestamp1_sec: float,
    timestamp2_sec: float,
) -> float:
    """Calculate pixel-wise difference between two frames.

    Useful for detecting scene changes or impact moments.

    Args:
        video_path: Path to video file.
        timestamp1_sec: First timestamp in seconds.
        timestamp2_sec: Second timestamp in seconds.

    Returns:
        Normalized difference score (0.0 to 1.0), where higher is more different.

    Raises:
        ValueError: If frames cannot be extracted.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Extract first frame
        frame1_number = int(timestamp1_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame1_number)
        ret1, frame1 = cap.read()

        if not ret1:
            raise ValueError(f"Cannot extract frame at {timestamp1_sec}s")

        # Extract second frame
        frame2_number = int(timestamp2_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame2_number)
        ret2, frame2 = cap.read()

        if not ret2:
            raise ValueError(f"Cannot extract frame at {timestamp2_sec}s")

        # Convert to grayscale for comparison
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Normalize to [0, 1]
        normalized_diff = np.mean(diff) / 255.0

        return normalized_diff

    finally:
        cap.release()


def detect_scene_changes(
    video_path: str,
    threshold: float = 0.3,
    min_gap_sec: float = 1.0,
) -> list[float]:
    """Detect scene changes in video using frame differencing.

    Args:
        video_path: Path to video file.
        threshold: Difference threshold for scene change (0.0 to 1.0).
        min_gap_sec: Minimum gap between detected scene changes (seconds).

    Returns:
        List of timestamps (in seconds) where scene changes occur.

    Raises:
        ValueError: If video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        scene_changes = []
        prev_gray = None
        last_change_sec = -float("inf")

        # Sample every 5 frames for efficiency
        sample_interval = 5

        for frame_idx in range(0, num_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Calculate difference
                diff = cv2.absdiff(gray, prev_gray)
                normalized_diff = np.mean(diff) / 255.0

                timestamp_sec = frame_idx / fps

                # Check if this is a scene change
                if normalized_diff > threshold and (timestamp_sec - last_change_sec) > min_gap_sec:
                    scene_changes.append(timestamp_sec)
                    last_change_sec = timestamp_sec

            prev_gray = gray

        logger.info(
            "Detected %d scene changes in %s (threshold=%.1f)", len(scene_changes), video_path, threshold,
        )

        return scene_changes

    finally:
        cap.release()


def estimate_motion_intensity(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_rate: int = 5,
) -> float:
    """Estimate motion intensity in a time segment using optical flow magnitude.

    Args:
        video_path: Path to video file.
        start_sec: Start of time segment (seconds).
        end_sec: End of time segment (seconds).
        sample_rate: Sample every N frames for efficiency.

    Returns:
        Average motion intensity (arbitrary units, higher = more motion).

    Raises:
        ValueError: If video file cannot be opened or time range invalid.
    """
    if start_sec >= end_sec:
        raise ValueError(f"Invalid time range: {start_sec} >= {end_sec}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        motion_magnitudes = []
        prev_gray = None

        for frame_idx in range(start_frame, end_frame, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Calculate optical flow using Farneback method
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )

                # Calculate flow magnitude
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                avg_magnitude = np.mean(magnitude)
                motion_magnitudes.append(avg_magnitude)

            prev_gray = gray

        if not motion_magnitudes:
            return 0.0

        avg_motion = float(np.mean(motion_magnitudes))

        logger.debug(
            "Motion intensity: %s [%.1f-%.1fs] avg=%.2f", video_path, start_sec, end_sec, avg_motion,
        )

        return avg_motion

    finally:
        cap.release()


def calculate_video_quality_score(video_path: str, num_samples: int = 10) -> dict[str, float]:
    """Calculate video quality metrics.

    Estimates:
    - Brightness
    - Contrast
    - Sharpness (Laplacian variance)

    Args:
        video_path: Path to video file.
        num_samples: Number of frames to sample for quality assessment.

    Returns:
        Dictionary with quality metrics:
        - brightness: Average brightness (0-255)
        - contrast: Average contrast (0-255)
        - sharpness: Average sharpness (higher = sharper)
        - overall_score: Normalized overall quality (0.0-1.0)

    Raises:
        ValueError: If video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    try:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        brightness_values = []
        contrast_values = []
        sharpness_values = []

        # Sample frames evenly distributed
        sample_indices = np.linspace(0, num_frames - 1, num_samples, dtype=int)

        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Brightness (mean pixel value)
            brightness = np.mean(gray)
            brightness_values.append(brightness)

            # Contrast (standard deviation)
            contrast = np.std(gray)
            contrast_values.append(contrast)

            # Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_values.append(sharpness)

        # Calculate averages
        avg_brightness = float(np.mean(brightness_values)) if brightness_values else 0.0
        avg_contrast = float(np.mean(contrast_values)) if contrast_values else 0.0
        avg_sharpness = float(np.mean(sharpness_values)) if sharpness_values else 0.0

        # Overall score (normalized)
        # Optimal brightness: 100-150, contrast: >30, sharpness: >100
        brightness_score = 1.0 - abs(avg_brightness - 125) / 125
        contrast_score = min(avg_contrast / 50.0, 1.0)
        sharpness_score = min(avg_sharpness / 200.0, 1.0)

        overall_score = (brightness_score + contrast_score + sharpness_score) / 3.0

        quality_metrics = {
            "brightness": avg_brightness,
            "contrast": avg_contrast,
            "sharpness": avg_sharpness,
            "overall_score": overall_score,
        }

        logger.info(
            "Video quality: %s brightness=%.1f contrast=%.1f sharpness=%.1f score=%.3f",
            video_path, avg_brightness, avg_contrast, avg_sharpness, overall_score,
        )

        return quality_metrics

    finally:
        cap.release()
