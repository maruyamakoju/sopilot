"""Optical Flow Analysis for Insurance Video Review

Analyzes visual motion patterns to detect danger signals:
- Sudden movements (hard braking, swerving)
- Irregular movements (multi-vehicle collisions)
- High-speed activity

Design:
- Use cv2.calcOpticalFlowFarneback for dense optical flow
- Process every N frames (not every frame - too slow)
- Return normalized danger scores [0.0, 1.0]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MotionConfig:
    """Configuration for motion analysis"""

    # Optical flow parameters (Farneback algorithm)
    pyr_scale: float = 0.5  # Pyramid scale
    levels: int = 3  # Pyramid levels
    winsize: int = 15  # Window size
    iterations: int = 3  # Iterations per pyramid level
    poly_n: int = 5  # Polynomial expansion neighborhood
    poly_sigma: float = 1.2  # Gaussian sigma for polynomial expansion
    flags: int = 0  # Operation flags

    # Frame sampling (process every N frames)
    frame_skip: int = 5  # Process every 5 frames (e.g., 30fps → 6 samples/sec)

    # Danger detection thresholds
    magnitude_percentile: float = 95.0  # Percentile for normalization
    variance_percentile: float = 95.0  # Percentile for normalization


class MotionAnalyzer:
    """
    Analyze optical flow for danger signals.

    Extracts:
    1. Flow magnitude - overall motion intensity (speed, sudden movements)
    2. Flow variance - motion irregularity (chaotic multi-vehicle scenarios)
    """

    def __init__(self, config: Optional[MotionConfig] = None):
        self.config = config or MotionConfig()

    def analyze(self, video_path: Path | str) -> np.ndarray:
        """
        Analyze optical flow and return danger scores per second.

        Args:
            video_path: Path to video file

        Returns:
            np.ndarray: Danger scores per second [0.0, 1.0], shape (n_seconds,)

        Raises:
            RuntimeError: If video file cannot be read
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise RuntimeError(f"Video file not found: {video_path}")

        logger.info(f"Analyzing motion: {video_path.name}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            cap.release()
            raise RuntimeError(f"Invalid FPS: {fps}")

        duration_sec = frame_count / fps
        n_seconds = int(np.ceil(duration_sec))

        logger.info(f"Video: {frame_count} frames, {fps:.1f} fps, {duration_sec:.1f}s, {width}x{height}")

        # Process video
        magnitude_per_frame = []
        variance_per_frame = []
        frame_timestamps = []

        prev_gray = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Skip frames for performance
            if frame_idx % self.config.frame_skip != 0:
                frame_idx += 1
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compute optical flow
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,
                    pyr_scale=self.config.pyr_scale,
                    levels=self.config.levels,
                    winsize=self.config.winsize,
                    iterations=self.config.iterations,
                    poly_n=self.config.poly_n,
                    poly_sigma=self.config.poly_sigma,
                    flags=self.config.flags,
                )

                # Extract flow magnitude (overall motion intensity)
                fx, fy = flow[..., 0], flow[..., 1]
                magnitude = np.sqrt(fx**2 + fy**2)
                mag_mean = magnitude.mean()
                magnitude_per_frame.append(mag_mean)

                # Extract flow variance (motion irregularity)
                var = magnitude.var()
                variance_per_frame.append(var)

                # Record timestamp
                timestamp_sec = frame_idx / fps
                frame_timestamps.append(timestamp_sec)

            prev_gray = gray
            frame_idx += 1

            # Log progress every 100 frames
            if frame_idx % 100 == 0:
                logger.debug(f"Processed {frame_idx}/{frame_count} frames")

        cap.release()

        if len(magnitude_per_frame) == 0:
            logger.warning(f"No optical flow computed for {video_path.name}")
            return np.zeros(n_seconds, dtype=np.float32)

        # Convert to numpy arrays
        magnitude_arr = np.array(magnitude_per_frame, dtype=np.float32)
        variance_arr = np.array(variance_per_frame, dtype=np.float32)
        timestamps_arr = np.array(frame_timestamps, dtype=np.float32)

        # Aggregate to per-second scores
        magnitude_per_sec = self._aggregate_to_seconds(timestamps_arr, magnitude_arr, n_seconds)
        variance_per_sec = self._aggregate_to_seconds(timestamps_arr, variance_arr, n_seconds)

        # Normalize
        magnitude_norm = self._normalize(magnitude_per_sec, self.config.magnitude_percentile)
        variance_norm = self._normalize(variance_per_sec, self.config.variance_percentile)

        # Fuse magnitude and variance (max fusion)
        danger_scores = np.maximum(magnitude_norm, variance_norm)

        logger.info(
            f"Motion analysis complete: {n_seconds}s, "
            f"mean_danger={danger_scores.mean():.3f}, "
            f"max_danger={danger_scores.max():.3f}, "
            f"peaks={np.sum(danger_scores > 0.5)}"
        )

        return danger_scores

    def _aggregate_to_seconds(self, timestamps: np.ndarray, values: np.ndarray, n_seconds: int) -> np.ndarray:
        """
        Aggregate frame-level values to per-second values using max pooling.

        Args:
            timestamps: Frame timestamps (seconds)
            values: Frame-level values
            n_seconds: Total number of seconds

        Returns:
            Per-second aggregated values
        """
        per_sec = np.zeros(n_seconds, dtype=np.float32)

        for ts, val in zip(timestamps, values):
            sec_idx = int(np.floor(ts))
            if 0 <= sec_idx < n_seconds:
                # Use max pooling (keep highest motion within each second)
                per_sec[sec_idx] = max(per_sec[sec_idx], val)

        return per_sec

    def _normalize(self, values: np.ndarray, percentile: float) -> np.ndarray:
        """
        Normalize values to [0.0, 1.0] using percentile-based scaling.

        Args:
            values: Raw values
            percentile: Percentile for normalization (avoid outlier saturation)

        Returns:
            Normalized values
        """
        if len(values) == 0 or values.max() == 0:
            return values

        # Use percentile to avoid outlier saturation
        percentile_val = np.percentile(values, percentile)

        if percentile_val > 0:
            normalized = values / percentile_val
            normalized = np.clip(normalized, 0.0, 1.0)
            return normalized
        else:
            return values


def compute_flow_visualization(
    video_path: Path | str, output_path: Optional[Path | str] = None, frame_skip: int = 5
) -> Optional[Path]:
    """
    Generate optical flow visualization video for debugging.

    Args:
        video_path: Input video path
        output_path: Output video path (optional, auto-generated if None)
        frame_skip: Process every N frames

    Returns:
        Path to output video, or None if failed
    """
    video_path = Path(video_path)

    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_flow_viz.mp4"
    else:
        output_path = Path(output_path)

    logger.info(f"Generating flow visualization: {video_path.name} -> {output_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps / frame_skip, (width, height))

    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Compute flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Convert flow to HSV color representation
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Blend with original frame
            viz = cv2.addWeighted(frame, 0.6, flow_rgb, 0.4, 0)

            out.write(viz)

        prev_gray = gray
        frame_idx += 1

    cap.release()
    out.release()

    logger.info(f"Flow visualization saved: {output_path}")
    return output_path
