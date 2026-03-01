"""Frame extractor — samples video at N fps and yields JPEG frames."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def iter_frames(
    video_path: Path,
    sample_fps: float = 1.0,
    max_frames: int = 7200,  # 2 hours at 1 fps
    jpeg_quality: int = 85,
    output_dir: Path | None = None,
) -> Iterator[tuple[int, float, Path]]:
    """Yield (frame_number, timestamp_sec, jpeg_path) for sampled frames.

    Parameters
    ----------
    video_path:
        Path to the video file.
    sample_fps:
        How many frames per second to sample (1.0 = 1 frame per second).
    max_frames:
        Hard cap on total frames to extract.
    jpeg_quality:
        JPEG compression quality (1–100).
    output_dir:
        Directory to write JPEG files. Defaults to a `frames/` subdirectory
        next to the video file.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_video_frames / video_fps

    logger.info(
        "video=%s  duration=%.1fs  video_fps=%.1f  sample_fps=%.1f  max_frames=%d",
        video_path.name, duration_sec, video_fps, sample_fps, max_frames,
    )

    if output_dir is None:
        output_dir = video_path.parent / f"frames_{video_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Interval between sampled frames (in video frames)
    frame_interval = max(1, round(video_fps / sample_fps))

    frame_count = 0
    sampled_count = 0

    try:
        while sampled_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp_sec = frame_count / video_fps
                out_path = output_dir / f"frame_{sampled_count:06d}.jpg"

                # Resize to 1280×720 max to keep VLM costs reasonable
                h, w = frame.shape[:2]
                if w > 1280 or h > 720:
                    scale = min(1280 / w, 720 / h)
                    frame = cv2.resize(
                        frame,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                yield sampled_count, timestamp_sec, out_path
                sampled_count += 1

            frame_count += 1
    finally:
        cap.release()

    logger.info("extracted %d frames from %s", sampled_count, video_path.name)


def count_frames(video_path: Path, sample_fps: float = 1.0) -> int:
    """Estimate number of frames that will be extracted without actually extracting."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_sec = total / video_fps
    return int(duration_sec * sample_fps)
