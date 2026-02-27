"""Stage 1: Multimodal Signal Mining → Top-K danger clips."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def run_mining(signal_fuser: Any, video_path: str, video_id: str, top_k: int) -> list[Any]:
    """Run signal mining stage to extract danger clips.

    Args:
        signal_fuser: Initialized SignalFuser instance.
        video_path: Path to video file.
        video_id: Unique video identifier.
        top_k: Number of top clips to extract.

    Returns:
        List of danger clip dicts.
    """
    try:
        return signal_fuser.extract_danger_clips(video_path=video_path, top_k=top_k)
    except Exception as e:
        logger.error("Mining failed: %s", e)
        raise


def mock_danger_clips(video_path: str, video_id: str, top_k: int = 3) -> list[dict]:
    """Generate mock danger clips for testing.

    Signal scores are pattern-matched from video filename for
    recalibration compatibility.
    """
    from pathlib import Path

    filename = Path(video_path).stem.lower()

    # Derive signal scores from filename pattern
    if "collision" in filename or "crash" in filename:
        danger_base, motion_score, proximity_score = 0.9, 0.9, 0.9
    elif "near_miss" in filename or "near-miss" in filename:
        danger_base, motion_score, proximity_score = 0.75, 0.8, 0.7
    elif "swerve" in filename:
        danger_base, motion_score, proximity_score = 0.7, 0.85, 0.6
    elif "normal" in filename or "safe" in filename:
        danger_base, motion_score, proximity_score = 0.15, 0.1, 0.1
    elif "parking" in filename or "bump" in filename:
        danger_base, motion_score, proximity_score = 0.4, 0.3, 0.8
    elif "hard_braking" in filename:
        danger_base, motion_score, proximity_score = 0.5, 0.7, 0.2
    else:
        danger_base, motion_score, proximity_score = 0.5, 0.5, 0.5

    return [
        {
            "clip_id": f"{video_id}_clip_{i}",
            "start_sec": i * 10.0,
            "end_sec": (i * 10.0) + 5.0,
            "peak_sec": (i * 10.0) + 2.5,
            "danger_score": max(0.0, danger_base - (i * 0.1)),
            "audio_score": 0.0,
            "video_path": video_path,
            "motion_score": motion_score,
            "proximity_score": proximity_score,
        }
        for i in range(min(3, top_k))
    ]
