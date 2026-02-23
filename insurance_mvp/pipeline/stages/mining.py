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
    """Generate mock danger clips for testing."""
    return [
        {
            "clip_id": f"{video_id}_clip_{i}",
            "start_sec": i * 10.0,
            "end_sec": (i * 10.0) + 5.0,
            "danger_score": 0.8 - (i * 0.1),
            "video_path": video_path,
        }
        for i in range(min(3, top_k))
    ]
