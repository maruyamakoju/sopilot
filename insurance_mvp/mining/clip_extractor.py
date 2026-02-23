"""FFmpeg-based Clip Extraction for Insurance Video Review

Extracts video clips identified by the mining pipeline as separate files
using ffmpeg subprocess calls.

Usage:
    from insurance_mvp.mining.clip_extractor import ClipExtractor, ClipExtractorConfig

    extractor = ClipExtractor()
    clips_with_paths = extractor.extract_clips("dashcam.mp4", danger_clips)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ClipExtractorConfig:
    """Configuration for ffmpeg clip extraction."""

    output_dir: str = "extracted_clips"
    codec: str = "copy"  # stream copy for speed, "libx264" for re-encode
    padding_sec: float = 1.0  # extra seconds before/after clip boundaries
    max_clips: int = 20


class ClipExtractor:
    """Extract video clips using ffmpeg subprocess calls.

    Handles the case where ffmpeg is not installed by logging a warning
    and returning None for individual clips.
    """

    def __init__(self, config: ClipExtractorConfig | None = None):
        self.config = config or ClipExtractorConfig()
        self._ffmpeg_path = shutil.which("ffmpeg")
        if self._ffmpeg_path is None:
            logger.warning("ffmpeg not found on PATH. Clip extraction will be unavailable.")

    def extract_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_path: str | None = None,
    ) -> str | None:
        """Extract a single clip using ffmpeg.

        Args:
            video_path: Path to the source video file.
            start_time: Clip start time in seconds.
            end_time: Clip end time in seconds.
            output_path: Destination file path. Auto-generated if None.

        Returns:
            Path to the extracted clip file, or None if extraction failed.
        """
        if self._ffmpeg_path is None:
            logger.warning("ffmpeg not available. Skipping clip extraction.")
            return None

        # Apply padding (clamp start to >= 0)
        padded_start = max(0.0, start_time - self.config.padding_sec)
        padded_end = end_time + self.config.padding_sec

        # Generate output path if not provided
        if output_path is None:
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            video_stem = Path(video_path).stem
            suffix = Path(video_path).suffix or ".mp4"
            output_path = str(out_dir / f"{video_stem}_{padded_start:.1f}s_{padded_end:.1f}s{suffix}")

        # Build ffmpeg command
        cmd = [
            self._ffmpeg_path,
            "-y",  # overwrite without asking
            "-ss", f"{padded_start:.3f}",
            "-to", f"{padded_end:.3f}",
            "-i", str(video_path),
            "-c", self.config.codec,
            str(output_path),
        ]

        logger.info("Extracting clip: %.1fs-%.1fs -> %s", padded_start, padded_end, output_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.error("ffmpeg failed (rc=%d): %s", result.returncode, result.stderr[:500])
                return None
            return output_path
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timed out extracting clip %.1fs-%.1fs", padded_start, padded_end)
            return None
        except FileNotFoundError:
            logger.error("ffmpeg binary not found at %s", self._ffmpeg_path)
            self._ffmpeg_path = None
            return None

    def extract_clips(
        self,
        video_path: str,
        clips: list[dict],
        output_dir: str | None = None,
    ) -> list[dict]:
        """Extract all clips, adding 'extracted_path' to each clip dict.

        Respects max_clips limit. Clips are processed in order; those beyond
        the limit get extracted_path=None.

        Args:
            video_path: Path to the source video file.
            clips: List of clip dicts with 'start_sec' and 'end_sec' keys.
            output_dir: Override output directory for this batch.

        Returns:
            The same list of clip dicts, each augmented with 'extracted_path'.
        """
        if output_dir is not None:
            original_output_dir = self.config.output_dir
            self.config.output_dir = output_dir

        try:
            for i, clip in enumerate(clips):
                if i >= self.config.max_clips:
                    logger.info("Reached max_clips limit (%d). Skipping remaining.", self.config.max_clips)
                    clip["extracted_path"] = None
                    continue

                start_sec = clip.get("start_sec", 0.0)
                end_sec = clip.get("end_sec", 5.0)
                extracted = self.extract_clip(video_path, start_sec, end_sec)
                clip["extracted_path"] = extracted
        finally:
            if output_dir is not None:
                self.config.output_dir = original_output_dir

        return clips
