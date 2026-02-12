"""Multi-scale video chunking service for VIGIL-RAG.

This module implements hierarchical video segmentation:
- Shot level: Scene detection via PySceneDetect (adaptive)
- Micro level: 2-4 second chunks for detail
- Meso level: 8-16 second chunks for context
- Macro level: 32-64 second chunks for overview

Design principles:
- Hierarchical structure: macro contains meso, meso contains micro, micro contains shots
- Keyframe extraction at each level (1 per shot, 3 per micro, 5 per meso, 7 per macro)
- Domain-specific configuration (factory/surveillance/sports)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

try:
    from scenedetect import detect, AdaptiveDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for multi-scale chunking."""

    # Shot detection
    shot_detector: Literal["adaptive", "content", "threshold"] = "adaptive"
    shot_threshold: float = 3.0  # Adaptive detector threshold

    # Chunk durations (in seconds)
    micro_min: float = 2.0
    micro_max: float = 4.0
    meso_min: float = 8.0
    meso_max: float = 16.0
    macro_min: float = 32.0
    macro_max: float = 64.0

    # Keyframe extraction
    keyframes_per_shot: int = 1
    keyframes_per_micro: int = 3
    keyframes_per_meso: int = 5
    keyframes_per_macro: int = 7

    @classmethod
    def for_domain(cls, domain: str) -> ChunkConfig:
        """Get domain-specific chunking configuration.

        Args:
            domain: One of 'factory', 'surveillance', 'sports', 'generic'

        Returns:
            ChunkConfig optimized for the domain
        """
        if domain == "factory":
            # Factory: Shorter chunks for detailed work analysis
            return cls(
                shot_threshold=2.5,  # More sensitive to scene changes
                micro_min=2.0,
                micro_max=3.0,
                meso_min=8.0,
                meso_max=12.0,
                macro_min=30.0,
                macro_max=60.0,
                keyframes_per_micro=4,  # More keyframes for detail
            )
        elif domain == "surveillance":
            # Surveillance: Longer chunks (less frequent activity)
            return cls(
                shot_threshold=4.0,  # Less sensitive (reduce false positives)
                micro_min=3.0,
                micro_max=5.0,
                meso_min=10.0,
                meso_max=20.0,
                macro_min=40.0,
                macro_max=80.0,
                keyframes_per_micro=2,  # Fewer keyframes (save space)
            )
        elif domain == "sports":
            # Sports: Medium chunks with more keyframes (dynamic content)
            return cls(
                shot_threshold=2.0,  # Very sensitive (fast action)
                micro_min=2.0,
                micro_max=4.0,
                meso_min=8.0,
                meso_max=15.0,
                macro_min=30.0,
                macro_max=60.0,
                keyframes_per_micro=5,  # Many keyframes (dynamic)
                keyframes_per_meso=7,
            )
        else:
            # Generic/default
            return cls()


@dataclass
class Chunk:
    """Represents a video chunk at any level."""

    level: Literal["shot", "micro", "meso", "macro"]
    start_sec: float
    end_sec: float
    start_frame: int
    end_frame: int
    keyframe_indices: list[int]  # Global frame indices
    parent_chunks: list[Chunk] | None = None  # For hierarchical nesting


@dataclass
class ChunkingResult:
    """Result of multi-scale chunking."""

    shots: list[Chunk]
    micro: list[Chunk]
    meso: list[Chunk]
    macro: list[Chunk]
    video_fps: float
    video_duration_sec: float
    total_frames: int


class ChunkingService:
    """Multi-scale video chunking service."""

    def __init__(self, config: ChunkConfig | None = None) -> None:
        """Initialize chunking service.

        Args:
            config: Chunking configuration (defaults to generic if None)
        """
        self.config = config or ChunkConfig()

    def chunk_video(
        self,
        video_path: Path | str,
        *,
        domain: str = "generic",
        keyframe_dir: Path | None = None,
    ) -> ChunkingResult:
        """Perform multi-scale chunking on a video.

        Args:
            video_path: Path to video file
            domain: Domain for config selection ('factory', 'surveillance', 'sports', 'generic')
            keyframe_dir: Optional directory to save keyframes (if None, keyframes not saved)

        Returns:
            ChunkingResult with all levels of chunks

        Raises:
            ValueError: If video cannot be opened or has invalid properties
            RuntimeError: If PySceneDetect is not available
        """
        if not SCENEDETECT_AVAILABLE:
            raise RuntimeError(
                "scenedetect package not installed. "
                "Install with: pip install scenedetect[opencv]"
            )

        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"Video not found: {video_path}")

        # Use domain-specific config if provided
        config = ChunkConfig.for_domain(domain) if domain != "generic" else self.config

        # Open video to get metadata
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0
        cap.release()

        if fps <= 0 or total_frames <= 0:
            raise ValueError(f"Invalid video properties: fps={fps}, frames={total_frames}")

        # Step 1: Shot detection
        shots = self._detect_shots(video_path, fps, config)

        # Step 2: Generate hierarchical chunks
        micro_chunks = self._generate_fixed_chunks(
            shots,
            fps,
            duration_sec,
            level="micro",
            min_dur=config.micro_min,
            max_dur=config.micro_max,
            keyframes=config.keyframes_per_micro,
        )
        meso_chunks = self._generate_fixed_chunks(
            micro_chunks,
            fps,
            duration_sec,
            level="meso",
            min_dur=config.meso_min,
            max_dur=config.meso_max,
            keyframes=config.keyframes_per_meso,
        )
        macro_chunks = self._generate_fixed_chunks(
            meso_chunks,
            fps,
            duration_sec,
            level="macro",
            min_dur=config.macro_min,
            max_dur=config.macro_max,
            keyframes=config.keyframes_per_macro,
        )

        # Step 3: Extract and save keyframes if requested
        if keyframe_dir is not None:
            keyframe_dir.mkdir(parents=True, exist_ok=True)
            self._extract_keyframes(video_path, shots, keyframe_dir, "shot")
            self._extract_keyframes(video_path, micro_chunks, keyframe_dir, "micro")
            self._extract_keyframes(video_path, meso_chunks, keyframe_dir, "meso")
            self._extract_keyframes(video_path, macro_chunks, keyframe_dir, "macro")

        return ChunkingResult(
            shots=shots,
            micro=micro_chunks,
            meso=meso_chunks,
            macro=macro_chunks,
            video_fps=fps,
            video_duration_sec=duration_sec,
            total_frames=total_frames,
        )

    def _detect_shots(
        self,
        video_path: Path,
        fps: float,
        config: ChunkConfig,
    ) -> list[Chunk]:
        """Detect shot boundaries using PySceneDetect.

        Args:
            video_path: Path to video file
            fps: Video frame rate
            config: Chunking configuration

        Returns:
            List of shot-level chunks
        """
        logger.info("Detecting shots in %s (threshold=%.1f)", video_path, config.shot_threshold)

        try:
            # Use PySceneDetect's detect() API
            scene_list = detect(
                str(video_path),
                AdaptiveDetector(adaptive_threshold=config.shot_threshold),
            )
        except Exception as exc:
            logger.warning("Shot detection failed, falling back to single shot: %s", exc)
            # Fallback: treat entire video as one shot
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            scene_list = [(0, total_frames)]

        shots: list[Chunk] = []
        for idx, (start_frame, end_frame) in enumerate(scene_list):
            start_sec = start_frame / fps
            end_sec = end_frame / fps

            # Extract 1 keyframe at the middle of the shot
            mid_frame = (start_frame + end_frame) // 2

            shots.append(
                Chunk(
                    level="shot",
                    start_sec=start_sec,
                    end_sec=end_sec,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    keyframe_indices=[mid_frame],
                )
            )

        logger.info("Detected %d shots", len(shots))
        return shots

    def _generate_fixed_chunks(
        self,
        base_chunks: list[Chunk],
        fps: float,
        duration_sec: float,
        *,
        level: Literal["micro", "meso", "macro"],
        min_dur: float,
        max_dur: float,
        keyframes: int,
    ) -> list[Chunk]:
        """Generate fixed-duration chunks by grouping base chunks.

        Args:
            base_chunks: Lower-level chunks to group
            fps: Video frame rate
            duration_sec: Total video duration
            level: Chunk level to generate
            min_dur: Minimum chunk duration in seconds
            max_dur: Maximum chunk duration in seconds
            keyframes: Number of keyframes to extract per chunk

        Returns:
            List of fixed-duration chunks
        """
        target_dur = (min_dur + max_dur) / 2.0
        chunks: list[Chunk] = []
        current_start = 0.0

        while current_start < duration_sec:
            current_end = min(current_start + target_dur, duration_sec)

            # Adjust to align with base chunk boundaries if possible
            if base_chunks:
                # Find base chunks overlapping with [current_start, current_end]
                overlapping = [
                    c for c in base_chunks
                    if c.start_sec < current_end and c.end_sec > current_start
                ]
                if overlapping:
                    # Snap to the last overlapping chunk's end
                    current_end = max(c.end_sec for c in overlapping)

            # Ensure minimum duration
            if current_end - current_start < min_dur and current_end < duration_sec:
                current_end = min(current_start + min_dur, duration_sec)

            start_frame = int(current_start * fps)
            end_frame = int(current_end * fps)

            # Distribute keyframes evenly across the chunk
            keyframe_indices = self._distribute_keyframes(start_frame, end_frame, keyframes)

            chunks.append(
                Chunk(
                    level=level,
                    start_sec=current_start,
                    end_sec=current_end,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    keyframe_indices=keyframe_indices,
                )
            )

            current_start = current_end

        logger.info("Generated %d %s chunks", len(chunks), level)
        return chunks

    def _distribute_keyframes(
        self,
        start_frame: int,
        end_frame: int,
        num_keyframes: int,
    ) -> list[int]:
        """Distribute keyframes evenly across a frame range.

        Args:
            start_frame: Start frame index
            end_frame: End frame index
            num_keyframes: Number of keyframes to distribute

        Returns:
            List of keyframe indices
        """
        if num_keyframes <= 0:
            return []

        total_frames = end_frame - start_frame
        if total_frames <= 0:
            return []

        if num_keyframes == 1:
            return [(start_frame + end_frame) // 2]

        # Distribute evenly
        step = total_frames / (num_keyframes + 1)
        return [int(start_frame + step * (i + 1)) for i in range(num_keyframes)]

    def _extract_keyframes(
        self,
        video_path: Path,
        chunks: list[Chunk],
        keyframe_dir: Path,
        level: str,
    ) -> None:
        """Extract keyframes for chunks and save to disk.

        Args:
            video_path: Path to video file
            chunks: List of chunks to extract keyframes from
            keyframe_dir: Directory to save keyframes
            level: Chunk level (for naming)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Cannot open video for keyframe extraction: %s", video_path)
            return

        # Collect all unique frame indices needed
        frame_indices_needed: set[int] = set()
        for chunk in chunks:
            frame_indices_needed.update(chunk.keyframe_indices)

        frame_indices_sorted = sorted(frame_indices_needed)
        current_frame_idx = 0
        frames_extracted = 0

        for target_frame in frame_indices_sorted:
            # Seek to the target frame
            if target_frame < current_frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                current_frame_idx = target_frame
            else:
                # Skip frames to reach target
                while current_frame_idx < target_frame:
                    ret = cap.grab()
                    if not ret:
                        break
                    current_frame_idx += 1

            ret, frame = cap.retrieve()
            if not ret:
                logger.warning("Failed to extract frame %d", target_frame)
                continue

            # Save keyframe
            keyframe_path = keyframe_dir / f"{level}_frame_{target_frame:08d}.jpg"
            cv2.imwrite(str(keyframe_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames_extracted += 1

        cap.release()
        logger.info("Extracted %d keyframes for %s level", frames_extracted, level)
