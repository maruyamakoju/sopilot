"""Signal Fusion and Peak Detection for Insurance Video Review

Combines audio, motion, and proximity signals to identify top hazard clips.

Pipeline:
1. Fuse signals with weighted sum (audio=0.3, motion=0.4, proximity=0.3)
2. Find top-K danger peaks using scipy.signal.find_peaks
3. Merge nearby peaks (within 3 seconds)
4. Extract ±5 second clips around peaks
5. Return list of (start_sec, end_sec, score) tuples

Design:
- Production-ready with edge case handling
- Configurable weights and parameters
- Returns empty list if video too short or no peaks found
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """Configuration for signal fusion"""

    # Signal weights (must sum to 1.0)
    audio_weight: float = 0.3
    motion_weight: float = 0.4
    proximity_weight: float = 0.3

    # Peak detection
    top_k_peaks: int = 20  # Number of top peaks to extract
    min_peak_score: float = 0.3  # Minimum score threshold for peaks
    min_peak_distance: int = 3  # Minimum distance between peaks (seconds)

    # Clip extraction
    clip_padding_sec: float = 5.0  # Extract ±5 seconds around each peak
    min_clip_duration: float = 2.0  # Minimum clip duration (seconds)
    max_clip_duration: float = 15.0  # Maximum clip duration (seconds)

    # Merge nearby peaks
    merge_gap_sec: float = 3.0  # Merge peaks within 3 seconds

    def __post_init__(self):
        # Validate weights sum to 1.0
        total_weight = self.audio_weight + self.motion_weight + self.proximity_weight
        if not np.isclose(total_weight, 1.0):
            logger.warning(
                f"Fusion weights sum to {total_weight:.3f}, not 1.0. "
                f"Normalizing weights."
            )
            self.audio_weight /= total_weight
            self.motion_weight /= total_weight
            self.proximity_weight /= total_weight


@dataclass
class HazardClip:
    """Hazard clip metadata"""

    start_sec: float
    end_sec: float
    peak_sec: float  # Peak timestamp within clip
    score: float  # Danger score at peak

    # Signal breakdown (for debugging/analysis)
    audio_score: float = 0.0
    motion_score: float = 0.0
    proximity_score: float = 0.0

    @property
    def duration_sec(self) -> float:
        """Clip duration in seconds"""
        return self.end_sec - self.start_sec


class SignalFuser:
    """
    Fuse multimodal signals and extract top hazard clips.

    Workflow:
    1. Weighted fusion of audio/motion/proximity
    2. Peak detection to find local maxima
    3. Peak merging to avoid duplicate clips
    4. Clip extraction with temporal padding
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()

    def fuse_and_extract(
        self,
        audio_scores: np.ndarray,
        motion_scores: np.ndarray,
        proximity_scores: np.ndarray,
        video_duration_sec: float,
    ) -> list[HazardClip]:
        """
        Fuse signals and extract top hazard clips.

        Args:
            audio_scores: Audio danger scores per second, shape (n_seconds,)
            motion_scores: Motion danger scores per second, shape (n_seconds,)
            proximity_scores: Proximity danger scores per second, shape (n_seconds,)
            video_duration_sec: Total video duration in seconds

        Returns:
            List of HazardClip objects, sorted by score (descending)

        Raises:
            ValueError: If input arrays have different lengths
        """
        # Validate inputs
        if not (len(audio_scores) == len(motion_scores) == len(proximity_scores)):
            raise ValueError(
                f"Signal arrays must have same length: "
                f"audio={len(audio_scores)}, motion={len(motion_scores)}, "
                f"proximity={len(proximity_scores)}"
            )

        n_seconds = len(audio_scores)

        if n_seconds == 0:
            logger.warning("Empty signal arrays, returning no clips")
            return []

        logger.info(f"Fusing signals: {n_seconds} seconds")

        # Weighted fusion
        fused_scores = (
            self.config.audio_weight * audio_scores
            + self.config.motion_weight * motion_scores
            + self.config.proximity_weight * proximity_scores
        )

        logger.info(
            f"Fused signal: mean={fused_scores.mean():.3f}, "
            f"max={fused_scores.max():.3f}, "
            f"std={fused_scores.std():.3f}"
        )

        # Find peaks
        peak_indices = self._find_peaks(fused_scores)

        if len(peak_indices) == 0:
            logger.warning("No peaks found, returning no clips")
            return []

        logger.info(f"Found {len(peak_indices)} peaks")

        # Get peak scores
        peak_scores = fused_scores[peak_indices]

        # Get top-K peaks
        top_k = min(self.config.top_k_peaks, len(peak_indices))
        top_k_idx = np.argsort(peak_scores)[::-1][:top_k]
        top_peak_indices = peak_indices[top_k_idx]
        top_peak_scores = peak_scores[top_k_idx]

        logger.info(
            f"Selected top-{top_k} peaks: "
            f"scores=[{top_peak_scores.min():.3f}, {top_peak_scores.max():.3f}]"
        )

        # Create initial clips
        clips = []
        for peak_idx, peak_score in zip(top_peak_indices, top_peak_scores):
            # Extract clip around peak
            peak_sec = float(peak_idx)

            start_sec = max(0.0, peak_sec - self.config.clip_padding_sec)
            end_sec = min(video_duration_sec, peak_sec + self.config.clip_padding_sec)

            # Skip if clip too short
            if end_sec - start_sec < self.config.min_clip_duration:
                logger.debug(f"Skipping short clip at {peak_sec:.1f}s")
                continue

            # Get signal breakdown at peak
            audio_score = float(audio_scores[peak_idx])
            motion_score = float(motion_scores[peak_idx])
            proximity_score = float(proximity_scores[peak_idx])

            clip = HazardClip(
                start_sec=start_sec,
                end_sec=end_sec,
                peak_sec=peak_sec,
                score=float(peak_score),
                audio_score=audio_score,
                motion_score=motion_score,
                proximity_score=proximity_score,
            )

            clips.append(clip)

        if len(clips) == 0:
            logger.warning("No valid clips after filtering, returning empty list")
            return []

        # Merge nearby clips
        merged_clips = self._merge_nearby_clips(clips)

        logger.info(
            f"Extracted {len(merged_clips)} clips after merging "
            f"(before: {len(clips)})"
        )

        # Sort by score (descending)
        merged_clips.sort(key=lambda c: c.score, reverse=True)

        # Log top clips
        for i, clip in enumerate(merged_clips[:5]):
            logger.info(
                f"Clip {i+1}: [{clip.start_sec:.1f}s, {clip.end_sec:.1f}s], "
                f"score={clip.score:.3f}, "
                f"audio={clip.audio_score:.3f}, "
                f"motion={clip.motion_score:.3f}, "
                f"proximity={clip.proximity_score:.3f}"
            )

        return merged_clips

    def _find_peaks(self, signal: np.ndarray) -> np.ndarray:
        """
        Find peaks in fused signal using scipy.signal.find_peaks.

        Args:
            signal: Fused danger scores per second

        Returns:
            Array of peak indices
        """
        # Find peaks with minimum distance constraint
        peaks, properties = find_peaks(
            signal,
            height=self.config.min_peak_score,
            distance=self.config.min_peak_distance,
        )

        return peaks

    def _merge_nearby_clips(self, clips: list[HazardClip]) -> list[HazardClip]:
        """
        Merge clips that are within merge_gap_sec of each other.

        Args:
            clips: List of clips sorted by peak time

        Returns:
            Merged clips
        """
        if len(clips) <= 1:
            return clips

        # Sort by peak time
        sorted_clips = sorted(clips, key=lambda c: c.peak_sec)

        merged = []
        current_clip = sorted_clips[0]

        for next_clip in sorted_clips[1:]:
            # Check if clips are close
            gap = next_clip.start_sec - current_clip.end_sec

            if gap <= self.config.merge_gap_sec:
                # Merge clips
                current_clip = HazardClip(
                    start_sec=current_clip.start_sec,
                    end_sec=max(current_clip.end_sec, next_clip.end_sec),
                    peak_sec=current_clip.peak_sec,  # Keep first peak
                    score=max(current_clip.score, next_clip.score),  # Keep max score
                    audio_score=max(current_clip.audio_score, next_clip.audio_score),
                    motion_score=max(current_clip.motion_score, next_clip.motion_score),
                    proximity_score=max(current_clip.proximity_score, next_clip.proximity_score),
                )
            else:
                # No merge, save current and move to next
                merged.append(current_clip)
                current_clip = next_clip

        # Add last clip
        merged.append(current_clip)

        return merged

    def visualize_signals(
        self,
        audio_scores: np.ndarray,
        motion_scores: np.ndarray,
        proximity_scores: np.ndarray,
        clips: list[HazardClip],
        output_path: Optional[Path | str] = None,
    ) -> Optional[Path]:
        """
        Generate signal visualization plot for debugging.

        Args:
            audio_scores: Audio scores
            motion_scores: Motion scores
            proximity_scores: Proximity scores
            clips: Extracted clips
            output_path: Output image path (optional)

        Returns:
            Path to output image, or None if matplotlib not available
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping visualization")
            return None

        # Compute fused signal
        fused_scores = (
            self.config.audio_weight * audio_scores
            + self.config.motion_weight * motion_scores
            + self.config.proximity_weight * proximity_scores
        )

        n_seconds = len(audio_scores)
        time_axis = np.arange(n_seconds)

        # Create figure
        fig, axes = plt.subplots(5, 1, figsize=(16, 10), sharex=True)

        # Plot individual signals
        axes[0].plot(time_axis, audio_scores, label="Audio", color="blue", alpha=0.7)
        axes[0].set_ylabel("Audio Score")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(time_axis, motion_scores, label="Motion", color="green", alpha=0.7)
        axes[1].set_ylabel("Motion Score")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(time_axis, proximity_scores, label="Proximity", color="orange", alpha=0.7)
        axes[2].set_ylabel("Proximity Score")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        # Plot fused signal
        axes[3].plot(time_axis, fused_scores, label="Fused", color="red", linewidth=2)
        axes[3].set_ylabel("Fused Score")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()

        # Mark peaks
        for clip in clips:
            axes[3].axvline(clip.peak_sec, color="red", linestyle="--", alpha=0.5)

        # Plot clip ranges
        axes[4].set_ylim([0, 1])
        for i, clip in enumerate(clips):
            axes[4].barh(
                0.5,
                clip.end_sec - clip.start_sec,
                left=clip.start_sec,
                height=0.8,
                alpha=0.5,
                color="red",
            )
            axes[4].text(
                clip.peak_sec,
                0.5,
                f"{clip.score:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

        axes[4].set_ylabel("Clips")
        axes[4].set_xlabel("Time (seconds)")
        axes[4].grid(True, alpha=0.3)
        axes[4].set_yticks([])

        plt.tight_layout()

        if output_path is None:
            output_path = Path("signal_fusion_viz.png")
        else:
            output_path = Path(output_path)

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Signal visualization saved: {output_path}")
        return output_path
