"""Multimodal Hazard Mining for Insurance Video Review

Production-ready pipeline for detecting dangerous events in dashcam footage.

Modules:
- audio: Audio signal analysis (brake sounds, horns, impacts)
- motion: Optical flow analysis (sudden movements, irregular motion)
- proximity: Object detection (nearby vehicles, pedestrians, cyclists)
- fuse: Signal fusion and peak detection (top-K hazard clips)

Example usage:

    from insurance_mvp.mining import (
        AudioAnalyzer,
        MotionAnalyzer,
        ProximityAnalyzer,
        SignalFuser,
        HazardClip,
    )

    # Initialize analyzers
    audio_analyzer = AudioAnalyzer()
    motion_analyzer = MotionAnalyzer()
    proximity_analyzer = ProximityAnalyzer()
    fuser = SignalFuser()

    # Analyze video
    video_path = "dashcam001.mp4"

    audio_scores = audio_analyzer.analyze(video_path)
    motion_scores = motion_analyzer.analyze(video_path)
    proximity_scores = proximity_analyzer.analyze(video_path)

    # Extract hazard clips
    clips = fuser.fuse_and_extract(
        audio_scores,
        motion_scores,
        proximity_scores,
        video_duration_sec=60.0
    )

    # Process clips
    for clip in clips:
        print(f"Hazard at {clip.peak_sec:.1f}s: score={clip.score:.3f}")
        print(f"  Clip: [{clip.start_sec:.1f}s, {clip.end_sec:.1f}s]")
        print(f"  Breakdown: audio={clip.audio_score:.3f}, "
              f"motion={clip.motion_score:.3f}, "
              f"proximity={clip.proximity_score:.3f}")
"""

from .audio import AudioAnalyzer, AudioConfig
from .clip_extractor import ClipExtractor, ClipExtractorConfig
from .fuse import FusionConfig, HazardClip, SignalFuser
from .motion import MotionAnalyzer, MotionConfig
from .proximity import ProximityAnalyzer, ProximityConfig

__all__ = [
    # Audio
    "AudioAnalyzer",
    "AudioConfig",
    # Motion
    "MotionAnalyzer",
    "MotionConfig",
    # Proximity
    "ProximityAnalyzer",
    "ProximityConfig",
    # Fusion
    "SignalFuser",
    "FusionConfig",
    "HazardClip",
    # Clip extraction
    "ClipExtractor",
    "ClipExtractorConfig",
]

__version__ = "0.1.0"
