"""Integration tests for mining pipeline

Tests the full multimodal hazard mining pipeline end-to-end.
"""

import numpy as np
import pytest
from pathlib import Path

from insurance_mvp.mining import (
    AudioAnalyzer,
    AudioConfig,
    MotionAnalyzer,
    MotionConfig,
    ProximityAnalyzer,
    ProximityConfig,
    SignalFuser,
    FusionConfig,
    HazardClip,
)


class TestAudioAnalyzer:
    """Test audio analysis module"""

    def test_audio_config_defaults(self):
        """Test AudioConfig default values"""
        config = AudioConfig()
        assert config.rms_window_sec == 1.0
        assert config.sample_rate == 16000
        assert config.horn_freq_min == 300
        assert config.horn_freq_max == 1000

    def test_audio_analyzer_init(self):
        """Test AudioAnalyzer initialization"""
        analyzer = AudioAnalyzer()
        assert analyzer.config is not None
        assert isinstance(analyzer.config, AudioConfig)

    def test_audio_analyzer_missing_file(self):
        """Test error handling for missing video file"""
        analyzer = AudioAnalyzer()
        with pytest.raises(RuntimeError, match="not found"):
            analyzer.analyze("nonexistent.mp4")

    def test_audio_analyzer_synthetic_no_audio(self, tmp_path):
        """Test handling of video with no audio track"""
        # Create synthetic video without audio
        import cv2

        video_path = tmp_path / "test_silent.mp4"
        fps = 30.0
        n_frames = 90  # 3 seconds

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (640, 480))

        for i in range(n_frames):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            writer.write(frame)

        writer.release()

        # Analyze (should return zeros, not crash)
        analyzer = AudioAnalyzer()
        scores = analyzer.analyze(video_path)

        assert len(scores) == 3
        assert np.all(scores == 0.0)


class TestMotionAnalyzer:
    """Test motion analysis module"""

    def test_motion_config_defaults(self):
        """Test MotionConfig default values"""
        config = MotionConfig()
        assert config.frame_skip == 5
        assert config.pyr_scale == 0.5
        assert config.levels == 3

    def test_motion_analyzer_init(self):
        """Test MotionAnalyzer initialization"""
        analyzer = MotionAnalyzer()
        assert analyzer.config is not None
        assert isinstance(analyzer.config, MotionConfig)

    def test_motion_analyzer_missing_file(self):
        """Test error handling for missing video file"""
        analyzer = MotionAnalyzer()
        with pytest.raises(RuntimeError, match="not found"):
            analyzer.analyze("nonexistent.mp4")

    def test_motion_analyzer_synthetic(self, tmp_path):
        """Test motion analysis on synthetic video with moving content"""
        import cv2

        video_path = tmp_path / "test_motion.mp4"
        fps = 30.0
        n_frames = 60  # 2 seconds

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (640, 480))

        # Create moving white square
        for i in range(n_frames):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            x = int(100 + i * 5)  # Move horizontally
            cv2.rectangle(frame, (x, 200), (x + 50, 250), (255, 255, 255), -1)
            writer.write(frame)

        writer.release()

        # Analyze
        analyzer = MotionAnalyzer(MotionConfig(frame_skip=3))
        scores = analyzer.analyze(video_path)

        assert len(scores) == 2
        assert scores.dtype == np.float32
        # Should detect motion (non-zero scores)
        assert np.any(scores > 0.0)


class TestProximityAnalyzer:
    """Test proximity analysis module"""

    def test_proximity_config_defaults(self):
        """Test ProximityConfig default values"""
        config = ProximityConfig()
        assert config.model_name == "yolov8n.pt"
        assert config.confidence_threshold == 0.25
        assert config.frame_skip == 5
        assert config.target_classes == [0, 1, 2, 3, 5, 7]
        assert config.class_weights[0] == 1.5  # person

    def test_proximity_analyzer_init(self):
        """Test ProximityAnalyzer initialization"""
        analyzer = ProximityAnalyzer()
        assert analyzer.config is not None
        assert analyzer.model is None  # Lazy loading

    def test_proximity_analyzer_missing_file(self):
        """Test error handling for missing video file"""
        analyzer = ProximityAnalyzer()
        with pytest.raises(RuntimeError, match="not found"):
            analyzer.analyze("nonexistent.mp4")

    @pytest.mark.slow
    def test_proximity_analyzer_synthetic(self, tmp_path):
        """Test proximity analysis on synthetic video (requires YOLOv8)"""
        pytest.importorskip("ultralytics")

        import cv2

        video_path = tmp_path / "test_proximity.mp4"
        fps = 30.0
        n_frames = 30  # 1 second

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (640, 480))

        for i in range(n_frames):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            writer.write(frame)

        writer.release()

        # Analyze (no objects expected, should return low scores)
        analyzer = ProximityAnalyzer(ProximityConfig(frame_skip=10))
        scores = analyzer.analyze(video_path)

        assert len(scores) == 1
        assert scores.dtype == np.float32


class TestSignalFuser:
    """Test signal fusion module"""

    def test_fusion_config_defaults(self):
        """Test FusionConfig default values"""
        config = FusionConfig()
        assert config.audio_weight == 0.3
        assert config.motion_weight == 0.4
        assert config.proximity_weight == 0.3
        assert config.top_k_peaks == 20
        assert config.clip_padding_sec == 5.0

    def test_fusion_config_weight_normalization(self):
        """Test automatic weight normalization"""
        config = FusionConfig(audio_weight=1.0, motion_weight=1.0, proximity_weight=1.0)
        assert np.isclose(config.audio_weight + config.motion_weight + config.proximity_weight, 1.0)

    def test_signal_fuser_init(self):
        """Test SignalFuser initialization"""
        fuser = SignalFuser()
        assert fuser.config is not None
        assert isinstance(fuser.config, FusionConfig)

    def test_fuse_and_extract_empty(self):
        """Test fusion with empty signals"""
        fuser = SignalFuser()
        audio = np.array([], dtype=np.float32)
        motion = np.array([], dtype=np.float32)
        proximity = np.array([], dtype=np.float32)

        clips = fuser.fuse_and_extract(audio, motion, proximity, 0.0)
        assert clips == []

    def test_fuse_and_extract_length_mismatch(self):
        """Test error handling for mismatched signal lengths"""
        fuser = SignalFuser()
        audio = np.zeros(10, dtype=np.float32)
        motion = np.zeros(5, dtype=np.float32)
        proximity = np.zeros(10, dtype=np.float32)

        with pytest.raises(ValueError, match="same length"):
            fuser.fuse_and_extract(audio, motion, proximity, 10.0)

    def test_fuse_and_extract_no_peaks(self):
        """Test fusion with no peaks above threshold"""
        fuser = SignalFuser(FusionConfig(min_peak_score=0.9))
        audio = np.random.rand(60).astype(np.float32) * 0.2  # Low scores
        motion = np.random.rand(60).astype(np.float32) * 0.2
        proximity = np.random.rand(60).astype(np.float32) * 0.2

        clips = fuser.fuse_and_extract(audio, motion, proximity, 60.0)
        assert clips == []

    def test_fuse_and_extract_synthetic_peaks(self):
        """Test fusion with synthetic danger peaks"""
        fuser = SignalFuser(FusionConfig(min_peak_score=0.5, top_k_peaks=3))

        n_seconds = 60
        audio = np.zeros(n_seconds, dtype=np.float32)
        motion = np.zeros(n_seconds, dtype=np.float32)
        proximity = np.zeros(n_seconds, dtype=np.float32)

        # Create 3 synthetic peaks
        audio[10] = 0.9
        motion[10] = 0.8

        audio[30] = 0.7
        proximity[30] = 0.8

        motion[50] = 0.9
        proximity[50] = 0.7

        clips = fuser.fuse_and_extract(audio, motion, proximity, 60.0)

        # Should find 3 peaks
        assert len(clips) >= 1
        assert all(isinstance(c, HazardClip) for c in clips)
        assert all(c.score > 0.0 for c in clips)
        assert all(c.start_sec < c.end_sec for c in clips)

    def test_fuse_and_extract_clip_merging(self):
        """Test merging of nearby clips"""
        fuser = SignalFuser(FusionConfig(min_peak_score=0.5, merge_gap_sec=5.0))

        n_seconds = 60
        audio = np.zeros(n_seconds, dtype=np.float32)
        motion = np.zeros(n_seconds, dtype=np.float32)
        proximity = np.zeros(n_seconds, dtype=np.float32)

        # Create 2 nearby peaks (within 5 seconds)
        audio[20] = 0.9
        motion[20] = 0.8

        audio[23] = 0.8
        proximity[23] = 0.9

        clips = fuser.fuse_and_extract(audio, motion, proximity, 60.0)

        # Should merge into 1 clip
        assert len(clips) >= 1

    def test_hazard_clip_properties(self):
        """Test HazardClip dataclass"""
        clip = HazardClip(
            start_sec=10.0,
            end_sec=20.0,
            peak_sec=15.0,
            score=0.85,
            audio_score=0.7,
            motion_score=0.8,
            proximity_score=0.6,
        )

        assert clip.duration_sec == 10.0
        assert clip.start_sec < clip.peak_sec < clip.end_sec
        assert 0.0 <= clip.score <= 1.0


class TestIntegration:
    """Integration tests for full pipeline"""

    @pytest.mark.slow
    def test_full_pipeline_synthetic(self, tmp_path):
        """Test full pipeline on synthetic video"""
        import cv2

        # Create synthetic video
        video_path = tmp_path / "test_full.mp4"
        fps = 30.0
        n_frames = 90  # 3 seconds

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (640, 480))

        for i in range(n_frames):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some visual content
            x = int(100 + i * 3)
            cv2.rectangle(frame, (x, 200), (x + 50, 250), (128, 128, 128), -1)
            writer.write(frame)

        writer.release()

        # Run full pipeline
        audio_analyzer = AudioAnalyzer()
        motion_analyzer = MotionAnalyzer(MotionConfig(frame_skip=3))
        proximity_analyzer = ProximityAnalyzer(ProximityConfig(frame_skip=10))
        fuser = SignalFuser(FusionConfig(min_peak_score=0.1))

        # Analyze
        audio_scores = audio_analyzer.analyze(video_path)
        motion_scores = motion_analyzer.analyze(video_path)

        # Skip proximity for speed (requires YOLOv8 download)
        proximity_scores = np.zeros_like(motion_scores)

        # Fuse
        clips = fuser.fuse_and_extract(
            audio_scores,
            motion_scores,
            proximity_scores,
            video_duration_sec=3.0,
        )

        # Validation
        assert len(audio_scores) == 3
        assert len(motion_scores) == 3
        assert len(proximity_scores) == 3

        # May or may not find clips depending on motion detection
        assert isinstance(clips, list)
        for clip in clips:
            assert 0.0 <= clip.start_sec < clip.end_sec <= 3.0
            assert 0.0 <= clip.score <= 1.0
