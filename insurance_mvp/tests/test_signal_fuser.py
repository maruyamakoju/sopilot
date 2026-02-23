"""Tests for insurance_mvp.mining.fuse — signal fusion and peak detection.

Tests cover:
- FusionConfig: defaults, weight normalization, custom parameters
- HazardClip: dataclass, duration_sec property
- SignalFuser: fuse_and_extract, _find_peaks, _merge_nearby_clips
- Edge cases: empty signals, length mismatch, no peaks, boundary padding
- extract_danger_clips without analyzers -> RuntimeError
- Top-K selection
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from insurance_mvp.mining.fuse import (
    FusionConfig,
    HazardClip,
    SignalFuser,
)


# ---------------------------------------------------------------------------
# FusionConfig
# ---------------------------------------------------------------------------


class TestFusionConfig:

    def test_defaults(self):
        config = FusionConfig()
        assert config.audio_weight == pytest.approx(0.3)
        assert config.motion_weight == pytest.approx(0.4)
        assert config.proximity_weight == pytest.approx(0.3)
        total = config.audio_weight + config.motion_weight + config.proximity_weight
        assert total == pytest.approx(1.0)

    def test_weight_normalization(self):
        """Weights that do not sum to 1.0 are auto-normalized."""
        config = FusionConfig(audio_weight=2.0, motion_weight=2.0, proximity_weight=1.0)
        total = config.audio_weight + config.motion_weight + config.proximity_weight
        assert total == pytest.approx(1.0)
        assert config.audio_weight == pytest.approx(0.4)
        assert config.motion_weight == pytest.approx(0.4)
        assert config.proximity_weight == pytest.approx(0.2)

    def test_equal_weights_normalize(self):
        config = FusionConfig(audio_weight=1.0, motion_weight=1.0, proximity_weight=1.0)
        assert config.audio_weight == pytest.approx(1.0 / 3)

    def test_custom_parameters(self):
        config = FusionConfig(top_k_peaks=5, min_peak_score=0.5, clip_padding_sec=3.0)
        assert config.top_k_peaks == 5
        assert config.min_peak_score == 0.5
        assert config.clip_padding_sec == 3.0


# ---------------------------------------------------------------------------
# HazardClip
# ---------------------------------------------------------------------------


class TestHazardClip:

    def test_basic_properties(self):
        clip = HazardClip(
            start_sec=5.0, end_sec=15.0, peak_sec=10.0, score=0.9,
            audio_score=0.8, motion_score=0.7, proximity_score=0.6,
        )
        assert clip.start_sec == 5.0
        assert clip.end_sec == 15.0
        assert clip.peak_sec == 10.0
        assert clip.score == 0.9

    def test_duration_sec(self):
        clip = HazardClip(start_sec=3.0, end_sec=10.0, peak_sec=6.0, score=0.5)
        assert clip.duration_sec == pytest.approx(7.0)

    def test_zero_duration(self):
        clip = HazardClip(start_sec=5.0, end_sec=5.0, peak_sec=5.0, score=0.5)
        assert clip.duration_sec == pytest.approx(0.0)

    def test_default_scores(self):
        clip = HazardClip(start_sec=0.0, end_sec=10.0, peak_sec=5.0, score=0.5)
        assert clip.audio_score == 0.0
        assert clip.motion_score == 0.0
        assert clip.proximity_score == 0.0


# ---------------------------------------------------------------------------
# SignalFuser — fuse_and_extract
# ---------------------------------------------------------------------------


class TestSignalFuserFuseAndExtract:

    def test_empty_signals_return_empty(self):
        fuser = SignalFuser()
        audio = np.array([], dtype=np.float32)
        motion = np.array([], dtype=np.float32)
        proximity = np.array([], dtype=np.float32)

        clips = fuser.fuse_and_extract(audio, motion, proximity, 0.0)
        assert clips == []

    def test_length_mismatch_raises(self):
        fuser = SignalFuser()
        audio = np.zeros(10, dtype=np.float32)
        motion = np.zeros(5, dtype=np.float32)
        proximity = np.zeros(10, dtype=np.float32)

        with pytest.raises(ValueError, match="same length"):
            fuser.fuse_and_extract(audio, motion, proximity, 10.0)

    def test_no_peaks_above_threshold(self):
        """All signals well below min_peak_score -> empty result."""
        fuser = SignalFuser(FusionConfig(min_peak_score=0.9))
        n = 60
        audio = np.random.RandomState(42).rand(n).astype(np.float32) * 0.1
        motion = np.random.RandomState(43).rand(n).astype(np.float32) * 0.1
        proximity = np.random.RandomState(44).rand(n).astype(np.float32) * 0.1

        clips = fuser.fuse_and_extract(audio, motion, proximity, 60.0)
        assert clips == []

    def test_single_peak_detection(self):
        """One clear spike should produce exactly one clip."""
        config = FusionConfig(min_peak_score=0.3, top_k_peaks=10, clip_padding_sec=5.0)
        fuser = SignalFuser(config)

        n = 60
        audio = np.zeros(n, dtype=np.float32)
        motion = np.zeros(n, dtype=np.float32)
        proximity = np.zeros(n, dtype=np.float32)

        # Single spike at t=30
        audio[30] = 1.0
        motion[30] = 1.0
        proximity[30] = 1.0

        clips = fuser.fuse_and_extract(audio, motion, proximity, 60.0)
        assert len(clips) == 1
        assert clips[0].peak_sec == pytest.approx(30.0)
        assert clips[0].score > 0.3
        assert clips[0].start_sec >= 0.0
        assert clips[0].end_sec <= 60.0

    def test_multiple_peaks(self):
        """Three well-separated spikes should produce three clips."""
        config = FusionConfig(
            min_peak_score=0.3, top_k_peaks=10,
            min_peak_distance=3, clip_padding_sec=2.0, merge_gap_sec=0.5,
        )
        fuser = SignalFuser(config)

        n = 60
        audio = np.zeros(n, dtype=np.float32)
        motion = np.zeros(n, dtype=np.float32)
        proximity = np.zeros(n, dtype=np.float32)

        for t in [10, 30, 50]:
            audio[t] = 0.9
            motion[t] = 0.8
            proximity[t] = 0.7

        clips = fuser.fuse_and_extract(audio, motion, proximity, 60.0)
        assert len(clips) == 3
        # Sorted by score descending
        assert all(clips[i].score >= clips[i + 1].score for i in range(len(clips) - 1))

    def test_clip_padding_at_boundaries(self):
        """Peak at t=0 should clip start at 0, peak at end should clip to duration."""
        config = FusionConfig(min_peak_score=0.3, clip_padding_sec=5.0, min_peak_distance=10)
        fuser = SignalFuser(config)

        n = 30
        audio = np.zeros(n, dtype=np.float32)
        motion = np.zeros(n, dtype=np.float32)
        proximity = np.zeros(n, dtype=np.float32)

        # Peaks at boundaries
        audio[2] = 1.0
        motion[2] = 1.0
        audio[27] = 1.0
        motion[27] = 1.0

        clips = fuser.fuse_and_extract(audio, motion, proximity, 30.0)
        assert len(clips) >= 1
        # All clips should respect boundaries
        for clip in clips:
            assert clip.start_sec >= 0.0
            assert clip.end_sec <= 30.0

    def test_top_k_selection(self):
        """With top_k_peaks=2, at most 2 clips returned."""
        config = FusionConfig(min_peak_score=0.2, top_k_peaks=2, min_peak_distance=3, clip_padding_sec=1.0, merge_gap_sec=0.1)
        fuser = SignalFuser(config)

        n = 60
        audio = np.zeros(n, dtype=np.float32)
        motion = np.zeros(n, dtype=np.float32)
        proximity = np.zeros(n, dtype=np.float32)

        for t in [10, 25, 40, 55]:
            audio[t] = 0.8
            motion[t] = 0.9

        clips = fuser.fuse_and_extract(audio, motion, proximity, 60.0)
        assert len(clips) <= 2

    def test_min_peak_score_filtering(self):
        """Peaks below min_peak_score are ignored."""
        config = FusionConfig(min_peak_score=0.7, top_k_peaks=10)
        fuser = SignalFuser(config)

        n = 40
        audio = np.zeros(n, dtype=np.float32)
        motion = np.zeros(n, dtype=np.float32)
        proximity = np.zeros(n, dtype=np.float32)

        # Low peak at t=10 (fused ~ 0.3*0.5 + 0.4*0.5 + 0.3*0.5 = 0.5 < 0.7)
        audio[10] = 0.5
        motion[10] = 0.5
        proximity[10] = 0.5

        clips = fuser.fuse_and_extract(audio, motion, proximity, 40.0)
        assert clips == []


# ---------------------------------------------------------------------------
# SignalFuser — _merge_nearby_clips
# ---------------------------------------------------------------------------


class TestMergeNearbyClips:

    def test_single_clip_unchanged(self):
        fuser = SignalFuser()
        clips = [HazardClip(5.0, 15.0, 10.0, 0.8)]
        merged = fuser._merge_nearby_clips(clips)
        assert len(merged) == 1
        assert merged[0].score == 0.8

    def test_two_overlapping_clips_merged(self):
        config = FusionConfig(merge_gap_sec=3.0)
        fuser = SignalFuser(config)
        clips = [
            HazardClip(5.0, 12.0, 8.0, 0.7, audio_score=0.5),
            HazardClip(13.0, 20.0, 16.0, 0.9, audio_score=0.8),
        ]
        merged = fuser._merge_nearby_clips(clips)
        assert len(merged) == 1
        # Merged clip takes max score and widest range
        assert merged[0].score == 0.9
        assert merged[0].start_sec == 5.0
        assert merged[0].end_sec == 20.0
        assert merged[0].audio_score == 0.8

    def test_two_distant_clips_not_merged(self):
        config = FusionConfig(merge_gap_sec=1.0)
        fuser = SignalFuser(config)
        clips = [
            HazardClip(0.0, 5.0, 2.0, 0.8),
            HazardClip(20.0, 25.0, 22.0, 0.7),
        ]
        merged = fuser._merge_nearby_clips(clips)
        assert len(merged) == 2

    def test_empty_list(self):
        fuser = SignalFuser()
        merged = fuser._merge_nearby_clips([])
        assert merged == []


# ---------------------------------------------------------------------------
# SignalFuser — extract_danger_clips error
# ---------------------------------------------------------------------------


class TestExtractDangerClipsError:

    def test_no_analyzers_raises(self):
        fuser = SignalFuser()
        with pytest.raises(RuntimeError, match="Analyzers not set"):
            fuser.extract_danger_clips("video.mp4")


# ---------------------------------------------------------------------------
# SignalFuser — MiningConfig-style init
# ---------------------------------------------------------------------------


class TestSignalFuserMiningConfigInit:

    def test_accepts_arbitrary_config_object(self):
        """SignalFuser should accept a non-FusionConfig object with compatible attrs."""
        mock_config = MagicMock()
        mock_config.audio_weight = 0.5
        mock_config.motion_weight = 0.3
        mock_config.proximity_weight = 0.2
        mock_config.top_k_clips = 10

        fuser = SignalFuser(config=mock_config)
        assert isinstance(fuser.config, FusionConfig)
        total = fuser.config.audio_weight + fuser.config.motion_weight + fuser.config.proximity_weight
        assert total == pytest.approx(1.0)

    def test_none_config_uses_defaults(self):
        fuser = SignalFuser(config=None)
        assert isinstance(fuser.config, FusionConfig)
        assert fuser.config.audio_weight == pytest.approx(0.3)
