"""Tests for insurance_mvp.insurance.utils — video processing utilities.

Tests cover:
- VideoMetadata class
- extract_video_metadata() with mocked OpenCV
- extract_keyframes() with mocked OpenCV
- format_timestamp() / parse_timestamp() — pure functions
- calculate_frame_difference() with mocked OpenCV
- detect_scene_changes() with mocked OpenCV
- estimate_motion_intensity() with mocked OpenCV
- calculate_video_quality_score() with mocked OpenCV
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from insurance_mvp.insurance.utils import (
    VideoMetadata,
    calculate_frame_difference,
    calculate_video_quality_score,
    detect_scene_changes,
    estimate_motion_intensity,
    extract_keyframes,
    extract_video_metadata,
    format_timestamp,
    parse_timestamp,
)


# ---------------------------------------------------------------------------
# Helpers: reusable mock factories
# ---------------------------------------------------------------------------


def _make_mock_cap(
    *,
    opened: bool = True,
    fps: float = 30.0,
    width: int = 1920,
    height: int = 1080,
    num_frames: int = 300,
    fourcc: int = 0x34363248,  # "H264"
    frames: list[np.ndarray] | None = None,
    read_success: bool = True,
):
    """Build a MagicMock that behaves like cv2.VideoCapture."""
    cap = MagicMock()
    cap.isOpened.return_value = opened

    prop_map = {
        1: fps,          # CAP_PROP_FPS
        3: float(width),  # CAP_PROP_FRAME_WIDTH
        4: float(height),  # CAP_PROP_FRAME_HEIGHT
        7: float(num_frames),  # CAP_PROP_FRAME_COUNT
        6: float(fourcc),  # CAP_PROP_FOURCC
    }
    cap.get.side_effect = lambda prop: prop_map.get(prop, 0.0)

    if frames is not None:
        frame_iter = iter(frames)
        def _read():
            try:
                return True, next(frame_iter)
            except StopIteration:
                return False, None
        cap.read.side_effect = _read
    else:
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cap.read.return_value = (read_success, dummy_frame)

    return cap


# ---------------------------------------------------------------------------
# VideoMetadata
# ---------------------------------------------------------------------------


class TestVideoMetadata:
    """Tests for the VideoMetadata dataclass."""

    def test_basic_properties(self):
        m = VideoMetadata(
            duration_sec=10.0, fps=30.0, width=1920, height=1080,
            num_frames=300, file_size_mb=50.0, codec="h264",
        )
        assert m.duration_sec == 10.0
        assert m.fps == 30.0
        assert m.width == 1920
        assert m.height == 1080
        assert m.num_frames == 300
        assert m.file_size_mb == 50.0
        assert m.codec == "h264"

    def test_default_codec(self):
        m = VideoMetadata(10.0, 30.0, 640, 480, 300, 5.0)
        assert m.codec == "unknown"

    def test_repr(self):
        m = VideoMetadata(10.0, 30.0, 640, 480, 300, 5.0, "h264")
        r = repr(m)
        assert "640x480" in r
        assert "10.0s" in r
        assert "300" in r


# ---------------------------------------------------------------------------
# format_timestamp / parse_timestamp (pure functions)
# ---------------------------------------------------------------------------


class TestFormatTimestamp:

    def test_basic(self):
        assert format_timestamp(0.0) == "00:00:00.00"

    def test_with_minutes(self):
        result = format_timestamp(90.5)
        assert result == "00:01:30.50"

    def test_with_hours(self):
        result = format_timestamp(3661.25)
        assert result == "01:01:01.25"

    def test_no_hours_flag_with_zero_hours(self):
        result = format_timestamp(90.5, include_hours=False)
        assert result == "01:30.50"

    def test_no_hours_flag_but_hours_present(self):
        # When hours > 0, include_hours is effectively forced True
        result = format_timestamp(3661.25, include_hours=False)
        assert result == "01:01:01.25"


class TestParseTimestamp:

    def test_seconds_only(self):
        assert parse_timestamp("42.50") == 42.50

    def test_minutes_seconds(self):
        assert parse_timestamp("01:30.50") == 90.50

    def test_hours_minutes_seconds(self):
        assert parse_timestamp("01:01:01.25") == 3661.25

    def test_whitespace_stripped(self):
        assert parse_timestamp("  42.50  ") == 42.50

    def test_invalid_format_too_many_colons(self):
        with pytest.raises(ValueError, match="Cannot parse timestamp"):
            parse_timestamp("1:2:3:4")

    def test_invalid_non_numeric(self):
        with pytest.raises(ValueError, match="Cannot parse timestamp"):
            parse_timestamp("abc")

    def test_roundtrip(self):
        """format -> parse should be identity (within precision)."""
        for secs in [0.0, 5.5, 61.25, 3723.99]:
            ts = format_timestamp(secs)
            recovered = parse_timestamp(ts)
            assert abs(recovered - secs) < 0.02  # centisecond precision


# ---------------------------------------------------------------------------
# extract_video_metadata (mocked OpenCV)
# ---------------------------------------------------------------------------


class TestExtractVideoMetadata:

    @patch("insurance_mvp.insurance.utils.cv2")
    @patch("insurance_mvp.insurance.utils.Path")
    def test_happy_path(self, mock_path_cls, mock_cv2):
        cap = _make_mock_cap(fps=30.0, width=1920, height=1080, num_frames=300, fourcc=0x34363248)
        mock_cv2.VideoCapture.return_value = cap
        # Wire up cv2 property constants
        mock_cv2.CAP_PROP_FPS = 1
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FOURCC = 6

        # Mock Path(...).exists() and stat()
        mock_file_path = MagicMock()
        mock_file_path.exists.return_value = True
        mock_stat = MagicMock()
        mock_stat.st_size = 50 * 1024 * 1024  # 50 MB
        mock_file_path.stat.return_value = mock_stat
        mock_path_cls.return_value = mock_file_path

        meta = extract_video_metadata("test.mp4")

        assert isinstance(meta, VideoMetadata)
        assert meta.fps == 30.0
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.num_frames == 300
        assert meta.duration_sec == pytest.approx(10.0)
        assert meta.file_size_mb == pytest.approx(50.0)
        cap.release.assert_called_once()

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_cannot_open_video(self, mock_cv2):
        cap = _make_mock_cap(opened=False)
        mock_cv2.VideoCapture.return_value = cap

        with pytest.raises(ValueError, match="Cannot open video"):
            extract_video_metadata("bad.mp4")

    @patch("insurance_mvp.insurance.utils.cv2")
    @patch("insurance_mvp.insurance.utils.Path")
    def test_zero_fps(self, mock_path_cls, mock_cv2):
        cap = _make_mock_cap(fps=0.0, num_frames=100)
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FPS = 1
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FOURCC = 6

        mock_file_path = MagicMock()
        mock_file_path.exists.return_value = False
        mock_path_cls.return_value = mock_file_path

        meta = extract_video_metadata("zero_fps.mp4")
        assert meta.duration_sec == 0.0
        assert meta.file_size_mb == 0.0


# ---------------------------------------------------------------------------
# extract_keyframes (mocked OpenCV)
# ---------------------------------------------------------------------------


class TestExtractKeyframes:

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_happy_path_no_output_dir(self, mock_cv2):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap = _make_mock_cap(fps=30.0)
        cap.read.return_value = (True, dummy_frame)
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FPS = 1
        mock_cv2.CAP_PROP_POS_FRAMES = 2  # needed for .set()

        evidence_list = extract_keyframes("video.mp4", [1.0, 2.5])

        assert len(evidence_list) == 2
        assert evidence_list[0].timestamp_sec == 1.0
        assert evidence_list[1].timestamp_sec == 2.5
        # No output dir -> no frame_path
        assert evidence_list[0].frame_path is None
        cap.release.assert_called_once()

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_cannot_open_video(self, mock_cv2):
        cap = _make_mock_cap(opened=False)
        mock_cv2.VideoCapture.return_value = cap

        with pytest.raises(ValueError, match="Cannot open video"):
            extract_keyframes("bad.mp4", [1.0])

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_failed_frame_read_skipped(self, mock_cv2):
        cap = _make_mock_cap(fps=30.0)
        cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FPS = 1
        mock_cv2.CAP_PROP_POS_FRAMES = 2

        evidence_list = extract_keyframes("video.mp4", [1.0, 2.0])
        assert len(evidence_list) == 0

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_empty_timestamps_returns_empty(self, mock_cv2):
        cap = _make_mock_cap(fps=30.0)
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FPS = 1

        evidence_list = extract_keyframes("video.mp4", [])
        assert evidence_list == []


# ---------------------------------------------------------------------------
# calculate_frame_difference (mocked OpenCV)
# ---------------------------------------------------------------------------


class TestCalculateFrameDifference:

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_identical_frames_zero_diff(self, mock_cv2):
        gray = np.full((100, 100), 128, dtype=np.uint8)
        cap = _make_mock_cap(fps=30.0)
        cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FPS = 1
        mock_cv2.CAP_PROP_POS_FRAMES = 2
        mock_cv2.cvtColor.return_value = gray
        mock_cv2.absdiff.return_value = np.zeros((100, 100), dtype=np.uint8)

        diff = calculate_frame_difference("video.mp4", 0.0, 1.0)
        assert diff == pytest.approx(0.0)

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_cannot_open_video(self, mock_cv2):
        cap = _make_mock_cap(opened=False)
        mock_cv2.VideoCapture.return_value = cap

        with pytest.raises(ValueError, match="Cannot open video"):
            calculate_frame_difference("bad.mp4", 0.0, 1.0)

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_first_frame_read_fails(self, mock_cv2):
        cap = _make_mock_cap(fps=30.0)
        cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FPS = 1
        mock_cv2.CAP_PROP_POS_FRAMES = 2

        with pytest.raises(ValueError, match="Cannot extract frame"):
            calculate_frame_difference("video.mp4", 0.0, 1.0)


# ---------------------------------------------------------------------------
# detect_scene_changes (mocked OpenCV)
# ---------------------------------------------------------------------------


class TestDetectSceneChanges:

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_cannot_open_video(self, mock_cv2):
        cap = _make_mock_cap(opened=False)
        mock_cv2.VideoCapture.return_value = cap

        with pytest.raises(ValueError, match="Cannot open video"):
            detect_scene_changes("bad.mp4")

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_no_scene_changes(self, mock_cv2):
        """All identical frames -> no scene changes detected."""
        gray = np.full((100, 100), 128, dtype=np.uint8)
        cap = _make_mock_cap(fps=30.0, num_frames=30)
        cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FPS = 1
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_POS_FRAMES = 2
        mock_cv2.cvtColor.return_value = gray
        mock_cv2.absdiff.return_value = np.zeros((100, 100), dtype=np.uint8)

        changes = detect_scene_changes("video.mp4", threshold=0.3)
        assert changes == []

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_scene_change_detected(self, mock_cv2):
        """Big pixel difference triggers a scene change."""
        cap = _make_mock_cap(fps=10.0, num_frames=20)
        cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FPS = 1
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_POS_FRAMES = 2

        # Alternate between dark and bright gray frames
        call_count = [0]
        def fake_cvtcolor(frame, code):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return np.full((100, 100), 255, dtype=np.uint8)
            return np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.cvtColor.side_effect = fake_cvtcolor

        # Return high diff every time
        mock_cv2.absdiff.return_value = np.full((100, 100), 200, dtype=np.uint8)

        changes = detect_scene_changes("video.mp4", threshold=0.3, min_gap_sec=0.1)
        assert len(changes) >= 1


# ---------------------------------------------------------------------------
# estimate_motion_intensity (mocked OpenCV)
# ---------------------------------------------------------------------------


class TestEstimateMotionIntensity:

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_cannot_open_video(self, mock_cv2):
        cap = _make_mock_cap(opened=False)
        mock_cv2.VideoCapture.return_value = cap

        with pytest.raises(ValueError, match="Cannot open video"):
            estimate_motion_intensity("bad.mp4", 0.0, 5.0)

    def test_invalid_time_range(self):
        with pytest.raises(ValueError, match="Invalid time range"):
            estimate_motion_intensity("video.mp4", 5.0, 3.0)

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_no_frames_read_returns_zero(self, mock_cv2):
        cap = _make_mock_cap(fps=30.0)
        cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FPS = 1
        mock_cv2.CAP_PROP_POS_FRAMES = 2

        result = estimate_motion_intensity("video.mp4", 0.0, 1.0)
        assert result == 0.0

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_motion_detected(self, mock_cv2):
        cap = _make_mock_cap(fps=30.0)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cap.read.return_value = (True, frame)
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FPS = 1
        mock_cv2.CAP_PROP_POS_FRAMES = 2

        gray = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = gray

        # Simulate non-zero optical flow
        flow = np.ones((100, 100, 2), dtype=np.float32) * 5.0
        mock_cv2.calcOpticalFlowFarneback.return_value = flow

        result = estimate_motion_intensity("video.mp4", 0.0, 1.0, sample_rate=1)
        assert result > 0.0


# ---------------------------------------------------------------------------
# calculate_video_quality_score (mocked OpenCV)
# ---------------------------------------------------------------------------


class TestCalculateVideoQualityScore:

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_cannot_open_video(self, mock_cv2):
        cap = _make_mock_cap(opened=False)
        mock_cv2.VideoCapture.return_value = cap

        with pytest.raises(ValueError, match="Cannot open video"):
            calculate_video_quality_score("bad.mp4")

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_happy_path(self, mock_cv2):
        cap = _make_mock_cap(fps=30.0, num_frames=100, width=640, height=480)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cap.read.return_value = (True, frame)
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_POS_FRAMES = 2
        mock_cv2.CV_64F = 6

        gray = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = gray
        mock_cv2.Laplacian.return_value = np.random.randn(480, 640).astype(np.float64)

        metrics = calculate_video_quality_score("video.mp4", num_samples=3)

        assert "brightness" in metrics
        assert "contrast" in metrics
        assert "sharpness" in metrics
        assert "overall_score" in metrics
        assert metrics["brightness"] >= 0.0
        assert metrics["contrast"] >= 0.0

    @patch("insurance_mvp.insurance.utils.cv2")
    def test_no_frames_returns_zero_metrics(self, mock_cv2):
        cap = _make_mock_cap(fps=30.0, num_frames=100)
        cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_POS_FRAMES = 2

        metrics = calculate_video_quality_score("video.mp4", num_samples=5)
        assert metrics["brightness"] == 0.0
        assert metrics["contrast"] == 0.0
        assert metrics["sharpness"] == 0.0
