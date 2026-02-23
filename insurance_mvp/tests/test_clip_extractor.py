"""Tests for insurance_mvp.mining.clip_extractor — ffmpeg-based clip extraction."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from insurance_mvp.mining.clip_extractor import ClipExtractor, ClipExtractorConfig


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestClipExtractorConfig:
    def test_defaults(self):
        cfg = ClipExtractorConfig()
        assert cfg.output_dir == "extracted_clips"
        assert cfg.codec == "copy"
        assert cfg.padding_sec == 1.0
        assert cfg.max_clips == 20

    def test_custom_values(self):
        cfg = ClipExtractorConfig(output_dir="/tmp/clips", codec="libx264", padding_sec=2.5, max_clips=10)
        assert cfg.output_dir == "/tmp/clips"
        assert cfg.codec == "libx264"
        assert cfg.padding_sec == 2.5
        assert cfg.max_clips == 10


# ---------------------------------------------------------------------------
# ClipExtractor — extract_clip
# ---------------------------------------------------------------------------


class TestExtractClip:
    """Test single clip extraction builds correct ffmpeg command."""

    @patch("insurance_mvp.mining.clip_extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("insurance_mvp.mining.clip_extractor.subprocess.run")
    def test_builds_correct_ffmpeg_command(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        extractor = ClipExtractor(ClipExtractorConfig(padding_sec=1.0, codec="copy"))

        result = extractor.extract_clip("video.mp4", 10.0, 20.0, output_path="out.mp4")

        assert result == "out.mp4"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/ffmpeg"
        assert "-y" in cmd
        # Padded start: 10.0 - 1.0 = 9.0
        ss_idx = cmd.index("-ss")
        assert cmd[ss_idx + 1] == "9.000"
        # Padded end: 20.0 + 1.0 = 21.0
        to_idx = cmd.index("-to")
        assert cmd[to_idx + 1] == "21.000"
        # Codec
        c_idx = cmd.index("-c")
        assert cmd[c_idx + 1] == "copy"
        # Input
        i_idx = cmd.index("-i")
        assert cmd[i_idx + 1] == "video.mp4"

    @patch("insurance_mvp.mining.clip_extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("insurance_mvp.mining.clip_extractor.subprocess.run")
    def test_padding_clamps_start_to_zero(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        extractor = ClipExtractor(ClipExtractorConfig(padding_sec=5.0))

        extractor.extract_clip("video.mp4", 2.0, 8.0, output_path="out.mp4")

        cmd = mock_run.call_args[0][0]
        ss_idx = cmd.index("-ss")
        # 2.0 - 5.0 = -3.0 clamped to 0.0
        assert cmd[ss_idx + 1] == "0.000"
        to_idx = cmd.index("-to")
        # 8.0 + 5.0 = 13.0
        assert cmd[to_idx + 1] == "13.000"

    @patch("insurance_mvp.mining.clip_extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("insurance_mvp.mining.clip_extractor.subprocess.run")
    def test_auto_generates_output_path(self, mock_run, mock_which, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        out_dir = str(tmp_path / "clips")
        extractor = ClipExtractor(ClipExtractorConfig(output_dir=out_dir, padding_sec=0.0))

        result = extractor.extract_clip("dashcam001.mp4", 5.0, 10.0)

        assert result is not None
        assert "dashcam001" in result
        assert "5.0s" in result
        assert "10.0s" in result
        assert result.endswith(".mp4")

    @patch("insurance_mvp.mining.clip_extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("insurance_mvp.mining.clip_extractor.subprocess.run")
    def test_returns_none_on_ffmpeg_failure(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=1, stderr="Error: something went wrong")
        extractor = ClipExtractor()

        result = extractor.extract_clip("video.mp4", 0.0, 5.0, output_path="out.mp4")
        assert result is None

    @patch("insurance_mvp.mining.clip_extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("insurance_mvp.mining.clip_extractor.subprocess.run")
    def test_returns_none_on_timeout(self, mock_run, mock_which):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120)
        extractor = ClipExtractor()

        result = extractor.extract_clip("video.mp4", 0.0, 5.0, output_path="out.mp4")
        assert result is None

    @patch("insurance_mvp.mining.clip_extractor.shutil.which", return_value=None)
    def test_returns_none_when_ffmpeg_missing(self, mock_which):
        extractor = ClipExtractor()
        assert extractor._ffmpeg_path is None

        result = extractor.extract_clip("video.mp4", 0.0, 5.0, output_path="out.mp4")
        assert result is None

    @patch("insurance_mvp.mining.clip_extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("insurance_mvp.mining.clip_extractor.subprocess.run")
    def test_file_not_found_disables_ffmpeg(self, mock_run, mock_which):
        """If subprocess.run raises FileNotFoundError, ffmpeg path is cleared."""
        mock_run.side_effect = FileNotFoundError("No such file")
        extractor = ClipExtractor()
        assert extractor._ffmpeg_path == "/usr/bin/ffmpeg"

        result = extractor.extract_clip("video.mp4", 0.0, 5.0, output_path="out.mp4")
        assert result is None
        assert extractor._ffmpeg_path is None

    @patch("insurance_mvp.mining.clip_extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("insurance_mvp.mining.clip_extractor.subprocess.run")
    def test_codec_libx264(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        extractor = ClipExtractor(ClipExtractorConfig(codec="libx264"))

        extractor.extract_clip("video.mp4", 0.0, 5.0, output_path="out.mp4")

        cmd = mock_run.call_args[0][0]
        c_idx = cmd.index("-c")
        assert cmd[c_idx + 1] == "libx264"


# ---------------------------------------------------------------------------
# ClipExtractor — extract_clips (batch)
# ---------------------------------------------------------------------------


class TestExtractClips:
    """Test batch clip extraction."""

    @patch("insurance_mvp.mining.clip_extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("insurance_mvp.mining.clip_extractor.subprocess.run")
    def test_processes_all_clips_and_adds_extracted_path(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        extractor = ClipExtractor(ClipExtractorConfig(padding_sec=0.0))

        clips = [
            {"clip_id": "c0", "start_sec": 0.0, "end_sec": 5.0, "danger_score": 0.9},
            {"clip_id": "c1", "start_sec": 10.0, "end_sec": 15.0, "danger_score": 0.7},
            {"clip_id": "c2", "start_sec": 20.0, "end_sec": 25.0, "danger_score": 0.5},
        ]

        result = extractor.extract_clips("video.mp4", clips)

        assert len(result) == 3
        for clip in result:
            assert "extracted_path" in clip
            assert clip["extracted_path"] is not None
        assert mock_run.call_count == 3

    @patch("insurance_mvp.mining.clip_extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("insurance_mvp.mining.clip_extractor.subprocess.run")
    def test_max_clips_limit(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        extractor = ClipExtractor(ClipExtractorConfig(max_clips=2))

        clips = [
            {"clip_id": f"c{i}", "start_sec": i * 10.0, "end_sec": i * 10.0 + 5.0}
            for i in range(5)
        ]

        result = extractor.extract_clips("video.mp4", clips)

        assert len(result) == 5
        # First 2 should have extracted paths
        assert result[0]["extracted_path"] is not None
        assert result[1]["extracted_path"] is not None
        # Remaining should be None (beyond max_clips)
        assert result[2]["extracted_path"] is None
        assert result[3]["extracted_path"] is None
        assert result[4]["extracted_path"] is None
        # Only 2 ffmpeg calls
        assert mock_run.call_count == 2

    @patch("insurance_mvp.mining.clip_extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("insurance_mvp.mining.clip_extractor.subprocess.run")
    def test_output_dir_override(self, mock_run, mock_which, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        extractor = ClipExtractor(ClipExtractorConfig(output_dir="default_dir"))

        override_dir = str(tmp_path / "override")
        clips = [{"clip_id": "c0", "start_sec": 0.0, "end_sec": 5.0}]

        extractor.extract_clips("video.mp4", clips, output_dir=override_dir)

        # Output path should use override dir
        cmd = mock_run.call_args[0][0]
        output_arg = cmd[-1]
        assert "override" in output_arg

        # Config should be restored after call
        assert extractor.config.output_dir == "default_dir"

    @patch("insurance_mvp.mining.clip_extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("insurance_mvp.mining.clip_extractor.subprocess.run")
    def test_empty_clips_list(self, mock_run, mock_which):
        extractor = ClipExtractor()
        result = extractor.extract_clips("video.mp4", [])
        assert result == []
        mock_run.assert_not_called()

    @patch("insurance_mvp.mining.clip_extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("insurance_mvp.mining.clip_extractor.subprocess.run")
    def test_preserves_existing_clip_keys(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        extractor = ClipExtractor()

        clips = [
            {
                "clip_id": "c0",
                "start_sec": 0.0,
                "end_sec": 5.0,
                "danger_score": 0.9,
                "video_path": "dashcam.mp4",
                "motion_score": 0.8,
            }
        ]

        result = extractor.extract_clips("dashcam.mp4", clips)

        assert result[0]["clip_id"] == "c0"
        assert result[0]["danger_score"] == 0.9
        assert result[0]["motion_score"] == 0.8
        assert "extracted_path" in result[0]

    @patch("insurance_mvp.mining.clip_extractor.shutil.which", return_value=None)
    def test_all_clips_none_when_ffmpeg_missing(self, mock_which):
        extractor = ClipExtractor()
        clips = [
            {"clip_id": "c0", "start_sec": 0.0, "end_sec": 5.0},
            {"clip_id": "c1", "start_sec": 10.0, "end_sec": 15.0},
        ]

        result = extractor.extract_clips("video.mp4", clips)

        assert len(result) == 2
        assert result[0]["extracted_path"] is None
        assert result[1]["extracted_path"] is None


# ---------------------------------------------------------------------------
# Padding calculation
# ---------------------------------------------------------------------------


class TestPaddingCalculation:
    @patch("insurance_mvp.mining.clip_extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("insurance_mvp.mining.clip_extractor.subprocess.run")
    def test_zero_padding(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        extractor = ClipExtractor(ClipExtractorConfig(padding_sec=0.0))

        extractor.extract_clip("video.mp4", 10.0, 20.0, output_path="out.mp4")

        cmd = mock_run.call_args[0][0]
        ss_idx = cmd.index("-ss")
        assert cmd[ss_idx + 1] == "10.000"
        to_idx = cmd.index("-to")
        assert cmd[to_idx + 1] == "20.000"

    @patch("insurance_mvp.mining.clip_extractor.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("insurance_mvp.mining.clip_extractor.subprocess.run")
    def test_large_padding(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        extractor = ClipExtractor(ClipExtractorConfig(padding_sec=10.0))

        extractor.extract_clip("video.mp4", 3.0, 7.0, output_path="out.mp4")

        cmd = mock_run.call_args[0][0]
        ss_idx = cmd.index("-ss")
        # 3.0 - 10.0 = -7.0 clamped to 0.0
        assert cmd[ss_idx + 1] == "0.000"
        to_idx = cmd.index("-to")
        # 7.0 + 10.0 = 17.0
        assert cmd[to_idx + 1] == "17.000"
