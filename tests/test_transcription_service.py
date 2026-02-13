"""Tests for TranscriptionService."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sopilot.transcription_service import (
    TranscriptionConfig,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionService,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dummy_file(suffix=".mp4") -> Path:
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(b"\x00" * 64)
    return Path(tmp.name)


def _make_mock_service() -> TranscriptionService:
    """Create a mock TranscriptionService."""
    config = TranscriptionConfig(backend="mock")
    return TranscriptionService(config)


def _make_sample_segments() -> list[TranscriptionSegment]:
    """Create sample segments for testing helper methods."""
    return [
        TranscriptionSegment(start_sec=0.0, end_sec=2.5, text="Hello world", language="en"),
        TranscriptionSegment(start_sec=2.5, end_sec=5.0, text="This is a test", language="en"),
        TranscriptionSegment(start_sec=5.0, end_sec=8.0, text="of the transcription", language="en"),
        TranscriptionSegment(start_sec=8.0, end_sec=10.0, text="service module", language="en"),
    ]


# ---------------------------------------------------------------------------
# TranscriptionConfig tests
# ---------------------------------------------------------------------------

class TestTranscriptionConfig:
    def test_defaults(self):
        config = TranscriptionConfig()
        assert config.backend == "openai-whisper"
        assert config.model_size == "base"
        assert config.device == "cuda"
        assert config.language is None
        assert config.fp16 is True

    def test_custom_values(self):
        config = TranscriptionConfig(
            backend="mock",
            model_size="large",
            device="cpu",
            language="ja",
            fp16=False,
        )
        assert config.backend == "mock"
        assert config.model_size == "large"
        assert config.device == "cpu"
        assert config.language == "ja"
        assert config.fp16 is False


# ---------------------------------------------------------------------------
# TranscriptionSegment tests
# ---------------------------------------------------------------------------

class TestTranscriptionSegment:
    def test_create(self):
        seg = TranscriptionSegment(start_sec=1.0, end_sec=3.5, text="hello", language="en")
        assert seg.start_sec == 1.0
        assert seg.end_sec == 3.5
        assert seg.text == "hello"
        assert seg.language == "en"

    def test_defaults(self):
        seg = TranscriptionSegment(start_sec=0.0, end_sec=1.0, text="test")
        assert seg.language == "unknown"


# ---------------------------------------------------------------------------
# TranscriptionResult tests
# ---------------------------------------------------------------------------

class TestTranscriptionResult:
    def test_empty(self):
        result = TranscriptionResult()
        assert result.segments == []
        assert result.language == "unknown"
        assert result.duration_sec == 0.0

    def test_with_segments(self):
        segs = _make_sample_segments()
        result = TranscriptionResult(segments=segs, language="en", duration_sec=10.0)
        assert len(result.segments) == 4
        assert result.language == "en"
        assert result.duration_sec == 10.0


# ---------------------------------------------------------------------------
# TranscriptionService: Mock backend
# ---------------------------------------------------------------------------

class TestTranscriptionServiceMock:
    def test_mock_init_no_deps(self):
        """Mock backend should not require whisper."""
        service = _make_mock_service()
        assert service.config.backend == "mock"
        assert service._model is None

    def test_mock_transcribe(self):
        service = _make_mock_service()
        dummy = _make_dummy_file()
        try:
            result = service.transcribe(dummy)
            assert isinstance(result, TranscriptionResult)
            assert result.segments == []
            assert result.language == "unknown"
        finally:
            dummy.unlink(missing_ok=True)

    def test_file_not_found(self):
        service = _make_mock_service()
        with pytest.raises(ValueError, match="File not found"):
            service.transcribe(Path("/nonexistent/video.mp4"))


# ---------------------------------------------------------------------------
# TranscriptionService: Whisper backend (mocked import)
# ---------------------------------------------------------------------------

class TestTranscriptionServiceWhisper:
    def test_missing_whisper_raises(self):
        """Should raise RuntimeError if whisper not installed."""
        config = TranscriptionConfig(backend="openai-whisper")
        with patch.dict("sys.modules", {"whisper": None}):
            with pytest.raises(RuntimeError, match="openai-whisper is required"):
                TranscriptionService(config)

    def test_whisper_transcribe(self):
        """Test Whisper transcription with mocked model."""
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": " Hello world"},
                {"start": 2.5, "end": 5.0, "text": " Testing"},
            ],
        }

        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            config = TranscriptionConfig(backend="openai-whisper", device="cpu")
            service = TranscriptionService(config)

            dummy = _make_dummy_file()
            try:
                result = service.transcribe(dummy)
                assert len(result.segments) == 2
                assert result.segments[0].text == "Hello world"
                assert result.segments[1].text == "Testing"
                assert result.language == "en"
                assert result.duration_sec == 5.0
            finally:
                dummy.unlink(missing_ok=True)

    def test_whisper_fp16_disabled_on_cpu(self):
        """FP16 should be disabled when device is CPU."""
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {"language": "en", "segments": []}

        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            config = TranscriptionConfig(backend="openai-whisper", device="cpu", fp16=True)
            service = TranscriptionService(config)

            dummy = _make_dummy_file()
            try:
                service.transcribe(dummy)
                # fp16 should be False when device=cpu
                call_kwargs = mock_model.transcribe.call_args[1]
                assert call_kwargs["fp16"] is False
            finally:
                dummy.unlink(missing_ok=True)

    def test_whisper_empty_segments(self):
        """Handle empty transcription result."""
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {"language": "ja", "segments": []}

        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            config = TranscriptionConfig(backend="openai-whisper", device="cpu")
            service = TranscriptionService(config)

            dummy = _make_dummy_file()
            try:
                result = service.transcribe(dummy)
                assert result.segments == []
                assert result.language == "ja"
                assert result.duration_sec == 0.0
            finally:
                dummy.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------

class TestSegmentsForRange:
    def test_full_overlap(self):
        service = _make_mock_service()
        segs = _make_sample_segments()
        result = service.segments_for_range(segs, 0.0, 10.0)
        assert len(result) == 4

    def test_partial_overlap(self):
        service = _make_mock_service()
        segs = _make_sample_segments()
        # [3.0, 6.0] should overlap with seg[1] (2.5-5.0) and seg[2] (5.0-8.0)
        result = service.segments_for_range(segs, 3.0, 6.0)
        assert len(result) == 2
        assert result[0].text == "This is a test"
        assert result[1].text == "of the transcription"

    def test_no_overlap(self):
        service = _make_mock_service()
        segs = _make_sample_segments()
        result = service.segments_for_range(segs, 20.0, 30.0)
        assert result == []

    def test_boundary_exact(self):
        service = _make_mock_service()
        segs = _make_sample_segments()
        # [2.5, 5.0] — seg[0] ends at 2.5 (not overlapping), seg[1] starts at 2.5 (overlapping)
        result = service.segments_for_range(segs, 2.5, 5.0)
        assert len(result) == 1
        assert result[0].text == "This is a test"

    def test_empty_segments(self):
        service = _make_mock_service()
        result = service.segments_for_range([], 0.0, 10.0)
        assert result == []


class TestSegmentsToText:
    def test_join(self):
        service = _make_mock_service()
        segs = _make_sample_segments()
        text = service.segments_to_text(segs)
        assert text == "Hello world This is a test of the transcription service module"

    def test_empty(self):
        service = _make_mock_service()
        assert service.segments_to_text([]) == ""

    def test_skip_empty_text(self):
        service = _make_mock_service()
        segs = [
            TranscriptionSegment(start_sec=0.0, end_sec=1.0, text="hello"),
            TranscriptionSegment(start_sec=1.0, end_sec=2.0, text=""),
            TranscriptionSegment(start_sec=2.0, end_sec=3.0, text="world"),
        ]
        assert service.segments_to_text(segs) == "hello world"
