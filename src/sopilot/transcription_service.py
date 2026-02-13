"""Audio transcription service for VIGIL-RAG.

This module provides speech-to-text transcription with timestamps,
enabling audio-based retrieval and evidence in the RAG pipeline.

Supported backends:
- openai-whisper: OpenAI Whisper (local, requires `openai-whisper` package)
- mock: Returns empty segments (no dependencies, for testing)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

TranscriptionBackend = Literal["openai-whisper", "mock"]


@dataclass
class TranscriptionConfig:
    """Configuration for transcription service."""

    backend: TranscriptionBackend = "openai-whisper"
    model_size: str = "base"  # tiny, base, small, medium, large
    device: str = "cuda"  # cuda / cpu
    language: str | None = None  # None = auto-detect
    fp16: bool = True  # Use FP16 (faster on GPU, ignored on CPU)


@dataclass
class TranscriptionSegment:
    """A single transcription segment with timestamps."""

    start_sec: float
    end_sec: float
    text: str
    language: str = "unknown"


@dataclass
class TranscriptionResult:
    """Complete transcription result for a video/audio file."""

    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str = "unknown"
    duration_sec: float = 0.0


class TranscriptionService:
    """Audio transcription service with Whisper backend.

    Usage:
        config = TranscriptionConfig(backend="openai-whisper", model_size="base")
        service = TranscriptionService(config)
        result = service.transcribe(Path("video.mp4"))
        for seg in result.segments:
            print(f"[{seg.start_sec:.1f}-{seg.end_sec:.1f}] {seg.text}")
    """

    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config
        self._model = None

        if config.backend != "mock":
            self._load_model()

    def _load_model(self) -> None:
        """Load Whisper model.

        Raises:
            RuntimeError: If whisper is not installed.
        """
        try:
            import whisper  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "openai-whisper is required for transcription. "
                "Install with: pip install openai-whisper"
            ) from None

        logger.info("Loading Whisper model: %s (device=%s)", self.config.model_size, self.config.device)
        import whisper

        self._model = whisper.load_model(self.config.model_size, device=self.config.device)
        logger.info("Whisper model loaded successfully")

    def transcribe(
        self,
        video_path: Path | str,
        *,
        start_sec: float = 0.0,
        end_sec: float | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio from a video file.

        Args:
            video_path: Path to video (or audio) file.
            start_sec: Start time in seconds (for future clip-level transcription).
            end_sec: End time in seconds (None = entire file).

        Returns:
            TranscriptionResult with time-aligned segments.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If file does not exist.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"File not found: {video_path}")

        if self.config.backend == "mock":
            return self._transcribe_mock(video_path)

        return self._transcribe_whisper(video_path)

    def _transcribe_mock(self, video_path: Path) -> TranscriptionResult:
        """Mock transcription: return empty result."""
        logger.debug("Mock transcription for %s", video_path)
        return TranscriptionResult(segments=[], language="unknown", duration_sec=0.0)

    def _transcribe_whisper(self, video_path: Path) -> TranscriptionResult:
        """Transcribe using Whisper.

        Whisper accepts video files directly (extracts audio via ffmpeg internally).
        """
        if self._model is None:
            raise RuntimeError("Whisper model not loaded")

        logger.info("Transcribing: %s", video_path)

        fp16 = self.config.fp16 and self.config.device != "cpu"

        result = self._model.transcribe(
            str(video_path),
            language=self.config.language,
            fp16=fp16,
            verbose=False,
        )

        detected_language = result.get("language", "unknown")
        segments = []
        for seg in result.get("segments", []):
            segments.append(
                TranscriptionSegment(
                    start_sec=float(seg["start"]),
                    end_sec=float(seg["end"]),
                    text=seg["text"].strip(),
                    language=detected_language,
                )
            )

        # Compute duration from last segment end
        duration_sec = segments[-1].end_sec if segments else 0.0

        logger.info(
            "Transcribed %d segments (%.1f sec, language=%s)",
            len(segments), duration_sec, detected_language,
        )

        return TranscriptionResult(
            segments=segments,
            language=detected_language,
            duration_sec=duration_sec,
        )

    def segments_for_range(
        self,
        segments: list[TranscriptionSegment],
        start_sec: float,
        end_sec: float,
    ) -> list[TranscriptionSegment]:
        """Filter segments that overlap with a time range.

        Args:
            segments: Full list of transcription segments.
            start_sec: Range start.
            end_sec: Range end.

        Returns:
            Segments that overlap [start_sec, end_sec].
        """
        return [
            seg for seg in segments
            if seg.end_sec > start_sec and seg.start_sec < end_sec
        ]

    def segments_to_text(self, segments: list[TranscriptionSegment]) -> str:
        """Join segment texts into a single string.

        Args:
            segments: List of transcription segments.

        Returns:
            Concatenated text with space separation.
        """
        return " ".join(seg.text for seg in segments if seg.text)
