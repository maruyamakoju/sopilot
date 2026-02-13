"""Tests for VIGIL-RAG helper utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from sopilot.chunking_service import Chunk, ChunkingResult
from sopilot.transcription_service import (
    TranscriptionConfig,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionService,
)
from sopilot.vigil_helpers import (
    chunk_to_clip_record,
    chunking_result_to_clip_records,
    compute_video_checksum,
    index_video_micro,
)


class TestComputeChecksum:
    """Tests for checksum computation."""

    def test_compute_checksum_basic(self):
        """Test checksum computation for a file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = Path(f.name)

        try:
            checksum = compute_video_checksum(temp_path)
            assert isinstance(checksum, str)
            assert len(checksum) == 64  # SHA-256 hex is 64 characters
            # Verify deterministic (same content -> same checksum)
            checksum2 = compute_video_checksum(temp_path)
            assert checksum == checksum2
        finally:
            temp_path.unlink()

    def test_compute_checksum_empty_file(self):
        """Test checksum computation for empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)

        try:
            checksum = compute_video_checksum(temp_path)
            assert isinstance(checksum, str)
            assert len(checksum) == 64
            # SHA-256 of empty string
            assert checksum == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        finally:
            temp_path.unlink()


class TestChunkToClipRecord:
    """Tests for chunk to clip record conversion."""

    def test_chunk_to_clip_record_basic(self):
        """Test basic chunk to record conversion."""
        chunk = Chunk(
            level="shot",
            start_sec=0.0,
            end_sec=2.5,
            start_frame=0,
            end_frame=75,
            keyframe_indices=[37],
        )
        video_id = "test-video-123"

        record = chunk_to_clip_record(chunk, video_id)

        assert record["video_id"] == video_id
        assert record["level"] == "shot"
        assert record["start_sec"] == 0.0
        assert record["end_sec"] == 2.5
        assert record["keyframe_paths"] is None  # No keyframe_dir provided
        assert record["transcript_text"] is None
        assert record["embedding_id"] is None
        assert record["clip_idx"] is None

    def test_chunk_to_clip_record_with_keyframes(self):
        """Test chunk conversion with keyframe directory."""
        chunk = Chunk(
            level="micro",
            start_sec=0.0,
            end_sec=3.0,
            start_frame=0,
            end_frame=90,
            keyframe_indices=[30, 60],
        )
        video_id = "test-video-456"
        keyframe_dir = Path("/tmp/keyframes")

        record = chunk_to_clip_record(chunk, video_id, keyframe_dir)

        assert record["keyframe_paths"] is not None
        assert len(record["keyframe_paths"]) == 2
        assert "micro_frame_00000030.jpg" in record["keyframe_paths"][0]
        assert "micro_frame_00000060.jpg" in record["keyframe_paths"][1]


class TestChunkingResultToClipRecords:
    """Tests for chunking result to records conversion."""

    def test_chunking_result_to_clip_records(self):
        """Test converting full chunking result to records."""
        shot1 = Chunk(
            level="shot",
            start_sec=0.0,
            end_sec=2.0,
            start_frame=0,
            end_frame=60,
            keyframe_indices=[30],
        )
        shot2 = Chunk(
            level="shot",
            start_sec=2.0,
            end_sec=4.0,
            start_frame=60,
            end_frame=120,
            keyframe_indices=[90],
        )
        micro1 = Chunk(
            level="micro",
            start_sec=0.0,
            end_sec=3.0,
            start_frame=0,
            end_frame=90,
            keyframe_indices=[30, 60],
        )
        meso1 = Chunk(
            level="meso",
            start_sec=0.0,
            end_sec=10.0,
            start_frame=0,
            end_frame=300,
            keyframe_indices=[60, 150, 240],
        )
        macro1 = Chunk(
            level="macro",
            start_sec=0.0,
            end_sec=32.0,
            start_frame=0,
            end_frame=960,
            keyframe_indices=[120, 240, 360, 480, 600, 720, 840],
        )

        result = ChunkingResult(
            shots=[shot1, shot2],
            micro=[micro1],
            meso=[meso1],
            macro=[macro1],
            video_fps=30.0,
            video_duration_sec=60.0,
            total_frames=1800,
        )

        video_id = "test-video-789"
        records = chunking_result_to_clip_records(result, video_id)

        # Should have 2 shots + 1 micro + 1 meso + 1 macro = 5 records
        assert len(records) == 5

        # Verify levels
        levels = [r["level"] for r in records]
        assert levels.count("shot") == 2
        assert levels.count("micro") == 1
        assert levels.count("meso") == 1
        assert levels.count("macro") == 1

        # Verify all records have the same video_id
        assert all(r["video_id"] == video_id for r in records)

    def test_chunking_result_empty(self):
        """Test with empty chunking result."""
        result = ChunkingResult(
            shots=[],
            micro=[],
            meso=[],
            macro=[],
            video_fps=30.0,
            video_duration_sec=0.0,
            total_frames=0,
        )

        records = chunking_result_to_clip_records(result, "test-video")
        assert len(records) == 0


# ---------------------------------------------------------------------------
# index_video_micro + transcription integration tests
# ---------------------------------------------------------------------------


def _make_mock_chunker(micro_chunks):
    """Create a mock chunker that returns given micro chunks."""
    chunker = MagicMock()
    chunker.chunk_video.return_value = ChunkingResult(
        shots=[],
        micro=micro_chunks,
        meso=[],
        macro=[],
        video_fps=30.0,
        video_duration_sec=10.0,
        total_frames=300,
    )
    return chunker


def _make_mock_embedder(dim=512):
    """Create a mock embedder."""
    embedder = MagicMock()
    embedder.config = MagicMock()
    embedder.config.embedding_dim = dim
    embedder.encode_images.return_value = np.random.randn(1, dim).astype(np.float32)
    embedder.encode_text.side_effect = lambda texts: np.random.randn(len(texts), dim).astype(np.float32)
    return embedder


def _make_mock_qdrant():
    """Create a mock Qdrant service (FAISS fallback)."""
    from sopilot.qdrant_service import QdrantConfig, QdrantService

    config = QdrantConfig(host="localhost", port=19999)
    return QdrantService(config, use_faiss_fallback=True)


def _make_test_video():
    """Create a minimal video file with OpenCV for testing."""
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        pass
    path = Path(tmp.name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30, (64, 64))
    for i in range(300):  # 10 seconds at 30fps
        frame = np.full((64, 64, 3), fill_value=(i % 256), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class TestIndexVideoMicroWithTranscription:
    """Tests for index_video_micro with TranscriptionService."""

    def test_without_transcription(self):
        """Index without transcription service should work as before."""
        video_path = _make_test_video()
        try:
            micro_chunk = Chunk(
                level="micro",
                start_sec=0.0,
                end_sec=3.0,
                start_frame=0,
                end_frame=90,
                keyframe_indices=[45],
            )
            result = index_video_micro(
                video_path,
                "test-vid-id",
                _make_mock_chunker([micro_chunk]),
                _make_mock_embedder(),
                _make_mock_qdrant(),
            )
            assert result["num_added"] == 1
            assert result["num_text_added"] == 0
            assert "transcript_segments" in result
            assert result["transcript_segments"] == []
            # No transcript_text in metadata
            for meta in result["micro_metadata"]:
                assert "transcript_text" not in meta
        finally:
            video_path.unlink(missing_ok=True)

    def test_with_mock_transcription(self):
        """Index with mock transcription (empty segments)."""
        video_path = _make_test_video()
        try:
            micro_chunk = Chunk(
                level="micro",
                start_sec=0.0,
                end_sec=3.0,
                start_frame=0,
                end_frame=90,
                keyframe_indices=[45],
            )
            tx_service = TranscriptionService(TranscriptionConfig(backend="mock"))

            result = index_video_micro(
                video_path,
                "test-vid-id",
                _make_mock_chunker([micro_chunk]),
                _make_mock_embedder(),
                _make_mock_qdrant(),
                transcription_service=tx_service,
            )
            assert result["num_added"] == 1
            assert result["transcript_segments"] == []
        finally:
            video_path.unlink(missing_ok=True)

    def test_with_transcript_segments(self):
        """Index with real segments: transcript_text should appear in metadata."""
        video_path = _make_test_video()
        try:
            micro_chunks = [
                Chunk(
                    level="micro",
                    start_sec=0.0,
                    end_sec=3.0,
                    start_frame=0,
                    end_frame=90,
                    keyframe_indices=[45],
                ),
                Chunk(
                    level="micro",
                    start_sec=3.0,
                    end_sec=6.0,
                    start_frame=90,
                    end_frame=180,
                    keyframe_indices=[135],
                ),
            ]

            # Create a transcription service with pre-canned segments
            tx_service = TranscriptionService(TranscriptionConfig(backend="mock"))
            # Monkey-patch transcribe to return segments
            fake_segments = [
                TranscriptionSegment(0.0, 1.5, "Step one", "en"),
                TranscriptionSegment(1.5, 3.5, "Step two", "en"),
                TranscriptionSegment(4.0, 5.5, "Step three", "en"),
            ]
            tx_service.transcribe = lambda _path: TranscriptionResult(
                segments=fake_segments,
                language="en",
                duration_sec=6.0,
            )

            qdrant = _make_mock_qdrant()
            result = index_video_micro(
                video_path,
                "test-vid-id",
                _make_mock_chunker(micro_chunks),
                _make_mock_embedder(),
                qdrant,
                transcription_service=tx_service,
            )
            assert result["num_added"] == 2
            assert len(result["transcript_segments"]) == 3

            # First micro [0-3]: overlaps seg[0] and seg[1]
            meta0 = result["micro_metadata"][0]
            assert "transcript_text" in meta0
            assert "Step one" in meta0["transcript_text"]
            assert "Step two" in meta0["transcript_text"]

            # Second micro [3-6]: overlaps seg[1] (3.5>3.0) and seg[2]
            meta1 = result["micro_metadata"][1]
            assert "transcript_text" in meta1
            assert "Step three" in meta1["transcript_text"]

            # micro_text embeddings should have been stored
            assert result["num_text_added"] == 2
            # Verify micro_text collection exists in FAISS
            assert "micro_text" in qdrant._faiss_indexes

        finally:
            video_path.unlink(missing_ok=True)

    def test_transcription_failure_continues(self):
        """If transcription fails, indexing should continue without transcript."""
        video_path = _make_test_video()
        try:
            micro_chunk = Chunk(
                level="micro",
                start_sec=0.0,
                end_sec=3.0,
                start_frame=0,
                end_frame=90,
                keyframe_indices=[45],
            )

            tx_service = TranscriptionService(TranscriptionConfig(backend="mock"))
            tx_service.transcribe = MagicMock(side_effect=RuntimeError("ffmpeg not found"))

            result = index_video_micro(
                video_path,
                "test-vid-id",
                _make_mock_chunker([micro_chunk]),
                _make_mock_embedder(),
                _make_mock_qdrant(),
                transcription_service=tx_service,
            )
            # Should still index successfully
            assert result["num_added"] == 1
            assert result["transcript_segments"] == []
        finally:
            video_path.unlink(missing_ok=True)
