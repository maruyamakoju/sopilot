"""Tests for VIGIL-RAG helper utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path
from sopilot.vigil_helpers import (
    chunk_to_clip_record,
    chunking_result_to_clip_records,
    compute_video_checksum,
)
from sopilot.chunking_service import Chunk, ChunkingResult


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
