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
    index_video_all_levels,
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


# ---------------------------------------------------------------------------
# index_video_all_levels tests
# ---------------------------------------------------------------------------


def _make_mock_chunker_all_levels(micro_chunks, meso_chunks, macro_chunks):
    """Create a mock chunker that returns micro, meso, and macro chunks."""
    chunker = MagicMock()
    chunker.chunk_video.return_value = ChunkingResult(
        shots=[],
        micro=micro_chunks,
        meso=meso_chunks,
        macro=macro_chunks,
        video_fps=30.0,
        video_duration_sec=30.0,
        total_frames=900,
    )
    return chunker


class TestIndexVideoAllLevels:
    """Tests for index_video_all_levels (multi-level indexing)."""

    def test_all_three_levels_stored(self):
        """All three levels (micro, meso, macro) should be stored."""
        video_path = _make_test_video()
        try:
            micro = [
                Chunk(level="micro", start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, keyframe_indices=[45]),
                Chunk(level="micro", start_sec=3.0, end_sec=6.0, start_frame=90, end_frame=180, keyframe_indices=[135]),
            ]
            meso = [
                Chunk(
                    level="meso", start_sec=0.0, end_sec=6.0, start_frame=0, end_frame=180, keyframe_indices=[45, 135]
                ),
            ]
            macro = [
                Chunk(
                    level="macro",
                    start_sec=0.0,
                    end_sec=10.0,
                    start_frame=0,
                    end_frame=300,
                    keyframe_indices=[45, 135, 225],
                ),
            ]

            qdrant = _make_mock_qdrant()
            result = index_video_all_levels(
                video_path,
                "test-vid-id",
                _make_mock_chunker_all_levels(micro, meso, macro),
                _make_mock_embedder(),
                qdrant,
            )

            # Micro
            assert result["num_added"] == 2
            assert len(result["micro_metadata"]) == 2

            # Meso
            assert result["num_meso_added"] == 1
            assert len(result["meso_metadata"]) == 1

            # Macro
            assert result["num_macro_added"] == 1
            assert len(result["macro_metadata"]) == 1

            # Verify all levels in FAISS
            assert "micro" in qdrant._faiss_indexes
            assert "meso" in qdrant._faiss_indexes
            assert "macro" in qdrant._faiss_indexes

        finally:
            video_path.unlink(missing_ok=True)

    def test_backward_compat_micro_only(self):
        """When there are no meso/macro chunks, behaves like index_video_micro."""
        video_path = _make_test_video()
        try:
            micro = [
                Chunk(level="micro", start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, keyframe_indices=[45]),
            ]

            qdrant = _make_mock_qdrant()
            result = index_video_all_levels(
                video_path,
                "test-vid-id",
                _make_mock_chunker_all_levels(micro, [], []),
                _make_mock_embedder(),
                qdrant,
            )

            assert result["num_added"] == 1
            assert result["num_meso_added"] == 0
            assert result["num_macro_added"] == 0
            assert len(result["meso_metadata"]) == 0
            assert len(result["macro_metadata"]) == 0

        finally:
            video_path.unlink(missing_ok=True)

    def test_transcript_assigned_to_micro(self):
        """Transcripts should be assigned to micro chunks only."""
        video_path = _make_test_video()
        try:
            micro = [
                Chunk(level="micro", start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, keyframe_indices=[45]),
            ]
            meso = [
                Chunk(level="meso", start_sec=0.0, end_sec=6.0, start_frame=0, end_frame=180, keyframe_indices=[45]),
            ]

            tx_service = TranscriptionService(TranscriptionConfig(backend="mock"))
            fake_segments = [
                TranscriptionSegment(0.0, 2.0, "Hello world", "en"),
            ]
            tx_service.transcribe = lambda _path: TranscriptionResult(
                segments=fake_segments,
                language="en",
                duration_sec=6.0,
            )

            qdrant = _make_mock_qdrant()
            result = index_video_all_levels(
                video_path,
                "test-vid-id",
                _make_mock_chunker_all_levels(micro, meso, []),
                _make_mock_embedder(),
                qdrant,
                transcription_service=tx_service,
            )

            # Micro metadata should have transcript
            assert "transcript_text" in result["micro_metadata"][0]
            assert "Hello world" in result["micro_metadata"][0]["transcript_text"]

            # Meso metadata should NOT have transcript_text
            for meta in result["meso_metadata"]:
                assert "transcript_text" not in meta

        finally:
            video_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# search() time_range filter tests (FAISS fallback)
# ---------------------------------------------------------------------------


class TestSearchTimeRange:
    """Tests for QdrantService.search() with time_range parameter."""

    def test_time_range_filters_clips(self):
        """time_range should restrict results to overlapping clips."""
        qdrant = _make_mock_qdrant()
        dim = 8

        embeddings = np.random.randn(5, dim).astype(np.float32)
        metadata = [
            {"clip_id": "c1", "video_id": "v1", "start_sec": 0.0, "end_sec": 5.0},
            {"clip_id": "c2", "video_id": "v1", "start_sec": 5.0, "end_sec": 10.0},
            {"clip_id": "c3", "video_id": "v1", "start_sec": 10.0, "end_sec": 15.0},
            {"clip_id": "c4", "video_id": "v1", "start_sec": 15.0, "end_sec": 20.0},
            {"clip_id": "c5", "video_id": "v1", "start_sec": 20.0, "end_sec": 25.0},
        ]

        qdrant.ensure_collections(levels=["micro"], embedding_dim=dim)
        qdrant.add_embeddings("micro", embeddings, metadata)

        query = np.random.randn(dim).astype(np.float32)

        # Search without time_range: should return all 5
        all_results = qdrant.search("micro", query, top_k=10)
        assert len(all_results) == 5

        # Search with time_range [4, 11]: clips c1 (end=5>4), c2, c3 (start=10<11)
        filtered = qdrant.search("micro", query, top_k=10, time_range=(4.0, 11.0))
        clip_ids = {r.clip_id for r in filtered}
        assert "c1" in clip_ids  # [0,5) overlaps [4,11)
        assert "c2" in clip_ids  # [5,10) overlaps [4,11)
        assert "c3" in clip_ids  # [10,15) overlaps [4,11) — start=10 < 11
        assert "c4" not in clip_ids  # [15,20) does not overlap [4,11)
        assert "c5" not in clip_ids  # [20,25) does not overlap [4,11)

    def test_time_range_no_overlap(self):
        """time_range with no overlapping clips should return empty."""
        qdrant = _make_mock_qdrant()
        dim = 8

        embeddings = np.random.randn(2, dim).astype(np.float32)
        metadata = [
            {"clip_id": "c1", "video_id": "v1", "start_sec": 0.0, "end_sec": 5.0},
            {"clip_id": "c2", "video_id": "v1", "start_sec": 5.0, "end_sec": 10.0},
        ]

        qdrant.add_embeddings("micro", embeddings, metadata)

        query = np.random.randn(dim).astype(np.float32)
        results = qdrant.search("micro", query, top_k=10, time_range=(50.0, 60.0))
        assert len(results) == 0

    def test_time_range_none_returns_all(self):
        """time_range=None should return all clips (default behavior)."""
        qdrant = _make_mock_qdrant()
        dim = 8

        embeddings = np.random.randn(3, dim).astype(np.float32)
        metadata = [
            {"clip_id": f"c{i}", "video_id": "v1", "start_sec": i * 5.0, "end_sec": (i + 1) * 5.0} for i in range(3)
        ]

        qdrant.add_embeddings("micro", embeddings, metadata)

        query = np.random.randn(dim).astype(np.float32)
        results = qdrant.search("micro", query, top_k=10, time_range=None)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# coarse_to_fine_search with temporal filtering tests
# ---------------------------------------------------------------------------


class TestCoarseToFineSearch:
    """Tests for coarse_to_fine_search with temporal filtering."""

    def _setup_multi_level(self):
        """Create QdrantService with data at macro, meso, and micro levels."""
        qdrant = _make_mock_qdrant()
        dim = 8
        rng = np.random.RandomState(42)

        # Macro: 2 chunks spanning full video
        macro_embs = rng.randn(2, dim).astype(np.float32)
        macro_meta = [
            {"clip_id": "mac1", "video_id": "v1", "start_sec": 0.0, "end_sec": 50.0},
            {"clip_id": "mac2", "video_id": "v1", "start_sec": 50.0, "end_sec": 100.0},
        ]
        qdrant.add_embeddings("macro", macro_embs, macro_meta)

        # Meso: 5 chunks
        meso_embs = rng.randn(5, dim).astype(np.float32)
        meso_meta = [
            {"clip_id": f"mes{i}", "video_id": "v1", "start_sec": i * 20.0, "end_sec": (i + 1) * 20.0} for i in range(5)
        ]
        qdrant.add_embeddings("meso", meso_embs, meso_meta)

        # Micro: 10 chunks
        micro_embs = rng.randn(10, dim).astype(np.float32)
        micro_meta = [
            {"clip_id": f"mic{i}", "video_id": "v1", "start_sec": i * 10.0, "end_sec": (i + 1) * 10.0}
            for i in range(10)
        ]
        qdrant.add_embeddings("micro", micro_embs, micro_meta)

        # Shot: empty (to test graceful handling)
        return qdrant, dim

    def test_temporal_filtering_narrows_micro_results(self):
        """With temporal filtering, micro results should be a subset."""
        qdrant, dim = self._setup_multi_level()
        query = np.random.randn(dim).astype(np.float32)

        # Without temporal filtering
        results_flat = qdrant.coarse_to_fine_search(
            query,
            macro_k=2,
            meso_k=5,
            micro_k=10,
            shot_k=0,
            enable_temporal_filtering=False,
        )
        micro_flat = results_flat["micro"]

        # With temporal filtering
        results_hier = qdrant.coarse_to_fine_search(
            query,
            macro_k=1,
            meso_k=5,
            micro_k=10,
            shot_k=0,
            enable_temporal_filtering=True,
        )
        micro_hier = results_hier["micro"]

        # Hierarchical should return <= flat (narrowed by macro window)
        assert len(micro_hier) <= len(micro_flat)

    def test_empty_macro_falls_through(self):
        """If macro returns nothing, micro should still get full results."""
        qdrant = _make_mock_qdrant()
        dim = 8

        # Only micro data, no macro
        micro_embs = np.random.randn(5, dim).astype(np.float32)
        micro_meta = [
            {"clip_id": f"mic{i}", "video_id": "v1", "start_sec": i * 5.0, "end_sec": (i + 1) * 5.0} for i in range(5)
        ]
        qdrant.add_embeddings("micro", micro_embs, micro_meta)

        query = np.random.randn(dim).astype(np.float32)
        results = qdrant.coarse_to_fine_search(
            query,
            macro_k=5,
            meso_k=5,
            micro_k=10,
            shot_k=0,
            enable_temporal_filtering=True,
        )

        # macro empty → no time window → micro searched without filter
        assert len(results["macro"]) == 0
        assert len(results["micro"]) == 5

    def test_expand_factor_includes_boundary_clips(self):
        """expand_factor should include clips near the boundary."""
        qdrant = _make_mock_qdrant()
        dim = 8
        rng = np.random.RandomState(99)

        # One macro chunk: [10, 20]
        macro_emb = rng.randn(1, dim).astype(np.float32)
        qdrant.add_embeddings(
            "macro",
            macro_emb,
            [
                {"clip_id": "mac1", "video_id": "v1", "start_sec": 10.0, "end_sec": 20.0},
            ],
        )

        # Micro clips: one at boundary [9, 11], one far away [50, 55]
        micro_embs = rng.randn(2, dim).astype(np.float32)
        qdrant.add_embeddings(
            "micro",
            micro_embs,
            [
                {"clip_id": "mic_near", "video_id": "v1", "start_sec": 9.0, "end_sec": 11.0},
                {"clip_id": "mic_far", "video_id": "v1", "start_sec": 50.0, "end_sec": 55.0},
            ],
        )

        query = rng.randn(dim).astype(np.float32)

        # With expand_factor=0.2, macro window [10,20] → pad 2.0 → [8, 22]
        results = qdrant.coarse_to_fine_search(
            query,
            macro_k=1,
            meso_k=0,
            micro_k=10,
            shot_k=0,
            enable_temporal_filtering=True,
            time_expand_factor=0.2,
        )

        micro_ids = {r.clip_id for r in results["micro"]}
        assert "mic_near" in micro_ids  # [9,11] overlaps [8,22]
        assert "mic_far" not in micro_ids  # [50,55] does not overlap [8,22]
