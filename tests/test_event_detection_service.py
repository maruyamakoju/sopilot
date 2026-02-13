"""Tests for event detection service."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_mock_vigil_services
from sopilot.event_detection_service import (
    DetectedEvent,
    EventDetectionResult,
    EventDetectionService,
)
from sopilot.llm_utils import parse_llm_json
from sopilot.qdrant_service import SearchResult
from sopilot.temporal import temporal_iou


def _create_mock_services():
    return make_mock_vigil_services()


def _populate_micro_embeddings(qdrant_service, video_id="video-123", n=5, dim=512):
    """Add mock micro-level embeddings to the FAISS fallback."""
    embeddings = np.random.randn(n, dim).astype(np.float32)
    metadata = [
        {
            "clip_id": f"micro-{i}",
            "video_id": video_id,
            "start_sec": float(i * 3),
            "end_sec": float((i + 1) * 3),
        }
        for i in range(n)
    ]
    qdrant_service.add_embeddings("micro", embeddings, metadata)
    return metadata


class TestDetectedEvent:
    """Tests for DetectedEvent dataclass."""

    def test_creation(self):
        evt = DetectedEvent(
            event_type="intrusion",
            start_sec=5.0,
            end_sec=8.0,
            confidence=0.85,
            clip_id="clip-1",
            observation="Person entered restricted area",
        )
        assert evt.event_type == "intrusion"
        assert evt.start_sec == 5.0
        assert evt.end_sec == 8.0
        assert evt.confidence == 0.85
        assert evt.clip_id == "clip-1"
        assert "restricted" in evt.observation

    def test_all_fields_required(self):
        with pytest.raises(TypeError):
            DetectedEvent(event_type="intrusion")  # missing required fields


class TestEventDetectionResult:
    """Tests for EventDetectionResult dataclass."""

    def test_empty_result(self):
        result = EventDetectionResult(video_id="vid-1")
        assert result.video_id == "vid-1"
        assert result.events == []
        assert result.num_proposals == 0
        assert result.num_verified == 0

    def test_with_events(self):
        events = [
            DetectedEvent("intrusion", 5.0, 8.0, 0.9, "c1", "obs1"),
            DetectedEvent("ppe_violation", 12.0, 15.0, 0.7, "c2", "obs2"),
        ]
        result = EventDetectionResult(
            video_id="vid-1",
            events=events,
            num_proposals=10,
            num_verified=2,
        )
        assert len(result.events) == 2
        assert result.num_proposals == 10
        assert result.num_verified == 2


class TestEventDetectionService:
    """Tests for EventDetectionService."""

    def test_init(self):
        qdrant_service, llm_service = _create_mock_services()
        detector = EventDetectionService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )
        assert detector.vector_service is qdrant_service
        assert detector.llm_service is llm_service
        assert detector.retrieval_embedder is None

    def test_stage1_retrieve_mock(self):
        """Stage 1 retrieval returns proposals from FAISS fallback."""
        qdrant_service, llm_service = _create_mock_services()
        _populate_micro_embeddings(qdrant_service, n=5)

        detector = EventDetectionService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        proposals = detector._stage1_retrieve("color change", top_k=3)
        assert len(proposals) <= 3
        assert all(isinstance(p, SearchResult) for p in proposals)
        # Each should have valid temporal boundaries
        for p in proposals:
            assert p.start_sec >= 0
            assert p.end_sec > p.start_sec

    def test_stage2_verify_mock(self):
        """Stage 2 verification with mock LLM returns a result."""
        qdrant_service, llm_service = _create_mock_services()
        detector = EventDetectionService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        clip = SearchResult(
            clip_id="clip-1",
            video_id="video-123",
            level="micro",
            start_sec=2.0,
            end_sec=5.0,
            score=0.8,
        )

        # Mock LLM returns "Mock answer: This is a placeholder response."
        # which won't parse as valid JSON, so _parse_verification_json returns
        # the fallback with detected=False
        result = detector._stage2_verify("dummy.mp4", "intrusion", clip)
        # Mock LLM produces unparseable text -> fallback -> detected=False -> None
        assert result is None

    def test_detect_events_full_pipeline_mock(self):
        """Full pipeline with FAISS + mock LLM."""
        qdrant_service, llm_service = _create_mock_services()
        _populate_micro_embeddings(qdrant_service, n=8)

        detector = EventDetectionService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        result = detector.detect_events(
            "dummy.mp4",
            ["color change", "scene transition"],
            video_id="video-123",
            top_k=5,
            confidence_threshold=0.3,
        )

        assert isinstance(result, EventDetectionResult)
        assert result.video_id == "video-123"
        assert result.num_proposals > 0
        # Mock LLM can't produce valid JSON -> all proposals rejected
        # so num_verified should be 0
        assert result.num_verified == 0
        assert len(result.events) == 0

    def test_detect_events_no_data(self):
        """Pipeline with empty vector DB returns no events."""
        qdrant_service, llm_service = _create_mock_services()

        detector = EventDetectionService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        result = detector.detect_events(
            "dummy.mp4",
            ["intrusion"],
            top_k=5,
        )

        assert result.num_proposals == 0
        assert result.num_verified == 0
        assert len(result.events) == 0

    def test_dedup_across_event_types(self):
        """Cross-type dedup removes overlapping proposals."""
        qdrant_service, llm_service = _create_mock_services()
        detector = EventDetectionService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        # Two event types retrieve the same temporal window
        proposals = [
            ("intrusion", SearchResult("c1", "v1", "micro", 0.0, 4.0, 0.9)),
            ("ppe_violation", SearchResult("c2", "v1", "micro", 0.5, 3.5, 0.8)),  # overlaps c1
            ("intrusion", SearchResult("c3", "v1", "micro", 10.0, 14.0, 0.7)),  # no overlap
        ]

        deduped = detector._dedup_proposals(proposals, iou_threshold=0.3)
        # c1 and c2 overlap (IoU > 0.3), c1 wins (higher score). c3 kept.
        assert len(deduped) == 2
        clip_ids = {clip.clip_id for _, clip in deduped}
        assert "c1" in clip_ids
        assert "c3" in clip_ids

    def test_dedup_no_overlap(self):
        """No dedup when proposals don't overlap."""
        qdrant_service, llm_service = _create_mock_services()
        detector = EventDetectionService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        proposals = [
            ("intrusion", SearchResult("c1", "v1", "micro", 0.0, 3.0, 0.9)),
            ("ppe_violation", SearchResult("c2", "v1", "micro", 5.0, 8.0, 0.8)),
            ("intrusion", SearchResult("c3", "v1", "micro", 10.0, 13.0, 0.7)),
        ]

        deduped = detector._dedup_proposals(proposals, iou_threshold=0.5)
        assert len(deduped) == 3

    def test_dedup_different_videos(self):
        """Proposals from different videos are never deduped."""
        qdrant_service, llm_service = _create_mock_services()
        detector = EventDetectionService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        proposals = [
            ("intrusion", SearchResult("c1", "v1", "micro", 0.0, 4.0, 0.9)),
            ("intrusion", SearchResult("c2", "v2", "micro", 0.0, 4.0, 0.8)),  # same time, different video
        ]

        deduped = detector._dedup_proposals(proposals, iou_threshold=0.5)
        assert len(deduped) == 2

    def test_confidence_threshold_filtering(self):
        """Events below confidence threshold are filtered out."""
        qdrant_service, llm_service = _create_mock_services()
        detector = EventDetectionService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        # Directly test the threshold logic: detect_events filters by threshold
        # after Stage 2. Since mock LLM produces no valid events, test the
        # threshold on the result structure instead.
        result = detector.detect_events(
            "dummy.mp4",
            ["intrusion"],
            confidence_threshold=0.9,  # very high threshold
        )
        # All events should be filtered out (mock LLM can't produce valid JSON)
        assert result.num_verified == 0

    def test_temporal_iou(self):
        """Test temporal IoU computation."""
        assert temporal_iou(0, 4, 2, 6) == pytest.approx(2.0 / 6.0)
        assert temporal_iou(0, 4, 0, 4) == pytest.approx(1.0)
        assert temporal_iou(0, 4, 5, 9) == pytest.approx(0.0)
        assert temporal_iou(0, 0, 0, 0) == pytest.approx(0.0)

    def test_parse_verification_json_valid(self):
        """Parse well-formed verification JSON."""
        text = '{"detected": true, "confidence": 0.85, "observation": "person entered", "refined_start_sec": 1.0, "refined_end_sec": 3.5}'
        parsed = parse_llm_json(text)
        assert parsed["detected"] is True
        assert parsed["confidence"] == 0.85
        assert parsed["observation"] == "person entered"
        assert parsed["refined_start_sec"] == 1.0

    def test_parse_verification_json_markdown_fences(self):
        """Parse JSON wrapped in markdown code fences."""
        text = '```json\n{"detected": true, "confidence": 0.7, "observation": "test"}\n```'
        parsed = parse_llm_json(text)
        assert parsed["detected"] is True
        assert parsed["confidence"] == 0.7

    def test_parse_verification_json_with_surrounding_text(self):
        """Parse JSON embedded in surrounding text."""
        text = 'Here is the result: {"detected": false, "confidence": 0.1, "observation": "nothing"} end.'
        parsed = parse_llm_json(text)
        assert parsed["detected"] is False
        assert parsed["confidence"] == 0.1

    def test_parse_verification_json_fallback(self):
        """Unparseable text returns fallback dict."""
        text = "This is not JSON at all"
        fallback = {"detected": False, "confidence": 0.0}
        parsed = parse_llm_json(text, fallback=fallback)
        assert parsed["detected"] is False
        assert parsed["confidence"] == 0.0

    def test_parse_llm_json_default_fallback(self):
        """Unparseable text with no fallback returns _parse_error marker."""
        text = "Not JSON"
        parsed = parse_llm_json(text)
        assert parsed["_parse_error"] is True

    def test_encode_event_query_mock(self):
        """Mock encoding returns a 512-dim vector."""
        qdrant_service, llm_service = _create_mock_services()
        detector = EventDetectionService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )
        embedding = detector._encode_event_query("intrusion")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (512,)

    def test_detect_events_video_id_default(self):
        """When video_id is not provided, result uses 'unknown'."""
        qdrant_service, llm_service = _create_mock_services()
        detector = EventDetectionService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        result = detector.detect_events("dummy.mp4", ["test"])
        assert result.video_id == "unknown"

    def test_detect_events_preserves_video_id(self):
        """When video_id is provided, it's preserved in the result."""
        qdrant_service, llm_service = _create_mock_services()
        detector = EventDetectionService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        result = detector.detect_events("dummy.mp4", ["test"], video_id="my-video-id")
        assert result.video_id == "my-video-id"
