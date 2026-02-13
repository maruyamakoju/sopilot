"""2-Stage Event Detection Service for VIGIL-RAG.

This module implements a 2-stage event detection pipeline:
  Stage 1 (Retrieval): For each event type, encode the description as a text
      query and retrieve top-K matching micro-level clips from the vector DB.
  Stage 2 (Verification): For each proposal clip, run the Video-LLM to verify
      whether the event actually occurs, extract confidence, temporal refinement,
      and a natural-language observation.

The pipeline reuses existing infrastructure:
- QdrantService.search() for vector retrieval
- RetrievalEmbedder.encode_text() for text-to-embedding
- VideoLLMService.answer_question() for per-clip verification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .llm_utils import parse_llm_json
from .qdrant_service import QdrantService, SearchResult
from .temporal import temporal_iou
from .video_llm_service import VideoLLMService

try:
    from .retrieval_embeddings import RetrievalEmbedder

    RETRIEVAL_AVAILABLE = True
except ImportError:
    RETRIEVAL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DetectedEvent:
    """A single detected event with temporal boundaries and confidence."""

    event_type: str
    start_sec: float
    end_sec: float
    confidence: float  # 0.0-1.0 from LLM Stage 2
    clip_id: str  # Source clip ID from retrieval
    observation: str  # LLM description of what was seen


@dataclass
class EventDetectionResult:
    """Result of the 2-stage event detection pipeline."""

    video_id: str
    events: list[DetectedEvent] = field(default_factory=list)
    num_proposals: int = 0  # Stage 1 proposal count (before verification)
    num_verified: int = 0  # Stage 2 verified count (after filtering)


class EventDetectionService:
    """2-stage event detection: retrieval proposals + LLM verification."""

    def __init__(
        self,
        vector_service: QdrantService,
        llm_service: VideoLLMService,
        retrieval_embedder: RetrievalEmbedder | None = None,
    ) -> None:
        self.vector_service = vector_service
        self.llm_service = llm_service
        self.retrieval_embedder = retrieval_embedder

    def detect_events(
        self,
        video_path: Path | str,
        event_types: list[str],
        *,
        video_id: str | None = None,
        top_k: int = 10,
        confidence_threshold: float = 0.3,
        overlap_threshold: float = 0.5,
    ) -> EventDetectionResult:
        """Run 2-stage event detection pipeline.

        Args:
            video_path: Path to the video file.
            event_types: List of event type descriptions (e.g. ["intrusion", "PPE violation"]).
            video_id: Optional video ID filter for retrieval.
            top_k: Number of proposals per event type (Stage 1).
            confidence_threshold: Minimum confidence to keep a detection (Stage 2).
            overlap_threshold: Temporal IoU threshold for cross-type dedup.

        Returns:
            EventDetectionResult with verified events.
        """
        video_path = Path(video_path)
        result_video_id = video_id or "unknown"

        logger.info(
            "Event detection: %d types, top_k=%d, threshold=%.2f",
            len(event_types),
            top_k,
            confidence_threshold,
        )

        # Stage 1: Retrieve proposals for each event type
        all_proposals: list[tuple[str, SearchResult]] = []
        for event_type in event_types:
            proposals = self._stage1_retrieve(event_type, video_id=video_id, top_k=top_k)
            for p in proposals:
                all_proposals.append((event_type, p))

        logger.info("Stage 1: %d total proposals across %d event types", len(all_proposals), len(event_types))

        # Dedup overlapping proposals across event types
        deduped = self._dedup_proposals(all_proposals, overlap_threshold)
        num_proposals = len(deduped)

        logger.info("After dedup: %d proposals (from %d)", num_proposals, len(all_proposals))

        # Stage 2: Verify each proposal with LLM
        verified_events: list[DetectedEvent] = []
        for event_type, clip in deduped:
            detected = self._stage2_verify(video_path, event_type, clip)
            if detected is not None and detected.confidence >= confidence_threshold:
                verified_events.append(detected)

        # Sort by confidence descending
        verified_events.sort(key=lambda e: e.confidence, reverse=True)

        logger.info(
            "Stage 2: %d verified events (from %d proposals, threshold=%.2f)",
            len(verified_events),
            num_proposals,
            confidence_threshold,
        )

        return EventDetectionResult(
            video_id=result_video_id,
            events=verified_events,
            num_proposals=num_proposals,
            num_verified=len(verified_events),
        )

    def _stage1_retrieve(
        self,
        event_type_description: str,
        *,
        video_id: str | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Stage 1: Retrieve clip proposals for an event type.

        Args:
            event_type_description: Text description of the event (e.g. "intrusion").
            video_id: Optional video ID filter.
            top_k: Number of results to return.

        Returns:
            List of SearchResult clip proposals.
        """
        query_embedding = self._encode_event_query(event_type_description)
        results = self.vector_service.search(
            level="micro",
            query_vector=query_embedding,
            top_k=top_k,
            video_id=video_id,
        )
        logger.debug("Stage 1 [%s]: %d proposals", event_type_description, len(results))
        return results

    def _stage2_verify(
        self,
        video_path: Path,
        event_type: str,
        clip: SearchResult,
    ) -> DetectedEvent | None:
        """Stage 2: Verify a proposal clip with the Video-LLM.

        Args:
            video_path: Path to the video file.
            event_type: Event type being verified.
            clip: The clip proposal from Stage 1.

        Returns:
            DetectedEvent if the LLM confirms the event, None otherwise.
        """
        # Build prompt with optional transcript context
        transcript_ctx = ""
        if clip.transcript_text:
            excerpt = clip.transcript_text[:400]
            transcript_ctx = f'\nAudio transcript for this clip: "{excerpt}"\n'

        verification_prompt = (
            f'Does this video clip contain the event: "{event_type}"?\n'
            f"{transcript_ctx}\n"
            "Respond ONLY with a JSON object (no markdown fences):\n"
            "{\n"
            '  "detected": true or false,\n'
            '  "confidence": <0.0-1.0>,\n'
            '  "observation": "<describe what you see AND hear>",\n'
            '  "refined_start_sec": <float or null>,\n'
            '  "refined_end_sec": <float or null>\n'
            "}"
        )

        try:
            qa_result = self.llm_service.answer_question(
                video_path,
                verification_prompt,
                start_sec=clip.start_sec,
                end_sec=clip.end_sec,
                enable_cot=False,
            )
            parsed = parse_llm_json(
                qa_result.answer,
                fallback={
                    "detected": False,
                    "confidence": 0.0,
                    "observation": "",
                    "refined_start_sec": None,
                    "refined_end_sec": None,
                },
            )
        except Exception as exc:
            logger.warning(
                "Stage 2 verification failed [%.1f-%.1f]: %s",
                clip.start_sec,
                clip.end_sec,
                exc,
            )
            return None

        if not parsed.get("detected", False):
            return None

        # Use refined boundaries if provided, otherwise fall back to clip boundaries
        start_sec = parsed.get("refined_start_sec")
        if start_sec is None or not isinstance(start_sec, (int, float)):
            start_sec = clip.start_sec

        end_sec = parsed.get("refined_end_sec")
        if end_sec is None or not isinstance(end_sec, (int, float)):
            end_sec = clip.end_sec

        return DetectedEvent(
            event_type=event_type,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            confidence=float(parsed.get("confidence", 0.0)),
            clip_id=clip.clip_id,
            observation=str(parsed.get("observation", "")),
        )

    def _encode_event_query(self, event_type_description: str) -> np.ndarray:
        """Encode event type description to embedding vector."""
        if self.retrieval_embedder is not None:
            embeddings = self.retrieval_embedder.encode_text([event_type_description])
            return embeddings[0]

        # Mock fallback: random embedding
        logger.debug("Encoding event query (mock): %s", event_type_description)
        return np.random.randn(512).astype(np.float32)

    def _dedup_proposals(
        self,
        proposals: list[tuple[str, SearchResult]],
        iou_threshold: float = 0.5,
    ) -> list[tuple[str, SearchResult]]:
        """Remove temporally overlapping proposals across event types.

        Keeps the higher-scored proposal when two overlap.

        Args:
            proposals: List of (event_type, SearchResult) tuples.
            iou_threshold: Temporal IoU threshold for overlap.

        Returns:
            Deduplicated proposals.
        """
        if not proposals:
            return []

        # Sort by score descending so higher-scored proposals are kept
        sorted_proposals = sorted(proposals, key=lambda x: x[1].score, reverse=True)

        kept: list[tuple[str, SearchResult]] = []
        for candidate_type, candidate in sorted_proposals:
            is_dup = False
            for _, existing in kept:
                if candidate.video_id != existing.video_id:
                    continue
                iou = temporal_iou(
                    candidate.start_sec,
                    candidate.end_sec,
                    existing.start_sec,
                    existing.end_sec,
                )
                if iou >= iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append((candidate_type, candidate))
        return kept
