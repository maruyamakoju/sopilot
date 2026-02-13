"""RAG service for VIGIL-RAG long-form video question answering.

This module implements the full RAG pipeline:
1. Query encoding (text/video → embedding)
2. Coarse-to-fine retrieval (Qdrant multi-level search)
3. Re-ranking (optional cross-encoder or LLM scoring)
4. Context assembly (gather clips + metadata)
5. LLM inference (Video-LLM QA)
6. Answer generation with evidence citations

Design principles:
- Hierarchical retrieval: Start broad (macro), refine to specific (shot)
- Evidence tracking: Maintain clip_id → time range mapping for citations
- Uncertainty quantification: Track confidence scores
- Fallback gracefully: Degrade to available levels if some fail
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .llm_utils import parse_llm_json
from .qdrant_service import QdrantService, SearchResult
from .temporal import temporal_iou
from .video_llm_service import VideoLLMService, VideoQAResult

try:
    from .retrieval_embeddings import RetrievalEmbedder

    RETRIEVAL_AVAILABLE = True
except ImportError:
    RETRIEVAL_AVAILABLE = False

logger = logging.getLogger(__name__)


def compute_video_id(video_path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    """Compute a stable, deterministic video ID from the file's SHA-256.

    Args:
        video_path: Path to the video file.
        chunk_size: Read buffer size (default 1 MB).

    Returns:
        SHA-256 hex digest (64 chars).
    """
    h = hashlib.sha256()
    with open(video_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline."""

    # Retrieval depths per level
    macro_k: int = 5
    meso_k: int = 10
    micro_k: int = 20
    shot_k: int = 30

    # Re-ranking
    enable_rerank: bool = False
    rerank_top_k: int = 10

    # Filtering
    min_score: float | None = None
    video_id_filter: str | None = None


@dataclass
class Evidence:
    """Evidence clip supporting an answer."""

    clip_id: str
    video_id: str
    level: Literal["shot", "micro", "meso", "macro"]
    start_sec: float
    end_sec: float
    score: float  # Retrieval score
    rerank_score: float | None = None  # Re-ranking score (if enabled)
    transcript_text: str | None = None  # Audio transcript (Whisper)


@dataclass
class ClipObservation:
    """Per-clip observation from Video-LLM."""

    clip_id: str
    start_sec: float
    end_sec: float
    relevance: float  # 0.0-1.0 relevance to the question
    observation: str  # What is visible in the clip
    answer_candidate: str  # Partial answer based on this clip
    confidence: float  # 0.0-1.0 self-assessed confidence


@dataclass
class RAGResult:
    """Result of RAG question answering."""

    question: str
    answer: str
    evidence: list[Evidence]  # Supporting clips
    confidence: float | None = None
    reasoning: str | None = None  # Chain-of-thought (if enabled)
    clip_observations: list[ClipObservation] | None = None  # Per-clip observations (4.3)


class RAGService:
    """RAG service for long-form video QA."""

    def __init__(
        self,
        *,
        vector_service: QdrantService,
        llm_service: VideoLLMService,
        retrieval_config: RetrievalConfig | None = None,
        retrieval_embedder: RetrievalEmbedder | None = None,
    ) -> None:
        """Initialize RAG service.

        Args:
            vector_service: Qdrant service for vector search
            llm_service: Video-LLM service for embeddings and QA
            retrieval_config: Retrieval configuration (defaults to RetrievalConfig())
            retrieval_embedder: Optional retrieval embedder (OpenCLIP). If None, falls back to mock.
        """
        self.vector_service = vector_service
        self.llm_service = llm_service
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.retrieval_embedder = retrieval_embedder

    def answer_question(
        self,
        question: str,
        *,
        video_id: str | None = None,
        enable_cot: bool = False,
        return_top_k: int = 5,
    ) -> RAGResult:
        """Answer a question about video(s) using RAG.

        Pipeline:
        1. Encode question to embedding
        2. Retrieve relevant clips (coarse-to-fine)
        3. Re-rank clips (if enabled)
        4. Assemble context from top clips
        5. Generate answer with LLM
        6. Return answer + evidence

        Args:
            question: Question to answer
            video_id: Optional video ID filter
            enable_cot: Enable chain-of-thought reasoning
            return_top_k: Number of evidence clips to return

        Returns:
            RAGResult with answer and supporting evidence
        """
        logger.info("RAG question: %s", question)

        # Step 1: Encode question
        # TODO: For text-only queries, we'd use a text encoder
        # For now, using a mock embedding (same as video embeddings)
        query_embedding = self._encode_query(question)

        # Step 2: Retrieve relevant clips (coarse-to-fine)
        search_results = self.vector_service.coarse_to_fine_search(
            query_vector=query_embedding,
            video_id=video_id,
            macro_k=self.retrieval_config.macro_k,
            meso_k=self.retrieval_config.meso_k,
            micro_k=self.retrieval_config.micro_k,
            shot_k=self.retrieval_config.shot_k,
        )

        # Step 3: Flatten and re-rank
        all_results = self._flatten_search_results(search_results)
        if self.retrieval_config.enable_rerank:
            all_results = self._rerank_results(question, all_results)

        # Step 4: Take top-k evidence
        top_results = all_results[:return_top_k]

        # Convert to Evidence objects
        evidence = [
            Evidence(
                clip_id=r.clip_id,
                video_id=r.video_id,
                level=r.level,
                start_sec=r.start_sec,
                end_sec=r.end_sec,
                score=r.score,
                transcript_text=r.transcript_text,
            )
            for r in top_results
        ]

        # Step 5: Generate answer (for now, just use the top video clip)
        if top_results:
            # TODO: Use actual video path from database
            # For now, mock answer
            qa_result = VideoQAResult(
                question=question,
                answer=f"Based on the retrieved clips, the answer is: [placeholder answer]. "
                f"Evidence found in {len(evidence)} clips.",
                confidence=0.85,
                reasoning="Coarse-to-fine retrieval identified relevant clips." if enable_cot else None,
            )
        else:
            qa_result = VideoQAResult(
                question=question,
                answer="No relevant clips found to answer this question.",
                confidence=0.0,
            )

        logger.info("Generated answer with %d evidence clips", len(evidence))

        return RAGResult(
            question=question,
            answer=qa_result.answer,
            evidence=evidence,
            confidence=qa_result.confidence,
            reasoning=qa_result.reasoning,
        )

    def answer_question_from_clip(
        self,
        video_path: Path | str,
        question: str,
        *,
        start_sec: float = 0.0,
        end_sec: float | None = None,
        enable_cot: bool = False,
    ) -> RAGResult:
        """Answer a question about a specific video clip (no retrieval).

        This is useful when you already know the relevant clip.

        Args:
            video_path: Path to video file
            question: Question to answer
            start_sec: Start time in seconds
            end_sec: End time in seconds (None = end of video)
            enable_cot: Enable chain-of-thought reasoning

        Returns:
            RAGResult with answer (no evidence since no retrieval)
        """
        logger.info("Direct clip QA: %s", question)

        qa_result = self.llm_service.answer_question(
            video_path,
            question,
            start_sec=start_sec,
            end_sec=end_sec,
            enable_cot=enable_cot,
        )

        # No evidence (direct clip, no retrieval)
        evidence = [
            Evidence(
                clip_id="direct-clip",
                video_id="unknown",
                level="micro",
                start_sec=start_sec,
                end_sec=end_sec or 0.0,
                score=1.0,  # Perfect match (user specified)
            )
        ]

        return RAGResult(
            question=question,
            answer=qa_result.answer,
            evidence=evidence,
            confidence=qa_result.confidence,
            reasoning=qa_result.reasoning,
        )

    def answer_question_topk(
        self,
        video_path: Path | str,
        question: str,
        *,
        video_id: str | None = None,
        top_k: int = 5,
        overlap_threshold: float = 0.5,
    ) -> RAGResult:
        """Answer a question using Top-K composite approach.

        Pipeline:
        1. Encode question → retrieve top clips
        2. Dedup overlapping clips
        3. Per-clip observation (Qwen on each clip → JSON)
        4. Synthesis (aggregate observations → final answer)

        Args:
            video_path: Path to video file
            question: Question to answer
            video_id: Optional video ID filter
            top_k: Number of clips to observe
            overlap_threshold: IoU threshold for dedup (0-1)

        Returns:
            RAGResult with composite answer, evidence, and clip_observations
        """
        video_path = Path(video_path)
        logger.info("Top-K RAG: question=%s, k=%d", question[:50], top_k)

        # Step 1: Retrieve (micro-level search, most granular available)
        query_embedding = self._encode_query(question)
        all_results = self.vector_service.search(
            level="micro",
            query_vector=query_embedding,
            top_k=self.retrieval_config.micro_k,
            video_id=video_id,
        )

        # Step 2: Dedup overlapping clips
        deduped = self._dedup_clips(all_results, overlap_threshold)
        top_clips = deduped[:top_k]

        if not top_clips:
            return RAGResult(
                question=question,
                answer="No relevant clips found to answer this question.",
                evidence=[],
                confidence=0.0,
                clip_observations=[],
            )

        logger.info("Using %d clips after dedup (from %d)", len(top_clips), len(all_results))

        # Step 3: Per-clip observation
        observations: list[ClipObservation] = []
        for clip in top_clips:
            obs = self._observe_clip(video_path, question, clip)
            observations.append(obs)

        # Step 4: Synthesize final answer
        final_answer = self._synthesize_answer(question, observations)

        # Build evidence
        evidence = [
            Evidence(
                clip_id=clip.clip_id,
                video_id=clip.video_id,
                level=clip.level,
                start_sec=clip.start_sec,
                end_sec=clip.end_sec,
                score=clip.score,
                transcript_text=clip.transcript_text,
            )
            for clip in top_clips
        ]

        # Compute aggregate confidence
        if observations:
            avg_conf = sum(o.confidence for o in observations) / len(observations)
        else:
            avg_conf = 0.0

        return RAGResult(
            question=question,
            answer=final_answer,
            evidence=evidence,
            confidence=avg_conf,
            clip_observations=observations,
        )

    def _dedup_clips(
        self,
        results: list[SearchResult],
        iou_threshold: float = 0.5,
    ) -> list[SearchResult]:
        """Remove temporally overlapping clips, keeping higher-scored ones.

        Args:
            results: Sorted search results (score desc)
            iou_threshold: Temporal IoU threshold for overlap

        Returns:
            Deduplicated results
        """
        if not results:
            return []

        kept: list[SearchResult] = []
        for candidate in results:
            is_dup = False
            for existing in kept:
                # Only dedup within same video
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
                kept.append(candidate)
        return kept

    def _observe_clip(
        self,
        video_path: Path,
        question: str,
        clip: SearchResult,
    ) -> ClipObservation:
        """Run Qwen on a single clip and extract structured observation.

        Args:
            video_path: Path to video file
            question: User's question
            clip: The clip to observe

        Returns:
            ClipObservation with structured output
        """
        observation_prompt = (
            f'You are analyzing a short video clip to answer: "{question}"\n\n'
            "Respond ONLY with a JSON object (no markdown fences):\n"
            "{\n"
            '  "relevance": <0.0-1.0 how relevant this clip is to the question>,\n'
            '  "observation": "<describe what you see in this clip>",\n'
            '  "answer_candidate": "<partial answer based on this clip only>",\n'
            '  "confidence": <0.0-1.0 how confident you are>\n'
            "}"
        )

        try:
            qa_result = self.llm_service.answer_question(
                video_path,
                observation_prompt,
                start_sec=clip.start_sec,
                end_sec=clip.end_sec,
                enable_cot=False,
            )
            parsed = parse_llm_json(
                qa_result.answer,
                fallback={
                    "relevance": 0.0,
                    "observation": "",
                    "answer_candidate": "",
                    "confidence": 0.0,
                },
            )
        except Exception as exc:
            logger.warning(
                "Clip observation failed [%.1f-%.1f]: %s",
                clip.start_sec,
                clip.end_sec,
                exc,
            )
            parsed = {
                "relevance": 0.0,
                "observation": f"Observation failed: {exc}",
                "answer_candidate": "",
                "confidence": 0.0,
            }

        return ClipObservation(
            clip_id=clip.clip_id,
            start_sec=clip.start_sec,
            end_sec=clip.end_sec,
            relevance=float(parsed.get("relevance", 0.0)),
            observation=str(parsed.get("observation", "")),
            answer_candidate=str(parsed.get("answer_candidate", "")),
            confidence=float(parsed.get("confidence", 0.0)),
        )

    def _synthesize_answer(
        self,
        question: str,
        observations: list[ClipObservation],
    ) -> str:
        """Synthesize a final answer from multiple clip observations.

        This is a text-only step — no video inference required.

        Args:
            question: Original question
            observations: Per-clip observations

        Returns:
            Final synthesized answer
        """
        if not observations:
            return "No relevant clips found to answer this question."

        # Filter to relevant observations
        relevant = [o for o in observations if o.relevance > 0.2]
        if not relevant:
            relevant = observations  # Use all if none above threshold

        # Build observation summary for synthesis
        obs_lines = []
        for i, obs in enumerate(relevant, 1):
            obs_lines.append(
                f"Clip {i} [{obs.start_sec:.1f}-{obs.end_sec:.1f}s] "
                f"(relevance={obs.relevance:.1f}, confidence={obs.confidence:.1f}):\n"
                f"  Observation: {obs.observation}\n"
                f"  Partial answer: {obs.answer_candidate}"
            )
        obs_text = "\n\n".join(obs_lines)

        synthesis_prompt = (
            f"Question: {question}\n\n"
            f"Evidence from {len(relevant)} video clips:\n\n"
            f"{obs_text}\n\n"
            "Based on ALL the evidence above, provide a comprehensive final answer. "
            "Cite specific clip timestamps when referencing evidence."
        )

        try:
            # Text-only synthesis: use LLM without video
            # We pass a dummy call through answer_question with the original question
            # but really we want text-only. For now, use the mock path or
            # a simple join approach if model is mock
            if self.llm_service.config.model_name == "mock":
                # Mock: join candidates
                candidates = [o.answer_candidate for o in relevant if o.answer_candidate]
                if candidates:
                    return " ".join(candidates)
                return "Based on the video clips, no clear answer could be determined."

            # Real model: use text-only synthesis via Qwen
            # Pass the synthesis prompt through the most relevant clip
            # (Qwen needs video input, so we use the top clip as visual context)
            best_obs = max(relevant, key=lambda o: o.relevance)
            # Find the matching clip info — just reuse best observation's time range
            # We need a video path, which the caller provides via answer_question_topk
            # For now, we'll raise if we can't synthesize
            logger.info("Synthesis: using %d observations", len(relevant))
            return f"Based on {len(relevant)} video clips:\n\n" + "\n\n".join(
                f"[{o.start_sec:.1f}-{o.end_sec:.1f}s]: {o.answer_candidate}" for o in relevant if o.answer_candidate
            )

        except Exception as exc:
            logger.warning("Synthesis failed: %s", exc)
            candidates = [o.answer_candidate for o in relevant if o.answer_candidate]
            if candidates:
                return " ".join(candidates)
            return "Answer synthesis failed."

    def _encode_query(self, question: str) -> np.ndarray:
        """Encode question to embedding vector.

        Args:
            question: Question text

        Returns:
            Query embedding vector (normalized for cosine similarity)
        """
        if self.retrieval_embedder is not None:
            # Use OpenCLIP text encoder (real implementation)
            logger.debug("Encoding query with OpenCLIP: %s", question[:50])
            embeddings = self.retrieval_embedder.encode_text([question])
            return embeddings[0]  # Return single embedding

        # Fallback: mock embedding
        # Determine dimension from vector service or default to 512 (OpenCLIP ViT-B-32)
        dim = 512

        logger.debug("Encoding query (mock): %s", question[:50])
        return np.random.randn(dim).astype(np.float32)

    def _flatten_search_results(
        self,
        search_results: dict[str, list[SearchResult]],
    ) -> list[SearchResult]:
        """Flatten multi-level search results to single list.

        Args:
            search_results: Dict of level -> results

        Returns:
            Flattened list of results, sorted by score (descending)
        """
        all_results: list[SearchResult] = []
        for level in ["shot", "micro", "meso", "macro"]:
            if level in search_results:
                all_results.extend(search_results[level])

        # Sort by score descending
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results

    def _rerank_results(
        self,
        question: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Re-rank results using cross-encoder or LLM scoring.

        Args:
            question: Original question
            results: Search results to re-rank

        Returns:
            Re-ranked results (sorted by new score)

        Note:
            This is a placeholder. Real implementation would:
            - Use cross-encoder model (e.g., ms-marco-MiniLM-L-6-v2)
            - Or use LLM to score relevance (slower but more accurate)
        """
        logger.debug("Re-ranking %d results (mock)", len(results))

        # TODO: Implement real re-ranking
        # Option 1: Cross-encoder
        # from sentence_transformers import CrossEncoder
        # model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        # pairs = [(question, clip_text) for clip_text in ...]
        # scores = model.predict(pairs)
        #
        # Option 2: LLM scoring
        # for result in results:
        #     prompt = f"How relevant is this clip to the question?\nQ: {question}\nClip: ..."
        #     score = llm.score(prompt)

        # Mock: just return as-is (no re-ranking)
        return results


def create_rag_service(
    *,
    vector_service: QdrantService,
    llm_service: VideoLLMService,
    retrieval_embedder: RetrievalEmbedder | None = None,
    macro_k: int = 5,
    meso_k: int = 10,
    micro_k: int = 20,
    shot_k: int = 30,
    enable_rerank: bool = False,
) -> RAGService:
    """Factory function to create RAG service with common configuration.

    Args:
        vector_service: Qdrant service
        llm_service: Video-LLM service
        retrieval_embedder: Optional retrieval embedder (OpenCLIP)
        macro_k: Top-k for macro level
        meso_k: Top-k for meso level
        micro_k: Top-k for micro level
        shot_k: Top-k for shot level
        enable_rerank: Enable re-ranking

    Returns:
        Configured RAGService
    """
    config = RetrievalConfig(
        macro_k=macro_k,
        meso_k=meso_k,
        micro_k=micro_k,
        shot_k=shot_k,
        enable_rerank=enable_rerank,
    )

    return RAGService(
        vector_service=vector_service,
        llm_service=llm_service,
        retrieval_config=config,
        retrieval_embedder=retrieval_embedder,
    )
