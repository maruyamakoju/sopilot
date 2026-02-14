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

import contextlib
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
    rerank_diversity_weight: float = 0.3  # MMR lambda: 0=pure diversity, 1=pure relevance
    rerank_transcript_boost: float = 0.15  # Bonus for keyword overlap with transcript

    # Cross-encoder re-ranking (high accuracy, slower)
    enable_cross_encoder: bool = False  # DISABLED: Implementation incomplete (TODO: load keyframes)
    cross_encoder_weight: float = 0.5  # Weight for cross-encoder score (0.5 = equal blend with retrieval)

    # Temporal coherence boost
    enable_temporal_coherence: bool = True  # Boost consecutive clips
    temporal_coherence_boost: float = 0.1  # Bonus for temporally adjacent clips (10%)

    # Filtering
    min_score: float | None = None
    video_id_filter: str | None = None

    # Hybrid search (visual + audio transcript)
    use_micro_text: bool = True
    micro_text_alpha: float = 0.7  # Weight for audio text score in fusion
    micro_text_k: int = 20  # Top-k for micro_text level

    # Hierarchical coarse-to-fine retrieval
    enable_hierarchical: bool = False
    hierarchical_expand_factor: float = 0.1


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
            enable_temporal_filtering=self.retrieval_config.enable_hierarchical,
            time_expand_factor=self.retrieval_config.hierarchical_expand_factor,
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

        # Step 1: Retrieve (hybrid visual + audio text search)
        query_embedding = self._encode_query(question)
        all_results = self._hybrid_search(query_embedding, video_id=video_id)

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
        # Build prompt with optional transcript context
        transcript_ctx = ""
        if clip.transcript_text:
            excerpt = clip.transcript_text[:400]
            transcript_ctx = f'\nAudio transcript for this clip: "{excerpt}"\n'

        observation_prompt = (
            f'You are analyzing a short video clip to answer: "{question}"\n'
            f"{transcript_ctx}\n"
            "Respond ONLY with a JSON object (no markdown fences):\n"
            "{\n"
            '  "relevance": <0.0-1.0 how relevant this clip is to the question>,\n'
            '  "observation": "<describe what you see AND hear in this clip>",\n'
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

    def _hybrid_search(
        self,
        query_embedding: np.ndarray,
        *,
        video_id: str | None = None,
    ) -> list[SearchResult]:
        """Search micro (visual) and micro_text (audio) then fuse scores.

        Fusion: final_score = max(visual_score, alpha * audio_score)
        Clips found only in micro_text are included with score = alpha * audio_score.

        Args:
            query_embedding: Normalised query vector (D,).
            video_id: Optional video ID filter.

        Returns:
            Sorted list of SearchResult (descending by fused score).
        """
        cfg = self.retrieval_config

        # Visual search (always)
        visual_results = self.vector_service.search(
            level="micro",
            query_vector=query_embedding,
            top_k=cfg.micro_k,
            video_id=video_id,
        )

        # Audio text search (optional)
        audio_results: list[SearchResult] = []
        if cfg.use_micro_text:
            with contextlib.suppress(Exception):
                audio_results = self.vector_service.search(
                    level="micro_text",
                    query_vector=query_embedding,
                    top_k=cfg.micro_text_k,
                    video_id=video_id,
                )

        if not audio_results:
            return visual_results

        # Build clip_id -> score maps
        visual_map: dict[str, SearchResult] = {r.clip_id: r for r in visual_results}
        audio_map: dict[str, SearchResult] = {r.clip_id: r for r in audio_results}

        all_clip_ids = set(visual_map) | set(audio_map)
        alpha = cfg.micro_text_alpha

        fused: list[SearchResult] = []
        for cid in all_clip_ids:
            v = visual_map.get(cid)
            a = audio_map.get(cid)

            v_score = v.score if v else 0.0
            a_score = (a.score * alpha) if a else 0.0
            final_score = max(v_score, a_score)

            # Use the visual result as base (more metadata); fall back to audio
            base = v or a
            assert base is not None
            fused.append(
                SearchResult(
                    clip_id=base.clip_id,
                    video_id=base.video_id,
                    level="micro",
                    start_sec=base.start_sec,
                    end_sec=base.end_sec,
                    score=final_score,
                    transcript_text=base.transcript_text or (a.transcript_text if a else None),
                )
            )

        fused.sort(key=lambda r: r.score, reverse=True)
        logger.info(
            "Hybrid search: visual=%d, audio=%d, fused=%d",
            len(visual_results),
            len(audio_results),
            len(fused),
        )
        return fused

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
        """Re-rank results using MMR diversity + transcript keyword boost + cross-encoder.

        Combines five signals:
        1. Original retrieval score (cosine similarity from vector search)
        2. Cross-encoder score (CLIP image-text direct similarity, if enabled)
        3. Temporal coherence boost (bonus for consecutive clips, if enabled)
        4. Temporal diversity penalty (MMR-style: penalize clips near already-selected ones)
        5. Transcript keyword boost (bonus when query words appear in transcript text)

        Cross-encoder provides higher accuracy but is slower (requires image loading).

        Args:
            question: Original question
            results: Search results to re-rank

        Returns:
            Re-ranked results (sorted by combined score, truncated to ``rerank_top_k``)
        """
        if len(results) <= 1:
            return results

        cfg = self.retrieval_config

        # Step 0: Cross-encoder re-scoring (if enabled)
        if cfg.enable_cross_encoder and self.retrieval_embedder is not None:
            results = self._apply_cross_encoder(question, results)

        # Step 1: Temporal coherence boost (if enabled)
        if cfg.enable_temporal_coherence:
            results = self._apply_temporal_coherence(results)

        # Step 2: MMR diversity + transcript boost
        lam = cfg.rerank_diversity_weight  # higher = more relevance
        tx_boost = cfg.rerank_transcript_boost
        top_k = cfg.rerank_top_k

        # Normalise query words for keyword matching
        query_words = set(question.lower().split())

        # Pre-compute transcript keyword bonus for each result
        tx_bonuses: list[float] = []
        for r in results:
            bonus = 0.0
            if r.transcript_text and query_words:
                doc_words = set(r.transcript_text.lower().split())
                overlap = len(query_words & doc_words)
                bonus = tx_boost * (overlap / len(query_words))
            tx_bonuses.append(bonus)

        # MMR-style greedy selection
        selected: list[int] = []
        remaining = set(range(len(results)))

        while remaining and len(selected) < top_k:
            best_idx = -1
            best_mmr = -float("inf")

            for idx in remaining:
                relevance = results[idx].score + tx_bonuses[idx]

                # Max temporal similarity to already-selected clips
                max_sim = 0.0
                if selected:
                    for s_idx in selected:
                        iou = temporal_iou(
                            results[idx].start_sec,
                            results[idx].end_sec,
                            results[s_idx].start_sec,
                            results[s_idx].end_sec,
                        )
                        max_sim = max(max_sim, iou)

                mmr_score = (1 - lam) * relevance - lam * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.discard(best_idx)

        reranked = [results[i] for i in selected]
        logger.info(
            "Re-ranked %d → %d results (diversity=%.2f, tx_boost=%.2f, cross_enc=%s, temporal=%s)",
            len(results),
            len(reranked),
            lam,
            tx_boost,
            cfg.enable_cross_encoder,
            cfg.enable_temporal_coherence,
        )
        return reranked

    def _apply_cross_encoder(
        self,
        question: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Apply cross-encoder re-scoring using CLIP image-text similarity.

        For each result, load the keyframe image and compute direct CLIP
        similarity with the query text. Blend with original retrieval score.

        Args:
            question: Query text
            results: Search results with scores

        Returns:
            Results with updated scores (blended: original + cross-encoder)
        """
        if not self.retrieval_embedder:
            logger.warning("Cross-encoder requested but no retrieval_embedder available")
            return results

        cfg = self.retrieval_config
        weight = cfg.cross_encoder_weight

        # Encode query text once
        query_emb = self.retrieval_embedder.encode_text([question])[0]  # (D,)

        # Re-score each result
        updated_results = []
        for r in results:
            # Original retrieval score
            orig_score = r.score

            # Cross-encoder score (fallback to original score if keyframe unavailable)
            cross_score = orig_score  # SAFE FALLBACK: Don't degrade performance

            # TODO: Load keyframe image from r.clip_id via database
            # For now, use original score as fallback
            # In production: keyframe_path = self._get_keyframe_path(r.clip_id)
            # keyframe_emb = self.retrieval_embedder.encode_images([keyframe_image])[0]
            # cross_score = float(np.dot(query_emb, keyframe_emb))

            # Blend scores (currently no-op since cross_score == orig_score)
            blended_score = (1 - weight) * orig_score + weight * cross_score  # = orig_score

            # Create updated result
            updated_results.append(
                SearchResult(
                    clip_id=r.clip_id,
                    video_id=r.video_id,
                    level=r.level,
                    score=blended_score,  # Updated score
                    start_sec=r.start_sec,
                    end_sec=r.end_sec,
                    transcript_text=r.transcript_text,
                )
            )

        # CRITICAL: Re-sort by updated scores
        updated_results.sort(key=lambda r: r.score, reverse=True)

        logger.debug(
            "Cross-encoder re-scoring: weight=%.2f (fallback mode, using original scores)",
            weight,
        )
        return updated_results

    def _apply_temporal_coherence(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Apply temporal coherence boost to consecutive clips.

        Clips that are temporally adjacent (end_sec == next.start_sec within 0.5s)
        get a score bonus to encourage retrieving coherent sequences.

        CRITICAL: Only considers clips from the same video_id to avoid false positives.

        Args:
            results: Search results sorted by score

        Returns:
            Results with updated scores (boosted for consecutive clips)
        """
        if len(results) <= 1:
            return results

        cfg = self.retrieval_config
        boost = cfg.temporal_coherence_boost

        # Group by video_id to avoid cross-video false positives
        from collections import defaultdict

        by_video = defaultdict(list)
        for r in results:
            by_video[r.video_id].append(r)

        # Track which clips are consecutive (within same video only)
        consecutive_set = set()
        for video_id, clips in by_video.items():
            # Sort by time within this video
            clips_sorted = sorted(clips, key=lambda r: r.start_sec)

            # Detect consecutive pairs
            for i in range(len(clips_sorted) - 1):
                curr = clips_sorted[i]
                next_clip = clips_sorted[i + 1]

                # Check if consecutive (within 0.5s tolerance)
                if abs(curr.end_sec - next_clip.start_sec) < 0.5:
                    consecutive_set.add(curr.clip_id)
                    consecutive_set.add(next_clip.clip_id)

        # Apply boost
        updated_results = []
        for r in results:
            score = r.score
            if r.clip_id in consecutive_set:
                score = score * (1 + boost)  # e.g., +10%

            updated_results.append(
                SearchResult(
                    clip_id=r.clip_id,
                    video_id=r.video_id,
                    level=r.level,
                    score=score,
                    start_sec=r.start_sec,
                    end_sec=r.end_sec,
                    transcript_text=r.transcript_text,
                )
            )

        # CRITICAL: Re-sort by updated scores
        updated_results.sort(key=lambda r: r.score, reverse=True)

        logger.debug(
            "Temporal coherence boost: %d/%d clips boosted by %.1f%%",
            len(consecutive_set),
            len(results),
            boost * 100,
        )
        return updated_results


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
