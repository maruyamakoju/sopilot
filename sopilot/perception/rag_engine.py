"""RAG Engine — hybrid visual+text retrieval over video memory.

Phase 43: Implements retrieval-augmented generation for natural language
queries over the camera fleet's visual memory.

Pipeline::

    User query: "過去1時間でヘルメット未着用の事例を見せて"
        ↓
    1. Visual search: embed query frame (if provided) → cosine sim
    2. Text search: BM25 over LLaVA captions
    3. Hybrid fusion: α × visual_score + (1-α) × normalised_text_score
    4. Answer generation: LLaVA generates JP answer over top-k captions

Usage::

    engine = RAGEngine(
        memory_store=store,
        llava_client=llava,
        alpha=0.6,      # visual weight (0=text-only, 1=visual-only)
    )
    result = engine.full_query(
        query_text="エリアBで転倒した人物は誰？",
        top_k=5,
    )
    print(result["answer"])   # Japanese LLaVA-generated answer
    for r in result["results"]:
        print(r["caption"], r["combined_score"])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .video_memory import MemoryFrame, VideoMemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RAGResult:
    """A single retrieval result from RAGEngine.query()."""

    frame_id: str
    camera_id: str
    timestamp: float
    caption: str
    visual_score: float = 0.0
    text_score: float = 0.0
    combined_score: float = 0.0
    incident_id: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "caption": self.caption,
            "visual_score": round(self.visual_score, 4),
            "text_score": round(self.text_score, 4),
            "combined_score": round(self.combined_score, 4),
            "incident_id": self.incident_id,
            "tags": list(self.tags),
        }


# ---------------------------------------------------------------------------
# RAGEngine
# ---------------------------------------------------------------------------


class RAGEngine:
    """Hybrid visual+text RAG engine over VideoMemoryStore.

    Parameters
    ----------
    memory_store:
        :class:`VideoMemoryStore` instance.
    llava_client:
        Optional LLaVA client for answer generation.
    alpha:
        Visual weight in hybrid score: ``alpha × visual + (1-alpha) × text``.
        Default 0.6 (visual-dominant for camera queries).
    answer_max_contexts:
        Maximum retrieved captions to feed into LLaVA for answer generation.
    """

    def __init__(
        self,
        memory_store: VideoMemoryStore,
        llava_client: Any = None,
        alpha: float = 0.6,
        answer_max_contexts: int = 5,
    ) -> None:
        self._store = memory_store
        self._llava = llava_client
        self._alpha = max(0.0, min(1.0, alpha))
        self._max_ctx = answer_max_contexts
        self._query_count = 0
        self._avg_query_ms = 0.0
        self._created_at = time.time()

        logger.info(
            "RAGEngine created (alpha=%.2f, max_ctx=%d, llava=%s)",
            self._alpha,
            self._max_ctx,
            "yes" if llava_client else "no",
        )

    # ── Core query ─────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        query_visual: np.ndarray | None = None,
        top_k: int = 10,
        camera_filter: str | None = None,
        time_after: float | None = None,
        time_before: float | None = None,
    ) -> list[RAGResult]:
        """Retrieve relevant frames using hybrid visual+text search.

        Parameters
        ----------
        query_text:
            Natural language query (Japanese or English).
        query_visual:
            Optional 768-dim DINOv2 query embedding for visual search.
        top_k:
            Number of results to return.
        camera_filter:
            Restrict to a specific camera ID.
        time_after:
            Only include frames after this Unix timestamp.
        time_before:
            Only include frames before this Unix timestamp.

        Returns
        -------
        list[RAGResult]
            Sorted by combined_score descending.
        """
        candidate_k = top_k * 3  # over-retrieve then fuse

        # --- Visual search ---
        visual_map: dict[str, float] = {}
        if query_visual is not None and self._alpha > 0:
            vis_results = self._store.search_visual(
                query_visual,
                top_k=candidate_k,
                camera_filter=camera_filter,
                time_after=time_after,
                time_before=time_before,
            )
            for frame, score in vis_results:
                visual_map[frame.frame_id] = float(score)

        # --- Text search ---
        text_map: dict[str, float] = {}
        text_raw: dict[str, float] = {}
        if self._alpha < 1.0 and query_text:
            text_results = self._store.search_text(
                query_text,
                top_k=candidate_k,
                camera_filter=camera_filter,
                time_after=time_after,
                time_before=time_before,
            )
            if text_results:
                max_score = max(s for _, s in text_results)
                for frame, score in text_results:
                    norm_score = score / max(max_score, 1e-8)
                    text_map[frame.frame_id] = norm_score
                    text_raw[frame.frame_id] = score

        # --- Collect union of frame IDs ---
        all_ids = set(visual_map.keys()) | set(text_map.keys())

        results: list[RAGResult] = []
        for fid in all_ids:
            frame = self._store.get_frame(fid)
            if frame is None:
                continue
            v_score = visual_map.get(fid, 0.0)
            t_score = text_map.get(fid, 0.0)

            if query_visual is not None and self._alpha > 0:
                combined = self._alpha * v_score + (1 - self._alpha) * t_score
            else:
                combined = t_score  # text-only when no visual query

            results.append(
                RAGResult(
                    frame_id=fid,
                    camera_id=frame.camera_id,
                    timestamp=frame.timestamp,
                    caption=frame.caption,
                    visual_score=v_score,
                    text_score=text_raw.get(fid, t_score),
                    combined_score=combined,
                    incident_id=frame.incident_id,
                    tags=frame.tags,
                )
            )

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]

    # ── Answer generation ──────────────────────────────────────────────

    def generate_answer(
        self,
        query_text: str,
        results: list[RAGResult],
    ) -> str:
        """Generate a Japanese answer over retrieved captions using LLaVA.

        If LLaVA is unavailable, returns a structured text summary.

        Parameters
        ----------
        query_text:
            Original user query.
        results:
            Retrieved RAGResult list (will use top ``answer_max_contexts``).

        Returns
        -------
        str
            Japanese answer or summary.
        """
        if not results:
            return f"「{query_text}」に該当するフレームは見つかりませんでした。"

        top = results[: self._max_ctx]
        context_parts = []
        for i, r in enumerate(top, 1):
            ts = _fmt_ts(r.timestamp)
            context_parts.append(
                f"[{i}] {ts} {r.camera_id}: {r.caption}"
            )
        context_text = "\n".join(context_parts)

        if self._llava is not None:
            try:
                prompt = (
                    f"以下はカメラ映像の観測記録です。\n\n"
                    f"{context_text}\n\n"
                    f"質問: {query_text}\n"
                    f"上記の記録に基づき、日本語で簡潔に回答してください。"
                )
                answer = self._llava.answer_question(None, prompt)
                if answer and len(answer) > 5:
                    return answer
            except Exception:
                logger.debug("RAGEngine: LLaVA answer_question failed", exc_info=True)

        # Fallback: structured JP summary
        lines = [f"「{query_text}」の検索結果（{len(results)}件）:"]
        for r in top:
            ts = _fmt_ts(r.timestamp)
            lines.append(f"  • {ts} [{r.camera_id}] {r.caption} (スコア: {r.combined_score:.3f})")
        return "\n".join(lines)

    # ── Full pipeline ──────────────────────────────────────────────────

    def full_query(
        self,
        query_text: str,
        query_visual: np.ndarray | None = None,
        top_k: int = 10,
        camera_filter: str | None = None,
        time_after: float | None = None,
        time_before: float | None = None,
    ) -> dict[str, Any]:
        """Retrieve results and generate an answer in one call.

        Returns
        -------
        dict
            Keys: ``query``, ``results``, ``answer``, ``total_results``,
            ``query_time_ms``, ``memory_size``.
        """
        t0 = time.perf_counter()
        results = self.query(
            query_text=query_text,
            query_visual=query_visual,
            top_k=top_k,
            camera_filter=camera_filter,
            time_after=time_after,
            time_before=time_before,
        )
        answer = self.generate_answer(query_text, results)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Running average query latency
        self._query_count += 1
        self._avg_query_ms = (
            self._avg_query_ms * (self._query_count - 1) + elapsed_ms
        ) / self._query_count

        logger.info(
            "RAGEngine: query=%r → %d results in %.1fms",
            query_text[:40],
            len(results),
            elapsed_ms,
        )

        return {
            "query": query_text,
            "results": [r.to_dict() for r in results],
            "answer": answer,
            "total_results": len(results),
            "query_time_ms": round(elapsed_ms, 1),
            "memory_size": self._store.get_count(),
        }

    # ── Stats ──────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        return {
            "query_count": self._query_count,
            "avg_query_ms": round(self._avg_query_ms, 1),
            "alpha": self._alpha,
            "memory_size": self._store.get_count(),
            "llava_available": self._llava is not None,
            "uptime_seconds": round(time.time() - self._created_at, 1),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_ts(ts: float) -> str:
    """Format Unix timestamp as HH:MM:SS."""
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
