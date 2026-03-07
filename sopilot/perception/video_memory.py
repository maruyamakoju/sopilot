"""Video Memory Store — multi-modal frame indexing for RAG.

Phase 43: Stores DINOv2 visual embeddings (768-dim) + LLaVA text
captions for each processed frame, enabling hybrid visual+text
retrieval for the RAG engine.

Architecture::

    PerceptionEngine.process_frame()
        → Stage 4p: video_memory.add_frame(cls_token, caption, ...)
    RAGEngine.query("ヘルメット未着用を探して")
        → VideoMemoryStore.search_visual(dinov2_query)  # cosine sim
        + VideoMemoryStore.search_text("ヘルメット 未着用")  # BM25
        → RAGEngine.generate_answer(top_k_results)       # LLaVA JP

Text scoring uses BM25 (Robertson & Zaragoza, 2009):
    score(Q, D) = Σ IDF(qi) · (tf(qi,D)·(k1+1)) / (tf(qi,D)+k1·(1-b+b·|D|/avgdl))
    k1=1.5, b=0.75

Usage::

    store = VideoMemoryStore()
    fid = store.add_frame(emb_768d, caption="ヘルメット未着用の作業員を検出", cam="cam01")
    results = store.search_visual(query_emb, top_k=5)
    results = store.search_text("ヘルメット", top_k=5)
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_MAX_FRAMES = 10_000         # rolling window
_EMBEDDING_DIM = 768         # DINOv2 ViT-B CLS token
_BM25_K1 = 1.5
_BM25_B = 0.75
_TOKEN_RE = re.compile(r"[^\w\u3000-\u9fff\uff00-\uffef]+", re.UNICODE)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class MemoryFrame:
    """A single indexed perception frame."""

    frame_id: str
    camera_id: str
    timestamp: float
    visual_embedding: np.ndarray   # 768-dim, L2-normalised
    caption: str                   # LLaVA-generated Japanese/English text
    incident_id: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    tokens: tuple[str, ...] = field(default_factory=tuple, repr=False)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> list[str]:
    """Split text into lowercase tokens, including CJK characters."""
    raw = _TOKEN_RE.sub(" ", text.lower()).split()
    # Also split CJK sequences into individual characters as tokens
    tokens: list[str] = []
    for w in raw:
        if any("\u3000" <= c <= "\u9fff" or "\uff00" <= c <= "\uffef" for c in w):
            tokens.extend(list(w))  # CJK: character-level tokens
        else:
            tokens.append(w)
    return [t for t in tokens if t.strip()]


# ---------------------------------------------------------------------------
# VideoMemoryStore
# ---------------------------------------------------------------------------


class VideoMemoryStore:
    """Multi-modal vector store for perception frames.

    Maintains a rolling window of up to ``max_frames`` frames with:
    - Visual index: L2-normalised DINOv2 embeddings for cosine similarity
    - Text index: BM25 inverted index over LLaVA captions

    Parameters
    ----------
    max_frames:
        Maximum frames to retain (rolling window, oldest dropped first).
    embedding_dim:
        Visual embedding dimension (default 768 for DINOv2 ViT-B).
    persist_path:
        Optional JSON path for persistence (embeddings stored as base64).
    """

    def __init__(
        self,
        max_frames: int = _MAX_FRAMES,
        embedding_dim: int = _EMBEDDING_DIM,
        persist_path: Path | None = None,
    ) -> None:
        self._max_frames = max_frames
        self._embedding_dim = embedding_dim
        self._persist_path = persist_path

        self._lock = threading.Lock()
        self._frames: deque[MemoryFrame] = deque(maxlen=max_frames)
        self._frame_index: dict[str, MemoryFrame] = {}  # frame_id → frame

        # BM25 text index
        self._inverted: dict[str, list[tuple[str, int]]] = defaultdict(list)
        # term → [(frame_id, tf), ...]
        self._doc_len: dict[str, int] = {}   # frame_id → token count
        self._df: dict[str, int] = defaultdict(int)  # term → document freq
        self._avg_dl: float = 0.0
        self._total_docs: int = 0

        self._add_count = 0
        self._search_count = 0
        self._created_at = time.time()

        if persist_path:
            self._load()

        logger.info(
            "VideoMemoryStore created (max=%d, dim=%d, persist=%s)",
            max_frames,
            embedding_dim,
            persist_path,
        )

    # ── Add ────────────────────────────────────────────────────────────

    def add_frame(
        self,
        visual_embedding: np.ndarray,
        caption: str,
        camera_id: str = "cam",
        timestamp: float | None = None,
        incident_id: str | None = None,
        tags: tuple[str, ...] = (),
        frame_id: str | None = None,
    ) -> str:
        """Index a new frame.

        Parameters
        ----------
        visual_embedding:
            DINOv2 CLS token (768-dim float32). Will be L2-normalised.
        caption:
            LLaVA-generated description of the frame.
        camera_id:
            Source camera identifier.
        timestamp:
            Unix timestamp (default: now).
        incident_id:
            Associated incident UUID, if any.
        tags:
            Free-form tags for filtering.

        Returns
        -------
        str
            Unique frame ID (UUID4).
        """
        fid = frame_id or str(uuid.uuid4())
        ts = timestamp or time.time()

        # L2-normalise visual embedding
        emb = np.asarray(visual_embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm

        tokens = tuple(_tokenise(caption))

        frame = MemoryFrame(
            frame_id=fid,
            camera_id=camera_id,
            timestamp=ts,
            visual_embedding=emb,
            caption=caption,
            incident_id=incident_id,
            tags=tags,
            tokens=tokens,
        )

        with self._lock:
            # Evict oldest if needed
            if len(self._frames) == self._max_frames:
                old = self._frames[0]  # leftmost (oldest)
                self._evict(old)

            self._frames.append(frame)
            self._frame_index[fid] = frame
            self._index_text(frame)
            self._add_count += 1

        logger.debug(
            "VideoMemoryStore: added frame %s (cam=%s, tokens=%d, total=%d)",
            fid[:8],
            camera_id,
            len(tokens),
            len(self._frames),
        )
        return fid

    def _evict(self, frame: MemoryFrame) -> None:
        """Remove a frame from text + visual indexes (called under lock)."""
        fid = frame.frame_id
        self._frame_index.pop(fid, None)
        self._doc_len.pop(fid, None)
        # Rebuild DF for evicted tokens (best-effort: decrement, floor 0)
        for token in set(frame.tokens):
            if self._df[token] > 0:
                self._df[token] -= 1
            # Rebuild inverted entry (remove this frame_id)
            self._inverted[token] = [
                (fid2, tf) for fid2, tf in self._inverted[token] if fid2 != fid
            ]

    def _index_text(self, frame: MemoryFrame) -> None:
        """Update BM25 text index (called under lock)."""
        fid = frame.frame_id
        tf_map: dict[str, int] = defaultdict(int)
        for t in frame.tokens:
            tf_map[t] += 1
        self._doc_len[fid] = len(frame.tokens)
        # Update DF and inverted index
        for term, tf in tf_map.items():
            if not any(f == fid for f, _ in self._inverted[term]):
                self._df[term] += 1
            self._inverted[term].append((fid, tf))
        # Recalculate avg_dl
        n = len(self._doc_len)
        self._total_docs = n
        self._avg_dl = sum(self._doc_len.values()) / max(1, n)

    # ── Visual search ──────────────────────────────────────────────────

    def search_visual(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        camera_filter: str | None = None,
        time_after: float | None = None,
        time_before: float | None = None,
    ) -> list[tuple[MemoryFrame, float]]:
        """Search by visual similarity (cosine).

        Returns
        -------
        list[tuple[MemoryFrame, float]]
            Sorted by similarity descending, score in [-1, 1].
        """
        q = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 1e-8:
            q = q / norm

        with self._lock:
            frames = list(self._frames)
            self._search_count += 1

        # Apply filters
        if camera_filter:
            frames = [f for f in frames if f.camera_id == camera_filter]
        if time_after is not None:
            frames = [f for f in frames if f.timestamp >= time_after]
        if time_before is not None:
            frames = [f for f in frames if f.timestamp <= time_before]

        if not frames:
            return []

        # Vectorised cosine similarity
        emb_matrix = np.stack([f.visual_embedding for f in frames])  # (N, D)
        scores = (emb_matrix @ q).tolist()

        ranked = sorted(zip(frames, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ── Text search (BM25) ─────────────────────────────────────────────

    def search_text(
        self,
        query: str,
        top_k: int = 10,
        camera_filter: str | None = None,
        time_after: float | None = None,
        time_before: float | None = None,
    ) -> list[tuple[MemoryFrame, float]]:
        """Search by text similarity (BM25).

        Returns
        -------
        list[tuple[MemoryFrame, float]]
            Sorted by BM25 score descending.
        """
        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        with self._lock:
            frames_set = set(self._frame_index.keys())
            total = max(1, self._total_docs)
            avg_dl = max(1.0, self._avg_dl)
            # Collect scores per frame
            scores: dict[str, float] = defaultdict(float)

            for token in query_tokens:
                df = self._df.get(token, 0)
                if df == 0:
                    continue
                idf = np.log((total - df + 0.5) / (df + 0.5) + 1.0)
                for fid, tf in self._inverted.get(token, []):
                    if fid not in frames_set:
                        continue
                    dl = self._doc_len.get(fid, 1)
                    denom = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avg_dl)
                    scores[fid] += idf * (tf * (_BM25_K1 + 1)) / denom

            snapshot = dict(self._frame_index)
            self._search_count += 1

        # Apply filters
        ranked = []
        for fid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            frame = snapshot.get(fid)
            if frame is None:
                continue
            if camera_filter and frame.camera_id != camera_filter:
                continue
            if time_after is not None and frame.timestamp < time_after:
                continue
            if time_before is not None and frame.timestamp > time_before:
                continue
            ranked.append((frame, score))

        return ranked[:top_k]

    # ── Accessors ──────────────────────────────────────────────────────

    def get_frame(self, frame_id: str) -> MemoryFrame | None:
        with self._lock:
            return self._frame_index.get(frame_id)

    def get_count(self) -> int:
        with self._lock:
            return len(self._frames)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "frame_count": len(self._frames),
                "max_frames": self._max_frames,
                "total_added": self._add_count,
                "total_searches": self._search_count,
                "vocabulary_size": len(self._df),
                "avg_caption_tokens": round(self._avg_dl, 1),
                "uptime_seconds": round(time.time() - self._created_at, 1),
            }

    # ── Persistence ────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist recent frames to JSON (best-effort, base64 embeddings)."""
        import base64

        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            frames_data = [
                {
                    "frame_id": f.frame_id,
                    "camera_id": f.camera_id,
                    "timestamp": f.timestamp,
                    "visual_b64": base64.b64encode(
                        f.visual_embedding.astype(np.float32).tobytes()
                    ).decode("ascii"),
                    "caption": f.caption,
                    "incident_id": f.incident_id,
                    "tags": list(f.tags),
                }
                for f in self._frames
            ]

        try:
            with open(self._persist_path, "w", encoding="utf-8") as fp:
                json.dump({"frames": frames_data}, fp)
            logger.debug("VideoMemoryStore: saved %d frames", len(frames_data))
        except Exception:
            logger.debug("VideoMemoryStore: save failed", exc_info=True)

    def _load(self) -> None:
        """Load persisted frames from JSON."""
        import base64

        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            for e in data.get("frames", []):
                raw = base64.b64decode(e["visual_b64"])
                emb = np.frombuffer(raw, dtype=np.float32).copy()
                if len(emb) != self._embedding_dim:
                    continue
                self.add_frame(
                    visual_embedding=emb,
                    caption=e["caption"],
                    camera_id=e["camera_id"],
                    timestamp=e["timestamp"],
                    incident_id=e.get("incident_id"),
                    tags=tuple(e.get("tags", [])),
                    frame_id=e["frame_id"],
                )
            logger.info(
                "VideoMemoryStore: loaded %d frames from %s",
                len(self._frames),
                self._persist_path,
            )
        except Exception:
            logger.debug("VideoMemoryStore: load failed", exc_info=True)
