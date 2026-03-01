"""Clip similarity search — vectorized with numpy batch matmul."""

from typing import Any

import numpy as np

from sopilot.database import Database
from sopilot.exceptions import NotFoundError


class SearchService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def search(
        self,
        *,
        query_video_id: int,
        query_clip_index: int,
        k: int,
        task_id: str | None,
    ) -> dict[str, Any]:
        clip = self.database.get_clip(query_video_id, query_clip_index)
        if clip is None:
            raise NotFoundError("Query clip not found")

        query_vec = np.asarray(clip["embedding"], dtype=np.float32)
        task_filter = task_id if task_id else clip["task_id"]
        all_clips = self.database.iter_clips(task_id=task_filter, exclude_video_id=None)

        # Exclude the query clip itself
        candidates = [
            c for c in all_clips
            if not (c["video_id"] == query_video_id and c["clip_index"] == query_clip_index)
        ]

        if not candidates:
            return {
                "query_video_id": query_video_id,
                "query_clip_index": query_clip_index,
                "results": [],
            }

        # ── Vectorized batch cosine similarity ──────────────────────────
        # Build (N, D) embedding matrix in one pass
        embeddings = np.stack(
            [np.asarray(c["embedding"], dtype=np.float32) for c in candidates]
        )

        # L2-normalise query
        q_norm = float(np.linalg.norm(query_vec))
        query_normed = query_vec / q_norm if q_norm > 1e-12 else query_vec

        # L2-normalise all candidates (broadcast)
        row_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        row_norms = np.where(row_norms > 1e-12, row_norms, 1.0)
        normed = embeddings / row_norms

        # Single matrix-vector multiply → similarities for all candidates
        similarities = np.clip(normed @ query_normed, -1.0, 1.0)

        # Build result list and sort
        scored = [
            {
                "video_id": int(c["video_id"]),
                "clip_index": int(c["clip_index"]),
                "task_id": c["task_id"],
                "is_gold": bool(c["is_gold"]),
                "similarity": round(float(similarities[i]), 6),
                "start_sec": float(c["start_sec"]),
                "end_sec": float(c["end_sec"]),
            }
            for i, c in enumerate(candidates)
        ]

        scored.sort(key=lambda x: x["similarity"], reverse=True)  # type: ignore[arg-type, return-value]
        return {
            "query_video_id": query_video_id,
            "query_clip_index": query_clip_index,
            "results": scored[: max(k, 1)],
        }
