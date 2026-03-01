"""Video and clip CRUD operations."""

from __future__ import annotations

import json
from typing import Any

from sopilot.repositories.base import ConnectFactory, RepositoryBase
from sopilot.types import (
    ClipRow,
    JoinedClipRow,
    VideoListRow,
    VideoRow,
)


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


class VideoRepository(RepositoryBase):
    """Video and clip persistence layer."""

    def __init__(self, connect: ConnectFactory) -> None:
        super().__init__(connect)

    # ------------------------------------------------------------------
    # Videos
    # ------------------------------------------------------------------

    def insert_video(
        self,
        task_id: str,
        site_id: str | None,
        camera_id: str | None,
        operator_id_hash: str | None,
        recorded_at: str | None,
        is_gold: bool,
        original_filename: str | None = None,
    ) -> int:
        now = _utc_now_iso()
        cursor = self._execute(
            """
            INSERT INTO videos (
                task_id, site_id, camera_id, operator_id_hash, recorded_at,
                is_gold, status, original_filename, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'processing', ?, ?, ?)
            """,
            (task_id, site_id, camera_id, operator_id_hash, recorded_at, int(is_gold), original_filename, now, now),
        )
        return cursor.lastrowid or 0

    def finalize_video(
        self,
        video_id: int,
        file_path: str,
        step_boundaries: list[int],
        clips: list[dict[str, Any]],
        embedding_model: str,
    ) -> None:
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute("DELETE FROM clips WHERE video_id = ?", (video_id,))
            conn.executemany(
                """
                INSERT INTO clips (
                    video_id, clip_index, start_sec, end_sec, embedding_json, quality_flag, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        video_id,
                        clip["clip_index"],
                        clip["start_sec"],
                        clip["end_sec"],
                        json.dumps(clip["embedding"]),
                        clip.get("quality_flag"),
                        now,
                    )
                    for clip in clips
                ],
            )
            conn.execute(
                """
                UPDATE videos
                SET file_path = ?, status = 'ready', clip_count = ?, step_boundaries_json = ?,
                    embedding_model = ?, updated_at = ?, error = NULL
                WHERE id = ?
                """,
                (file_path, len(clips), json.dumps(step_boundaries), embedding_model, now, video_id),
            )

    def fail_video(self, video_id: int, error: str) -> None:
        self._execute(
            "UPDATE videos SET status = 'failed', error = ?, updated_at = ? WHERE id = ?",
            (error, _utc_now_iso(), video_id),
        )

    def get_video(self, video_id: int) -> VideoRow | None:
        item = self._fetch_one("SELECT * FROM videos WHERE id = ?", (video_id,))
        if item is None:
            return None
        self._parse_bool(item, "is_gold")
        return item  # type: ignore[return-value]

    def get_gold_version(self, video_id: int, task_id: str) -> int | None:
        """Return the 1-based chronological version number of a gold video within its task."""
        row = self._fetch_one(
            """SELECT COUNT(*) AS cnt FROM videos
               WHERE task_id = ? AND is_gold = 1 AND id <= ?""",
            (task_id, video_id),
        )
        return int(row["cnt"]) if row else None

    def list_videos(
        self,
        *,
        task_id: str | None = None,
        site_id: str | None = None,
        is_gold: bool | None = None,
        limit: int = 200,
    ) -> list[VideoListRow]:
        query = """
            SELECT id, task_id, is_gold, status, site_id, camera_id, operator_id_hash, recorded_at, created_at, clip_count, original_filename,
                   CASE WHEN is_gold = 1 THEN (
                       SELECT COUNT(*) FROM videos v2
                       WHERE v2.task_id = videos.task_id AND v2.is_gold = 1 AND v2.id <= videos.id
                   ) ELSE NULL END AS gold_version
            FROM videos
            WHERE 1=1
        """
        params: list[Any] = []
        if task_id:
            query += " AND task_id = ?"
            params.append(task_id)
        if site_id:
            query += " AND site_id = ?"
            params.append(site_id)
        if is_gold is not None:
            query += " AND is_gold = ?"
            params.append(int(is_gold))
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(max(1, limit))

        rows = self._fetch_all(query, tuple(params))
        for item in rows:
            self._parse_bool(item, "is_gold")
        return rows  # type: ignore[return-value]

    def delete_video(self, video_id: int, *, force: bool = False) -> bool:
        """Delete a video and its clips. Returns True if the video existed.

        When ``force=True``, also deletes any score jobs and reviews that
        reference this video.  Without ``force``, raises ``ValueError``
        if score jobs reference the video.
        """
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM videos WHERE id = ?", (video_id,)).fetchone()
            if row is None:
                return False
            # Check for referenced score jobs
            ref = conn.execute(
                "SELECT COUNT(*) FROM score_jobs WHERE gold_video_id = ? OR trainee_video_id = ?",
                (video_id, video_id),
            ).fetchone()
            ref_count = int(ref[0]) if ref else 0
            if ref_count > 0 and not force:
                raise ValueError(f"Video {video_id} is referenced by {ref_count} score job(s)")
            if ref_count > 0 and force:
                # Delete reviews first (FK constraint), then score jobs
                job_ids = conn.execute(
                    "SELECT id FROM score_jobs WHERE gold_video_id = ? OR trainee_video_id = ?",
                    (video_id, video_id),
                ).fetchall()
                for jrow in job_ids:
                    conn.execute("DELETE FROM score_reviews WHERE job_id = ?", (jrow[0],))
                conn.execute(
                    "DELETE FROM score_jobs WHERE gold_video_id = ? OR trainee_video_id = ?",
                    (video_id, video_id),
                )
            conn.execute("DELETE FROM clips WHERE video_id = ?", (video_id,))
            conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
            return True

    def update_video_metadata(
        self,
        video_id: int,
        *,
        site_id: str | None = None,
        camera_id: str | None = None,
        operator_id_hash: str | None = None,
        recorded_at: str | None = None,
    ) -> bool:
        """Update mutable metadata fields on a video. Returns True if the video existed."""
        updates: list[str] = []
        params: list[Any] = []
        for col, val in [
            ("site_id", site_id),
            ("camera_id", camera_id),
            ("operator_id_hash", operator_id_hash),
            ("recorded_at", recorded_at),
        ]:
            if val is not None:
                updates.append(f"{col} = ?")
                params.append(val)
        if not updates:
            return self.get_video(video_id) is not None
        updates.append("updated_at = ?")
        params.append(_utc_now_iso())
        params.append(video_id)
        with self._connect() as conn:
            result = conn.execute(
                f"UPDATE videos SET {', '.join(updates)} WHERE id = ?",
                tuple(params),
            )
            return result.rowcount == 1

    def count_videos(
        self,
        *,
        task_id: str | None = None,
        is_gold: bool | None = None,
        status: str | None = None,
    ) -> int:
        query = "SELECT COUNT(*) FROM videos WHERE 1=1"
        params: list[Any] = []
        if task_id:
            query += " AND task_id = ?"
            params.append(task_id)
        if is_gold is not None:
            query += " AND is_gold = ?"
            params.append(int(is_gold))
        if status:
            query += " AND status = ?"
            params.append(status)
        with self._connect() as conn:
            row = conn.execute(query, tuple(params)).fetchone()
            return int(row[0]) if row else 0

    def count_videos_by_site(self, *, task_id: str | None = None) -> dict[str, int]:
        query = "SELECT site_id, COUNT(*) AS cnt FROM videos WHERE 1=1"
        params: list[Any] = []
        if task_id:
            query += " AND task_id = ?"
            params.append(task_id)
        query += " GROUP BY site_id"
        rows = self._fetch_all(query, tuple(params))
        return {(row["site_id"] or "unknown"): int(row["cnt"]) for row in rows}

    # ------------------------------------------------------------------
    # Clips
    # ------------------------------------------------------------------

    def get_video_clips(self, video_id: int) -> list[ClipRow]:
        rows = self._fetch_all(
            """
            SELECT clip_index, start_sec, end_sec, embedding_json, quality_flag
            FROM clips
            WHERE video_id = ?
            ORDER BY clip_index ASC
            """,
            (video_id,),
        )
        for item in rows:
            self._parse_json(item, "embedding_json", "embedding")
        return rows  # type: ignore[return-value]

    def get_clip(self, video_id: int, clip_index: int) -> JoinedClipRow | None:
        item = self._fetch_one(
            """
            SELECT c.video_id, c.clip_index, c.start_sec, c.end_sec, c.embedding_json,
                   v.task_id, v.is_gold
            FROM clips c
            JOIN videos v ON v.id = c.video_id
            WHERE c.video_id = ? AND c.clip_index = ?
            """,
            (video_id, clip_index),
        )
        if item is None:
            return None
        self._parse_bool(item, "is_gold")
        self._parse_json(item, "embedding_json", "embedding")
        return item  # type: ignore[return-value]

    def iter_clips(
        self,
        task_id: str | None = None,
        exclude_video_id: int | None = None,
    ) -> list[JoinedClipRow]:
        query = """
            SELECT c.video_id, c.clip_index, c.start_sec, c.end_sec, c.embedding_json,
                   v.task_id, v.is_gold
            FROM clips c
            JOIN videos v ON v.id = c.video_id
            WHERE 1=1
        """
        params: list[Any] = []
        if task_id:
            query += " AND v.task_id = ?"
            params.append(task_id)
        if exclude_video_id is not None:
            query += " AND c.video_id != ?"
            params.append(exclude_video_id)
        query += " ORDER BY c.video_id, c.clip_index"

        rows = self._fetch_all(query, tuple(params))
        for item in rows:
            self._parse_bool(item, "is_gold")
            self._parse_json(item, "embedding_json", "embedding")
        return rows  # type: ignore[return-value]
