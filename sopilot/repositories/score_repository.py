"""Score job lifecycle: create, claim, complete, fail, list, review."""

from __future__ import annotations

import json
from typing import Any

from sopilot.core.score_result import parse_score_json, summarize_score_result
from sopilot.repositories.base import ConnectFactory, RepositoryBase
from sopilot.types import (
    ScoreJobInputRow,
    ScoreJobRow,
    ScoreReviewRow,
)


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


class ScoreRepository(RepositoryBase):
    """Score job persistence and review management."""

    def __init__(self, connect: ConnectFactory) -> None:
        super().__init__(connect)

    # ------------------------------------------------------------------
    # Score jobs
    # ------------------------------------------------------------------

    def create_score_job(
        self,
        gold_video_id: int,
        trainee_video_id: int,
        weights: dict[str, float] | None = None,
    ) -> int:
        now = _utc_now_iso()
        weights_json = json.dumps(weights) if weights else None
        cursor = self._execute(
            """
            INSERT INTO score_jobs (
                gold_video_id, trainee_video_id, status, score_json, weights_json,
                created_at, updated_at, started_at, finished_at, error
            )
            VALUES (?, ?, 'queued', NULL, ?, ?, ?, NULL, NULL, NULL)
            """,
            (gold_video_id, trainee_video_id, weights_json, now, now),
        )
        return cursor.lastrowid or 0

    def list_pending_score_job_ids(self) -> list[int]:
        rows = self._fetch_all(
            "SELECT id FROM score_jobs WHERE status IN ('queued', 'running') ORDER BY id ASC"
        )
        return [int(row["id"]) for row in rows]

    def claim_score_job(self, job_id: int) -> bool:
        """Atomically claim a queued/running job for processing.

        Uses a single UPDATE with a WHERE guard instead of SELECT-then-UPDATE
        to eliminate the TOCTOU race condition.
        """
        now = _utc_now_iso()
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE score_jobs
                SET status = 'running',
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?,
                    error = NULL
                WHERE id = ? AND status IN ('queued', 'running')
                """,
                (now, now, job_id),
            )
            return result.rowcount == 1

    def get_score_job_input(self, job_id: int) -> ScoreJobInputRow | None:
        item = self._fetch_one(
            "SELECT id, gold_video_id, trainee_video_id, status, weights_json FROM score_jobs WHERE id = ?",
            (job_id,),
        )
        if item is None:
            return None
        self._parse_json(item, "weights_json", "weights")
        return item  # type: ignore[return-value]

    def complete_score_job(self, job_id: int, score_payload: dict[str, Any]) -> None:
        now = _utc_now_iso()
        self._execute(
            """
            UPDATE score_jobs
            SET status = 'completed', score_json = ?, updated_at = ?, finished_at = ?, error = NULL
            WHERE id = ?
            """,
            (json.dumps(score_payload), now, now, job_id),
        )

    def reset_score_job_for_retry(self, job_id: int) -> None:
        """Reset a failed score job to 'queued' so it can be retried."""
        now = _utc_now_iso()
        self._execute(
            "UPDATE score_jobs SET status='queued', error=NULL, updated_at=? WHERE id=?",
            (now, job_id),
        )

    def cancel_score_job(self, job_id: int) -> bool:
        """Cancel a queued or running score job. Returns True if cancelled."""
        now = _utc_now_iso()
        with self._connect() as conn:
            result = conn.execute(
                "UPDATE score_jobs SET status = 'cancelled', updated_at = ?, finished_at = ? "
                "WHERE id = ? AND status IN ('queued', 'running')",
                (now, now, job_id),
            )
            return result.rowcount == 1

    def fail_score_job(self, job_id: int, error: str) -> None:
        now = _utc_now_iso()
        self._execute(
            """
            UPDATE score_jobs
            SET status = 'failed', updated_at = ?, finished_at = ?, error = ?
            WHERE id = ?
            """,
            (now, now, error, job_id),
        )

    def get_score_job(self, job_id: int) -> ScoreJobRow | None:
        item = self._fetch_one("SELECT * FROM score_jobs WHERE id = ?", (job_id,))
        if item is None:
            return None
        item["score"] = parse_score_json(item.get("score_json"))
        item.pop("score_json", None)
        self._parse_json(item, "weights_json", "weights")
        return item  # type: ignore[return-value]

    def list_completed_score_jobs(self, task_id: str | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT sj.id, sj.gold_video_id, sj.trainee_video_id, sj.score_json, sj.created_at,
                   sj.updated_at, sj.started_at, sj.finished_at, gv.task_id AS task_id,
                   tv.operator_id_hash AS operator_id_hash, tv.site_id AS trainee_site_id
            FROM score_jobs sj
            LEFT JOIN videos gv ON gv.id = sj.gold_video_id
            LEFT JOIN videos tv ON tv.id = sj.trainee_video_id
            WHERE sj.status = 'completed' AND sj.score_json IS NOT NULL
        """
        params: list[Any] = []
        if task_id:
            query += " AND gv.task_id = ?"
            params.append(task_id)
        query += " ORDER BY sj.id ASC"

        rows = self._fetch_all(query, tuple(params))
        for item in rows:
            self._parse_json(item, "score_json", "score")
        return rows

    def list_score_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return score jobs with lightweight score summary (no full score_json)."""
        query = """
            SELECT sj.id, sj.gold_video_id, sj.trainee_video_id, sj.status,
                   sj.score_json, sj.created_at, sj.updated_at, sj.finished_at, sj.error,
                   gv.task_id
            FROM score_jobs sj
            LEFT JOIN videos gv ON gv.id = sj.gold_video_id
            WHERE 1=1
        """
        params: list[Any] = []
        if status:
            query += " AND sj.status = ?"
            params.append(status)
        query += " ORDER BY sj.id DESC LIMIT ? OFFSET ?"
        params.extend([max(1, limit), max(0, offset)])

        rows = self._fetch_all(query, tuple(params))
        for item in rows:
            parsed = parse_score_json(item.pop("score_json", None))
            item.update(summarize_score_result(parsed))
        return rows

    def update_score_json(self, job_id: int, score_payload: dict[str, Any]) -> None:
        """Overwrite score_json for a completed job without touching status or timestamps."""
        self._execute(
            "UPDATE score_jobs SET score_json = ? WHERE id = ? AND status = 'completed'",
            (json.dumps(score_payload), job_id),
        )

    def count_score_jobs(self, *, status: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM score_jobs WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status)
        with self._connect() as conn:
            row = conn.execute(query, tuple(params)).fetchone()
            return int(row[0]) if row else 0

    def get_operator_scores_chronological(
        self,
        operator_id: str,
        *,
        task_id: str | None = None,
        limit: int = 50,
    ) -> list[float]:
        """Return chronological list of completed scores for an operator (oldest first)."""
        with self._connect() as conn:
            filters = ["sj.status = 'completed'", "v.operator_id_hash = ?"]
            params: list[Any] = [operator_id]
            if task_id:
                filters.append("v.task_id = ?")
                params.append(task_id)
            where = " AND ".join(filters)
            params.append(limit)
            rows = conn.execute(
                f"""
                SELECT json_extract(sj.score_json, '$.score') AS score
                FROM score_jobs sj
                JOIN videos v ON sj.trainee_video_id = v.id
                WHERE {where}
                ORDER BY sj.created_at ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [float(r["score"]) for r in rows if r["score"] is not None]

    # ------------------------------------------------------------------
    # Score reviews
    # ------------------------------------------------------------------

    def upsert_score_review(self, job_id: int, verdict: str, note: str | None) -> ScoreReviewRow:
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO score_reviews (job_id, verdict, note, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    verdict = excluded.verdict,
                    note = excluded.note,
                    updated_at = excluded.updated_at
                """,
                (job_id, verdict, note, now, now),
            )
            row = conn.execute("SELECT * FROM score_reviews WHERE job_id = ?", (job_id,)).fetchone()
            return dict(row)  # type: ignore[return-value]

    def get_score_review(self, job_id: int) -> ScoreReviewRow | None:
        return self._fetch_one("SELECT * FROM score_reviews WHERE job_id = ?", (job_id,))  # type: ignore[return-value]
