"""Task profile and SOP step definition persistence."""

from __future__ import annotations

import json
from typing import Any

from sopilot.repositories.base import ConnectFactory, RepositoryBase
from sopilot.types import TaskProfileRow


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


class TaskProfileRepository(RepositoryBase):
    """Task profiles and SOP step CRUD."""

    def __init__(self, connect: ConnectFactory) -> None:
        super().__init__(connect)

    # ------------------------------------------------------------------
    # Task profiles
    # ------------------------------------------------------------------

    def upsert_task_profile(
        self,
        *,
        task_id: str,
        task_name: str,
        pass_score: float,
        retrain_score: float,
        default_weights: dict[str, float],
        deviation_policy: dict[str, str],
    ) -> None:
        now = _utc_now_iso()
        self._execute(
            """
            INSERT INTO task_profiles (
                task_id, task_name, pass_score, retrain_score,
                default_weights_json, deviation_policy_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                task_name = excluded.task_name,
                pass_score = excluded.pass_score,
                retrain_score = excluded.retrain_score,
                default_weights_json = excluded.default_weights_json,
                deviation_policy_json = excluded.deviation_policy_json,
                updated_at = excluded.updated_at
            """,
            (
                task_id,
                task_name,
                pass_score,
                retrain_score,
                json.dumps(default_weights),
                json.dumps(deviation_policy),
                now,
                now,
            ),
        )

    def get_task_profile(self, task_id: str) -> TaskProfileRow | None:
        item = self._fetch_one(
            """
            SELECT task_id, task_name, pass_score, retrain_score, default_weights_json, deviation_policy_json
            FROM task_profiles
            WHERE task_id = ?
            """,
            (task_id,),
        )
        if item is None:
            return None
        self._parse_json(item, "default_weights_json", "default_weights")
        self._parse_json(item, "deviation_policy_json", "deviation_policy")
        return item  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # SOP steps
    # ------------------------------------------------------------------

    def upsert_sop_steps(self, task_id: str, steps: list[dict]) -> int:
        """Insert or replace step definitions for a task. Returns count upserted."""
        now = _utc_now_iso()
        with self._connect() as conn:
            upserted = 0
            for step in steps:
                conn.execute(
                    """
                    INSERT INTO sop_steps
                        (task_id, step_index, name_ja, name_en, expected_duration_sec,
                         min_duration_sec, max_duration_sec, is_critical, description,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(task_id, step_index) DO UPDATE SET
                        name_ja=excluded.name_ja,
                        name_en=excluded.name_en,
                        expected_duration_sec=excluded.expected_duration_sec,
                        min_duration_sec=excluded.min_duration_sec,
                        max_duration_sec=excluded.max_duration_sec,
                        is_critical=excluded.is_critical,
                        description=excluded.description,
                        updated_at=excluded.updated_at
                    """,
                    (
                        task_id,
                        step["step_index"],
                        step.get("name_ja", ""),
                        step.get("name_en", ""),
                        step.get("expected_duration_sec"),
                        step.get("min_duration_sec"),
                        step.get("max_duration_sec"),
                        1 if step.get("is_critical") else 0,
                        step.get("description"),
                        now,
                        now,
                    ),
                )
                upserted += 1
        return upserted

    def get_sop_steps(self, task_id: str) -> list[dict]:
        """Return all step definitions for a task, ordered by step_index."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT step_index, name_ja, name_en, expected_duration_sec,
                       min_duration_sec, max_duration_sec, is_critical, description,
                       created_at, updated_at
                FROM sop_steps
                WHERE task_id = ?
                ORDER BY step_index ASC
                """,
                (task_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_sop_steps(self, task_id: str) -> int:
        """Delete all step definitions for a task. Returns count deleted."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM sop_steps WHERE task_id = ?", (task_id,))
        return cur.rowcount
