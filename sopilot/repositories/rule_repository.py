"""Repository for task-level VigilPilot auto-generated monitoring rules.

Rules are stored in ``task_vigil_rules`` and are associated with a task ID.
Each rule carries a source tag (``"auto"`` for VLM-generated, ``"manual"`` for
operator-defined) and an optional confidence score (0–1).
"""

from __future__ import annotations

import time
from typing import Any

from sopilot.repositories.base import RepositoryBase


class RuleRepository(RepositoryBase):
    """CRUD for task_vigil_rules."""

    def __init__(self, connect: Any) -> None:
        super().__init__(connect)
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_vigil_rules (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id    TEXT    NOT NULL,
                    rule       TEXT    NOT NULL,
                    source     TEXT    NOT NULL DEFAULT 'auto',
                    confidence REAL    NOT NULL DEFAULT 0.0,
                    created_at REAL    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tvr_task ON task_vigil_rules(task_id)"
            )

    # ── Write ──────────────────────────────────────────────────────────────────

    def save_rules(
        self,
        task_id: str,
        rules: list[str],
        source: str = "auto",
        confidences: list[float] | None = None,
    ) -> int:
        """Replace all rules for *task_id* with *rules*.

        Returns the number of rules saved.
        """
        now = time.time()
        confs = confidences if confidences and len(confidences) == len(rules) else [0.0] * len(rules)
        with self._connect() as conn:
            conn.execute("DELETE FROM task_vigil_rules WHERE task_id = ?", (task_id,))
            for rule, conf in zip(rules, confs):
                conn.execute(
                    "INSERT INTO task_vigil_rules (task_id, rule, source, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
                    (task_id, rule, source, conf, now),
                )
        return len(rules)

    def add_rule(
        self,
        task_id: str,
        rule: str,
        source: str = "manual",
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """Append a single rule without clearing existing ones."""
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO task_vigil_rules (task_id, rule, source, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
                (task_id, rule, source, confidence, now),
            )
            row_id = cur.lastrowid
        return {"id": row_id, "task_id": task_id, "rule": rule, "source": source, "confidence": confidence}

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_rules(self, task_id: str) -> list[dict[str, Any]]:
        """Return all rules for *task_id*, oldest first."""
        return self._fetch_all(
            "SELECT id, task_id, rule, source, confidence, created_at FROM task_vigil_rules WHERE task_id = ? ORDER BY created_at ASC",
            (task_id,),
        )

    def get_rule_strings(self, task_id: str) -> list[str]:
        """Return just the rule text strings for *task_id*."""
        rows = self.get_rules(task_id)
        return [r["rule"] for r in rows]

    # ── Delete ─────────────────────────────────────────────────────────────────

    def delete_rules(self, task_id: str) -> int:
        """Delete all rules for *task_id*. Returns deleted count."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM task_vigil_rules WHERE task_id = ?", (task_id,)
            )
            return cur.rowcount
