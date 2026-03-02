"""SQLite repository for global VigilPilot webhook registrations."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class WebhookRepository:
    """Data-access layer for the vigil_webhooks table.

    Follows the same pattern as :class:`~sopilot.vigil.repository.VigilRepository`:
    each method opens its own WAL-mode connection, commits on success, and rolls
    back on error.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── CRUD ───────────────────────────────────────────────────────────────

    def create(
        self,
        url: str,
        name: str = "",
        secret: str = "",
        min_severity: str = "critical",
    ) -> dict:
        """Insert a new webhook registration and return the created row as a dict."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO vigil_webhooks (url, name, secret, min_severity, enabled)
                VALUES (?, ?, ?, ?, 1)
                """,
                (url, name, secret, min_severity),
            )
            webhook_id = int(cur.lastrowid)  # type: ignore[arg-type]
            row = conn.execute(
                "SELECT * FROM vigil_webhooks WHERE id = ?", (webhook_id,)
            ).fetchone()
            return dict(row)

    def list_all(self) -> list[dict]:
        """Return all webhook registrations ordered by creation date."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM vigil_webhooks ORDER BY created_at ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get(self, webhook_id: int) -> dict | None:
        """Return a single webhook by ID, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM vigil_webhooks WHERE id = ?", (webhook_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def update_enabled(self, webhook_id: int, enabled: bool) -> bool:
        """Enable or disable a webhook. Returns True if the row was found."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE vigil_webhooks SET enabled = ? WHERE id = ?",
                (1 if enabled else 0, webhook_id),
            )
            return cur.rowcount > 0

    def delete(self, webhook_id: int) -> bool:
        """Delete a webhook by ID. Returns True if a row was deleted."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM vigil_webhooks WHERE id = ?", (webhook_id,)
            )
            return cur.rowcount > 0

    def update_triggered(self, webhook_id: int) -> None:
        """Record a successful dispatch: increment trigger_count and set last_triggered_at."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE vigil_webhooks
                SET last_triggered_at = datetime('now'),
                    trigger_count = trigger_count + 1
                WHERE id = ?
                """,
                (webhook_id,),
            )
