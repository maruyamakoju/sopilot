"""Audit log repository.

Write-only append from middleware; read by admin endpoint.
Schema: audit_log table (see migrations/sqlite/012_audit_log.py).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sopilot.repositories.base import ConnectFactory, RepositoryBase

logger = logging.getLogger(__name__)


class AuditRepository(RepositoryBase):
    """Append audit log entries and query them for the admin endpoint."""

    def __init__(self, connect: ConnectFactory) -> None:
        super().__init__(connect)
        self._ensure_table()

    def _ensure_table(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts          TEXT NOT NULL,
                        method      TEXT NOT NULL,
                        path        TEXT NOT NULL,
                        status_code INTEGER,
                        duration_ms REAL,
                        client_id   TEXT,
                        role        TEXT,
                        ip_address  TEXT,
                        request_id  TEXT
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts     ON audit_log(ts)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_client ON audit_log(client_id)")
        except Exception:
            logger.exception("Failed to create audit_log table")

    def log_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int | None = None,
        duration_ms: float | None = None,
        client_id: str | None = None,
        role: str | None = None,
        ip_address: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Insert a single audit log row.  Never raises — errors are swallowed."""
        try:
            ts = datetime.now(UTC).isoformat()
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO audit_log
                        (ts, method, path, status_code, duration_ms,
                         client_id, role, ip_address, request_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (ts, method, path, status_code, duration_ms,
                     client_id, role, ip_address, request_id),
                )
        except Exception:
            logger.exception("audit_log insert failed")

    def list_audit_log(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        client_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if client_id:
            return self._fetch_all(
                "SELECT * FROM audit_log WHERE client_id = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (client_id, limit, offset),
            )
        return self._fetch_all(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )

    def count_audit_log(self, *, client_id: str | None = None) -> int:
        if client_id:
            row = self._fetch_one(
                "SELECT COUNT(*) AS n FROM audit_log WHERE client_id = ?", (client_id,)
            )
        else:
            row = self._fetch_one("SELECT COUNT(*) AS n FROM audit_log")
        return int(row["n"]) if row else 0
