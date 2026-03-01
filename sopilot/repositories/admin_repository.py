"""Administrative operations: backup, vacuum, database statistics."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from sopilot.repositories.base import ConnectFactory, RepositoryBase


class AdminRepository(RepositoryBase):
    """Database maintenance and monitoring operations."""

    _logger = logging.getLogger(__name__)

    def __init__(self, connect: ConnectFactory, db_path: str) -> None:
        super().__init__(connect)
        self._db_path = db_path

    def backup(self, dest_path: str) -> None:
        """Create a hot backup of the database using SQLite's backup API."""
        src = sqlite3.connect(self._db_path, timeout=10.0)
        dst = sqlite3.connect(dest_path)
        try:
            src.backup(dst)
            self._logger.info("database backup created dest=%s", dest_path)
        finally:
            dst.close()
            src.close()

    def vacuum(self) -> None:
        """Reclaim disk space and rebuild indices."""
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        try:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
            self._logger.info("database vacuum+analyze completed")
        finally:
            conn.close()

    def get_stats(self) -> dict[str, Any]:
        """Return table row counts and database file size for monitoring."""
        with self._connect() as conn:
            tables = ["videos", "clips", "score_jobs", "score_reviews", "task_profiles"]
            stats: dict[str, Any] = {}
            for t in tables:
                row = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
                stats[t] = int(row[0]) if row else 0
        # Database file size
        db_path = Path(self._db_path)
        if db_path.exists():
            size_bytes = db_path.stat().st_size
            stats["_db_size_bytes"] = size_bytes
            if size_bytes < 1024:
                stats["_db_size_human"] = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                stats["_db_size_human"] = f"{size_bytes / 1024:.1f} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                stats["_db_size_human"] = f"{size_bytes / (1024 * 1024):.1f} MB"
            else:
                stats["_db_size_human"] = f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
        return stats
