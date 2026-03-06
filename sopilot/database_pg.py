"""PostgreSQL connection pool and adapter for SOPilot.

Provides a drop-in replacement for ``Database`` that targets PostgreSQL
via ``psycopg2``.  Repositories receive the same ``connect()`` interface,
so all existing repository code works without modification.

Key design decisions:
- ``PostgresConnectionAdapter`` translates SQLite's ``?`` placeholders to
  PostgreSQL's ``%s`` via a simple regex substitution.
- ``DictCursorAdapter`` wraps psycopg2's DictCursor so that ``dict(row)``
  returns a plain ``dict``, identical to ``dict(sqlite3.Row)``.
- ``DatabasePG`` exposes the same ``connect()`` context-manager interface
  as ``Database``, allowing ``main.py`` to swap backends with one ``if``.
- All PostgreSQL-specific errors are propagated unchanged; callers should
  treat any ``Exception`` as a DB error, consistent with the SQLite path.

Usage (via environment variable)::

    SOPILOT_DATABASE_URL=postgresql://user:pass@host:5432/sopilot

Requires: ``psycopg2-binary>=2.9.9`` (install with ``pip install sopilot[postgres]``).
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# Regex for SQLite → PostgreSQL placeholder translation
_PLACEHOLDER_RE = re.compile(r"\?")


def _translate_sql(sql: str) -> str:
    """Translate SQLite ``?`` placeholders to PostgreSQL ``%s``."""
    return _PLACEHOLDER_RE.sub("%s", sql)


class DictCursorAdapter:
    """Wraps a psycopg2 DictCursor to mimic sqlite3.Row dict-conversion.

    ``dict(row)`` returns a plain ``{column: value}`` mapping, matching
    the behaviour of ``dict(sqlite3.Row)``.
    """

    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor
        self.lastrowid: int | None = None
        self.rowcount: int = 0

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> "DictCursorAdapter":
        self._cursor.execute(_translate_sql(sql), params)
        self.rowcount = self._cursor.rowcount
        # For INSERT … RETURNING id or similar patterns, try to get lastrowid.
        try:
            row = self._cursor.fetchone()
            if row is not None:
                self.lastrowid = row[0]
        except Exception:
            pass
        return self

    def executescript(self, script: str) -> None:
        """Execute multiple semicolon-separated statements (DDL)."""
        for stmt in script.split(";"):
            stmt = stmt.strip()
            if stmt:
                self._cursor.execute(stmt)

    def fetchone(self) -> dict[str, Any] | None:
        row = self._cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def fetchall(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._cursor.fetchall()]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._cursor:
            yield dict(row)


class PostgresConnectionAdapter:
    """Wraps a psycopg2 connection to match the sqlite3.Connection interface
    expected by all SOPilot repositories.

    Key compatibility shims:
    - ``execute()`` / ``executescript()`` delegate to a DictCursor.
    - ``commit()`` / ``rollback()`` delegate to the underlying connection.
    - ``row_factory`` attribute exists (but is unused; DictCursor handles it).
    """

    def __init__(self, conn: Any) -> None:
        import psycopg2.extras  # type: ignore[import-untyped]

        self._conn = conn
        self._cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        self._adapter = DictCursorAdapter(self._cursor)
        # Sentinel to keep repositories from breaking on hasattr checks
        self.row_factory = None

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> DictCursorAdapter:
        self._cursor.execute(_translate_sql(sql), params)
        self._adapter.rowcount = self._cursor.rowcount
        return self._adapter

    def executescript(self, script: str) -> None:
        for stmt in script.split(";"):
            stmt = stmt.strip()
            if stmt:
                self._cursor.execute(stmt)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        try:
            self._cursor.close()
        except Exception:
            pass
        self._conn.close()


class DatabasePG:
    """PostgreSQL-backed database facade.

    Implements the same public ``connect()`` interface as ``Database``
    (SQLite), allowing repositories to be used unchanged.

    Parameters
    ----------
    database_url:
        PostgreSQL DSN, e.g. ``"postgresql://user:pass@localhost:5432/sopilot"``.
    pool_size:
        Maximum number of connections to keep in the pool.
    """

    def __init__(self, database_url: str, *, pool_size: int = 5) -> None:
        self._url = database_url
        self._pool_size = pool_size
        self._pool: list[Any] = []
        self._pool_lock = threading.Lock()
        logger.info("DatabasePG initialising (pool_size=%d)", pool_size)
        self._init_pool()

    def _init_pool(self) -> None:
        try:
            import psycopg2  # type: ignore[import-untyped]
            from psycopg2 import pool as pg_pool  # type: ignore[import-untyped]

            self._pg_pool = pg_pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=self._pool_size,
                dsn=self._url,
            )
            logger.info("psycopg2 ThreadedConnectionPool ready")
        except ImportError as exc:
            raise ImportError(
                "psycopg2-binary is required for PostgreSQL support. "
                "Install it with: pip install sopilot[postgres]"
            ) from exc

    @contextmanager
    def connect(self) -> Iterator[PostgresConnectionAdapter]:
        """Yield a ``PostgresConnectionAdapter`` from the pool.

        Automatically commits on success and rolls back on exception,
        then returns the raw connection to the pool.
        """
        raw_conn = self._pg_pool.getconn()
        adapter = PostgresConnectionAdapter(raw_conn)
        try:
            yield adapter
            adapter.commit()
        except Exception:
            try:
                adapter.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                self._pg_pool.putconn(raw_conn)
            except Exception:
                pass

    def close(self) -> None:
        """Shut down the connection pool."""
        try:
            self._pg_pool.closeall()
            logger.info("DatabasePG connection pool closed")
        except Exception:
            logger.exception("Error closing DatabasePG pool")
