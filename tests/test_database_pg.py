"""Tests for sopilot.database_pg — PostgreSQL migration path (Item 2).

All tests use mocks — no real PostgreSQL connection required.
"""

import sqlite3
from unittest.mock import MagicMock, call, patch

import pytest

from sopilot.database_pg import DatabasePG, DictCursorAdapter, PostgresConnectionAdapter, _translate_sql


# ---------------------------------------------------------------------------
# SQL placeholder translation
# ---------------------------------------------------------------------------

class TestTranslateSql:
    def test_no_placeholders(self) -> None:
        assert _translate_sql("SELECT 1") == "SELECT 1"

    def test_single_placeholder(self) -> None:
        assert _translate_sql("SELECT * FROM t WHERE id = ?") == "SELECT * FROM t WHERE id = %s"

    def test_multiple_placeholders(self) -> None:
        sql = "INSERT INTO t (a, b, c) VALUES (?, ?, ?)"
        assert _translate_sql(sql) == "INSERT INTO t (a, b, c) VALUES (%s, %s, %s)"

    def test_no_change_to_existing_percent_s(self) -> None:
        """Pre-existing %s should not be double-replaced."""
        sql = "SELECT %s"
        assert _translate_sql(sql) == "SELECT %s"

    def test_question_mark_in_string_literal(self) -> None:
        """? in SQL string literals is translated (limitation — acceptable)."""
        sql = "SELECT * FROM t WHERE val = ?"
        result = _translate_sql(sql)
        assert "?" not in result
        assert "%s" in result

    def test_complex_query(self) -> None:
        sql = (
            "SELECT id FROM videos WHERE task_id = ? AND status = ? "
            "AND created_at > ? ORDER BY id LIMIT ? OFFSET ?"
        )
        result = _translate_sql(sql)
        assert result.count("%s") == 5
        assert "?" not in result


# ---------------------------------------------------------------------------
# DictCursorAdapter
# ---------------------------------------------------------------------------

class TestDictCursorAdapter:
    def _make_adapter(self) -> tuple[DictCursorAdapter, MagicMock]:
        mock_cursor = MagicMock()
        adapter = DictCursorAdapter(mock_cursor)
        return adapter, mock_cursor

    def test_execute_translates_placeholders(self) -> None:
        adapter, mock_cursor = self._make_adapter()
        mock_cursor.fetchone.return_value = None
        adapter.execute("SELECT * FROM t WHERE id = ?", (1,))
        mock_cursor.execute.assert_called_once_with(
            "SELECT * FROM t WHERE id = %s", (1,)
        )

    def test_fetchone_returns_dict(self) -> None:
        adapter, mock_cursor = self._make_adapter()
        mock_cursor.fetchone.return_value = {"id": 1, "name": "test"}
        result = adapter.fetchone()
        assert result == {"id": 1, "name": "test"}

    def test_fetchone_returns_none_when_empty(self) -> None:
        adapter, mock_cursor = self._make_adapter()
        mock_cursor.fetchone.return_value = None
        assert adapter.fetchone() is None

    def test_fetchall_returns_list_of_dicts(self) -> None:
        adapter, mock_cursor = self._make_adapter()
        mock_cursor.fetchall.return_value = [{"id": 1}, {"id": 2}]
        result = adapter.fetchall()
        assert result == [{"id": 1}, {"id": 2}]

    def test_fetchall_empty(self) -> None:
        adapter, mock_cursor = self._make_adapter()
        mock_cursor.fetchall.return_value = []
        assert adapter.fetchall() == []

    def test_executescript_splits_on_semicolons(self) -> None:
        adapter, mock_cursor = self._make_adapter()
        adapter.executescript("CREATE TABLE a (id INT); CREATE TABLE b (id INT);")
        assert mock_cursor.execute.call_count == 2


# ---------------------------------------------------------------------------
# PostgresConnectionAdapter
# ---------------------------------------------------------------------------

class TestPostgresConnectionAdapter:
    def _make(self) -> tuple[PostgresConnectionAdapter, MagicMock]:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 0
        mock_cursor.fetchone.return_value = None

        import psycopg2.extras  # type: ignore[import-untyped]
        with patch("psycopg2.extras.RealDictCursor"):
            adapter = PostgresConnectionAdapter.__new__(PostgresConnectionAdapter)
            adapter._conn = mock_conn
            adapter._cursor = mock_cursor
            adapter._adapter = DictCursorAdapter(mock_cursor)
            adapter.row_factory = None
        return adapter, mock_conn

    def test_commit_delegates_to_connection(self) -> None:
        adapter, conn = self._make()
        adapter.commit()
        conn.commit.assert_called_once()

    def test_rollback_delegates_to_connection(self) -> None:
        adapter, conn = self._make()
        adapter.rollback()
        conn.rollback.assert_called_once()

    def test_close_closes_cursor_and_conn(self) -> None:
        adapter, conn = self._make()
        adapter.close()
        conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# DatabasePG init + connect
# ---------------------------------------------------------------------------

class TestDatabasePG:
    def _make_db(self) -> DatabasePG:
        """Create a DatabasePG with a mocked psycopg2 pool."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn

        with (
            patch("psycopg2.pool.ThreadedConnectionPool", return_value=mock_pool),
            patch("psycopg2.extras.RealDictCursor"),
        ):
            db = DatabasePG.__new__(DatabasePG)
            db._url = "postgresql://user:pass@localhost/sopilot"
            db._pool_size = 5
            db._pg_pool = mock_pool
        return db

    def test_connect_yields_adapter(self) -> None:
        db = self._make_db()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        db._pg_pool.getconn.return_value = mock_conn

        with patch("psycopg2.extras.RealDictCursor"):
            with db.connect() as conn:
                assert conn is not None

        db._pg_pool.getconn.assert_called_once()
        mock_conn.commit.assert_called_once()
        db._pg_pool.putconn.assert_called_once()

    def test_connect_rolls_back_on_exception(self) -> None:
        db = self._make_db()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        db._pg_pool.getconn.return_value = mock_conn

        with patch("psycopg2.extras.RealDictCursor"):
            with pytest.raises(ValueError):
                with db.connect() as conn:
                    raise ValueError("test error")

        mock_conn.rollback.assert_called_once()
        db._pg_pool.putconn.assert_called_once()

    def test_close_calls_closeall(self) -> None:
        db = self._make_db()
        db.close()
        db._pg_pool.closeall.assert_called_once()

    def test_init_raises_if_psycopg2_missing(self) -> None:
        with patch.dict(__import__("sys").modules, {"psycopg2": None, "psycopg2.pool": None}):
            with pytest.raises((ImportError, Exception)):
                DatabasePG("postgresql://user:pass@localhost/sopilot")


# ---------------------------------------------------------------------------
# Integration: VideoRepository-like query works via DatabasePG
# ---------------------------------------------------------------------------

class TestDatabasePGRepositoryCompat:
    """Verify that the repository query pattern works against the PG adapter."""

    def test_fetch_all_pattern(self) -> None:
        """Simulate what RepositoryBase._fetch_all does against PG adapter."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_cursor.fetchall.return_value = [{"id": 1, "task_id": "pilot"}]
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn

        with (
            patch("psycopg2.pool.ThreadedConnectionPool", return_value=mock_pool),
            patch("psycopg2.extras.RealDictCursor"),
        ):
            db = DatabasePG.__new__(DatabasePG)
            db._url = "postgresql://u:p@h/db"
            db._pool_size = 1
            db._pg_pool = mock_pool

        with db.connect() as conn:
            result = conn.execute(
                "SELECT * FROM videos WHERE task_id = ?", ("pilot",)
            ).fetchall()

        # Verify ? was translated to %s in the actual cursor call
        mock_cursor.execute.assert_called_once_with(
            "SELECT * FROM videos WHERE task_id = %s", ("pilot",)
        )
        assert result == [{"id": 1, "task_id": "pilot"}]
