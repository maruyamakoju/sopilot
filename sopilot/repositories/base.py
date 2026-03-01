"""Shared base for all repository classes.

Every repository receives a ``connect`` callable -- a context-manager
factory that yields ``sqlite3.Connection`` with row_factory already set
to ``sqlite3.Row`` and foreign keys enabled.  This decouples repositories
from the concrete ``Database`` class (and its lifecycle concerns such as
WAL mode, schema init and migrations).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

# Type alias for the connect callable shared across all repositories.
ConnectFactory = Callable[[], AbstractContextManager[sqlite3.Connection]]


class RepositoryBase:
    """Mixin providing connection access and common parse helpers."""

    def __init__(self, connect: ConnectFactory) -> None:
        self._connect = connect

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
            return dict(row) if row is not None else None

    def _fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _execute(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._connect() as conn:
            return conn.execute(query, params)

    # ------------------------------------------------------------------
    # Parse helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_bool(item: dict[str, Any], key: str) -> None:
        if key in item:
            item[key] = bool(item[key])

    @staticmethod
    def _parse_json(item: dict[str, Any], json_key: str, target_key: str | None = None) -> None:
        raw = item.get(json_key)
        dest = target_key if target_key else json_key
        if target_key and json_key in item:
            del item[json_key]
        if raw is not None:
            item[dest] = json.loads(raw)
        else:
            item[dest] = None
