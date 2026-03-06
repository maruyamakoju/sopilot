#!/usr/bin/env python3
"""Run SOPilot PostgreSQL migrations in order.

Usage::

    python migrations/postgres/run_pg_migrations.py postgresql://user:pass@localhost/sopilot

Each .sql file in this directory is executed in lexicographic order.
Already-applied tables are skipped via ``IF NOT EXISTS`` guards in the DDL.
"""

import sys
from pathlib import Path


def run_migrations(database_url: str) -> None:
    try:
        import psycopg2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "psycopg2-binary is required. Install with: pip install sopilot[postgres]"
        ) from exc

    sql_dir = Path(__file__).parent
    sql_files = sorted(sql_dir.glob("*.sql"))

    if not sql_files:
        print("No .sql migration files found.")
        return

    conn = psycopg2.connect(database_url)
    try:
        conn.autocommit = False
        cur = conn.cursor()
        for sql_file in sql_files:
            print(f"Applying {sql_file.name} ...", end=" ")
            sql = sql_file.read_text(encoding="utf-8")
            cur.execute(sql)
            conn.commit()
            print("done")
        print(f"All {len(sql_files)} migration(s) applied successfully.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <database_url>")
        sys.exit(1)
    run_migrations(sys.argv[1])
