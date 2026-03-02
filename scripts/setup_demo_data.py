#!/usr/bin/env python3
"""SOPilot Demo Setup Script.

Sets up a complete VigilPilot demo environment with realistic violation data.
The SOPilot scoring side is already covered by the running server and smoke_test.py;
this script focuses on creating three VigilPilot sessions with pre-populated
violation events so the 監視 tab is immediately impressive.

Strategy
--------
1. Sessions are created via the REST API (POST /vigil/sessions).
2. Violation events are inserted directly into SQLite — bypassing the VLM — so
   the demo looks rich without requiring an Anthropic API key or real video files.
3. Session statistics (violation_count, total_frames_analyzed, status) are
   updated via SQL to reflect the injected events.

Usage
-----
    # Defaults (localhost:8000, API key from SOPILOT_API_KEY env var or "demo-key"):
    python scripts/setup_demo_data.py

    # Override host / key:
    python scripts/setup_demo_data.py --host http://localhost:8000 --api-key YOUR_KEY

    # Use an alternate DB path:
    python scripts/setup_demo_data.py --db-path /path/to/sopilot.db

Exit codes: 0 = success, 1 = failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    import httpx
except ImportError:
    print("ERROR: httpx is not installed. Run: pip install httpx")
    sys.exit(1)

# ── Terminal colour helpers ──────────────────────────────────────────────────

GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def _green(s: str) -> str:  return f"{GREEN}{s}{RESET}"
def _cyan(s: str) -> str:   return f"{CYAN}{s}{RESET}"
def _yellow(s: str) -> str: return f"{YELLOW}{s}{RESET}"
def _red(s: str) -> str:    return f"{RED}{s}{RESET}"
def _bold(s: str) -> str:   return f"{BOLD}{s}{RESET}"
def _dim(s: str) -> str:    return f"{DIM}{s}{RESET}"

def _ok(msg: str)   -> None: print(f"  {_green('✓')} {msg}")
def _fail(msg: str) -> None: print(f"  {_red('✗')} {msg}")
def _info(msg: str) -> None: print(f"  {_dim('·')} {msg}")

def _section(title: str) -> None:
    print(f"\n{_bold(title)}")
    print("─" * len(title))


def _print_header() -> None:
    width = 62
    border = "═" * width
    print(f"\n{_cyan('╔' + border + '╗')}")
    print(_cyan("║") + _bold(f"{'SOPilot v1.2.0 — Demo Setup':^{width}}") + _cyan("║"))
    print(_cyan("║") + f"{'VigilPilot Session Initializer':^{width}}" + _cyan("║"))
    print(f"{_cyan('╚' + border + '╝')}\n")


# ── Session definitions ──────────────────────────────────────────────────────

SESSION_SPECS: list[dict] = [
    {
        "name": "建設現場安全監視",
        "rules": [
            "ヘルメット未着用の作業者を検出",
            "安全ベルトなしで高所作業している人を検出",
            "立入禁止エリアへの侵入を検出",
            "重機の安全距離違反を検出",
        ],
        "sample_fps": 1.0,
        "severity_threshold": "warning",
        # Synthetic frames analyzed: 120 (2 min @ 1 fps)
        "total_frames_analyzed": 120,
        "events": [
            {
                "timestamp_sec": 12.5,
                "frame_number": 12,
                "violations": [
                    {
                        "rule_index": 0,
                        "rule": "ヘルメット未着用の作業者を検出",
                        "description_ja": "作業者のヘルメット未着用が検出されました。即時着用が必要です。",
                        "severity": "critical",
                        "confidence": 0.96,
                        "bboxes": None,
                    }
                ],
            },
            {
                "timestamp_sec": 28.0,
                "frame_number": 28,
                "violations": [
                    {
                        "rule_index": 1,
                        "rule": "安全ベルトなしで高所作業している人を検出",
                        "description_ja": "高さ3m以上の足場で安全ベルトを未装着の作業者が確認されました。",
                        "severity": "critical",
                        "confidence": 0.91,
                        "bboxes": None,
                    }
                ],
            },
            {
                "timestamp_sec": 45.5,
                "frame_number": 45,
                "violations": [
                    {
                        "rule_index": 2,
                        "rule": "立入禁止エリアへの侵入を検出",
                        "description_ja": "立入禁止エリアへの人物侵入が検出されました。",
                        "severity": "warning",
                        "confidence": 0.88,
                        "bboxes": None,
                    }
                ],
            },
            {
                "timestamp_sec": 67.0,
                "frame_number": 67,
                "violations": [
                    {
                        "rule_index": 0,
                        "rule": "ヘルメット未着用の作業者を検出",
                        "description_ja": "別の作業者がヘルメットなしで資材搬入作業中です。",
                        "severity": "critical",
                        "confidence": 0.94,
                        "bboxes": None,
                    }
                ],
            },
            {
                "timestamp_sec": 89.5,
                "frame_number": 89,
                "violations": [
                    {
                        "rule_index": 3,
                        "rule": "重機の安全距離違反を検出",
                        "description_ja": "クレーン稼働中に作業者が安全距離（3m）以内に接近しています。",
                        "severity": "warning",
                        "confidence": 0.82,
                        "bboxes": None,
                    }
                ],
            },
            {
                "timestamp_sec": 112.0,
                "frame_number": 112,
                "violations": [
                    {
                        "rule_index": 1,
                        "rule": "安全ベルトなしで高所作業している人を検出",
                        "description_ja": "屋上作業中の作業者が安全ハーネスを未装着です。転落リスクが高い状態です。",
                        "severity": "critical",
                        "confidence": 0.93,
                        "bboxes": None,
                    }
                ],
            },
        ],
    },
    {
        "name": "食品工場衛生管理",
        "rules": [
            "手袋未着用での食品取り扱いを検出",
            "マスク未着用の作業者を検出",
            "異物混入リスクのある行動を検出",
        ],
        "sample_fps": 0.5,
        "severity_threshold": "info",
        # Synthetic frames analyzed: 45 (90 sec @ 0.5 fps)
        "total_frames_analyzed": 45,
        "events": [
            {
                "timestamp_sec": 8.0,
                "frame_number": 4,
                "violations": [
                    {
                        "rule_index": 0,
                        "rule": "手袋未着用での食品取り扱いを検出",
                        "description_ja": "素手での食品直接接触が確認されました。衛生手袋の着用が義務付けられています。",
                        "severity": "warning",
                        "confidence": 0.89,
                        "bboxes": None,
                    }
                ],
            },
            {
                "timestamp_sec": 25.5,
                "frame_number": 12,
                "violations": [
                    {
                        "rule_index": 1,
                        "rule": "マスク未着用の作業者を検出",
                        "description_ja": "食品加工エリアでマスク未着用の作業者が検出されました。",
                        "severity": "warning",
                        "confidence": 0.92,
                        "bboxes": None,
                    }
                ],
            },
            {
                "timestamp_sec": 41.0,
                "frame_number": 20,
                "violations": [
                    {
                        "rule_index": 2,
                        "rule": "異物混入リスクのある行動を検出",
                        "description_ja": "作業者が金属製アクセサリーを着用したまま製造ラインに接近しています。異物混入のリスクがあります。",
                        "severity": "critical",
                        "confidence": 0.87,
                        "bboxes": None,
                    }
                ],
            },
            {
                "timestamp_sec": 58.5,
                "frame_number": 29,
                "violations": [
                    {
                        "rule_index": 0,
                        "rule": "手袋未着用での食品取り扱いを検出",
                        "description_ja": "別の作業者がグローブなしで原材料を計量しています。",
                        "severity": "warning",
                        "confidence": 0.85,
                        "bboxes": None,
                    }
                ],
            },
            {
                "timestamp_sec": 74.0,
                "frame_number": 37,
                "violations": [
                    {
                        "rule_index": 1,
                        "rule": "マスク未着用の作業者を検出",
                        "description_ja": "包装ライン担当者のマスクが顎まで下げられた状態での作業が確認されました。",
                        "severity": "warning",
                        "confidence": 0.90,
                        "bboxes": None,
                    }
                ],
            },
        ],
    },
    {
        "name": "倉庫作業安全",
        "rules": [
            "フォークリフトの安全速度超過を検出",
            "歩行者とフォークリフトの接近を検出",
            "積載物の不安定な積み方を検出",
        ],
        "sample_fps": 1.0,
        "severity_threshold": "critical",
        # Synthetic frames analyzed: 60 (1 min @ 1 fps)
        "total_frames_analyzed": 60,
        "events": [
            {
                "timestamp_sec": 15.0,
                "frame_number": 15,
                "violations": [
                    {
                        "rule_index": 0,
                        "rule": "フォークリフトの安全速度超過を検出",
                        "description_ja": "フォークリフトが倉庫内制限速度（6km/h）を超過して走行しています。推定速度: 約14km/h。",
                        "severity": "critical",
                        "confidence": 0.95,
                        "bboxes": None,
                    }
                ],
            },
            {
                "timestamp_sec": 38.5,
                "frame_number": 38,
                "violations": [
                    {
                        "rule_index": 1,
                        "rule": "歩行者とフォークリフトの接近を検出",
                        "description_ja": "作業員がフォークリフト走行ルート上に立入り、接近距離が安全基準（1.5m）を下回っています。",
                        "severity": "critical",
                        "confidence": 0.97,
                        "bboxes": None,
                    }
                ],
            },
        ],
    },
]


# ── Core logic ───────────────────────────────────────────────────────────────

def _resolve_db_path(override: str | None) -> Path:
    """Find the SQLite DB from env / override / default locations."""
    if override:
        return Path(override)
    env_dir = os.environ.get("SOPILOT_DATA_DIR")
    if env_dir:
        return Path(env_dir) / "sopilot.db"
    # Try common locations relative to the repo root
    candidates = [
        Path("data/sopilot.db"),
        Path("data_trip_96h_official_20260212/sopilot.db"),
        Path("/app/data/sopilot.db"),
    ]
    for c in candidates:
        if c.exists():
            return c
    # Return the default path even if it doesn't exist yet — the server
    # may create it after the first request (health check).
    return Path("data/sopilot.db")


def _check_health(client: httpx.Client, base_url: str) -> str:
    """GET /health and return the version string.  Raises on failure."""
    r = client.get(f"{base_url}/health", timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("version", "unknown")


def _create_session(client: httpx.Client, base_url: str, spec: dict) -> int:
    """POST /vigil/sessions and return the new session_id."""
    payload = {
        "name": spec["name"],
        "rules": spec["rules"],
        "sample_fps": spec["sample_fps"],
        "severity_threshold": spec["severity_threshold"],
    }
    r = client.post(
        f"{base_url}/vigil/sessions",
        json=payload,
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    return int(data["session_id"])


def _insert_events(db_path: Path, session_id: int, events: list[dict]) -> None:
    """Directly insert synthetic violation events into the SQLite database."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        for ev in events:
            now = datetime.now(UTC).isoformat()
            conn.execute(
                """
                INSERT INTO vigil_events
                    (session_id, timestamp_sec, frame_number,
                     violations_json, frame_path, created_at)
                VALUES (?, ?, ?, ?, NULL, ?)
                """,
                (
                    session_id,
                    ev["timestamp_sec"],
                    ev["frame_number"],
                    json.dumps(ev["violations"], ensure_ascii=False),
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _finalize_session(
    db_path: Path,
    session_id: int,
    violation_count: int,
    total_frames_analyzed: int,
) -> None:
    """Set session status to 'completed' and update aggregate counters."""
    now = datetime.now(UTC).isoformat()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        conn.execute(
            """
            UPDATE vigil_sessions
               SET status                = 'completed',
                   violation_count       = ?,
                   total_frames_analyzed = ?,
                   updated_at            = ?
             WHERE id = ?
            """,
            (violation_count, total_frames_analyzed, now, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def _severity_counts(events: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"critical": 0, "warning": 0, "info": 0}
    for ev in events:
        for v in ev["violations"]:
            sev = v.get("severity", "info")
            counts[sev] = counts.get(sev, 0) + 1
    return counts


def _print_summary(results: list[dict]) -> None:
    _section("Created VigilPilot Sessions")
    total_sessions = len(results)
    total_events = sum(r["event_count"] for r in results)
    for i, r in enumerate(results, 1):
        sc = r["severity_counts"]
        parts = []
        if sc["critical"] > 0:
            parts.append(f"{_red(str(sc['critical']) + ' critical')}")
        if sc["warning"] > 0:
            parts.append(f"{_yellow(str(sc['warning']) + ' warning')}")
        if sc["info"] > 0:
            parts.append(f"{sc['info']} info")
        breakdown = ", ".join(parts) if parts else "none"
        flag = "✓" if r["ok"] else "✗"
        colour = _green if r["ok"] else _red
        print(
            f"  {colour(flag)} [{i}] {_bold(r['name']):<28}"
            f" — session_id={_cyan(str(r['session_id']))},"
            f" {r['event_count']} violations ({breakdown})"
        )

    width = 62
    border = "═" * width
    print(f"\n{_cyan('╔' + border + '╗')}")
    print(_cyan("║") + _bold(f"{'Demo setup complete':^{width}}") + _cyan("║"))
    print(_cyan("║") + f"{'':^{width}}" + _cyan("║"))
    print(
        _cyan("║")
        + f"  {_green('✓')} {total_sessions} sessions, {total_events} violation events inserted{'':<{width - 50}}"
        + _cyan("║")
    )
    print(f"{_cyan('╚' + border + '╝')}")
    print()
    print(f"  {_bold('→')} Open {_cyan('http://localhost:8000')} and navigate to the {_bold('監視')} tab")
    print(f"  {_bold('→')} Demo is ready!")
    print()


# ── Entry point ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SOPilot demo setup — creates VigilPilot sessions with synthetic violation data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--host",
        default=os.environ.get("SOPILOT_HOST", "http://localhost:8000"),
        help="Base URL of the running SOPilot server (default: http://localhost:8000)",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("SOPILOT_API_KEY", "demo-key"),
        help="API key for X-API-Key header (default: env SOPILOT_API_KEY or 'demo-key')",
    )
    p.add_argument(
        "--db-path",
        default=None,
        help=(
            "Path to sopilot.db. Auto-detected from SOPILOT_DATA_DIR or "
            "common locations if omitted."
        ),
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip session creation if a session with the same name already exists.",
    )
    return p.parse_args()


def main() -> int:
    _print_header()
    args = _parse_args()

    base_url = args.host.rstrip("/")
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    db_path = _resolve_db_path(args.db_path)

    # ── 1. Health check ──────────────────────────────────────────────────────
    _section("Server Health")
    try:
        with httpx.Client(headers=headers) as client:
            version = _check_health(client, base_url)
        _ok(f"Server healthy (version: {_cyan(version)})")
    except Exception as exc:
        _fail(f"Cannot reach server at {base_url}: {exc}")
        print(
            f"\n  {_yellow('Hint:')} Make sure the server is running:\n"
            "         docker compose up -d\n"
            "         python -m uvicorn sopilot.main:create_app --factory --port 8000\n"
        )
        return 1

    # ── 2. Verify DB path ────────────────────────────────────────────────────
    _section("Database")
    if not db_path.exists():
        _fail(
            f"Database not found at {db_path}. "
            "Start the server first so it creates the DB, then re-run this script."
        )
        print(
            f"\n  {_yellow('Hint:')} Override the path with --db-path /your/path/to/sopilot.db\n"
        )
        return 1
    _ok(f"Using database: {_cyan(str(db_path))}")

    # Check that vigil_sessions table exists (migration 5 must have run)
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("SELECT 1 FROM vigil_sessions LIMIT 1")
        conn.close()
    except sqlite3.OperationalError:
        _fail(
            "vigil_sessions table not found. "
            "The server must run migrations first (start it with docker compose up -d)."
        )
        return 1
    _ok("vigil_sessions table present")

    # Optional: check existing sessions to support --skip-existing
    existing_names: set[str] = set()
    if args.skip_existing:
        try:
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute("SELECT name FROM vigil_sessions").fetchall()
            conn.close()
            existing_names = {r[0] for r in rows}
            if existing_names:
                _info(f"Existing sessions: {', '.join(sorted(existing_names))}")
        except Exception:
            pass

    # ── 3. Create sessions and inject events ─────────────────────────────────
    _section("Creating VigilPilot Sessions")
    results: list[dict] = []

    with httpx.Client(headers=headers, timeout=30) as client:
        for spec in SESSION_SPECS:
            name = spec["name"]

            if args.skip_existing and name in existing_names:
                _info(f"Skipping '{name}' (already exists, --skip-existing)")
                continue

            # 3a. Create session via API
            try:
                session_id = _create_session(client, base_url, spec)
                _ok(f"Created session [{_bold(name)}] → session_id={_cyan(str(session_id))}")
            except httpx.HTTPStatusError as exc:
                _fail(f"Failed to create session '{name}': HTTP {exc.response.status_code} — {exc.response.text[:200]}")
                results.append({"name": name, "session_id": -1, "event_count": 0, "severity_counts": {}, "ok": False})
                continue
            except Exception as exc:
                _fail(f"Failed to create session '{name}': {exc}")
                results.append({"name": name, "session_id": -1, "event_count": 0, "severity_counts": {}, "ok": False})
                continue

            # 3b. Insert violation events directly into DB
            events = spec["events"]
            try:
                _insert_events(db_path, session_id, events)
                _ok(f"  Inserted {len(events)} violation events into DB")
            except Exception as exc:
                _fail(f"  Failed to insert events for session {session_id}: {exc}")
                results.append({"name": name, "session_id": session_id, "event_count": 0, "severity_counts": {}, "ok": False})
                continue

            # 3c. Finalize session statistics
            sc = _severity_counts(events)
            try:
                _finalize_session(
                    db_path,
                    session_id,
                    violation_count=len(events),
                    total_frames_analyzed=spec["total_frames_analyzed"],
                )
                sev_str = ", ".join(f"{v} {k}" for k, v in sc.items() if v > 0)
                _ok(f"  Session finalized: status=completed, {sev_str}")
            except Exception as exc:
                _fail(f"  Failed to finalize session {session_id}: {exc}")
                results.append({"name": name, "session_id": session_id, "event_count": len(events), "severity_counts": sc, "ok": False})
                continue

            results.append({
                "name": name,
                "session_id": session_id,
                "event_count": len(events),
                "severity_counts": sc,
                "ok": True,
            })

    if not results:
        print(f"\n  {_yellow('Nothing was created.')} (all sessions may already exist with --skip-existing)")
        return 0

    # ── 4. Summary ───────────────────────────────────────────────────────────
    _print_summary(results)

    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"  {_red('WARNING:')} {len(failed)} session(s) had errors. Check output above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
