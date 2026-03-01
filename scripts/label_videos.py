"""label_videos.py — Interactive CLI for human expert video annotation.

Outputs a JSON file compatible with run_loso_evaluation.py --human-labels.

Usage
-----
    python scripts/label_videos.py \\
        --db data_release_baseline/sopilot.db \\
        --task-id filter_change \\
        --output human_labels_expert_A.json \\
        --annotator-id expert_A

Resuming interrupted sessions
------------------------------
Re-running with the same --output path will load existing annotations and skip
already-annotated video IDs, picking up where you left off.

Verdicts
--------
    p = pass
    f = fail
    r = retrain
    n = needs_review
    s = skip (not written to output)

Dependencies: stdlib + sqlite3 only (zero external dependencies).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# ANSI helpers (gracefully degraded when stdout is not a TTY)
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_DIM = "\033[2m"


def _c(text: str, *codes: str) -> str:
    """Wrap *text* in ANSI escape codes when stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + _RESET


def _hr(char: str = "=", width: int = 45) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _open_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise SystemExit(f"[ERROR] Database not found: {db_path}")
    conn = sqlite3.connect(str(db_path), timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA query_only=ON")
    return conn


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_jobs_for_annotation(
    conn: sqlite3.Connection,
    *,
    task_id: str | None,
) -> list[dict[str, Any]]:
    """Return one representative job per trainee video.

    When multiple score jobs exist for the same trainee (e.g. scored against
    multiple gold references), the job with the highest score is used for
    display purposes.  The trainee_video_id is what matters for labelling.
    """
    has_orig_filename = _has_column(conn, "videos", "original_filename")
    filename_expr = (
        "tv.original_filename AS display_filename,"
        if has_orig_filename
        else "NULL AS display_filename,"
    )

    query = f"""
        SELECT
            sj.id                   AS job_id,
            sj.trainee_video_id,
            sj.gold_video_id,
            sj.score_json,
            tv.file_path            AS file_path,
            {filename_expr}
            tv.operator_id_hash     AS operator_id,
            tv.is_gold              AS is_gold
        FROM score_jobs sj
        LEFT JOIN videos tv ON tv.id = sj.trainee_video_id
        WHERE sj.status = 'completed'
          AND sj.score_json IS NOT NULL
          AND (tv.is_gold = 0 OR tv.is_gold IS NULL)
    """
    params: list[Any] = []
    if task_id:
        query += """
          AND EXISTS (
              SELECT 1 FROM videos gv
              WHERE gv.id = sj.gold_video_id AND gv.task_id = ?
          )
        """
        params.append(task_id)
    query += " ORDER BY sj.trainee_video_id ASC, sj.id ASC"

    rows = conn.execute(query, params).fetchall()

    # Collapse to one representative row per trainee_video_id — highest score
    best: dict[int, dict[str, Any]] = {}
    for row in rows:
        item = dict(row)
        tid = item["trainee_video_id"]
        raw = item.get("score_json") or "{}"
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {}
        item["_parsed"] = parsed

        score_val = None
        try:
            score_val = float(parsed.get("score", 0) or 0)
        except (TypeError, ValueError):
            score_val = 0.0
        item["_score"] = score_val

        if tid not in best or score_val > best[tid]["_score"]:
            best[tid] = item

    return sorted(best.values(), key=lambda x: x["trainee_video_id"])


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display_filename(job: dict[str, Any]) -> str:
    """Best human-readable filename from the job record."""
    name = job.get("display_filename") or job.get("file_path") or ""
    if name:
        return Path(name).name
    return f"video_id_{job['trainee_video_id']}"


def _get_score_and_dtw(job: dict[str, Any]) -> tuple[float | None, float | None]:
    parsed = job.get("_parsed", {})
    score: float | None = None
    dtw: float | None = None
    try:
        score = float(parsed["score"])
    except (KeyError, TypeError, ValueError):
        pass
    metrics = parsed.get("metrics") or {}
    try:
        dtw = float(metrics["dtw_normalized_cost"])
    except (KeyError, TypeError, ValueError):
        # Fallback: alignment.normalized_cost
        alignment = parsed.get("alignment") or {}
        try:
            dtw = float(alignment["normalized_cost"])
        except (KeyError, TypeError, ValueError):
            pass
    return score, dtw


def _get_critical_flags(job: dict[str, Any]) -> list[str]:
    parsed = job.get("_parsed", {})
    flags: list[str] = []
    for dev in parsed.get("deviations", []):
        if isinstance(dev, dict) and dev.get("severity") == "critical":
            dtype = dev.get("type", "unknown")
            step = dev.get("step_index")
            detail = dev.get("detail", "")
            if step is not None:
                flags.append(f"{dtype} (step {step})")
            else:
                flags.append(dtype)
    return flags


def _get_step_contributions(job: dict[str, Any]) -> dict[str, float]:
    """Return step_contributions dict if present in score_json."""
    parsed = job.get("_parsed", {})
    sc = parsed.get("step_contributions")
    if isinstance(sc, dict):
        return {str(k): float(v) for k, v in sc.items()}
    # Some schemas may store it as a list of [name, value] pairs
    if isinstance(sc, list):
        return {f"step{i+1}": float(v) for i, v in enumerate(sc) if v is not None}
    return {}


def _fmt_step_contributions(sc: dict[str, float]) -> str:
    if not sc:
        return "(not available)"
    parts = [f"{k}={v:.1f}" for k, v in sc.items()]
    return ", ".join(parts)


def _print_job_card(
    job: dict[str, Any],
    *,
    index: int,
    total: int,
) -> None:
    fname = _display_filename(job)
    score, dtw = _get_score_and_dtw(job)
    critical = _get_critical_flags(job)
    sc = _get_step_contributions(job)

    score_str = f"{score:.1f}" if score is not None else "N/A"
    dtw_str = f"{dtw:.3f}" if dtw is not None else "N/A"
    flags_str = ", ".join(critical) if critical else "none"
    sc_str = _fmt_step_contributions(sc)

    print()
    print(_c(_hr("="), _BOLD, _CYAN))
    print(_c(f"Video {index}/{total}: {fname}", _BOLD))
    print(_c(f"System score: {score_str}  |  DTW cost: {dtw_str}  |  Critical flags: {flags_str}", _DIM))
    if sc:
        print(_c(f"Step contributions: {sc_str}", _DIM))
    print(_c(_hr("-"), _DIM))


# ---------------------------------------------------------------------------
# Input validators
# ---------------------------------------------------------------------------

_VERDICT_MAP: dict[str, str] = {
    "p": "pass",
    "f": "fail",
    "r": "retrain",
    "n": "needs_review",
    "s": "skip",
}


def _prompt_score() -> float | None:
    """Prompt for an expert score 0-100. Returns None on skip/empty."""
    while True:
        raw = input("  Your expert score (0-100, or Enter to skip): ").strip()
        if raw == "":
            return None
        try:
            val = float(raw)
        except ValueError:
            print(_c("  [!] Please enter a number between 0 and 100.", _YELLOW))
            continue
        if 0.0 <= val <= 100.0:
            return val
        print(_c("  [!] Score must be between 0 and 100.", _YELLOW))


def _prompt_verdict() -> str:
    """Prompt for a verdict key. Returns one of the _VERDICT_MAP keys."""
    keys_str = ", ".join(f"{k}={v}" for k, v in _VERDICT_MAP.items())
    while True:
        raw = input(f"  Your verdict [{keys_str}]: ").strip().lower()
        if raw in _VERDICT_MAP:
            return raw
        print(_c(f"  [!] Please enter one of: {', '.join(_VERDICT_MAP.keys())}", _YELLOW))


def _prompt_notes() -> str:
    return input("  Notes (optional): ").strip()


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


def _load_existing_labels(output_path: Path) -> list[dict[str, Any]]:
    """Load existing annotations from disk; return empty list if file absent."""
    if not output_path.exists():
        return []
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception as exc:
        print(_c(f"  [!] Could not parse existing file {output_path}: {exc}", _YELLOW))
    return []


def _save_labels(output_path: Path, labels: list[dict[str, Any]]) -> None:
    """Write labels to disk atomically using a temp file + rename."""
    tmp_path = output_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # On Windows os.replace works atomically when src and dst are on same volume
    os.replace(str(tmp_path), str(output_path))


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


class _AnnotationSession:
    """Manages incremental writes and graceful Ctrl+C handling."""

    def __init__(self, output_path: Path, annotator_id: str) -> None:
        self.output_path = output_path
        self.annotator_id = annotator_id
        self.labels: list[dict[str, Any]] = _load_existing_labels(output_path)
        self._already_done: set[int] = {int(lbl["video_id"]) for lbl in self.labels}
        self._n_annotated_this_session = 0
        self._n_skipped_this_session = 0
        self._interrupted = False

        # Register SIGINT handler for graceful exit
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum: int, frame: object) -> None:
        self._interrupted = True
        print()
        print(_c("\n[Ctrl+C] Saving and exiting...", _YELLOW, _BOLD))

    def already_done(self, video_id: int) -> bool:
        return video_id in self._already_done

    def record(
        self,
        *,
        video_id: int,
        human_score: float | None,
        verdict_key: str,
        notes: str,
    ) -> None:
        """Record one annotation (unless verdict is 'skip') and save to disk."""
        if verdict_key == "s":
            self._n_skipped_this_session += 1
            return

        entry: dict[str, Any] = {
            "video_id": video_id,
            "human_score": human_score,
            "human_verdict": _VERDICT_MAP[verdict_key],
            "annotator_id": self.annotator_id,
            "annotation_time": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
        self.labels.append(entry)
        self._already_done.add(video_id)
        self._n_annotated_this_session += 1
        _save_labels(self.output_path, self.labels)

    def print_summary(self) -> None:
        total = len(self.labels)
        print()
        print(_c(_hr("="), _BOLD, _CYAN))
        print(_c("Annotation Session Complete", _BOLD))
        print(f"  Annotated this session : {self._n_annotated_this_session}")
        print(f"  Skipped this session   : {self._n_skipped_this_session}")
        print(f"  Total in output file   : {total}")
        print(f"  Output path            : {self.output_path}")
        print(_c(_hr("="), _BOLD, _CYAN))
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive human expert annotation CLI for SOPilot videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", dest="db_path", required=True, metavar="PATH",
                        help="Path to sopilot.db SQLite database.")
    parser.add_argument("--task-id", default=None, metavar="ID",
                        help="Optional task_id filter (e.g. filter_change).")
    parser.add_argument("--output", required=True, metavar="PATH",
                        help="Output JSON file path for annotations.")
    parser.add_argument("--annotator-id", required=True, metavar="ID",
                        help="Annotator identifier (e.g. expert_A).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    db_path = Path(args.db_path).resolve()
    output_path = Path(args.output).resolve()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(_c("\nSOPilot Video Annotation Tool", _BOLD, _CYAN))
    print(_c("=" * 45, _CYAN))
    print(f"  DB:           {db_path}")
    print(f"  Task ID:      {args.task_id or '(all)'}")
    print(f"  Output:       {output_path}")
    print(f"  Annotator:    {args.annotator_id}")
    print()
    print("  Verdict keys: p=pass  f=fail  r=retrain  n=needs_review  s=skip")
    print()

    # Load jobs from DB
    conn = _open_db(db_path)
    jobs = _load_jobs_for_annotation(conn, task_id=args.task_id)
    conn.close()

    total = len(jobs)
    if total == 0:
        raise SystemExit(
            "[ERROR] No completed trainee score jobs found for this task. "
            "Check --task-id and database contents."
        )

    # Initialise session (loads existing labels if any)
    session = _AnnotationSession(output_path, annotator_id=args.annotator_id)

    n_already_done = sum(1 for job in jobs if session.already_done(job["trainee_video_id"]))
    if n_already_done > 0:
        print(
            _c(
                f"  Resuming: {n_already_done}/{total} videos already annotated — skipping them.",
                _YELLOW,
            )
        )
        print()

    # Main annotation loop
    displayed_index = 0
    for job in jobs:
        if session._interrupted:
            break

        vid_id = int(job["trainee_video_id"])

        if session.already_done(vid_id):
            continue

        displayed_index += 1
        # Count remaining (not already done)
        remaining_total = total - n_already_done
        _print_job_card(job, index=displayed_index, total=remaining_total)

        # Collect expert score
        try:
            human_score = _prompt_score()
        except EOFError:
            # Non-interactive stdin (e.g. piped input exhausted)
            break

        if session._interrupted:
            break

        # Collect verdict
        try:
            verdict_key = _prompt_verdict()
        except EOFError:
            break

        if session._interrupted:
            break

        # Collect optional notes
        try:
            notes = _prompt_notes()
        except EOFError:
            notes = ""

        # Record and save (skip verdict 's' does not write to file)
        session.record(
            video_id=vid_id,
            human_score=human_score,
            verdict_key=verdict_key,
            notes=notes,
        )

        if verdict_key == "s":
            print(_c("  [skipped — not saved]", _DIM))
        else:
            verdict_label = _VERDICT_MAP[verdict_key]
            score_display = f"{human_score:.1f}" if human_score is not None else "N/A"
            print(
                _c(
                    f"  Saved: verdict={verdict_label}, score={score_display}",
                    _GREEN,
                )
            )

        if session._interrupted:
            break

    session.print_summary()


if __name__ == "__main__":
    main()
