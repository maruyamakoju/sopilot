from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

try:
    from scripts.autopilot_common import (
        default_runner_pid_file,
        now_iso,
        process_exists,
        read_json,
        runner_pid,
        runner_run_id,
        write_json,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution path
    from autopilot_common import (
        default_runner_pid_file,
        now_iso,
        process_exists,
        read_json,
        runner_pid,
        runner_run_id,
        write_json,
    )


def resolve_pid_file_path(pid_file: str | None, data_dir: str | None) -> Path:
    pid_file_value = str(pid_file).strip() if pid_file else None
    data_dir_value = str(data_dir).strip() if data_dir else None
    if bool(pid_file_value) == bool(data_dir_value):
        raise SystemExit("specify exactly one of --pid-file or --data-dir")
    if pid_file_value:
        return Path(pid_file_value).resolve()
    return default_runner_pid_file(Path(data_dir_value).resolve())


def _soft_terminate(pid: int) -> None:
    os.kill(pid, signal.SIGTERM)


def _force_terminate(pid: int) -> None:
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=True, capture_output=True, text=True)
            return
        except FileNotFoundError:
            # taskkill may be unavailable in constrained shells.
            os.kill(pid, signal.SIGTERM)
            return
    os.kill(pid, signal.SIGKILL)


def _wait_until_stopped(pid: int, grace_seconds: float) -> bool:
    if grace_seconds <= 0:
        return not process_exists(pid)
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if not process_exists(pid):
            return True
        time.sleep(0.2)
    return not process_exists(pid)


def request_stop(pid: int, *, force: bool, grace_seconds: float) -> tuple[bool, str | None, str]:
    if not process_exists(pid):
        return True, None, "already_stopped"

    method = "graceful"
    error: str | None = None
    try:
        if force:
            method = "force"
            _force_terminate(pid)
        else:
            _soft_terminate(pid)
    except Exception as exc:
        error = str(exc)

    if _wait_until_stopped(pid, grace_seconds):
        return True, None, method

    if not force:
        method = "force_fallback"
        try:
            _force_terminate(pid)
        except Exception as exc:
            error = f"{error}; fallback: {exc}" if error else str(exc)
        if _wait_until_stopped(pid, max(0.5, grace_seconds)):
            return True, None, method

    if error is None:
        error = "process still running after stop attempt"
    return False, error, method


def main() -> None:
    parser = argparse.ArgumentParser(description="Stop detached SOPilot autopilot process.")
    parser.add_argument("--pid-file", default=None, help="Path to runner.json from start_autopilot_detached.py")
    parser.add_argument("--data-dir", default=None, help="Data directory; runner.json is resolved under <data-dir>/autopilot.")
    parser.add_argument("--run-id", default=None, help="Expected run_id; stop is rejected when it does not match.")
    parser.add_argument(
        "--require-run-id",
        action="store_true",
        help="Require --run-id to prevent accidental stop of a different run.",
    )
    parser.add_argument("--grace-seconds", type=float, default=10.0, help="Seconds to wait for graceful stop before forcing.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pid_path = resolve_pid_file_path(args.pid_file, args.data_dir)
    if not pid_path.exists():
        raise SystemExit(f"pid-file not found: {pid_path}")
    payload = read_json(pid_path)
    pid = runner_pid(payload)
    if payload is None or pid is None:
        raise SystemExit(f"invalid pid-file json: {pid_path}")
    file_run_id = runner_run_id(payload)
    expected_run_id = str(args.run_id).strip() if args.run_id else None

    if args.require_run_id and not expected_run_id:
        raise SystemExit("--require-run-id specified but --run-id is missing")
    if expected_run_id:
        if not file_run_id:
            raise SystemExit(f"run_id mismatch: pid-file has no run_id (expected {expected_run_id})")
        if file_run_id != expected_run_id:
            raise SystemExit(f"run_id mismatch: pid-file has {file_run_id} (expected {expected_run_id})")

    stopped, error, method = request_stop(pid, force=bool(args.force), grace_seconds=max(0.0, float(args.grace_seconds)))

    payload["stop_requested_at"] = now_iso()
    payload["stopped_at"] = now_iso()
    payload["stopped"] = stopped
    payload["stop_error"] = error
    payload["stop_method"] = method
    write_json(pid_path, payload)

    print(
        json.dumps(
            {
                "pid_file": str(pid_path),
                "pid": pid,
                "run_id": file_run_id,
                "stopped": stopped,
                "stop_method": method,
                "error": error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
