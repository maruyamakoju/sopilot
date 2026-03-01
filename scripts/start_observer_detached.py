from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from scripts.autopilot_common import now_iso, process_exists, read_json, runner_pid, write_json
except ModuleNotFoundError:  # pragma: no cover - script execution path
    from autopilot_common import now_iso, process_exists, read_json, runner_pid, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Start observe_autopilot.py in detached mode.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--interval-seconds", type=float, default=600.0)
    parser.add_argument("--stale-seconds", type=float, default=1800.0)
    parser.add_argument("--observations-dir", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--pid-file", default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--stop-when-final", action="store_true")
    parser.add_argument("--allow-multiple", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    autopilot_dir = data_dir / "autopilot"
    autopilot_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file).resolve() if args.log_file else (autopilot_dir / "observer.log")
    pid_file = Path(args.pid_file).resolve() if args.pid_file else (autopilot_dir / "observer.json")

    existing = read_json(pid_file)
    existing_pid = runner_pid(existing)
    if existing_pid and process_exists(existing_pid) and not args.allow_multiple:
        raise SystemExit(
            f"observer already running pid={existing_pid}. "
            f"stop it first or pass --allow-multiple."
        )

    observer_script = (Path(__file__).resolve().parent / "observe_autopilot.py").resolve()
    cmd = [
        sys.executable,
        str(observer_script),
        "--data-dir",
        str(data_dir),
        "--interval-seconds",
        str(max(1.0, float(args.interval_seconds))),
        "--stale-seconds",
        str(max(1.0, float(args.stale_seconds))),
    ]
    if args.observations_dir:
        cmd.extend(["--observations-dir", str(Path(args.observations_dir).resolve())])
    if args.max_iterations is not None:
        cmd.extend(["--max-iterations", str(int(args.max_iterations))])
    if args.stop_when_final:
        cmd.append("--stop-when-final")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log_fh:
        if os.name == "nt":
            detached = 0x00000008  # DETACHED_PROCESS
            new_group = 0x00000200  # CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                close_fds=True,
                creationflags=detached | new_group,
            )
        else:
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )

    payload = {
        "pid": int(proc.pid),
        "started_at": now_iso(),
        "data_dir": str(data_dir),
        "log_file": str(log_file),
        "cmd": cmd,
    }
    write_json(pid_file, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

