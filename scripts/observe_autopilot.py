from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

try:
    from scripts.autopilot_status import collect_status
except ModuleNotFoundError:  # pragma: no cover - script execution path
    from autopilot_status import collect_status


def _snapshot_stamp() -> str:
    now = datetime.now(UTC)
    return now.strftime("%Y%m%d_%H%M%S_%f")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def snapshot_once(
    *,
    data_dir: Path,
    observations_dir: Path,
    stale_seconds: float,
) -> dict:
    observations_dir.mkdir(parents=True, exist_ok=True)
    stamp = _snapshot_stamp()
    status = collect_status(data_dir, stale_seconds=stale_seconds)
    status_path = observations_dir / f"status_{stamp}.json"
    _write_json(status_path, status)

    copied: dict[str, str] = {}
    root_files = ("gate_report.json", "eval_report.json", "incremental_summary.json")
    for name in root_files:
        src = data_dir / name
        if not src.exists():
            continue
        dst = observations_dir / f"{Path(name).stem}_{stamp}.json"
        shutil.copy2(src, dst)
        copied[name] = str(dst)

    autopilot_files = ("latest_cycle.json", "heartbeat.json", "final_summary.json", "crash.json")
    for name in autopilot_files:
        src = data_dir / "autopilot" / name
        if not src.exists():
            continue
        dst = observations_dir / f"{Path(name).stem}_{stamp}.json"
        shutil.copy2(src, dst)
        copied[f"autopilot/{name}"] = str(dst)

    result = {
        "ts_epoch": time.time(),
        "stamp": stamp,
        "data_dir": str(data_dir),
        "status_path": str(status_path),
        "lifecycle_state": status.get("lifecycle_state"),
        "runner_active": status.get("runner_active"),
        "run_id_match": status.get("run_id_match"),
        "copied": copied,
    }
    _write_json(observations_dir / "latest_observation.json", result)
    return result


def _should_stop(*, stop_when_final: bool, status_payload: dict) -> bool:
    if not stop_when_final:
        return False
    state = str(status_payload.get("lifecycle_state") or "").strip()
    return state in {"finished", "stopped", "crashed", "not_running"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Periodic observation snapshots for detached autopilot runs.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--observations-dir",
        default=None,
        help="Output dir for snapshots (default: <data-dir>/autopilot/observations)",
    )
    parser.add_argument("--interval-seconds", type=float, default=600.0)
    parser.add_argument("--stale-seconds", type=float, default=1800.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--stop-when-final", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    observations_dir = (
        Path(args.observations_dir).resolve()
        if args.observations_dir
        else (data_dir / "autopilot" / "observations").resolve()
    )

    interval = max(1.0, float(args.interval_seconds))
    stale_seconds = max(1.0, float(args.stale_seconds))
    max_iterations = int(args.max_iterations) if args.max_iterations is not None else None

    iteration = 0
    while True:
        iteration += 1
        payload = snapshot_once(data_dir=data_dir, observations_dir=observations_dir, stale_seconds=stale_seconds)
        payload["iteration"] = iteration
        print(json.dumps(payload, ensure_ascii=False))

        status_path = Path(payload["status_path"])
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if args.once:
            break
        if max_iterations is not None and iteration >= max_iterations:
            break
        if _should_stop(stop_when_final=bool(args.stop_when_final), status_payload=status):
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()

