from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.autopilot_common import (
        heartbeat_age_seconds,
        read_json,
        runner_is_active,
        runner_is_stopped,
        runner_run_id,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution path
    from autopilot_common import heartbeat_age_seconds, read_json, runner_is_active, runner_is_stopped, runner_run_id


def _normalize_run_id(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _lifecycle_state(
    *,
    runner_active: bool,
    runner_stopped: bool,
    runner_run_id: str | None,
    heartbeat_state: str | None,
    heartbeat_stale: bool,
    crash_run_id: str | None,
    final_run_id: str | None,
) -> str:
    if runner_active:
        return "running"
    if runner_stopped:
        return "stopped"
    if runner_run_id and final_run_id and runner_run_id == final_run_id:
        return "finished"
    if runner_run_id and crash_run_id and runner_run_id == crash_run_id:
        return "crashed"
    if heartbeat_stale and heartbeat_state in {"started", "running_step"}:
        return "stale"
    return "not_running"


def _recommended_action(lifecycle_state: str) -> str:
    if lifecycle_state == "running":
        return "monitor"
    if lifecycle_state == "finished":
        return "collect_artifacts"
    if lifecycle_state == "stopped":
        return "restart_if_needed"
    if lifecycle_state == "crashed":
        return "inspect_crash_json"
    if lifecycle_state == "stale":
        return "check_runner_log_and_consider_stop"
    return "start_or_verify_runner"


def collect_status(root: Path, *, stale_seconds: float = 1800.0) -> dict:
    runner = read_json(root / "autopilot" / "runner.json")
    latest = read_json(root / "autopilot" / "latest_cycle.json")
    heartbeat = read_json(root / "autopilot" / "heartbeat.json")
    crash = read_json(root / "autopilot" / "crash.json")
    final = read_json(root / "autopilot" / "final_summary.json")
    gate = read_json(root / "gate_report.json")

    runner_active = runner_is_active(runner)
    runner_stopped = runner_is_stopped(runner)
    runner_state = "running" if runner_active else ("stopped" if runner_stopped else "not_running")
    runner_run = runner_run_id(runner)
    heartbeat_run = _normalize_run_id((heartbeat or {}).get("run_id"))
    latest_run = _normalize_run_id((latest or {}).get("run_id"))
    crash_run = _normalize_run_id((crash or {}).get("run_id"))
    final_run = _normalize_run_id((final or {}).get("run_id"))
    run_id_match = bool(runner_run and heartbeat_run and runner_run == heartbeat_run)
    heartbeat_state = _normalize_run_id((heartbeat or {}).get("state"))
    heartbeat_age_sec = heartbeat_age_seconds(heartbeat)
    heartbeat_stale = bool(
        heartbeat_age_sec is not None
        and heartbeat_age_sec > max(1.0, float(stale_seconds))
        and heartbeat_state in {"started", "running_step"}
    )
    lifecycle_state = _lifecycle_state(
        runner_active=runner_active,
        runner_stopped=runner_stopped,
        runner_run_id=runner_run,
        heartbeat_state=heartbeat_state,
        heartbeat_stale=heartbeat_stale,
        crash_run_id=crash_run,
        final_run_id=final_run,
    )
    current_run_id = runner_run or heartbeat_run or latest_run or final_run or crash_run

    return {
        "data_dir": str(root),
        "current_run_id": current_run_id,
        "lifecycle_state": lifecycle_state,
        "recommended_action": _recommended_action(lifecycle_state),
        "runner": runner,
        "runner_run_id": runner_run,
        "runner_active": runner_active,
        "runner_state": runner_state,
        "heartbeat_run_id": heartbeat_run,
        "latest_cycle_run_id": latest_run,
        "crash_run_id": crash_run,
        "final_run_id": final_run,
        "run_id_match": run_id_match,
        "heartbeat_state": heartbeat_state,
        "heartbeat_age_sec": heartbeat_age_sec,
        "heartbeat_stale": heartbeat_stale,
        "heartbeat": heartbeat,
        "latest_cycle": latest,
        "crash": crash,
        "final_summary": final,
        "gate": {
            "overall_pass": (gate or {}).get("gates", {}).get("overall_pass") if gate else None,
            "num_completed_jobs": (gate or {}).get("num_completed_jobs") if gate else None,
            "critical_miss_rate": (gate or {}).get("critical_miss_rate") if gate else None,
            "critical_false_positive_rate": (gate or {}).get("critical_false_positive_rate") if gate else None,
            "dtw_p90": ((gate or {}).get("dtw_normalized_cost_stats") or {}).get("p90") if gate else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Show detached autopilot status.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--stale-seconds",
        type=float,
        default=1800.0,
        help="Threshold for stale heartbeat while a cycle step is running.",
    )
    args = parser.parse_args()

    root = Path(args.data_dir).resolve()
    payload = collect_status(root, stale_seconds=max(1.0, float(args.stale_seconds)))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
