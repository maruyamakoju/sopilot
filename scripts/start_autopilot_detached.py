from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from sopilot.eval.gates import available_gate_profiles
from sopilot.eval.harness import available_critical_scoring_modes

try:
    from scripts.autopilot_common import now_iso, read_json, runner_is_active, runner_pid, write_json
except ModuleNotFoundError:  # pragma: no cover - script execution path
    from autopilot_common import now_iso, read_json, runner_is_active, runner_pid, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Start SOPilot autopilot in detached mode.")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--task-name", default="PoC Task")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--trainee-dir", required=True)
    parser.add_argument("--trainee-bad-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--site-id", default="site-a")
    parser.add_argument("--backend", choices=["vjepa2", "color-motion"], default="color-motion")
    parser.add_argument("--gold-id", type=int, default=2)
    parser.add_argument("--duration-hours", type=float, default=96.0)
    parser.add_argument("--interval-minutes", type=float, default=60.0)
    parser.add_argument(
        "--step-timeout-minutes",
        type=float,
        default=30.0,
        help="Timeout for each runner subprocess step (0 disables timeout).",
    )
    parser.add_argument("--bad-example-every", type=int, default=6)
    parser.add_argument("--rescore-every", type=int, default=6)
    parser.add_argument("--rescore-repeat", type=int, default=1)
    parser.add_argument("--backup-every", type=int, default=24)
    parser.add_argument("--max-source", type=int, default=6)
    parser.add_argument("--critical-pattern", action="append", default=None)
    parser.add_argument(
        "--gate-profile",
        choices=available_gate_profiles(),
        default="legacy_poc",
        help="Named evaluation gate profile passed to the runner.",
    )
    parser.add_argument(
        "--allow-profile-overrides",
        action="store_true",
        help="Allow overriding thresholds even if selected gate profile is locked.",
    )
    parser.add_argument(
        "--critical-scoring-mode",
        choices=available_critical_scoring_modes(),
        default="legacy_binary",
    )
    parser.add_argument("--critical-threshold", type=float, default=0.5)
    parser.add_argument("--critical-sweep-auto", action="store_true")
    parser.add_argument("--critical-sweep-values", default=None)
    parser.add_argument("--critical-sweep-start", type=float, default=None)
    parser.add_argument("--critical-sweep-stop", type=float, default=None)
    parser.add_argument("--critical-sweep-step", type=float, default=0.05)
    parser.add_argument(
        "--critical-sweep-scoring-mode",
        choices=available_critical_scoring_modes(),
        default="continuous_v1",
    )
    parser.add_argument("--max-critical-miss-rate", type=float, default=None)
    parser.add_argument("--max-critical-fpr", type=float, default=None)
    parser.add_argument("--max-critical-miss-ci95-high", type=float, default=None)
    parser.add_argument("--max-critical-fpr-ci95-high", type=float, default=None)
    parser.add_argument("--max-rescore-jitter", type=float, default=None)
    parser.add_argument("--max-dtw-p90", type=float, default=None)
    parser.add_argument("--min-completed-jobs", type=int, default=None)
    parser.add_argument("--min-labels-total-jobs", type=int, default=None)
    parser.add_argument("--min-labeled-jobs", type=int, default=None)
    parser.add_argument("--min-critical-positives", type=int, default=None)
    parser.add_argument("--min-critical-negatives", type=int, default=None)
    parser.add_argument("--min-coverage-rate", type=float, default=None)
    parser.add_argument("--min-rescore-pairs", type=int, default=None)
    parser.add_argument("--reset-data-dir", action="store_true")
    parser.add_argument("--skip-backup", action="store_true")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--pid-file", default=None)
    parser.add_argument("--run-id", default=None, help="Optional run identifier for correlation across status files.")
    parser.add_argument(
        "--allow-multiple",
        action="store_true",
        help="Allow starting a new detached runner even if runner.json points to an active process.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file).resolve() if args.log_file else (data_dir / "autopilot" / "runner.log")
    pid_file = Path(args.pid_file).resolve() if args.pid_file else (data_dir / "autopilot" / "runner.json")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    existing_runner = read_json(pid_file)
    if runner_is_active(existing_runner) and not args.allow_multiple:
        pid = runner_pid(existing_runner)
        started_at = (existing_runner or {}).get("started_at")
        raise SystemExit(
            f"autopilot already running pid={pid} started_at={started_at}. "
            f"Stop it first with scripts/stop_autopilot.py or pass --allow-multiple."
        )
    run_id = str(args.run_id).strip() if args.run_id else uuid4().hex

    runner_script = (Path(__file__).resolve().parent / "poc_autopilot_runner.py").resolve()

    cmd = [
        sys.executable,
        str(runner_script),
        "--task-id",
        args.task_id,
        "--run-id",
        run_id,
        "--task-name",
        args.task_name,
        "--base-dir",
        args.base_dir,
        "--trainee-dir",
        args.trainee_dir,
        "--trainee-bad-dir",
        args.trainee_bad_dir,
        "--data-dir",
        str(data_dir),
        "--site-id",
        args.site_id,
        "--backend",
        args.backend,
        "--gold-id",
        str(args.gold_id),
        "--duration-hours",
        str(args.duration_hours),
        "--interval-minutes",
        str(args.interval_minutes),
        "--step-timeout-minutes",
        str(args.step_timeout_minutes),
        "--bad-example-every",
        str(args.bad_example_every),
        "--rescore-every",
        str(args.rescore_every),
        "--rescore-repeat",
        str(args.rescore_repeat),
        "--backup-every",
        str(args.backup_every),
        "--max-source",
        str(args.max_source),
    ]
    if args.gate_profile:
        cmd.extend(["--gate-profile", args.gate_profile])
    if args.allow_profile_overrides:
        cmd.append("--allow-profile-overrides")
    if args.critical_scoring_mode:
        cmd.extend(["--critical-scoring-mode", args.critical_scoring_mode])
    cmd.extend(["--critical-threshold", str(args.critical_threshold)])
    if args.critical_sweep_auto:
        cmd.append("--critical-sweep-auto")
    if args.critical_sweep_values:
        cmd.extend(["--critical-sweep-values", str(args.critical_sweep_values)])
    if args.critical_sweep_start is not None:
        cmd.extend(["--critical-sweep-start", str(args.critical_sweep_start)])
    if args.critical_sweep_stop is not None:
        cmd.extend(["--critical-sweep-stop", str(args.critical_sweep_stop)])
    if args.critical_sweep_step is not None:
        cmd.extend(["--critical-sweep-step", str(args.critical_sweep_step)])
    if args.critical_sweep_scoring_mode:
        cmd.extend(["--critical-sweep-scoring-mode", args.critical_sweep_scoring_mode])
    if args.max_critical_miss_rate is not None:
        cmd.extend(["--max-critical-miss-rate", str(args.max_critical_miss_rate)])
    if args.max_critical_fpr is not None:
        cmd.extend(["--max-critical-fpr", str(args.max_critical_fpr)])
    if args.max_critical_miss_ci95_high is not None:
        cmd.extend(["--max-critical-miss-ci95-high", str(args.max_critical_miss_ci95_high)])
    if args.max_critical_fpr_ci95_high is not None:
        cmd.extend(["--max-critical-fpr-ci95-high", str(args.max_critical_fpr_ci95_high)])
    if args.max_rescore_jitter is not None:
        cmd.extend(["--max-rescore-jitter", str(args.max_rescore_jitter)])
    if args.max_dtw_p90 is not None:
        cmd.extend(["--max-dtw-p90", str(args.max_dtw_p90)])
    if args.min_completed_jobs is not None:
        cmd.extend(["--min-completed-jobs", str(args.min_completed_jobs)])
    if args.min_labels_total_jobs is not None:
        cmd.extend(["--min-labels-total-jobs", str(args.min_labels_total_jobs)])
    if args.min_labeled_jobs is not None:
        cmd.extend(["--min-labeled-jobs", str(args.min_labeled_jobs)])
    if args.min_critical_positives is not None:
        cmd.extend(["--min-critical-positives", str(args.min_critical_positives)])
    if args.min_critical_negatives is not None:
        cmd.extend(["--min-critical-negatives", str(args.min_critical_negatives)])
    if args.min_coverage_rate is not None:
        cmd.extend(["--min-coverage-rate", str(args.min_coverage_rate)])
    if args.min_rescore_pairs is not None:
        cmd.extend(["--min-rescore-pairs", str(args.min_rescore_pairs)])
    if args.reset_data_dir:
        cmd.append("--reset-data-dir")
    if args.skip_backup:
        cmd.append("--skip-backup")
    for pattern in (args.critical_pattern or []):
        cmd.extend(["--critical-pattern", pattern])

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
        "run_id": run_id,
        "started_at": now_iso(),
        "log_file": str(log_file),
        "data_dir": str(data_dir),
        "cmd": cmd,
    }
    write_json(pid_file, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
