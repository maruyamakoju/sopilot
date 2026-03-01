from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from uuid import uuid4

from sopilot.eval.gates import available_gate_profiles
from sopilot.eval.harness import available_critical_scoring_modes

try:
    from scripts.autopilot_common import now_iso
    from scripts.autopilot_common import write_json as _write_json_atomic
except ModuleNotFoundError:  # pragma: no cover - script execution path
    from autopilot_common import now_iso
    from autopilot_common import write_json as _write_json_atomic


def run(cmd: list[str], *, timeout_sec: float | None = None) -> subprocess.CompletedProcess[None]:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False, timeout=timeout_sec)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict) -> None:
    _write_json_atomic(path, payload)


def _int_or_none(value: object) -> int | None:
    try:
        pid = int(value)
    except Exception:
        return None
    return pid if pid > 0 else None


def sync_runner_metadata(
    runner_path: Path,
    *,
    run_id: str,
    data_dir: Path,
    log_file: Path,
    cmd: list[str] | None = None,
    current_pid: int | None = None,
) -> dict:
    payload = read_json(runner_path)
    existing_pid = _int_or_none(payload.get("pid"))
    pid = int(current_pid if current_pid is not None else os.getpid())
    started_at = payload.get("started_at") if isinstance(payload.get("started_at"), str) else None
    existing_cmd = payload.get("cmd")
    resolved_cmd = (
        cmd
        if cmd is not None
        else (existing_cmd if isinstance(existing_cmd, list) else [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    )
    synced = {
        **payload,
        "pid": pid,
        "run_id": run_id,
        "started_at": started_at or now_iso(),
        "runner_started_at": now_iso(),
        "pid_source": "runner",
        "log_file": str(log_file),
        "data_dir": str(data_dir),
        "cmd": resolved_cmd,
    }
    if existing_pid and existing_pid != pid:
        synced["launcher_pid"] = existing_pid
    write_json(runner_path, synced)
    return synced


def run_step(
    *,
    run_id: str,
    cycle: int,
    step_name: str,
    cmd: list[str],
    heartbeat_json: Path,
    cycle_row: dict,
    timeout_sec: float | None = None,
) -> None:
    write_json(
        heartbeat_json,
        {"run_id": run_id, "cycle": cycle, "state": "running_step", "step": step_name, "ts": time.time()},
    )
    started = time.time()
    try:
        cp = run(cmd, timeout_sec=timeout_sec)
        step = {
            "name": step_name,
            "returncode": cp.returncode,
            "duration_sec": round(time.time() - started, 2),
        }
    except subprocess.TimeoutExpired:
        step = {
            "name": step_name,
            "returncode": 124,
            "duration_sec": round(time.time() - started, 2),
            "timed_out": True,
            "timeout_sec": timeout_sec,
        }
    cycle_row["steps"].append(step)
    if step.get("returncode") != 0:
        cycle_row["ok"] = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-horizon PoC autopilot runner.")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-id", default=None)
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
    parser.add_argument("--bad-example-every", type=int, default=6, help="Generate bad examples every N cycles")
    parser.add_argument("--max-source", type=int, default=6)
    parser.add_argument("--rescore-every", type=int, default=6, help="Run rescore step every N cycles")
    parser.add_argument("--rescore-repeat", type=int, default=1)
    parser.add_argument("--backup-every", type=int, default=24, help="Run backup step every N cycles")
    parser.add_argument("--critical-pattern", action="append", default=None)
    parser.add_argument(
        "--gate-profile",
        choices=available_gate_profiles(),
        default="legacy_poc",
        help="Named evaluation gate profile passed to evaluate_poc.py.",
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
        help="Critical detection mode passed to evaluate_poc.py.",
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
    parser.add_argument(
        "--step-timeout-minutes",
        type=float,
        default=0.0,
        help="Optional timeout for each subprocess step (0 disables timeout).",
    )
    args = parser.parse_args()

    py = sys.executable
    scripts_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = data_dir / "autopilot"
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_jsonl = logs_dir / "cycles.jsonl"
    latest_json = logs_dir / "latest_cycle.json"
    heartbeat_json = logs_dir / "heartbeat.json"
    crash_json = logs_dir / "crash.json"
    runner_json = logs_dir / "runner.json"
    runner_log = logs_dir / "runner.log"

    labels_path = data_dir / "labels_template.json"
    eval_path = data_dir / "eval_report.json"
    gate_path = data_dir / "gate_report.json"
    summary_path = data_dir / "incremental_summary.json"
    db_path = data_dir / "sopilot.db"

    start_ts = time.time()
    run_id = str(args.run_id).strip() if args.run_id else uuid4().hex
    sync_runner_metadata(
        runner_json,
        run_id=run_id,
        data_dir=data_dir,
        log_file=runner_log,
        cmd=[py, str(Path(__file__).resolve()), *sys.argv[1:]],
    )
    deadline = start_ts + max(0.001, args.duration_hours) * 3600.0
    step_timeout_sec = max(0.0, float(args.step_timeout_minutes)) * 60.0
    step_timeout = step_timeout_sec if step_timeout_sec > 0 else None
    cycle = 0
    failures = 0
    patterns = args.critical_pattern or ["_bad_freeze"]

    while time.time() < deadline:
        cycle += 1
        cycle_started = time.time()
        cycle_row: dict = {
            "run_id": run_id,
            "cycle": cycle,
            "started_at_epoch": cycle_started,
            "ok": True,
            "steps": [],
        }
        print(f"\n=== cycle {cycle} ===", flush=True)
        write_json(
            heartbeat_json,
            {"run_id": run_id, "cycle": cycle, "state": "started", "ts": time.time(), "failures": failures},
        )

        try:
            if cycle == 1 and args.reset_data_dir and data_dir.exists():
                for item in data_dir.iterdir():
                    if item.name == "autopilot":
                        continue
                    if item.is_dir():
                        import shutil

                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        with contextlib.suppress(Exception):
                            item.unlink(missing_ok=True)

            run_bad = (cycle == 1) or (args.bad_example_every > 0 and cycle % args.bad_example_every == 0)
            if run_bad:
                cmd_bad = [
                    py,
                    str(scripts_dir / "generate_bad_examples.py"),
                    "--input-dir",
                    args.trainee_dir,
                    "--output-dir",
                    args.trainee_bad_dir,
                    "--max-source",
                    str(args.max_source),
                ]
                run_step(
                    run_id=run_id,
                    cycle=cycle,
                    step_name="generate_bad_examples",
                    cmd=cmd_bad,
                    heartbeat_json=heartbeat_json,
                    cycle_row=cycle_row,
                    timeout_sec=step_timeout,
                )

            cmd_inc = [
                py,
                str(scripts_dir / "poc_incremental_pipeline.py"),
                "--base-dir",
                args.base_dir,
                "--recursive",
                "--task-id",
                args.task_id,
                "--task-name",
                args.task_name,
                "--site-id",
                args.site_id,
                "--embedder-backend",
                args.backend,
                "--data-dir",
                str(data_dir),
                "--score-scope",
                "unscored",
                "--gold-id",
                str(args.gold_id),
                "--labels-output",
                str(labels_path),
                "--eval-output",
                str(eval_path),
                "--output",
                str(summary_path),
            ]
            run_step(
                run_id=run_id,
                cycle=cycle,
                step_name="incremental_pipeline",
                cmd=cmd_inc,
                heartbeat_json=heartbeat_json,
                cycle_row=cycle_row,
                timeout_sec=step_timeout,
            )

            cmd_prefill = [
                py,
                str(scripts_dir / "prefill_critical_labels.py"),
                "--summary",
                str(summary_path),
                "--labels",
                str(labels_path),
            ]
            for pat in patterns:
                cmd_prefill.extend(["--critical-pattern", pat])
            run_step(
                run_id=run_id,
                cycle=cycle,
                step_name="prefill_labels",
                cmd=cmd_prefill,
                heartbeat_json=heartbeat_json,
                cycle_row=cycle_row,
                timeout_sec=step_timeout,
            )

            run_rescore = cycle == 1 or (args.rescore_every > 0 and (cycle % args.rescore_every == 0))
            if run_rescore:
                cmd_rescore = [
                    py,
                    str(scripts_dir / "rescore_existing_pairs.py"),
                    "--data-dir",
                    str(data_dir),
                    "--task-id",
                    args.task_id,
                    "--task-name",
                    args.task_name,
                    "--gold-id",
                    str(args.gold_id),
                    "--backend",
                    args.backend,
                    "--repeat",
                    str(max(1, args.rescore_repeat)),
                    "--output",
                    str(data_dir / "rescore_results.json"),
                ]
                run_step(
                    run_id=run_id,
                    cycle=cycle,
                    step_name="rescore_pairs",
                    cmd=cmd_rescore,
                    heartbeat_json=heartbeat_json,
                    cycle_row=cycle_row,
                    timeout_sec=step_timeout,
                )
            else:
                cycle_row["steps"].append({"name": "rescore_pairs", "skipped": True, "reason": "interval"})

            cmd_gate = [
                py,
                str(scripts_dir / "evaluate_poc.py"),
                "--db-path",
                str(db_path),
                "--task-id",
                args.task_id,
                "--labels",
                str(labels_path),
                "--fail-on-gate",
                "--output",
                str(gate_path),
            ]
            if args.gate_profile:
                cmd_gate.extend(["--gate-profile", args.gate_profile])
            if args.allow_profile_overrides:
                cmd_gate.append("--allow-profile-overrides")
            if args.critical_scoring_mode:
                cmd_gate.extend(["--critical-scoring-mode", args.critical_scoring_mode])
            cmd_gate.extend(["--critical-threshold", str(args.critical_threshold)])
            if args.critical_sweep_auto:
                cmd_gate.append("--critical-sweep-auto")
            if args.critical_sweep_values:
                cmd_gate.extend(["--critical-sweep-values", str(args.critical_sweep_values)])
            if args.critical_sweep_start is not None:
                cmd_gate.extend(["--critical-sweep-start", str(args.critical_sweep_start)])
            if args.critical_sweep_stop is not None:
                cmd_gate.extend(["--critical-sweep-stop", str(args.critical_sweep_stop)])
            if args.critical_sweep_step is not None:
                cmd_gate.extend(["--critical-sweep-step", str(args.critical_sweep_step)])
            if args.critical_sweep_scoring_mode:
                cmd_gate.extend(["--critical-sweep-scoring-mode", args.critical_sweep_scoring_mode])
            if args.max_critical_miss_rate is not None:
                cmd_gate.extend(["--max-critical-miss-rate", str(args.max_critical_miss_rate)])
            if args.max_critical_fpr is not None:
                cmd_gate.extend(["--max-critical-fpr", str(args.max_critical_fpr)])
            if args.max_critical_miss_ci95_high is not None:
                cmd_gate.extend(["--max-critical-miss-ci95-high", str(args.max_critical_miss_ci95_high)])
            if args.max_critical_fpr_ci95_high is not None:
                cmd_gate.extend(["--max-critical-fpr-ci95-high", str(args.max_critical_fpr_ci95_high)])
            if args.max_rescore_jitter is not None:
                cmd_gate.extend(["--max-rescore-jitter", str(args.max_rescore_jitter)])
            if args.max_dtw_p90 is not None:
                cmd_gate.extend(["--max-dtw-p90", str(args.max_dtw_p90)])
            if args.min_completed_jobs is not None:
                cmd_gate.extend(["--min-completed-jobs", str(args.min_completed_jobs)])
            if args.min_labels_total_jobs is not None:
                cmd_gate.extend(["--min-labels-total-jobs", str(args.min_labels_total_jobs)])
            if args.min_labeled_jobs is not None:
                cmd_gate.extend(["--min-labeled-jobs", str(args.min_labeled_jobs)])
            if args.min_critical_positives is not None:
                cmd_gate.extend(["--min-critical-positives", str(args.min_critical_positives)])
            if args.min_critical_negatives is not None:
                cmd_gate.extend(["--min-critical-negatives", str(args.min_critical_negatives)])
            if args.min_coverage_rate is not None:
                cmd_gate.extend(["--min-coverage-rate", str(args.min_coverage_rate)])
            if args.min_rescore_pairs is not None:
                cmd_gate.extend(["--min-rescore-pairs", str(args.min_rescore_pairs)])
            run_step(
                run_id=run_id,
                cycle=cycle,
                step_name="evaluate_gate",
                cmd=cmd_gate,
                heartbeat_json=heartbeat_json,
                cycle_row=cycle_row,
                timeout_sec=step_timeout,
            )

            run_backup = (not args.skip_backup) and (
                cycle == 1 or (args.backup_every > 0 and cycle % args.backup_every == 0)
            )
            if run_backup:
                backup_cmd = [
                    py,
                    str(scripts_dir / "backup_onprem.py"),
                    "--data-dir",
                    str(data_dir),
                    "--out-dir",
                    str(data_dir / "backups"),
                ]
                run_step(
                    run_id=run_id,
                    cycle=cycle,
                    step_name="backup",
                    cmd=backup_cmd,
                    heartbeat_json=heartbeat_json,
                    cycle_row=cycle_row,
                    timeout_sec=step_timeout,
                )
            else:
                cycle_row["steps"].append({"name": "backup", "skipped": True, "reason": "interval_or_disabled"})

            gate_payload = read_json(gate_path)
            cycle_row["gate_overall_pass"] = bool((gate_payload.get("gates") or {}).get("overall_pass"))
            cycle_row["completed_jobs"] = gate_payload.get("num_completed_jobs")
            cycle_row["critical_miss_rate"] = gate_payload.get("critical_miss_rate")
            cycle_row["critical_false_positive_rate"] = gate_payload.get("critical_false_positive_rate")
            cycle_row["rescore_jitter_max_delta"] = ((gate_payload.get("rescore_jitter") or {}).get("max_delta"))
            cycle_row["dtw_p90"] = ((gate_payload.get("dtw_normalized_cost_stats") or {}).get("p90"))
            cycle_row["coverage_rate"] = gate_payload.get("coverage_rate")
            cycle_row["labels_labeled_jobs"] = gate_payload.get("labels_labeled_jobs")
            cycle_row["critical_positives"] = gate_payload.get("critical_positives")
            cycle_row["critical_negatives"] = gate_payload.get("critical_negatives")
            cycle_row["critical_fpr_ci95_high"] = (
                ((gate_payload.get("critical_confidence") or {}).get("false_positive_rate") or {}).get("ci95") or {}
            ).get("high")
            cycle_row["duration_sec"] = round(time.time() - cycle_started, 2)

        except Exception as exc:
            cycle_row["ok"] = False
            cycle_row["exception"] = str(exc)
            cycle_row["traceback"] = traceback.format_exc()[-4000:]
            write_json(
                crash_json,
                {
                    "run_id": run_id,
                    "cycle": cycle,
                    "ts": time.time(),
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

        if not cycle_row["ok"]:
            failures += 1
        cycle_row["failure_count_total"] = failures

        append_jsonl(report_jsonl, cycle_row)
        write_json(latest_json, cycle_row)
        write_json(
            heartbeat_json,
            {
                "run_id": run_id,
                "cycle": cycle,
                "state": "cycle_done",
                "ts": time.time(),
                "ok": cycle_row["ok"],
                "failures": failures,
            },
        )

        remaining = deadline - time.time()
        if remaining <= 0:
            break
        sleep_sec = min(remaining, max(5.0, args.interval_minutes * 60.0))
        print(f"cycle {cycle} done ok={cycle_row['ok']} sleep_sec={int(sleep_sec)}", flush=True)
        time.sleep(sleep_sec)

    final = {
        "run_id": run_id,
        "task_id": args.task_id,
        "started_at_epoch": start_ts,
        "ended_at_epoch": time.time(),
        "cycles": cycle,
        "failures": failures,
        "report_jsonl": str(report_jsonl),
        "latest_cycle": str(latest_json),
        "gate_report": str(gate_path),
    }
    write_json(logs_dir / "final_summary.json", final)
    write_json(
        heartbeat_json,
        {"run_id": run_id, "state": "finished", "ts": time.time(), "cycles": cycle, "failures": failures},
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
