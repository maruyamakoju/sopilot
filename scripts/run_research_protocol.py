from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _iso_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _cmd_text(cmd: list[str]) -> str:
    return " ".join([f'"{part}"' if " " in str(part) else str(part) for part in cmd])


def _run_step(*, name: str, cmd: list[str], logs_dir: Path, dry_run: bool) -> dict[str, Any]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{name}.stdout.txt"
    stderr_path = logs_dir / f"{name}.stderr.txt"

    started_at = _iso_now()
    if dry_run:
        payload = {
            "name": name,
            "command": cmd,
            "command_text": _cmd_text(cmd),
            "started_at": started_at,
            "finished_at": _iso_now(),
            "return_code": 0,
            "dry_run": True,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        stdout_path.write_text("dry-run\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return payload

    cp = subprocess.run(cmd, capture_output=True, text=True, check=False)
    stdout_path.write_text(cp.stdout or "", encoding="utf-8")
    stderr_path.write_text(cp.stderr or "", encoding="utf-8")
    payload = {
        "name": name,
        "command": cmd,
        "command_text": _cmd_text(cmd),
        "started_at": started_at,
        "finished_at": _iso_now(),
        "return_code": int(cp.returncode),
        "dry_run": False,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    if cp.returncode != 0:
        raise SystemExit(
            json.dumps(
                {
                    "failed_step": name,
                    "return_code": int(cp.returncode),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return payload


def _append_task_id(cmd: list[str], task_id: str | None) -> None:
    if task_id:
        cmd.extend(["--task-id", str(task_id)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research protocol end-to-end with fixed artifacts.")
    parser.add_argument("--db-path", "--db", dest="db_path", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--gate-profile", default="research_v2")
    parser.add_argument("--split-strategy", default="group_trainee")
    parser.add_argument("--seed", type=int, default=20260223)
    parser.add_argument("--critical-threshold", type=float, default=0.5)
    parser.add_argument("--use-ci-constraints", action="store_true")
    parser.add_argument("--git-commit", default="unknown")
    parser.add_argument("--baseline-mode", default="guarded_binary_v1")

    parser.add_argument("--holdout-axis", default="site")
    parser.add_argument("--auto-fallback-axis", action="store_true")
    parser.add_argument("--fallback-axis-order", default="site,gold,trainee")

    parser.add_argument("--target-miss-ci95-high", type=float, default=0.10)
    parser.add_argument("--target-fpr-ci95-high", type=float, default=0.20)
    parser.add_argument("--target-positive-hit-prob", type=float, default=0.90)
    parser.add_argument("--challenge-max-per-site", type=int, default=40)

    parser.add_argument("--skip-shadow", action="store_true")
    parser.add_argument("--skip-loso", action="store_true")
    parser.add_argument("--capture-env", action="store_true")
    parser.add_argument("--verify-artifacts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    split_raw_dir = out_dir / "split_eval_raw"
    split_locked_dir = out_dir / "split_eval_locked"
    policy_dir = out_dir / "policy"
    errors_dir = out_dir / "errors"
    loso_dir = out_dir / "loso"
    labeling_dir = out_dir / "labeling_plan"
    dossier_dir = out_dir / "evidence_dossier"
    logs_dir = out_dir / "logs"
    for d in (split_raw_dir, split_locked_dir, policy_dir, errors_dir, loso_dir, labeling_dir, dossier_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    split_manifest = split_raw_dir / "split_manifest.json"
    split_report_locked = split_locked_dir / "split_evaluation_report.json"
    policy_path = policy_dir / "critical_policy_devfit.json"
    policy_report_path = policy_dir / "critical_policy_devfit_report.json"
    gate_report_path = out_dir / "gate_report_policy_locked.json"
    shadow_path = policy_dir / "shadow_baseline_vs_candidate.json"
    fp_breakdown_path = errors_dir / "fp_breakdown.json"
    loso_report_path = loso_dir / "loso_table.json"
    labeling_plan_path = labeling_dir / "ci_labeling_plan.json"
    commands_path = out_dir / "repro_commands.txt"
    protocol_manifest_path = out_dir / "protocol_run_manifest.json"

    steps: list[tuple[str, list[str]]] = []

    split_cmd = [
        python,
        "scripts/evaluate_split_profiles.py",
        "--db-path",
        str(Path(args.db_path).resolve()),
        "--labels",
        str(Path(args.labels).resolve()),
        "--gate-profile",
        str(args.gate_profile),
        "--split-strategy",
        str(args.split_strategy),
        "--seed",
        str(int(args.seed)),
        "--output-dir",
        str(split_raw_dir),
    ]
    _append_task_id(split_cmd, args.task_id)
    steps.append(("01_split_eval_raw", split_cmd))

    fit_cmd = [
        python,
        "scripts/fit_critical_policy.py",
        "--db-path",
        str(Path(args.db_path).resolve()),
        "--labels",
        str(Path(args.labels).resolve()),
        "--split-manifest",
        str(split_manifest),
        "--gate-profile",
        str(args.gate_profile),
        "--critical-threshold",
        str(float(args.critical_threshold)),
        "--output-policy",
        str(policy_path),
        "--output-report",
        str(policy_report_path),
        "--git-commit",
        str(args.git_commit),
    ]
    _append_task_id(fit_cmd, args.task_id)
    if bool(args.use_ci_constraints):
        fit_cmd.append("--use-ci-constraints")
    steps.append(("02_fit_policy", fit_cmd))

    split_locked_cmd = [
        python,
        "scripts/evaluate_split_profiles.py",
        "--db-path",
        str(Path(args.db_path).resolve()),
        "--labels",
        str(Path(args.labels).resolve()),
        "--gate-profile",
        str(args.gate_profile),
        "--critical-policy",
        str(policy_path),
        "--split-manifest",
        str(split_manifest),
        "--output-dir",
        str(split_locked_dir),
    ]
    _append_task_id(split_locked_cmd, args.task_id)
    steps.append(("03_split_eval_locked", split_locked_cmd))

    poc_cmd = [
        python,
        "scripts/evaluate_poc.py",
        "--db-path",
        str(Path(args.db_path).resolve()),
        "--labels",
        str(Path(args.labels).resolve()),
        "--gate-profile",
        str(args.gate_profile),
        "--critical-policy",
        str(policy_path),
        "--output",
        str(gate_report_path),
    ]
    _append_task_id(poc_cmd, args.task_id)
    steps.append(("04_eval_poc", poc_cmd))

    if not bool(args.skip_shadow):
        shadow_cmd = [
            python,
            "scripts/evaluate_shadow_candidate.py",
            "--db-path",
            str(Path(args.db_path).resolve()),
            "--labels",
            str(Path(args.labels).resolve()),
            "--baseline-mode",
            str(args.baseline_mode),
            "--candidate-policy",
            str(policy_path),
            "--output",
            str(shadow_path),
        ]
        _append_task_id(shadow_cmd, args.task_id)
        steps.append(("05_shadow_eval", shadow_cmd))

    errors_cmd = [
        python,
        "scripts/extract_error_cases.py",
        "--db-path",
        str(Path(args.db_path).resolve()),
        "--labels",
        str(Path(args.labels).resolve()),
        "--critical-scoring-mode",
        "guarded_binary_v2",
        "--critical-threshold",
        str(float(args.critical_threshold)),
        "--critical-policy",
        str(policy_path),
        "--output-dir",
        str(errors_dir),
    ]
    _append_task_id(errors_cmd, args.task_id)
    steps.append(("06_extract_errors", errors_cmd))

    if not bool(args.skip_loso):
        loso_cmd = [
            python,
            "scripts/evaluate_loso_sweep.py",
            "--db",
            str(Path(args.db_path).resolve()),
            "--labels",
            str(Path(args.labels).resolve()),
            "--profile",
            str(args.gate_profile),
            "--scoring-mode",
            "guarded_binary_v2",
            "--holdout-axis",
            str(args.holdout_axis),
            "--fallback-axis-order",
            str(args.fallback_axis_order),
            "--output-dir",
            str(loso_dir),
        ]
        _append_task_id(loso_cmd, args.task_id)
        if bool(args.auto_fallback_axis):
            loso_cmd.append("--auto-fallback-axis")
        if bool(args.use_ci_constraints):
            loso_cmd.append("--use-ci-constraints")
        steps.append(("07_generalization_sweep", loso_cmd))

    labeling_cmd = [
        python,
        "scripts/plan_labeling_evidence.py",
        "--db",
        str(Path(args.db_path).resolve()),
        "--labels",
        str(Path(args.labels).resolve()),
        "--split-report",
        str(split_report_locked),
        "--target-miss-ci95-high",
        str(float(args.target_miss_ci95_high)),
        "--target-fpr-ci95-high",
        str(float(args.target_fpr_ci95_high)),
        "--target-positive-hit-prob",
        str(float(args.target_positive_hit_prob)),
        "--challenge-max-per-site",
        str(int(args.challenge_max_per_site)),
        "--output-dir",
        str(labeling_dir),
    ]
    _append_task_id(labeling_cmd, args.task_id)
    steps.append(("08_plan_labeling", labeling_cmd))

    dossier_cmd = [
        python,
        "scripts/build_evidence_dossier.py",
        "--split-report",
        str(split_report_locked),
        "--policy",
        str(policy_path),
        "--gate-report",
        str(gate_report_path),
        "--fp-breakdown",
        str(fp_breakdown_path),
        "--labeling-plan",
        str(labeling_plan_path),
        "--commands-file",
        str(commands_path),
        "--out-dir",
        str(dossier_dir),
    ]
    if not bool(args.skip_shadow):
        dossier_cmd.extend(["--shadow-report", str(shadow_path)])
    if not bool(args.skip_loso):
        dossier_cmd.extend(["--loso-report", str(loso_report_path)])
    if bool(args.capture_env):
        dossier_cmd.append("--capture-env")
    if bool(args.verify_artifacts):
        dossier_cmd.append("--verify-artifacts")
    steps.append(("09_build_dossier", dossier_cmd))

    commands_path.write_text(
        "\n".join([_cmd_text(step_cmd) for _, step_cmd in steps]) + "\n",
        encoding="utf-8",
    )

    executed: list[dict[str, Any]] = []
    started_at = _iso_now()
    for step_name, step_cmd in steps:
        executed.append(_run_step(name=step_name, cmd=step_cmd, logs_dir=logs_dir, dry_run=bool(args.dry_run)))

    payload = {
        "started_at": started_at,
        "finished_at": _iso_now(),
        "dry_run": bool(args.dry_run),
        "task_id": args.task_id,
        "db_path": str(Path(args.db_path).resolve()),
        "labels_path": str(Path(args.labels).resolve()),
        "out_dir": str(out_dir),
        "artifacts": {
            "split_manifest": str(split_manifest),
            "policy": str(policy_path),
            "split_report_locked": str(split_report_locked),
            "gate_report": str(gate_report_path),
            "shadow_report": (str(shadow_path) if not bool(args.skip_shadow) else None),
            "fp_breakdown": str(fp_breakdown_path),
            "loso_report": (str(loso_report_path) if not bool(args.skip_loso) else None),
            "labeling_plan": str(labeling_plan_path),
            "dossier_manifest": str(dossier_dir / "dossier_manifest.json"),
            "readiness": str(dossier_dir / "results" / "readiness.json"),
            "commands_file": str(commands_path),
        },
        "steps": executed,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    protocol_manifest_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
