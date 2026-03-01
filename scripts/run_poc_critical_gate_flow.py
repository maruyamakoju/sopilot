from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sopilot.eval.gates import available_gate_profiles
from sopilot.eval.harness import available_critical_scoring_modes

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _list_videos(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*")):
        if not item.is_file() or item.suffix.lower() not in VIDEO_EXTS:
            continue
        stat = item.stat()
        rows.append(
            {
                "path": str(item.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
            }
        )
    return rows


def run(cmd: list[str], *, executed_commands: list[dict[str, Any]]) -> None:
    print("$", " ".join(cmd))
    executed_commands.append({"ts": now_iso(), "cmd": list(cmd)})
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PoC critical-gate flow end-to-end.")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--task-name", default="PoC Task")
    parser.add_argument("--run-id", default=None, help="Optional run identifier used in failure capsules and manifests.")
    parser.add_argument("--base-dir", required=True, help="Dataset root with gold/trainee folders")
    parser.add_argument("--trainee-dir", required=True, help="Source trainee folder for bad-example generation")
    parser.add_argument("--trainee-bad-dir", required=True, help="Output bad trainee folder")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--site-id", default="site-a")
    parser.add_argument("--gold-id", type=int, default=2)
    parser.add_argument("--backend", choices=["vjepa2", "color-motion"], default="color-motion")
    parser.add_argument(
        "--vjepa2-pooling",
        choices=["mean_tokens", "first_token", "flatten"],
        default="mean_tokens",
        help="Pooling strategy when backend=vjepa2.",
    )
    parser.add_argument("--max-source", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0, help="Run seed used for reproducible data selection.")
    parser.add_argument("--skip-generate-bad", action="store_true")
    parser.add_argument("--skip-prefill-labels", action="store_true")
    parser.add_argument(
        "--disable-embedder-fallback",
        action="store_true",
        help="Disable embedder fallback for strict primary-only benchmarking.",
    )
    parser.add_argument(
        "--run-manifest-out",
        default=None,
        help="Path to write run manifest JSON (default: <data-dir>/run_manifest.json).",
    )
    parser.add_argument("--reset-data-dir", action="store_true", help="Delete data-dir before running")
    parser.add_argument(
        "--critical-pattern",
        action="append",
        default=None,
        help="Pattern used for prefill critical_expected (default: _bad_freeze)",
    )
    parser.add_argument(
        "--no-fail-on-gate",
        dest="fail_on_gate",
        action="store_false",
        help="Do not return non-zero when quality gate fails (useful for benchmarking).",
    )
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
    parser.set_defaults(fail_on_gate=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.reset_data_dir and data_dir.exists():
        shutil.rmtree(data_dir)
    labels_path = data_dir / "labels_template.json"
    eval_path = data_dir / "eval_report.json"
    summary_path = data_dir / "local_pipeline_summary.json"
    rescore_path = data_dir / "rescore_results.json"
    gate_path = data_dir / "gate_report.json"
    db_path = data_dir / "sopilot.db"
    bad_selection_path = data_dir / "generate_bad_selection.json"
    run_manifest_path = (
        Path(args.run_manifest_out).resolve()
        if args.run_manifest_out
        else (data_dir / "run_manifest.json").resolve()
    )

    py = sys.executable
    executed_commands: list[dict[str, Any]] = []
    run_manifest: dict[str, Any] = {
        "started_at_utc": now_iso(),
        "status": "started",
        "task_id": args.task_id,
        "task_name": args.task_name,
        "run_id": args.run_id or args.task_id,
        "seed": int(args.seed),
        "backend": args.backend,
        "vjepa2_pooling": args.vjepa2_pooling,
        "disable_embedder_fallback": bool(args.disable_embedder_fallback),
        "base_dir": str(Path(args.base_dir).resolve()),
        "trainee_dir": str(Path(args.trainee_dir).resolve()),
        "trainee_bad_dir": str(Path(args.trainee_bad_dir).resolve()),
        "input_fingerprint": {
            "base_dir_videos": _list_videos(Path(args.base_dir).resolve()),
            "trainee_dir_videos": _list_videos(Path(args.trainee_dir).resolve()),
            "trainee_bad_dir_videos_before_run": _list_videos(Path(args.trainee_bad_dir).resolve()),
        },
        "commands": [],
    }
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if not args.skip_generate_bad:
            run(
                [
                    py,
                    "scripts/generate_bad_examples.py",
                    "--input-dir",
                    args.trainee_dir,
                    "--output-dir",
                    args.trainee_bad_dir,
                    "--max-source",
                    str(args.max_source),
                    "--seed",
                    str(int(args.seed)),
                    "--selection-output",
                    str(bad_selection_path),
                ],
                executed_commands=executed_commands,
            )

        local_cmd = [
            py,
            "scripts/poc_local_pipeline.py",
            "--base-dir",
            args.base_dir,
            "--recursive",
            "--task-id",
            args.task_id,
            "--task-name",
            args.task_name,
            "--run-id",
            args.run_id or args.task_id,
            "--site-id",
            args.site_id,
            "--embedder-backend",
            args.backend,
            "--vjepa2-pooling",
            args.vjepa2_pooling,
            "--seed",
            str(int(args.seed)),
            "--labels-output",
            str(labels_path),
            "--eval-output",
            str(eval_path),
            "--output",
            str(summary_path),
            "--data-dir",
            str(data_dir),
        ]
        if args.disable_embedder_fallback:
            local_cmd.append("--disable-embedder-fallback")
        run(local_cmd, executed_commands=executed_commands)

        if not args.skip_prefill_labels:
            cmd = [
                py,
                "scripts/prefill_critical_labels.py",
                "--summary",
                str(summary_path),
                "--labels",
                str(labels_path),
            ]
            for pattern in (args.critical_pattern or []):
                cmd.extend(["--critical-pattern", pattern])
            run(cmd, executed_commands=executed_commands)

        run(
            [
                py,
                "scripts/rescore_existing_pairs.py",
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
                "1",
                "--output",
                str(rescore_path),
            ],
            executed_commands=executed_commands,
        )

        eval_cmd = [
            py,
            "scripts/evaluate_poc.py",
            "--db-path",
            str(db_path),
            "--task-id",
            args.task_id,
            "--labels",
            str(labels_path),
            "--output",
            str(gate_path),
        ]
        if args.gate_profile:
            eval_cmd.extend(["--gate-profile", args.gate_profile])
        if args.allow_profile_overrides:
            eval_cmd.append("--allow-profile-overrides")
        if args.critical_scoring_mode:
            eval_cmd.extend(["--critical-scoring-mode", args.critical_scoring_mode])
        eval_cmd.extend(["--critical-threshold", str(args.critical_threshold)])
        if args.critical_sweep_auto:
            eval_cmd.append("--critical-sweep-auto")
        if args.critical_sweep_values:
            eval_cmd.extend(["--critical-sweep-values", str(args.critical_sweep_values)])
        if args.critical_sweep_start is not None:
            eval_cmd.extend(["--critical-sweep-start", str(args.critical_sweep_start)])
        if args.critical_sweep_stop is not None:
            eval_cmd.extend(["--critical-sweep-stop", str(args.critical_sweep_stop)])
        if args.critical_sweep_step is not None:
            eval_cmd.extend(["--critical-sweep-step", str(args.critical_sweep_step)])
        if args.critical_sweep_scoring_mode:
            eval_cmd.extend(["--critical-sweep-scoring-mode", args.critical_sweep_scoring_mode])
        if args.max_critical_miss_rate is not None:
            eval_cmd.extend(["--max-critical-miss-rate", str(args.max_critical_miss_rate)])
        if args.max_critical_fpr is not None:
            eval_cmd.extend(["--max-critical-fpr", str(args.max_critical_fpr)])
        if args.max_critical_miss_ci95_high is not None:
            eval_cmd.extend(["--max-critical-miss-ci95-high", str(args.max_critical_miss_ci95_high)])
        if args.max_critical_fpr_ci95_high is not None:
            eval_cmd.extend(["--max-critical-fpr-ci95-high", str(args.max_critical_fpr_ci95_high)])
        if args.max_rescore_jitter is not None:
            eval_cmd.extend(["--max-rescore-jitter", str(args.max_rescore_jitter)])
        if args.max_dtw_p90 is not None:
            eval_cmd.extend(["--max-dtw-p90", str(args.max_dtw_p90)])
        if args.min_completed_jobs is not None:
            eval_cmd.extend(["--min-completed-jobs", str(args.min_completed_jobs)])
        if args.min_labels_total_jobs is not None:
            eval_cmd.extend(["--min-labels-total-jobs", str(args.min_labels_total_jobs)])
        if args.min_labeled_jobs is not None:
            eval_cmd.extend(["--min-labeled-jobs", str(args.min_labeled_jobs)])
        if args.min_critical_positives is not None:
            eval_cmd.extend(["--min-critical-positives", str(args.min_critical_positives)])
        if args.min_critical_negatives is not None:
            eval_cmd.extend(["--min-critical-negatives", str(args.min_critical_negatives)])
        if args.min_coverage_rate is not None:
            eval_cmd.extend(["--min-coverage-rate", str(args.min_coverage_rate)])
        if args.min_rescore_pairs is not None:
            eval_cmd.extend(["--min-rescore-pairs", str(args.min_rescore_pairs)])
        if args.fail_on_gate:
            eval_cmd.append("--fail-on-gate")
        run(eval_cmd, executed_commands=executed_commands)
        run_manifest["status"] = "completed"
        print(f"done gate_report={gate_path}")
    except Exception as exc:
        run_manifest["status"] = "failed"
        run_manifest["error"] = str(exc)
        raise
    finally:
        run_manifest["completed_at_utc"] = now_iso()
        run_manifest["commands"] = executed_commands
        run_manifest["outputs"] = {
            "labels": str(labels_path.resolve()),
            "eval": str(eval_path.resolve()),
            "summary": str(summary_path.resolve()),
            "rescore": str(rescore_path.resolve()),
            "gate_report": str(gate_path.resolve()),
            "bad_selection": str(bad_selection_path.resolve()),
        }
        run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"run_manifest={run_manifest_path}")


if __name__ == "__main__":
    main()
