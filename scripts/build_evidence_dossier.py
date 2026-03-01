from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from sopilot.eval.integrity import verify_payload_hash


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _extract_candidate_rows(split_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results = split_report.get("results", {}) or {}
    if not isinstance(results, dict):
        return rows
    for mode, mode_payload in results.items():
        if not isinstance(mode_payload, dict):
            continue
        for split_name in ("full", "dev", "test", "challenge"):
            report = mode_payload.get(split_name)
            if not isinstance(report, dict):
                continue
            ci = report.get("critical_confidence") or {}
            miss_ci = ((ci.get("miss_rate") or {}).get("ci95") or {}).get("high")
            fpr_ci = ((ci.get("false_positive_rate") or {}).get("ci95") or {}).get("high")
            rows.append(
                {
                    "mode": mode,
                    "split": split_name,
                    "miss_rate": report.get("critical_miss_rate"),
                    "fpr": report.get("critical_false_positive_rate"),
                    "miss_ci95_high": miss_ci,
                    "fpr_ci95_high": fpr_ci,
                    "positives": report.get("critical_positives"),
                    "negatives": report.get("critical_negatives"),
                    "overall_pass": ((report.get("gates") or {}).get("overall_pass")),
                }
            )
    return rows


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _split_failures(split_report: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    results = split_report.get("results", {}) or {}
    if not isinstance(results, dict):
        return out
    for mode, mode_payload in results.items():
        if not isinstance(mode_payload, dict):
            continue
        for split_name, report in mode_payload.items():
            if not isinstance(report, dict):
                continue
            gates = report.get("gates") or {}
            if bool(gates.get("overall_pass")):
                continue
            failed_checks: list[dict[str, Any]] = []
            for check in gates.get("checks", []) or []:
                if not isinstance(check, dict):
                    continue
                if bool(check.get("enabled")) and not bool(check.get("pass")):
                    failed_checks.append(
                        {
                            "name": check.get("name"),
                            "value": check.get("value"),
                            "threshold": check.get("threshold"),
                            "reason": check.get("reason"),
                        }
                    )
            out.append(
                {
                    "mode": mode,
                    "split": split_name,
                    "failed_checks": failed_checks,
                    "critical_positives": report.get("critical_positives"),
                    "critical_negatives": report.get("critical_negatives"),
                }
            )
    return out


def _build_readiness(
    *,
    split_report: dict[str, Any],
    labeling_plan: dict[str, Any] | None,
    loso_report: dict[str, Any] | None,
    hash_results: list[dict[str, Any]],
) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    notes: list[str] = []

    split_failures = _split_failures(split_report)
    for failure in split_failures:
        split_name = str(failure.get("split") or "")
        if split_name in {"dev", "test", "challenge"}:
            blockers.append(
                {
                    "type": "split_gate_fail",
                    "mode": failure.get("mode"),
                    "split": split_name,
                    "failed_checks": failure.get("failed_checks"),
                }
            )

    if labeling_plan is not None:
        deficits = labeling_plan.get("deficits") or {}
        for split_name in ("test", "challenge"):
            row = deficits.get(split_name) or {}
            pos_need = _safe_int(row.get("positives_needed")) or 0
            neg_need = _safe_int(row.get("negatives_needed")) or 0
            if pos_need > 0 or neg_need > 0:
                blockers.append(
                    {
                        "type": "evidence_deficit",
                        "split": split_name,
                        "positives_needed": pos_need,
                        "negatives_needed": neg_need,
                        "suggested_labels_to_add": row.get("suggested_labels_to_add"),
                    }
                )
    else:
        notes.append("labeling_plan_missing")

    if loso_report is not None:
        rows = loso_report.get("rows") or []
        ok_rows = [row for row in rows if isinstance(row, dict) and str(row.get("status")) == "ok"]
        if len(ok_rows) < 2:
            blockers.append(
                {
                    "type": "generalization_evidence_weak",
                    "reason": "insufficient_ok_holdouts",
                    "ok_rows": len(ok_rows),
                    "rows_total": len(rows),
                    "holdout_axis_used": loso_report.get("holdout_axis_used"),
                }
            )
        elif any(not bool(row.get("overall_pass")) for row in ok_rows):
            blockers.append(
                {
                    "type": "generalization_gate_fail",
                    "reason": "one_or_more_holdouts_failed",
                    "failed_holdouts": [
                        {
                            "axis": row.get("axis"),
                            "group": row.get("holdout_group"),
                            "constraint_violation": row.get("constraint_violation"),
                        }
                        for row in ok_rows
                        if not bool(row.get("overall_pass"))
                    ],
                }
            )
    else:
        notes.append("loso_report_missing")

    if hash_results:
        if not all(bool(row.get("verified")) for row in hash_results):
            blockers.append(
                {
                    "type": "artifact_hash_mismatch",
                    "files": [{"path": row.get("path"), "verified": row.get("verified")} for row in hash_results],
                }
            )
    else:
        notes.append("hash_verification_missing_or_not_requested")

    status = "ready_for_partner_review" if not blockers else "not_ready"
    score = max(0, 100 - min(100, 15 * len(blockers)))
    return {
        "status": status,
        "score_0_to_100": int(score),
        "num_blockers": int(len(blockers)),
        "blockers": blockers,
        "notes": notes,
    }


def _summary_markdown(split_report: dict[str, Any], *, readiness: dict[str, Any] | None = None) -> str:
    rows = _extract_candidate_rows(split_report)
    lines = [
        "# Evidence Summary",
        "",
        f"- task_id: `{split_report.get('task_id')}`",
        f"- split_strategy: `{split_report.get('split_strategy')}`",
        f"- gate_profile: `{split_report.get('gate_profile')}`",
    ]
    if isinstance(readiness, dict):
        lines.extend(
            [
                f"- readiness_status: `{readiness.get('status')}`",
                f"- readiness_score: `{readiness.get('score_0_to_100')}`",
                f"- readiness_blockers: `{readiness.get('num_blockers')}`",
            ]
        )
    lines.extend(
        [
            "",
            "|mode|split|miss|fpr|miss_ci95_hi|fpr_ci95_hi|pos|neg|pass|",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "|"
            + "|".join(
                [
                    str(row.get("mode")),
                    str(row.get("split")),
                    str(row.get("miss_rate")),
                    str(row.get("fpr")),
                    str(row.get("miss_ci95_high")),
                    str(row.get("fpr_ci95_high")),
                    str(row.get("positives")),
                    str(row.get("negatives")),
                    str(row.get("overall_pass")),
                ]
            )
            + "|"
        )
    if isinstance(readiness, dict):
        lines.extend(
            [
                "",
                "## Readiness",
                "",
            ]
        )
        blockers = readiness.get("blockers") or []
        if blockers:
            for blocker in blockers:
                btype = blocker.get("type")
                split = blocker.get("split")
                if split:
                    lines.append(f"- {btype}: split={split}")
                else:
                    lines.append(f"- {btype}")
        else:
            lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _safe_run(command: list[str]) -> tuple[int, str]:
    try:
        cp = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        text = (cp.stdout or "").strip()
        if not text:
            text = (cp.stderr or "").strip()
        return int(cp.returncode), text
    except Exception as exc:
        return 1, str(exc)


def _verify_hash(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    extra: set[str] = set()
    if "policy_id" in payload and "policy" in str(payload.get("version") or ""):
        extra.add("policy_id")
    ok = verify_payload_hash(payload, exclude_extra_keys=extra)
    return {
        "path": str(path),
        "verified": bool(ok),
        "artifact_hash_sha256": payload.get("artifact_hash_sha256"),
        "artifact_hash_method": payload.get("artifact_hash_method"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build partner-facing evidence dossier folder.")
    parser.add_argument("--split-report", required=True, help="split_evaluation_report.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--gate-report", action="append", default=None, help="Repeatable")
    parser.add_argument("--shadow-report", default=None)
    parser.add_argument("--fp-breakdown", default=None)
    parser.add_argument("--loso-report", default=None)
    parser.add_argument("--labeling-plan", default=None)
    parser.add_argument("--commands-file", default=None, help="Optional text file with reproduction commands")
    parser.add_argument("--capture-env", action="store_true", help="Capture pip freeze into repro/environment.txt")
    parser.add_argument("--verify-artifacts", action="store_true", help="Verify policy/manifest hashes into protocol/hash_verification.json")
    args = parser.parse_args()

    split_report_path = Path(args.split_report).resolve()
    split_report = json.loads(split_report_path.read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir).resolve()
    protocol_dir = out_dir / "protocol"
    results_dir = out_dir / "results"
    analysis_dir = out_dir / "analysis"
    repro_dir = out_dir / "repro"
    for d in (protocol_dir, results_dir, analysis_dir, repro_dir):
        d.mkdir(parents=True, exist_ok=True)

    _copy(split_report_path, results_dir / "split_evaluation_report.json")

    split_manifest = split_report.get("split_manifest")
    if isinstance(split_manifest, dict):
        (protocol_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.policy:
        _copy(Path(args.policy).resolve(), protocol_dir / "critical_policy.json")
    if args.gate_report:
        for idx, path in enumerate(args.gate_report, start=1):
            src = Path(path).resolve()
            _copy(src, results_dir / f"gate_report_{idx}.json")
    if args.shadow_report:
        _copy(Path(args.shadow_report).resolve(), analysis_dir / "shadow_report.json")
    if args.fp_breakdown:
        _copy(Path(args.fp_breakdown).resolve(), analysis_dir / "fp_breakdown.json")
    if args.loso_report:
        _copy(Path(args.loso_report).resolve(), results_dir / "loso_table.json")
    if args.labeling_plan:
        _copy(Path(args.labeling_plan).resolve(), analysis_dir / "labeling_plan.json")
    if args.commands_file:
        _copy(Path(args.commands_file).resolve(), repro_dir / "commands.txt")
    if args.capture_env:
        rc, text = _safe_run(["python", "-m", "pip", "freeze"])
        env_text = f"return_code={rc}\n\n{text}\n"
        (repro_dir / "environment.txt").write_text(env_text, encoding="utf-8")

    hash_results: list[dict[str, Any]] = []
    if args.verify_artifacts:
        split_manifest_path = protocol_dir / "split_manifest.json"
        policy_path = protocol_dir / "critical_policy.json"
        if split_manifest_path.exists():
            hash_results.append(_verify_hash(split_manifest_path))
        if policy_path.exists():
            hash_results.append(_verify_hash(policy_path))
        (protocol_dir / "hash_verification.json").write_text(
            json.dumps(
                {
                    "files": hash_results,
                    "all_verified": all(bool(row.get("verified")) for row in hash_results),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    labeling_plan_payload: dict[str, Any] | None = None
    if args.labeling_plan:
        labeling_plan_payload = json.loads(Path(args.labeling_plan).resolve().read_text(encoding="utf-8"))
    loso_payload: dict[str, Any] | None = None
    if args.loso_report:
        loso_payload = json.loads(Path(args.loso_report).resolve().read_text(encoding="utf-8"))

    readiness = _build_readiness(
        split_report=split_report,
        labeling_plan=labeling_plan_payload,
        loso_report=loso_payload,
        hash_results=hash_results,
    )
    (results_dir / "readiness.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (results_dir / "summary.md").write_text(_summary_markdown(split_report, readiness=readiness), encoding="utf-8")

    manifest = {
        "split_report": str(results_dir / "split_evaluation_report.json"),
        "summary": str(results_dir / "summary.md"),
        "readiness": str(results_dir / "readiness.json"),
        "split_manifest": str(protocol_dir / "split_manifest.json"),
        "policy": str(protocol_dir / "critical_policy.json") if args.policy else None,
        "shadow_report": str(analysis_dir / "shadow_report.json") if args.shadow_report else None,
        "fp_breakdown": str(analysis_dir / "fp_breakdown.json") if args.fp_breakdown else None,
        "loso_report": str(results_dir / "loso_table.json") if args.loso_report else None,
        "labeling_plan": str(analysis_dir / "labeling_plan.json") if args.labeling_plan else None,
        "hash_verification": str(protocol_dir / "hash_verification.json") if args.verify_artifacts else None,
        "environment": str(repro_dir / "environment.txt") if args.capture_env else None,
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    print(manifest_text)
    (out_dir / "dossier_manifest.json").write_text(manifest_text, encoding="utf-8")


if __name__ == "__main__":
    main()
