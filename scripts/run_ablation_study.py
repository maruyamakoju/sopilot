"""run_ablation_study.py — Ablation study comparing three classification systems.

Compares three systems on the scored data already in the DB:
  1. Random baseline   : pass/fail randomly at 50% (seed=42)
  2. Score threshold   : use stored system score >= pass_threshold
  3. DTW cost only     : normalized_cost <= 0.05 → pass

All metrics are computed against score-based ground truth
(score >= pass_threshold = pass), since human labels are not yet available.

Usage
-----
    python scripts/run_ablation_study.py \\
        --db data_release_baseline/sopilot.db \\
        --task-id filter_change \\
        --pass-threshold 70 \\
        --output-dir artifacts/ablation/

Dependencies: stdlib + numpy only (no FastAPI, no torch).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_DIM = "\033[2m"


def _c(text: str, *codes: str) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + _RESET


# ---------------------------------------------------------------------------
# Database
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
# Data loading — one representative record per trainee video
# ---------------------------------------------------------------------------


@dataclass
class TraineeRecord:
    """Aggregated record representing one unique trainee video."""
    trainee_video_id: int
    score: float               # representative score (average across gold references)
    dtw_cost: float | None     # representative DTW cost (average; None if unavailable)
    file_path: str | None
    operator_id: str | None
    critical_flag_count: int   # total critical deviations across all jobs for this trainee


def _load_trainee_records(
    conn: sqlite3.Connection,
    *,
    task_id: str | None,
) -> list[TraineeRecord]:
    """Load completed score jobs for non-gold videos, collapsed to one record per trainee.

    When a trainee is scored against multiple gold references, the scores and
    DTW costs are averaged so that each trainee contributes exactly one data
    point to the ablation — matching the 24-video denominator shown in the
    eval_report.json summary.
    """
    has_orig_filename = _has_column(conn, "videos", "original_filename")
    filename_expr = (
        "tv.original_filename AS original_filename,"
        if has_orig_filename
        else "NULL AS original_filename,"
    )

    query = f"""
        SELECT
            sj.id                                   AS job_id,
            sj.gold_video_id,
            sj.trainee_video_id,
            sj.score_json,
            tv.operator_id_hash                     AS operator_id,
            tv.file_path                            AS file_path,
            {filename_expr}
            tv.is_gold                              AS trainee_is_gold
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

    # Group by trainee_video_id
    from collections import defaultdict
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        item = dict(row)
        raw_json = item.pop("score_json", None)
        parsed: dict[str, Any] = {}
        if raw_json:
            try:
                parsed = json.loads(raw_json)
            except Exception:
                pass
        item["_parsed"] = parsed
        groups[item["trainee_video_id"]].append(item)

    records: list[TraineeRecord] = []
    for trainee_id, jobs in sorted(groups.items()):
        scores: list[float] = []
        dtw_costs: list[float] = []
        critical_count = 0
        file_path: str | None = None
        operator_id: str | None = None

        for job in jobs:
            parsed = job["_parsed"]
            if file_path is None:
                file_path = job.get("file_path") or job.get("original_filename")
            if operator_id is None:
                operator_id = job.get("operator_id")

            score_val = parsed.get("score")
            if score_val is not None:
                try:
                    scores.append(float(score_val))
                except (TypeError, ValueError):
                    pass

            # DTW cost from metrics.dtw_normalized_cost
            dtw_val: float | None = None
            metrics = parsed.get("metrics")
            if isinstance(metrics, dict):
                raw_dtw = metrics.get("dtw_normalized_cost")
                if raw_dtw is not None:
                    try:
                        dtw_val = float(raw_dtw)
                    except (TypeError, ValueError):
                        pass
            # fallback: alignment.normalized_cost
            if dtw_val is None:
                alignment = parsed.get("alignment")
                if isinstance(alignment, dict):
                    raw_dtw2 = alignment.get("normalized_cost")
                    if raw_dtw2 is not None:
                        try:
                            dtw_val = float(raw_dtw2)
                        except (TypeError, ValueError):
                            pass
            if dtw_val is not None:
                dtw_costs.append(dtw_val)

            # Count critical deviations
            for dev in parsed.get("deviations", []):
                if isinstance(dev, dict) and dev.get("severity") == "critical":
                    critical_count += 1

        if not scores:
            continue

        avg_score = float(np.mean(scores))
        avg_dtw: float | None = float(np.mean(dtw_costs)) if dtw_costs else None

        records.append(
            TraineeRecord(
                trainee_video_id=trainee_id,
                score=avg_score,
                dtw_cost=avg_dtw,
                file_path=file_path,
                operator_id=operator_id,
                critical_flag_count=critical_count,
            )
        )

    return records


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


@dataclass
class ConfusionCounts:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.tn + self.fp + self.fn


def _build_confusion(y_true: list[bool], y_pred: list[bool]) -> ConfusionCounts:
    cc = ConfusionCounts()
    for gt, pred in zip(y_true, y_pred):
        if gt and pred:
            cc.tp += 1
        elif gt and not pred:
            cc.fn += 1
        elif not gt and pred:
            cc.fp += 1
        else:
            cc.tn += 1
    return cc


def _accuracy(cc: ConfusionCounts) -> float | None:
    return (cc.tp + cc.tn) / cc.n if cc.n > 0 else None


def _precision(cc: ConfusionCounts) -> float | None:
    d = cc.tp + cc.fp
    return cc.tp / d if d > 0 else None


def _recall(cc: ConfusionCounts) -> float | None:
    d = cc.tp + cc.fn
    return cc.tp / d if d > 0 else None


def _f1(cc: ConfusionCounts) -> float | None:
    p = _precision(cc)
    r = _recall(cc)
    if p is None or r is None:
        return None
    d = p + r
    return 2 * p * r / d if d > 0 else 0.0


def _auc_roc(y_true: list[bool], y_score: list[float]) -> float | None:
    """Trapezoidal AUC-ROC (no sklearn required)."""
    if len(y_true) < 2:
        return None
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    tpr_pts: list[float] = [0.0]
    fpr_pts: list[float] = [0.0]
    cum_tp = 0
    cum_fp = 0
    for _, label in pairs:
        if label:
            cum_tp += 1
        else:
            cum_fp += 1
        tpr_pts.append(cum_tp / n_pos)
        fpr_pts.append(cum_fp / n_neg)
    auc = float(np.trapz(tpr_pts, fpr_pts))
    return max(0.0, min(1.0, auc))


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


@dataclass
class BootstrapCI:
    point: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    n_samples: int


def _bootstrap_metrics(
    y_true: list[bool],
    y_pred: list[bool],
    y_score: list[float],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, BootstrapCI | None]:
    """Bootstrap all four classification metrics plus AUC-ROC."""
    n = len(y_true)
    rng = np.random.default_rng(seed)

    # Point estimates
    cc_full = _build_confusion(y_true, y_pred)
    pt_acc = _accuracy(cc_full)
    pt_prec = _precision(cc_full)
    pt_rec = _recall(cc_full)
    pt_f1 = _f1(cc_full)
    pt_auc = _auc_roc(y_true, y_score)

    boot_acc: list[float] = []
    boot_prec: list[float] = []
    boot_rec: list[float] = []
    boot_f1: list[float] = []
    boot_auc: list[float] = []

    yt = np.array([int(x) for x in y_true], dtype=np.int8)
    yp = np.array([int(x) for x in y_pred], dtype=np.int8)
    ys = np.array(y_score, dtype=float)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bt = yt[idx].astype(bool).tolist()
        bp = yp[idx].astype(bool).tolist()
        bs = ys[idx].tolist()
        cc = _build_confusion(bt, bp)

        v = _accuracy(cc)
        if v is not None:
            boot_acc.append(v)
        v = _precision(cc)
        if v is not None:
            boot_prec.append(v)
        v = _recall(cc)
        if v is not None:
            boot_rec.append(v)
        v = _f1(cc)
        if v is not None:
            boot_f1.append(v)
        v = _auc_roc(bt, bs)
        if v is not None:
            boot_auc.append(v)

    def _ci(point: float | None, boots: list[float]) -> BootstrapCI | None:
        if point is None:
            return None
        if not boots:
            return BootstrapCI(point=point, ci_low=point, ci_high=point,
                               n_bootstrap=n_bootstrap, n_samples=n)
        lo = float(np.percentile(boots, 100 * alpha / 2))
        hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
        return BootstrapCI(point=point, ci_low=lo, ci_high=hi,
                           n_bootstrap=n_bootstrap, n_samples=n)

    return {
        "accuracy":  _ci(pt_acc, boot_acc),
        "precision": _ci(pt_prec, boot_prec),
        "recall":    _ci(pt_rec, boot_rec),
        "f1":        _ci(pt_f1, boot_f1),
        "auc_roc":   _ci(pt_auc, boot_auc),
    }


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------


def _standard_normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _mcnemar_test(
    y_true: list[bool],
    y_pred_a: list[bool],
    y_pred_b: list[bool],
    label_a: str = "System A",
    label_b: str = "System B",
) -> dict[str, Any]:
    """McNemar's test comparing two classifiers against the same ground truth.

    b = A correct, B wrong
    c = A wrong, B correct
    Uses continuity correction: chi2 = (|b-c| - 1)^2 / (b+c)
    """
    b = sum(1 for t, a, bb in zip(y_true, y_pred_a, y_pred_b) if a == t and bb != t)
    c = sum(1 for t, a, bb in zip(y_true, y_pred_a, y_pred_b) if a != t and bb == t)
    n_disc = b + c

    if n_disc == 0:
        return {
            "system_a": label_a,
            "system_b": label_b,
            "b": 0,
            "c": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_at_0.05": False,
            "note": "no_discordant_pairs — systems agree on every example",
        }

    statistic = (abs(b - c) - 1.0) ** 2 / n_disc  # with continuity correction
    # p-value via standard normal approximation (no scipy required)
    z = (abs(b - c) - 1.0) / math.sqrt(n_disc)
    p_value = 2.0 * (1.0 - _standard_normal_cdf(abs(z)))

    # Try scipy chi2 if available for a more accurate p-value
    try:
        from scipy import stats as _stats  # type: ignore[import-untyped]
        p_value = float(1.0 - _stats.chi2.cdf(statistic, df=1))
    except ImportError:
        pass

    return {
        "system_a": label_a,
        "system_b": label_b,
        "b": b,
        "c": c,
        "statistic": round(statistic, 6),
        "p_value": round(p_value, 6),
        "significant_at_0.05": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------------------
# Three systems
# ---------------------------------------------------------------------------


def _random_predictions(n: int, *, seed: int = 42) -> list[bool]:
    """50/50 random pass/fail assignments."""
    rng = np.random.default_rng(seed)
    return [bool(x) for x in rng.integers(0, 2, size=n)]


def _score_threshold_predictions(
    records: list[TraineeRecord], *, pass_threshold: float
) -> list[bool]:
    return [r.score >= pass_threshold for r in records]


def _dtw_threshold_predictions(
    records: list[TraineeRecord], *, dtw_pass_threshold: float = 0.05
) -> list[bool]:
    """Pass if dtw_normalized_cost <= dtw_pass_threshold.

    Records where dtw_cost is None are conservatively classified as FAIL.
    """
    preds: list[bool] = []
    for r in records:
        if r.dtw_cost is None:
            preds.append(False)  # missing cost → conservative fail
        else:
            preds.append(r.dtw_cost <= dtw_pass_threshold)
    return preds


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------


def _ci_to_dict(ci: BootstrapCI | None) -> dict[str, Any] | None:
    if ci is None:
        return None
    return {
        "point": round(ci.point, 6),
        "ci95_low": round(ci.ci_low, 6),
        "ci95_high": round(ci.ci_high, 6),
        "n_bootstrap": ci.n_bootstrap,
        "n_samples": ci.n_samples,
    }


def _system_result(
    name: str,
    y_true: list[bool],
    y_pred: list[bool],
    y_score: list[float],
    *,
    n_bootstrap: int,
    seed: int = 42,
) -> dict[str, Any]:
    cc = _build_confusion(y_true, y_pred)
    cis = _bootstrap_metrics(y_true, y_pred, y_score,
                             n_bootstrap=n_bootstrap, seed=seed)
    return {
        "system": name,
        "confusion_matrix": {"tp": cc.tp, "tn": cc.tn, "fp": cc.fp, "fn": cc.fn, "n": cc.n},
        "metrics": {k: _ci_to_dict(v) for k, v in cis.items()},
    }


# ---------------------------------------------------------------------------
# ASCII table output
# ---------------------------------------------------------------------------


def _fmt_ci_col(ci: BootstrapCI | None, *, width: int = 16) -> str:
    if ci is None:
        return "N/A".ljust(width)
    s = f"{ci.point:.2f} [{ci.ci_low:.2f},{ci.ci_high:.2f}]"
    return s.ljust(width)


def _build_summary_table(
    *,
    task_id: str | None,
    n: int,
    pass_threshold: float,
    n_bootstrap: int,
    systems: list[dict[str, Any]],
    mcnemar: dict[str, Any],
) -> str:
    lines: list[str] = []
    task_str = task_id or "all"
    lines.append(f"=== Ablation Study: {task_str} ===")
    lines.append(f"N={n} videos, pass_threshold={pass_threshold}, bootstrap n={n_bootstrap}")
    lines.append("")

    col_w = 16
    hdr = (
        f"{'System':<24}| {'Accuracy':<{col_w}}| {'Precision':<{col_w}}"
        f"| {'Recall':<{col_w}}| {'F1':<{col_w}}| {'AUC-ROC':<{col_w}}"
    )
    sep = "-" * len(hdr)
    lines.append(hdr)
    lines.append(sep)

    def _ci_from_d(d: dict[str, Any] | None) -> BootstrapCI | None:
        """Reconstruct a BootstrapCI from a serialised dict."""
        if d is None:
            return None
        return BootstrapCI(
            point=d["point"], ci_low=d["ci95_low"], ci_high=d["ci95_high"],
            n_bootstrap=d["n_bootstrap"], n_samples=d["n_samples"],
        )

    for sys_r in systems:
        m = sys_r["metrics"]
        row = (
            f"{sys_r['system']:<24}"
            f"| {_fmt_ci_col(_ci_from_d(m.get('accuracy')), width=col_w)}"
            f"| {_fmt_ci_col(_ci_from_d(m.get('precision')), width=col_w)}"
            f"| {_fmt_ci_col(_ci_from_d(m.get('recall')), width=col_w)}"
            f"| {_fmt_ci_col(_ci_from_d(m.get('f1')), width=col_w)}"
            f"| {_fmt_ci_col(_ci_from_d(m.get('auc_roc')), width=col_w)}"
        )
        lines.append(row)

    lines.append(sep)
    lines.append("")

    # Confusion matrices
    lines.append("Confusion matrices (TP/FP/FN/TN):")
    for sys_r in systems:
        cm = sys_r["confusion_matrix"]
        lines.append(
            f"  {sys_r['system']:<24}: "
            f"TP={cm['tp']}  FP={cm['fp']}  FN={cm['fn']}  TN={cm['tn']}"
        )
    lines.append("")

    # McNemar
    lines.append("McNemar's Test (DTW-only vs. SOPilot full system):")
    if "note" in mcnemar:
        lines.append(f"  {mcnemar['note']}")
    else:
        lines.append(f"  b (DTW correct, SOPilot wrong) = {mcnemar['b']}")
        lines.append(f"  c (DTW wrong, SOPilot correct) = {mcnemar['c']}")
        lines.append(f"  chi2 statistic (continuity-corrected) = {mcnemar['statistic']:.4f}")
        lines.append(f"  p-value = {mcnemar['p_value']:.4f}")
        sig_str = "YES (p < 0.05)" if mcnemar["significant_at_0.05"] else "NO (p >= 0.05)"
        lines.append(f"  Significant at alpha=0.05: {sig_str}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study: random / DTW-only / full-system classification comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", dest="db_path", required=True, metavar="PATH",
                        help="Path to sopilot.db SQLite database.")
    parser.add_argument("--task-id", default=None, metavar="ID",
                        help="Optional task_id filter (e.g. filter_change).")
    parser.add_argument("--pass-threshold", type=float, default=70.0, metavar="SCORE",
                        help="Score threshold: score >= threshold → PASS (ground truth + system).")
    parser.add_argument("--dtw-pass-threshold", type=float, default=0.05, metavar="COST",
                        help="DTW cost threshold: cost <= threshold → PASS for DTW-only system.")
    parser.add_argument("--output-dir", default="artifacts/ablation", metavar="DIR",
                        help="Directory to write output files.")
    parser.add_argument("--n-bootstrap", type=int, default=1000, metavar="N",
                        help="Number of bootstrap resamples for CI computation.")
    parser.add_argument("--seed", type=int, default=42, metavar="N",
                        help="Random seed for bootstrap and random baseline.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    db_path = Path(args.db_path).resolve()
    out_dir = Path(args.output_dir).resolve()

    print(_c("\nSOPilot Ablation Study", _BOLD, _CYAN))
    print(_c("=" * 60, _CYAN))
    print(f"  DB:              {db_path}")
    print(f"  Task ID:         {args.task_id or '(all)'}")
    print(f"  Pass threshold:  {args.pass_threshold}")
    print(f"  DTW threshold:   {args.dtw_pass_threshold}")
    print(f"  Bootstrap n:     {args.n_bootstrap}")
    print(f"  Output dir:      {out_dir}")

    # ---- Load data ----
    conn = _open_db(db_path)
    print("\n  Loading trainee records from database...")
    records = _load_trainee_records(conn, task_id=args.task_id)
    conn.close()

    n = len(records)
    if n == 0:
        raise SystemExit(
            "[ERROR] No completed trainee score jobs found. "
            "Check --task-id and ensure the database has scored data."
        )
    print(f"  Unique trainee videos: {n}")

    # ---- Ground truth: score >= pass_threshold ----
    y_true: list[bool] = [r.score >= args.pass_threshold for r in records]
    n_pass = sum(y_true)
    n_fail = n - n_pass
    print(f"  Ground truth (score >= {args.pass_threshold}): {n_pass} pass, {n_fail} fail")

    # ---- System predictions ----
    print("\n  Generating predictions for 3 systems...")

    # 1. Random baseline
    y_rand = _random_predictions(n, seed=args.seed)
    # Score used for AUC: random probabilities (0.5 for all → AUC ~= 0.5)
    rng_auc = np.random.default_rng(args.seed + 1)
    y_rand_score = rng_auc.uniform(0.0, 1.0, size=n).tolist()

    # 2. Score threshold (full system) — score in [0,100]; normalise to [0,1] for AUC
    y_sys = _score_threshold_predictions(records, pass_threshold=args.pass_threshold)
    y_sys_score = [r.score / 100.0 for r in records]

    # 3. DTW cost only — invert cost so high cost = low score for AUC
    y_dtw = _dtw_threshold_predictions(records, dtw_pass_threshold=args.dtw_pass_threshold)
    y_dtw_score = [
        (1.0 - r.dtw_cost) if r.dtw_cost is not None else 0.0
        for r in records
    ]

    # ---- Bootstrap metrics ----
    print("  Computing bootstrap CIs (this may take a moment)...")

    sys_rand = _system_result(
        "Random baseline", y_true, y_rand, y_rand_score,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    sys_dtw = _system_result(
        "DTW cost threshold", y_true, y_dtw, y_dtw_score,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    sys_full = _system_result(
        "SOPilot (full system)", y_true, y_sys, y_sys_score,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )

    systems = [sys_rand, sys_dtw, sys_full]

    # ---- McNemar: DTW-only vs. full system ----
    mcnemar = _mcnemar_test(
        y_true, y_dtw, y_sys,
        label_a="DTW cost threshold",
        label_b="SOPilot (full system)",
    )

    # ---- Assemble JSON output ----
    generated_at = datetime.now(timezone.utc).isoformat()
    json_out: dict[str, Any] = {
        "schema_version": "ablation_v1",
        "generated_at": generated_at,
        "db_path": str(db_path),
        "task_id": args.task_id,
        "pass_threshold": args.pass_threshold,
        "dtw_pass_threshold": args.dtw_pass_threshold,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "n_videos": n,
        "n_pass_gt": n_pass,
        "n_fail_gt": n_fail,
        "systems": systems,
        "mcnemar_dtw_vs_full": mcnemar,
        "per_video": [
            {
                "trainee_video_id": r.trainee_video_id,
                "file_path": r.file_path,
                "operator_id": r.operator_id,
                "score": round(r.score, 4),
                "dtw_cost": round(r.dtw_cost, 6) if r.dtw_cost is not None else None,
                "critical_flag_count": r.critical_flag_count,
                "gt_pass": y_true[i],
                "pred_random": y_rand[i],
                "pred_dtw": y_dtw[i],
                "pred_system": y_sys[i],
            }
            for i, r in enumerate(records)
        ],
    }

    # ---- ASCII summary table ----
    summary_txt = _build_summary_table(
        task_id=args.task_id,
        n=n,
        pass_threshold=args.pass_threshold,
        n_bootstrap=args.n_bootstrap,
        systems=systems,
        mcnemar=mcnemar,
    )

    # ---- Write outputs ----
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ablation_results.json"
    txt_path = out_dir / "ablation_summary.txt"

    json_path.write_text(
        json.dumps(json_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    txt_path.write_text(summary_txt, encoding="utf-8")

    # ---- Print summary to terminal ----
    print()
    print(summary_txt)
    print()
    print(_c("Output files written:", _BOLD))
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")
    print()


if __name__ == "__main__":
    main()
