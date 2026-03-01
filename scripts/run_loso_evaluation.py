"""run_loso_evaluation.py — Standalone Leave-One-Subject-Out (LOSO) scientific evaluation.

Usage
-----
    python scripts/run_loso_evaluation.py \
        --db data_release_baseline/sopilot.db \
        --task-id filter_change \
        --pass-threshold 70 \
        --output-dir artifacts/loso_eval/ \
        --human-labels path/to/human_labels.json   # optional

Dependencies: stdlib + numpy + scipy only (no FastAPI, no torch required).
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sqlite3
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# scipy is imported lazily so the error message is friendly
try:
    from scipy import stats as scipy_stats  # type: ignore[import-untyped]

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_DIM = "\033[2m"


def _c(text: str, *codes: str, tty: bool = True) -> str:
    if not tty or not sys.stdout.isatty():
        return text
    return "".join(codes) + text + _RESET


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\033\[[0-9;]*m", "", text)


# ---------------------------------------------------------------------------
# ASCII progress bar
# ---------------------------------------------------------------------------


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return f"[{'?' * width}]  ?/??"
    frac = min(1.0, current / total)
    filled = int(frac * width)
    bar = "#" * filled + "-" * (width - filled)
    pct = frac * 100.0
    return f"[{bar}] {pct:5.1f}% ({current}/{total})"


# ---------------------------------------------------------------------------
# Database access (raw sqlite3, no ORM)
# ---------------------------------------------------------------------------


def _open_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise SystemExit(f"[ERROR] Database not found: {db_path}")
    conn = sqlite3.connect(str(db_path), timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA query_only=ON")  # read-only session guard
    return conn


def _get_schema(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return {row["name"]: (row["sql"] or "") for row in rows}


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Return True if *column* exists in *table* (introspects PRAGMA table_info)."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


def _load_trainee_jobs(
    conn: sqlite3.Connection,
    *,
    task_id: str | None,
) -> list[dict[str, Any]]:
    """Return completed score_jobs for non-gold (trainee) videos with their scores.

    Handles databases that pre-date the ``original_filename`` migration by
    detecting column presence at runtime.
    """
    has_orig_filename = _has_column(conn, "videos", "original_filename")
    orig_filename_expr = (
        "tv.original_filename        AS original_filename,"
        if has_orig_filename
        else "NULL                        AS original_filename,"
    )

    query = f"""
        SELECT
            sj.id                       AS job_id,
            sj.gold_video_id,
            sj.trainee_video_id,
            sj.score_json,
            sj.created_at,
            sj.finished_at,
            tv.operator_id_hash         AS operator_id,
            tv.site_id                  AS site_id,
            tv.is_gold                  AS trainee_is_gold,
            {orig_filename_expr}
            sr.verdict                  AS review_verdict
        FROM score_jobs sj
        LEFT JOIN videos tv ON tv.id = sj.trainee_video_id
        LEFT JOIN score_reviews sr ON sr.job_id = sj.id
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
    query += " ORDER BY sj.id ASC"

    rows = conn.execute(query, params).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        raw_json = item.pop("score_json", None)
        score_val: float | None = None
        decision_val: str | None = None
        if raw_json:
            try:
                parsed = json.loads(raw_json)
                score_val = float(parsed["score"]) if parsed.get("score") is not None else None
                # decision lives in score_json["summary"]["decision"]
                _summary = parsed.get("summary") or {}
                _raw_dec = _summary.get("decision", "")
                decision_val = str(_raw_dec).lower().strip() if _raw_dec else None
            except Exception:
                pass
        item["score"] = score_val
        item["system_decision"] = decision_val  # extracted from summary.decision
        result.append(item)
    return result


def _load_gold_count(conn: sqlite3.Connection, *, task_id: str | None) -> int:
    query = "SELECT COUNT(*) FROM videos WHERE is_gold = 1"
    params: list[Any] = []
    if task_id:
        query += " AND task_id = ?"
        params.append(task_id)
    row = conn.execute(query, params).fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Human labels
# ---------------------------------------------------------------------------


@dataclass
class HumanLabel:
    video_id: int
    human_score: float | None
    human_verdict: str  # "pass" | "fail"
    annotator_id: str
    annotation_time: str


def _load_human_labels(path: Path) -> dict[int, HumanLabel]:
    """Load optional human labels JSON (list of label objects keyed by video_id)."""
    if not path.exists():
        raise SystemExit(f"[ERROR] Human labels file not found: {path}")
    raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("[ERROR] Human labels JSON must be a list of objects.")
    out: dict[int, HumanLabel] = {}
    for item in raw:
        vid = int(item["video_id"])
        out[vid] = HumanLabel(
            video_id=vid,
            human_score=float(item["human_score"]) if item.get("human_score") is not None else None,
            human_verdict=str(item.get("human_verdict", "")).lower().strip(),
            annotator_id=str(item.get("annotator_id", "")),
            annotation_time=str(item.get("annotation_time", "")),
        )
    return out


# ---------------------------------------------------------------------------
# Ground truth resolution
# ---------------------------------------------------------------------------


def _resolve_ground_truth(
    job: dict[str, Any],
    *,
    pass_threshold: float,
    human_labels: dict[int, HumanLabel] | None,
    skip_score_fallback: bool = False,
) -> bool | None:
    """Return True=pass, False=fail, None=unknown for a job.

    Args:
        skip_score_fallback: When True, Priority 3 (score >= threshold) is
            skipped.  Use this for product-mode evaluation to avoid circular
            ground truth: the product decision can legitimately return 'fail'
            for scores above the pass threshold (critical-deviation override),
            so using score-based GT would create spurious FNs.
    """
    vid = int(job["trainee_video_id"])

    # Priority 1: explicit human label
    if human_labels and vid in human_labels:
        verdict = human_labels[vid].human_verdict
        if verdict == "pass":
            return True
        if verdict in ("fail", "no_pass", "nopass"):
            return False

    # Priority 2: reviewer verdict from score_reviews table
    review = job.get("review_verdict")
    if review:
        rv = str(review).lower().strip()
        if rv in ("pass", "approved", "ok"):
            return True
        if rv in ("fail", "reject", "rejected", "failed"):
            return False

    # Priority 3: system score >= pass_threshold
    # Skipped for product-mode evaluation (see docstring).
    if not skip_score_fallback:
        score = job.get("score")
        if score is not None:
            return float(score) >= pass_threshold

    return None


# ---------------------------------------------------------------------------
# Core metric primitives
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

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def negatives(self) -> int:
        return self.fp + self.tn


def _build_confusion(
    y_true: list[bool],
    y_pred: list[bool],
) -> ConfusionCounts:
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
    if cc.n == 0:
        return None
    return (cc.tp + cc.tn) / cc.n


def _precision(cc: ConfusionCounts) -> float | None:
    denom = cc.tp + cc.fp
    if denom == 0:
        return None
    return cc.tp / denom


def _recall(cc: ConfusionCounts) -> float | None:
    denom = cc.tp + cc.fn
    if denom == 0:
        return None
    return cc.tp / denom


def _f1(cc: ConfusionCounts) -> float | None:
    p = _precision(cc)
    r = _recall(cc)
    if p is None or r is None:
        return None
    denom = p + r
    if denom == 0.0:
        return 0.0
    return 2 * p * r / denom


def _critical_miss_rate(cc: ConfusionCounts) -> float | None:
    """FN / (TP + FN) — fraction of true positives missed."""
    denom = cc.tp + cc.fn
    if denom == 0:
        return None
    return cc.fn / denom


def _false_alarm_rate(cc: ConfusionCounts) -> float | None:
    """FP / (FP + TN) — fraction of true negatives falsely alarmed."""
    denom = cc.fp + cc.tn
    if denom == 0:
        return None
    return cc.fp / denom


def _auc_roc(
    y_true: list[bool],
    y_score: list[float],
) -> float | None:
    """Trapezoidal AUC-ROC (manual, no sklearn required)."""
    if len(y_true) < 2:
        return None
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # Sort by descending score
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
    # Trapezoidal integration
    auc = float(np.trapz(tpr_pts, fpr_pts))
    return max(0.0, min(1.0, auc))


def _cohen_kappa(y_true: list[bool], y_pred: list[bool]) -> float | None:
    n = len(y_true)
    if n == 0:
        return None
    n_agree = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    p_o = n_agree / n
    p_yes_true = sum(y_true) / n
    p_yes_pred = sum(y_pred) / n
    p_e = p_yes_true * p_yes_pred + (1 - p_yes_true) * (1 - p_yes_pred)
    denom = 1.0 - p_e
    if abs(denom) < 1e-12:
        return 1.0 if abs(p_o - 1.0) < 1e-12 else None
    return (p_o - p_e) / denom


def _icc_two_way_mixed(scores_sys: list[float], scores_human: list[float]) -> float | None:
    """ICC(2,1) — two-way mixed, absolute agreement (Shrout & Fleiss)."""
    n = len(scores_sys)
    if n < 2:
        return None
    arr = np.array([[s, h] for s, h in zip(scores_sys, scores_human)], dtype=float)
    k = arr.shape[1]  # 2 raters
    grand_mean = arr.mean()
    row_means = arr.mean(axis=1)
    col_means = arr.mean(axis=0)

    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_total = np.sum((arr - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    df_rows = n - 1
    df_cols = k - 1
    df_error = df_rows * df_cols

    if df_rows == 0 or df_error == 0:
        return None

    ms_rows = ss_rows / df_rows
    ms_cols = ss_cols / df_cols if df_cols > 0 else 0.0
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    _ = ms_cols  # used only in full ICCs; simplified here
    denom = ms_rows + (k - 1) * ms_error
    if abs(denom) < 1e-12:
        return None
    return float((ms_rows - ms_error) / denom)


def _mae(a: list[float], b: list[float]) -> float | None:
    if not a:
        return None
    return float(np.mean(np.abs(np.array(a) - np.array(b))))


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


@dataclass
class BootstrapCI:
    point: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    n_samples: int


def _bootstrap_ci(
    values: np.ndarray,
    stat_fn: Any,  # callable(sample) -> float|None
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    show_progress: bool = False,
    label: str = "",
) -> BootstrapCI | None:
    """Generic bootstrap CI for a scalar statistic over a 1D array of indices."""
    n = len(values)
    if n == 0:
        return None
    point = stat_fn(values)
    if point is None:
        return None
    if rng is None:
        rng = np.random.default_rng(42)

    boot_stats: list[float] = []
    tick_every = max(1, n_bootstrap // 40)
    for i in range(n_bootstrap):
        if show_progress and (i % tick_every == 0 or i == n_bootstrap - 1):
            bar = _progress_bar(i + 1, n_bootstrap, width=30)
            tag = f" {label}" if label else ""
            sys.stderr.write(f"\r  bootstrap{tag}: {bar}")
            sys.stderr.flush()
        idx = rng.integers(0, n, size=n)
        sample = values[idx]
        stat = stat_fn(sample)
        if stat is not None:
            boot_stats.append(float(stat))

    if show_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    if not boot_stats:
        return BootstrapCI(point=float(point), ci_low=float(point), ci_high=float(point), n_bootstrap=n_bootstrap, n_samples=n)

    low = float(np.percentile(boot_stats, 100 * alpha / 2))
    high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return BootstrapCI(point=float(point), ci_low=low, ci_high=high, n_bootstrap=n_bootstrap, n_samples=n)


# ---------------------------------------------------------------------------
# Per-sample record for vectorised bootstrap
# ---------------------------------------------------------------------------


@dataclass
class EvalRecord:
    job_id: int
    trainee_video_id: int
    operator_id: str | None
    site_id: str | None
    score: float  # system score (0-100)
    pred_pass: bool  # system prediction
    gt_pass: bool  # ground truth
    human_score: float | None = None  # if human labels provided
    original_filename: str | None = None
    system_decision: str | None = None  # raw stored decision: pass/needs_review/retrain/fail


# ---------------------------------------------------------------------------
# Vectorised bootstrap over records
# ---------------------------------------------------------------------------


def _records_to_arrays(records: list[EvalRecord]) -> dict[str, np.ndarray]:
    n = len(records)
    scores = np.array([r.score for r in records], dtype=float)
    pred = np.array([int(r.pred_pass) for r in records], dtype=np.int8)
    gt = np.array([int(r.gt_pass) for r in records], dtype=np.int8)
    human_scores = np.array(
        [r.human_score if r.human_score is not None else float("nan") for r in records],
        dtype=float,
    )
    return {
        "scores": scores,
        "pred": pred,
        "gt": gt,
        "human_scores": human_scores,
        "n": np.array([n]),
    }


def _stat_from_indices(
    idx: np.ndarray,
    arrays: dict[str, np.ndarray],
    stat_name: str,
    pass_threshold: float,
) -> float | None:
    gt = arrays["gt"][idx].astype(bool).tolist()
    pred = arrays["pred"][idx].astype(bool).tolist()
    scores = arrays["scores"][idx].tolist()

    cc = _build_confusion(gt, pred)
    if stat_name == "accuracy":
        return _accuracy(cc)
    if stat_name == "precision":
        return _precision(cc)
    if stat_name == "recall":
        return _recall(cc)
    if stat_name == "f1":
        return _f1(cc)
    if stat_name == "auc_roc":
        return _auc_roc(gt, scores)
    if stat_name == "critical_miss_rate":
        return _critical_miss_rate(cc)
    if stat_name == "false_alarm_rate":
        return _false_alarm_rate(cc)
    return None


def _run_bootstrap(
    records: list[EvalRecord],
    arrays: dict[str, np.ndarray],
    stat_names: list[str],
    pass_threshold: float,
    *,
    n_bootstrap: int = 2000,
    show_progress: bool = True,
) -> dict[str, BootstrapCI | None]:
    n = len(records)
    rng = np.random.default_rng(42)
    results: dict[str, list[float]] = {name: [] for name in stat_names}

    tick_every = max(1, n_bootstrap // 50)
    for i in range(n_bootstrap):
        if show_progress and (i % tick_every == 0 or i == n_bootstrap - 1):
            bar = _progress_bar(i + 1, n_bootstrap, width=40)
            sys.stderr.write(f"\r  bootstrap: {bar}")
            sys.stderr.flush()
        idx = rng.integers(0, n, size=n)
        for name in stat_names:
            val = _stat_from_indices(idx, arrays, name, pass_threshold)
            if val is not None:
                results[name].append(val)

    if show_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    out: dict[str, BootstrapCI | None] = {}
    for name in stat_names:
        gt = arrays["gt"].astype(bool).tolist()
        pred = arrays["pred"].astype(bool).tolist()
        scores = arrays["scores"].tolist()
        cc = _build_confusion(gt, pred)

        if name == "accuracy":
            point = _accuracy(cc)
        elif name == "precision":
            point = _precision(cc)
        elif name == "recall":
            point = _recall(cc)
        elif name == "f1":
            point = _f1(cc)
        elif name == "auc_roc":
            point = _auc_roc(gt, scores)
        elif name == "critical_miss_rate":
            point = _critical_miss_rate(cc)
        elif name == "false_alarm_rate":
            point = _false_alarm_rate(cc)
        else:
            point = None

        if point is None or not results[name]:
            out[name] = None
            continue

        boots = results[name]
        low = float(np.percentile(boots, 2.5))
        high = float(np.percentile(boots, 97.5))
        out[name] = BootstrapCI(
            point=float(point),
            ci_low=low,
            ci_high=high,
            n_bootstrap=n_bootstrap,
            n_samples=n,
        )
    return out


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------


def _mcnemar_test(
    y_true: list[bool],
    y_pred_sys: list[bool],
    y_pred_human: list[bool],
) -> dict[str, Any]:
    """McNemar's test: does system and human baseline agree?"""
    # b: sys wrong, human right
    # c: sys right, human wrong
    b = sum(1 for t, s, h in zip(y_true, y_pred_sys, y_pred_human) if s != t and h == t)
    c = sum(1 for t, s, h in zip(y_true, y_pred_sys, y_pred_human) if s == t and h != t)
    n_discordant = b + c
    if n_discordant == 0:
        return {
            "b": 0,
            "c": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_at_0.05": False,
            "note": "no_discordant_pairs",
        }
    statistic = (abs(b - c) - 1.0) ** 2 / n_discordant  # continuity correction
    if _SCIPY_AVAILABLE:
        p_value = float(1.0 - scipy_stats.chi2.cdf(statistic, df=1))
    else:
        # Fallback: Bonferroni-like normal approximation
        z = (abs(b - c) - 1.0) / math.sqrt(n_discordant)
        p_value = 2.0 * (1.0 - _standard_normal_cdf(abs(z)))
    return {
        "b": b,
        "c": c,
        "statistic": round(statistic, 6),
        "p_value": round(p_value, 6),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def _standard_normal_cdf(z: float) -> float:
    """Approximation of the standard normal CDF (Abramowitz & Stegun)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass
class CalibrationBin:
    low: float
    high: float
    n_total: int
    n_pass: int
    pass_rate: float | None
    pred_pass_rate: float | None


def _compute_calibration(
    records: list[EvalRecord],
    *,
    n_bins: int = 10,
) -> list[CalibrationBin]:
    bin_width = 100.0 / n_bins
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        low = i * bin_width
        high = low + bin_width
        in_bin = [r for r in records if low <= r.score < high or (i == n_bins - 1 and r.score == 100.0)]
        n_total = len(in_bin)
        n_pass = sum(1 for r in in_bin if r.gt_pass)
        n_pred_pass = sum(1 for r in in_bin if r.pred_pass)
        pass_rate = n_pass / n_total if n_total > 0 else None
        pred_pass_rate = n_pred_pass / n_total if n_total > 0 else None
        bins.append(
            CalibrationBin(
                low=low,
                high=high,
                n_total=n_total,
                n_pass=n_pass,
                pass_rate=pass_rate,
                pred_pass_rate=pred_pass_rate,
            )
        )
    return bins


# ---------------------------------------------------------------------------
# Score distribution (ASCII histogram)
# ---------------------------------------------------------------------------


def _ascii_histogram(scores: list[float], *, n_bins: int = 10, width: int = 40) -> str:
    if not scores:
        return "  (no scores)"
    bin_width = 100.0 / n_bins
    counts = [0] * n_bins
    for s in scores:
        idx = min(int(s / bin_width), n_bins - 1)
        counts[idx] += 1
    max_count = max(counts) if counts else 1
    lines: list[str] = []
    for i, cnt in enumerate(counts):
        low = int(i * bin_width)
        high = int((i + 1) * bin_width)
        bar_len = int(cnt / max_count * width) if max_count > 0 else 0
        bar = "#" * bar_len
        lines.append(f"  [{low:3d}-{high:3d}) {bar:<{width}} {cnt:4d}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recommendations engine
# ---------------------------------------------------------------------------


def _generate_recommendations(
    cc: ConfusionCounts,
    cis: dict[str, BootstrapCI | None],
    *,
    n: int,
    pass_threshold: float,
    has_human_labels: bool,
    score_std: float | None,
    score_mean: float | None,
) -> list[str]:
    recs: list[str] = []

    if n < 10:
        recs.append(
            "SAMPLE SIZE: Fewer than 10 evaluation subjects detected. Collect more annotated "
            "samples before drawing conclusions. Statistical tests are unreliable below n=30."
        )

    miss_ci = cis.get("critical_miss_rate")
    if miss_ci is not None and miss_ci.ci_high > 0.10:
        recs.append(
            f"CRITICAL MISS RATE: Upper CI bound is {miss_ci.ci_high:.1%}, exceeding the 10% "
            "safety threshold. Investigate FN cases and consider lowering the pass threshold or "
            "retraining gold reference videos."
        )

    far_ci = cis.get("false_alarm_rate")
    if far_ci is not None and far_ci.ci_high > 0.30:
        recs.append(
            f"FALSE ALARM RATE: Upper CI bound is {far_ci.ci_high:.1%}, exceeding 30%. "
            "Consider raising the pass threshold or improving DTW reference alignment."
        )

    auc_ci = cis.get("auc_roc")
    if auc_ci is not None and auc_ci.point < 0.70:
        recs.append(
            f"AUC-ROC: Point estimate {auc_ci.point:.3f} is below 0.70, suggesting the system "
            "has limited discriminative ability at the current pass threshold ({pass_threshold}). "
            "Explore threshold sweep or improve embedding quality."
        )

    if score_std is not None and score_std < 5.0 and n > 5:
        recs.append(
            f"SCORE VARIANCE: Score std is very low ({score_std:.2f}). The scoring system may "
            "be poorly calibrated or the evaluation set lacks difficulty variety."
        )

    if not has_human_labels:
        recs.append(
            "INTER-RATER RELIABILITY: No human labels provided. Collect expert annotations to "
            "enable Cohen's kappa, ICC, and McNemar's test for rigorous validation."
        )

    if cc.fn > 0 and n > 0:
        fn_rate = cc.fn / n
        if fn_rate > 0.05:
            recs.append(
                f"FALSE NEGATIVES: {cc.fn} FN case(s) detected ({fn_rate:.1%} of all subjects). "
                "Review these videos to identify systematic failure patterns."
            )

    if not recs:
        recs.append(
            "All measured metrics are within acceptable bounds. Continue monitoring with "
            "periodic re-evaluation as new data is collected."
        )

    return recs


# ---------------------------------------------------------------------------
# HTML report builder
# ---------------------------------------------------------------------------


_HTML_STYLE = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #1a1a2e; background: #f8f9fa; }
  h1 { color: #16213e; border-bottom: 3px solid #0f3460; padding-bottom: 10px; }
  h2 { color: #0f3460; border-bottom: 1px solid #ccc; padding-bottom: 6px; margin-top: 40px; }
  h3 { color: #1a1a2e; margin-top: 25px; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; }
  th { background: #0f3460; color: #fff; padding: 10px 14px; text-align: left; }
  td { padding: 8px 14px; border-bottom: 1px solid #e0e0e0; }
  tr:nth-child(even) td { background: #f0f4fa; }
  .kpi-grid { display: flex; flex-wrap: wrap; gap: 16px; margin: 20px 0; }
  .kpi { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px 20px; min-width: 160px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
  .kpi .label { font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
  .kpi .value { font-size: 28px; font-weight: bold; color: #0f3460; margin: 6px 0 2px; }
  .kpi .ci { font-size: 11px; color: #888; }
  .pass { color: #2e7d32; font-weight: bold; }
  .fail { color: #c62828; font-weight: bold; }
  .warn { color: #f57c00; font-weight: bold; }
  pre { background: #1e1e2e; color: #cdd6f4; padding: 16px; border-radius: 8px; font-size: 13px; overflow-x: auto; }
  .rec-list { background: #fff8e1; border-left: 4px solid #f9a825; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 10px 0; }
  .rec-list li { margin: 8px 0; }
  .error-table { font-size: 13px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }
  .badge-fp { background: #ffcdd2; color: #b71c1c; }
  .badge-fn { background: #fff9c4; color: #f57f17; }
  .badge-tp { background: #c8e6c9; color: #1b5e20; }
  .badge-tn { background: #e3f2fd; color: #0d47a1; }
  .warn-box { background: #fff3e0; border: 1px solid #ffe082; padding: 12px 16px; border-radius: 8px; margin: 16px 0; }
  footer { margin-top: 60px; font-size: 12px; color: #888; text-align: center; border-top: 1px solid #ddd; padding-top: 16px; }
</style>
"""


def _fmt_ci(ci: BootstrapCI | None) -> str:
    if ci is None:
        return "N/A"
    return f"{ci.point:.4f} [{ci.ci_low:.4f}, {ci.ci_high:.4f}]"


def _fmt_pct_ci(ci: BootstrapCI | None) -> str:
    if ci is None:
        return "N/A"
    return f"{ci.point:.1%} [{ci.ci_low:.1%}, {ci.ci_high:.1%}]"


def _html_kpi(label: str, value_str: str, ci_str: str = "") -> str:
    return (
        f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value_str)}</div>'
        f'<div class="ci">{html.escape(ci_str)}</div></div>'
    )


def _build_html_report(
    *,
    task_id: str | None,
    pass_threshold: float,
    eval_mode: str = "threshold",
    records: list[EvalRecord],
    cc: ConfusionCounts,
    cis: dict[str, BootstrapCI | None],
    calibration: list[CalibrationBin],
    recommendations: list[str],
    score_mean: float | None,
    score_std: float | None,
    score_mae: float | None,
    cohen_kappa: float | None,
    icc: float | None,
    mcnemar: dict[str, Any] | None,
    gold_count: int,
    n_total_jobs: int,
    n_labeled: int,
    n_unknown_gt: int,
    has_human_labels: bool,
    small_n_warning: bool,
    generated_at: str,
) -> str:
    n = cc.n
    scores = [r.score for r in records]

    # Executive summary KPIs
    acc_ci = cis.get("accuracy")
    f1_ci = cis.get("f1")
    auc_ci = cis.get("auc_roc")
    miss_ci = cis.get("critical_miss_rate")
    far_ci = cis.get("false_alarm_rate")
    prec_ci = cis.get("precision")
    rec_ci = cis.get("recall")

    kpis = "".join(
        [
            _html_kpi("Subjects (N)", str(n)),
            _html_kpi("Pass Threshold", f"{pass_threshold:.0f}"),
            _html_kpi("Gold Videos", str(gold_count)),
            _html_kpi(
                "Accuracy",
                f"{acc_ci.point:.1%}" if acc_ci else "N/A",
                f"95% CI [{acc_ci.ci_low:.1%}, {acc_ci.ci_high:.1%}]" if acc_ci else "",
            ),
            _html_kpi(
                "F1 Score",
                f"{f1_ci.point:.3f}" if f1_ci else "N/A",
                f"95% CI [{f1_ci.ci_low:.3f}, {f1_ci.ci_high:.3f}]" if f1_ci else "",
            ),
            _html_kpi(
                "AUC-ROC",
                f"{auc_ci.point:.3f}" if auc_ci else "N/A",
                f"95% CI [{auc_ci.ci_low:.3f}, {auc_ci.ci_high:.3f}]" if auc_ci else "",
            ),
            _html_kpi(
                "Miss Rate",
                f"{miss_ci.point:.1%}" if miss_ci else "N/A",
                f"95% CI [{miss_ci.ci_low:.1%}, {miss_ci.ci_high:.1%}]" if miss_ci else "",
            ),
            _html_kpi(
                "False Alarm",
                f"{far_ci.point:.1%}" if far_ci else "N/A",
                f"95% CI [{far_ci.ci_low:.1%}, {far_ci.ci_high:.1%}]" if far_ci else "",
            ),
        ]
    )

    # Confusion matrix table
    confusion_html = f"""
<table>
<tr><th></th><th>Predicted PASS</th><th>Predicted FAIL</th></tr>
<tr><td><b>Actual PASS</b></td><td class="pass">{cc.tp} TP</td><td class="fail">{cc.fn} FN</td></tr>
<tr><td><b>Actual FAIL</b></td><td class="fail">{cc.fp} FP</td><td class="pass">{cc.tn} TN</td></tr>
</table>"""

    # Metrics table
    def _row(name: str, ci: BootstrapCI | None) -> str:
        if ci is None:
            return f"<tr><td>{html.escape(name)}</td><td>N/A</td><td>—</td><td>—</td></tr>"
        return (
            f"<tr><td>{html.escape(name)}</td>"
            f"<td><b>{ci.point:.4f}</b></td>"
            f"<td>{ci.ci_low:.4f}</td>"
            f"<td>{ci.ci_high:.4f}</td></tr>"
        )

    metrics_table = f"""
<table>
<tr><th>Metric</th><th>Point Estimate</th><th>95% CI Low</th><th>95% CI High</th></tr>
{_row("Accuracy", acc_ci)}
{_row("Precision", prec_ci)}
{_row("Recall / Sensitivity", rec_ci)}
{_row("F1 Score", f1_ci)}
{_row("AUC-ROC", auc_ci)}
{_row("Critical Miss Rate (FNR)", miss_ci)}
{_row("False Alarm Rate (FPR)", far_ci)}
</table>"""

    # Score distribution
    hist = _ascii_histogram(scores, n_bins=10)
    hist_html = f"<pre>{html.escape(hist)}</pre>"

    score_stats_html = ""
    if scores:
        arr = np.array(scores)
        score_stats_html = f"""
<table>
<tr><th>Statistic</th><th>Value</th></tr>
<tr><td>Mean</td><td>{float(arr.mean()):.2f}</td></tr>
<tr><td>Std Dev</td><td>{float(arr.std()):.2f}</td></tr>
<tr><td>Median (p50)</td><td>{float(np.percentile(arr,50)):.2f}</td></tr>
<tr><td>p10</td><td>{float(np.percentile(arr,10)):.2f}</td></tr>
<tr><td>p90</td><td>{float(np.percentile(arr,90)):.2f}</td></tr>
<tr><td>Min</td><td>{float(arr.min()):.2f}</td></tr>
<tr><td>Max</td><td>{float(arr.max()):.2f}</td></tr>
</table>"""

    # Error analysis
    fps = [r for r in records if r.pred_pass and not r.gt_pass]
    fns = [r for r in records if not r.pred_pass and r.gt_pass]

    def _err_rows(errs: list[EvalRecord], badge_class: str, badge_label: str) -> str:
        if not errs:
            return "<tr><td colspan='5'><em>(none)</em></td></tr>"
        rows_html = ""
        for r in errs:
            rows_html += (
                f"<tr>"
                f"<td>{r.job_id}</td>"
                f"<td>{r.trainee_video_id}</td>"
                f"<td>{r.score:.1f}</td>"
                f"<td>{html.escape(r.operator_id or '—')}</td>"
                f"<td><span class='badge {badge_class}'>{badge_label}</span></td>"
                f"</tr>"
            )
        return rows_html

    error_html = f"""
<h3>False Positives (system predicted PASS, actual FAIL) — {len(fps)} case(s)</h3>
<table class="error-table">
<tr><th>Job ID</th><th>Video ID</th><th>Score</th><th>Operator</th><th>Type</th></tr>
{_err_rows(fps, 'badge-fp', 'FP')}
</table>
<h3>False Negatives (system predicted FAIL, actual PASS) — {len(fns)} case(s)</h3>
<table class="error-table">
<tr><th>Job ID</th><th>Video ID</th><th>Score</th><th>Operator</th><th>Type</th></tr>
{_err_rows(fns, 'badge-fn', 'FN')}
</table>"""

    # Calibration table
    cal_rows = ""
    for b in calibration:
        pr_str = f"{b.pass_rate:.1%}" if b.pass_rate is not None else "—"
        ppr_str = f"{b.pred_pass_rate:.1%}" if b.pred_pass_rate is not None else "—"
        cal_rows += (
            f"<tr><td>{b.low:.0f}–{b.high:.0f}</td>"
            f"<td>{b.n_total}</td>"
            f"<td>{b.n_pass}</td>"
            f"<td>{pr_str}</td>"
            f"<td>{ppr_str}</td></tr>"
        )

    cal_html = f"""
<table>
<tr><th>Score Bin</th><th>N</th><th>N Pass (GT)</th><th>Actual Pass Rate</th><th>Pred Pass Rate</th></tr>
{cal_rows}
</table>"""

    # Inter-rater section
    interrater_html = ""
    if has_human_labels:
        kappa_str = f"{cohen_kappa:.4f}" if cohen_kappa is not None else "N/A"
        icc_str = f"{icc:.4f}" if icc is not None else "N/A"
        mae_str = f"{score_mae:.2f}" if score_mae is not None else "N/A"
        interrater_html = f"""
<h2>Inter-Rater Reliability</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Cohen's Kappa (system vs. human)</td><td><b>{kappa_str}</b></td></tr>
<tr><td>ICC(2,1) — two-way mixed absolute agreement</td><td><b>{icc_str}</b></td></tr>
<tr><td>Score MAE (system vs. human)</td><td><b>{mae_str}</b></td></tr>
</table>"""
        if mcnemar:
            sig = "YES" if mcnemar.get("significant_at_0.05") else "NO"
            sig_class = "fail" if mcnemar.get("significant_at_0.05") else "pass"
            interrater_html += f"""
<h3>McNemar's Test (system vs. human baseline)</h3>
<table>
<tr><th>Discordant b (sys wrong, human right)</th><td>{mcnemar.get("b", "—")}</td></tr>
<tr><th>Discordant c (sys right, human wrong)</th><td>{mcnemar.get("c", "—")}</td></tr>
<tr><th>Chi-square statistic</th><td>{mcnemar.get("statistic", "—")}</td></tr>
<tr><th>p-value</th><td>{mcnemar.get("p_value", "—")}</td></tr>
<tr><th>Significant at α=0.05</th><td class="{sig_class}">{sig}</td></tr>
</table>"""
    else:
        interrater_html = """
<h2>Inter-Rater Reliability</h2>
<div class="warn-box">No human labels provided — inter-rater section skipped.
Re-run with <code>--human-labels</code> to enable Cohen's kappa, ICC, and McNemar's test.</div>"""

    # Recommendations
    rec_items = "".join(f"<li>{html.escape(r)}</li>" for r in recommendations)
    recs_html = f'<div class="rec-list"><ul>{rec_items}</ul></div>'

    # Small-N warning
    warn_html = ""
    if small_n_warning:
        warn_html = f"""<div class="warn-box">
<b>Warning:</b> Only {n} evaluation subject(s) found (n &lt; 10).
Statistical tests may be unreliable. Bootstrap CIs are provided but interpret with caution.
</div>"""

    mode_label = (
        "score-threshold (research)" if eval_mode == "threshold"
        else "product decision (operational)"
    )
    title = f"LOSO Evaluation Report — {task_id or 'all tasks'} [{mode_label}]"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html.escape(title)}</title>
{_HTML_STYLE}
</head>
<body>
<h1>{html.escape(title)}</h1>
<p>Generated: {html.escape(generated_at)} &nbsp;|&nbsp;
   Pass threshold: <b>{pass_threshold:.0f}</b> &nbsp;|&nbsp;
   Eval mode: <b>{html.escape(mode_label)}</b> &nbsp;|&nbsp;
   Bootstrap resamples: <b>2000</b> &nbsp;|&nbsp;
   Total score jobs in DB: <b>{n_total_jobs}</b> &nbsp;|&nbsp;
   Labeled subjects: <b>{n_labeled}</b>
   {' &nbsp;|&nbsp; <span class="warn">Unknown GT: ' + str(n_unknown_gt) + '</span>' if n_unknown_gt else ''}
</p>

{warn_html}

<h2>Executive Summary</h2>
<div class="kpi-grid">{kpis}</div>
{confusion_html}

<h2>Classification Metrics (95% Bootstrap CI, n=2000)</h2>
{metrics_table}

<h2>Score Distribution</h2>
{hist_html}
{score_stats_html}

<h2>Error Analysis</h2>
{error_html}

<h2>Calibration</h2>
<p>Score bins vs. actual ground-truth pass rates and system predicted pass rates.</p>
{cal_html}

{interrater_html}

<h2>Recommendations</h2>
{recs_html}

<footer>
  SOPilot LOSO Evaluation &nbsp;|&nbsp; {html.escape(generated_at)} &nbsp;|&nbsp;
  Task: {html.escape(task_id or 'all')}
</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Terminal print helpers
# ---------------------------------------------------------------------------


def _print_section(title: str, tty: bool = True) -> None:
    line = "=" * 60
    print(_c(f"\n{line}", _BOLD, _CYAN, tty=tty))
    print(_c(f"  {title}", _BOLD, _CYAN, tty=tty))
    print(_c(line, _BOLD, _CYAN, tty=tty))


def _print_metric(label: str, ci: BootstrapCI | None, *, tty: bool = True, warn_if_above: float | None = None) -> None:
    if ci is None:
        print(f"  {label:<35} N/A")
        return
    line = f"  {label:<35} {ci.point:.4f}  ±  [{ci.ci_low:.4f}, {ci.ci_high:.4f}]  (95% CI)"
    if warn_if_above is not None and ci.point > warn_if_above:
        print(_c(line, _YELLOW, tty=tty))
    else:
        print(line)


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------


def _run_evaluation(
    conn: sqlite3.Connection,
    *,
    task_id: str | None,
    pass_threshold: float,
    human_labels: dict[int, HumanLabel] | None,
    n_bootstrap: int,
    show_progress: bool,
    eval_mode: str = "threshold",
) -> tuple[
    list[EvalRecord],
    ConfusionCounts,
    dict[str, BootstrapCI | None],
    list[CalibrationBin],
    dict[str, Any],
    int,
    int,
    int,
    int,
]:
    """Run LOSO evaluation. Returns (records, confusion, CIs, calibration, interrater, gold_count, n_total, n_labeled, n_unknown)."""
    tty = sys.stdout.isatty()

    # Load data
    print(_c("  Loading score jobs from database...", _DIM, tty=tty))
    jobs = _load_trainee_jobs(conn, task_id=task_id)
    gold_count = _load_gold_count(conn, task_id=task_id)
    n_total_jobs = len(jobs)

    print(f"  Total completed trainee score jobs: {n_total_jobs}")
    print(f"  Gold reference videos: {gold_count}")

    # Resolve ground truth and build records
    records: list[EvalRecord] = []
    n_unknown_gt = 0
    for job in jobs:
        score = job.get("score")
        if score is None:
            continue
        score_f = float(score)
        # In product mode, skip the score-threshold fallback for ground truth.
        # Using score >= threshold as GT is circular for product mode: the
        # system legitimately returns 'fail' for scores above threshold when
        # a critical deviation is present, which would be misclassified as FN.
        # Product mode is therefore evaluated only on human-labeled records.
        skip_score_fb = eval_mode == "product"
        gt = _resolve_ground_truth(
            job,
            pass_threshold=pass_threshold,
            human_labels=human_labels,
            skip_score_fallback=skip_score_fb,
        )
        if gt is None:
            n_unknown_gt += 1
            continue
        # Resolve prediction based on evaluation mode:
        #   threshold (research):  pred = score >= pass_threshold
        #     Evaluates raw scoring algorithm's discriminative power independently
        #     of threshold-at-scoring-time.  Correct for historical re-evaluation.
        #   product (operational): pred = stored decision == "pass"
        #     Evaluates full system behaviour including critical-deviation overrides
        #     and threshold applied at scoring time.  Falls back to threshold mode
        #     when system_decision is unavailable (e.g. old jobs).
        sys_dec = job.get("system_decision")
        if eval_mode == "product" and sys_dec is not None:
            pred_pass_val = sys_dec == "pass"
        else:
            pred_pass_val = score_f >= pass_threshold

        records.append(
            EvalRecord(
                job_id=int(job["job_id"]),
                trainee_video_id=int(job["trainee_video_id"]),
                operator_id=job.get("operator_id"),
                site_id=job.get("site_id"),
                score=score_f,
                pred_pass=pred_pass_val,
                gt_pass=bool(gt),
                human_score=(
                    human_labels[int(job["trainee_video_id"])].human_score
                    if human_labels and int(job["trainee_video_id"]) in human_labels
                    else None
                ),
                original_filename=job.get("original_filename"),
                system_decision=sys_dec,
            )
        )

    n_labeled = len(records)
    print(f"  Labeled (usable) subjects: {n_labeled}")
    if n_unknown_gt > 0:
        print(
            _c(
                f"  WARNING: {n_unknown_gt} jobs had no ground truth and were excluded.",
                _YELLOW,
                tty=tty,
            )
        )

    small_n = n_labeled < 10
    if small_n:
        print(
            _c(
                f"  WARNING: n={n_labeled} is below 10. Statistical reliability is limited.",
                _YELLOW,
                tty=tty,
            )
        )

    if n_labeled == 0:
        raise SystemExit(
            "[ERROR] No labeled subjects found. Provide --human-labels or ensure "
            "score_reviews table is populated in the database."
        )

    # Build confusion matrix
    y_true = [r.gt_pass for r in records]
    y_pred = [r.pred_pass for r in records]
    cc = _build_confusion(y_true, y_pred)

    # Bootstrap CIs
    stat_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "critical_miss_rate",
        "false_alarm_rate",
    ]
    arrays = _records_to_arrays(records)

    if small_n:
        print(_c("  Skipping bootstrap for n<10 (CIs will be point estimates only).", _DIM, tty=tty))
        cis: dict[str, BootstrapCI | None] = {}
        for name in stat_names:
            gt_b = [r.gt_pass for r in records]
            pred_b = [r.pred_pass for r in records]
            scores_b = [r.score for r in records]
            cc_b = _build_confusion(gt_b, pred_b)
            point: float | None = None
            if name == "accuracy":
                point = _accuracy(cc_b)
            elif name == "precision":
                point = _precision(cc_b)
            elif name == "recall":
                point = _recall(cc_b)
            elif name == "f1":
                point = _f1(cc_b)
            elif name == "auc_roc":
                point = _auc_roc(gt_b, scores_b)
            elif name == "critical_miss_rate":
                point = _critical_miss_rate(cc_b)
            elif name == "false_alarm_rate":
                point = _false_alarm_rate(cc_b)
            if point is not None:
                cis[name] = BootstrapCI(point=point, ci_low=point, ci_high=point, n_bootstrap=0, n_samples=n_labeled)
            else:
                cis[name] = None
    else:
        print(f"\n  Running bootstrap resampling (n_bootstrap={n_bootstrap})...")
        cis = _run_bootstrap(
            records,
            arrays,
            stat_names,
            pass_threshold,
            n_bootstrap=n_bootstrap,
            show_progress=show_progress,
        )

    # Score statistics
    scores_arr = np.array([r.score for r in records])
    score_mean: float | None = float(scores_arr.mean()) if len(scores_arr) > 0 else None
    score_std: float | None = float(scores_arr.std()) if len(scores_arr) > 0 else None

    # Inter-rater metrics
    interrater: dict[str, Any] = {}
    has_human_labels = bool(human_labels)
    if has_human_labels:
        matched_sys: list[float] = []
        matched_human: list[float] = []
        matched_y_true: list[bool] = []
        matched_y_pred: list[bool] = []
        matched_y_human: list[bool] = []

        for r in records:
            if r.human_score is not None:
                matched_sys.append(r.score)
                matched_human.append(r.human_score)
                matched_y_true.append(r.gt_pass)
                matched_y_pred.append(r.pred_pass)
                if human_labels and r.trainee_video_id in human_labels:
                    hverd = human_labels[r.trainee_video_id].human_verdict
                    matched_y_human.append(hverd == "pass")
                else:
                    # fallback: threshold on human score
                    matched_y_human.append(r.human_score >= pass_threshold)

        score_mae = _mae(matched_sys, matched_human) if matched_sys else None
        cohen_kappa = _cohen_kappa(matched_y_true, matched_y_pred) if matched_y_pred else None
        icc = _icc_two_way_mixed(matched_sys, matched_human) if len(matched_sys) >= 2 else None

        mcnemar_result: dict[str, Any] | None = None
        if len(matched_y_human) >= 2 and not small_n:
            mcnemar_result = _mcnemar_test(matched_y_true, matched_y_pred, matched_y_human)
        elif small_n:
            mcnemar_result = {"note": "skipped_small_n"}

        interrater = {
            "cohen_kappa": cohen_kappa,
            "icc": icc,
            "score_mae": score_mae,
            "mcnemar": mcnemar_result,
            "n_matched": len(matched_sys),
        }
    else:
        score_mae = None
        cohen_kappa = None
        icc = None
        mcnemar_result = None
        interrater = {}

    # Calibration
    calibration = _compute_calibration(records)

    return (
        records,
        cc,
        cis,
        calibration,
        interrater,
        gold_count,
        n_total_jobs,
        n_labeled,
        n_unknown_gt,
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _assemble_json_report(
    *,
    task_id: str | None,
    pass_threshold: float,
    eval_mode: str = "threshold",
    records: list[EvalRecord],
    cc: ConfusionCounts,
    cis: dict[str, BootstrapCI | None],
    interrater: dict[str, Any],
    calibration: list[CalibrationBin],
    recommendations: list[str],
    gold_count: int,
    n_total_jobs: int,
    n_labeled: int,
    n_unknown_gt: int,
    generated_at: str,
    db_path: str,
    n_bootstrap: int,
) -> dict[str, Any]:
    scores = [r.score for r in records]
    scores_arr = np.array(scores) if scores else np.array([])

    def _ci_dict(ci: BootstrapCI | None) -> dict[str, Any] | None:
        if ci is None:
            return None
        return {
            "point": round(ci.point, 6),
            "ci95_low": round(ci.ci_low, 6),
            "ci95_high": round(ci.ci_high, 6),
            "n_bootstrap": ci.n_bootstrap,
            "n_samples": ci.n_samples,
        }

    return {
        "schema_version": "loso_eval_v1",
        "generated_at": generated_at,
        "db_path": db_path,
        "task_id": task_id,
        "pass_threshold": pass_threshold,
        "eval_mode": eval_mode,
        "n_bootstrap": n_bootstrap,
        "data_summary": {
            "gold_videos": gold_count,
            "total_completed_trainee_jobs": n_total_jobs,
            "n_labeled": n_labeled,
            "n_unknown_gt": n_unknown_gt,
        },
        "confusion_matrix": {
            "tp": cc.tp,
            "tn": cc.tn,
            "fp": cc.fp,
            "fn": cc.fn,
            "n": cc.n,
        },
        "metrics": {
            name: _ci_dict(ci)
            for name, ci in cis.items()
        },
        "score_stats": {
            "mean": round(float(scores_arr.mean()), 4) if len(scores_arr) > 0 else None,
            "std": round(float(scores_arr.std()), 4) if len(scores_arr) > 0 else None,
            "p10": round(float(np.percentile(scores_arr, 10)), 4) if len(scores_arr) > 0 else None,
            "p50": round(float(np.percentile(scores_arr, 50)), 4) if len(scores_arr) > 0 else None,
            "p90": round(float(np.percentile(scores_arr, 90)), 4) if len(scores_arr) > 0 else None,
            "min": round(float(scores_arr.min()), 4) if len(scores_arr) > 0 else None,
            "max": round(float(scores_arr.max()), 4) if len(scores_arr) > 0 else None,
        },
        "interrater": interrater,
        "recommendations": recommendations,
        "records": [
            {
                "job_id": r.job_id,
                "trainee_video_id": r.trainee_video_id,
                "operator_id": r.operator_id,
                "site_id": r.site_id,
                "score": r.score,
                "pred_pass": r.pred_pass,
                "gt_pass": r.gt_pass,
                "human_score": r.human_score,
                "original_filename": r.original_filename,
                "outcome": (
                    "TP" if r.gt_pass and r.pred_pass
                    else "TN" if not r.gt_pass and not r.pred_pass
                    else "FP" if not r.gt_pass and r.pred_pass
                    else "FN"
                ),
            }
            for r in records
        ],
    }


def _assemble_calibration_json(
    calibration: list[CalibrationBin],
    *,
    pass_threshold: float,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": "loso_calibration_v1",
        "generated_at": generated_at,
        "pass_threshold": pass_threshold,
        "bins": [
            {
                "score_low": b.low,
                "score_high": b.high,
                "n_total": b.n_total,
                "n_pass": b.n_pass,
                "actual_pass_rate": round(b.pass_rate, 6) if b.pass_rate is not None else None,
                "predicted_pass_rate": round(b.pred_pass_rate, 6) if b.pred_pass_rate is not None else None,
            }
            for b in calibration
        ],
    }


# ---------------------------------------------------------------------------
# Terminal summary printer
# ---------------------------------------------------------------------------


def _print_terminal_summary(
    *,
    cc: ConfusionCounts,
    cis: dict[str, BootstrapCI | None],
    interrater: dict[str, Any],
    score_mean: float | None,
    score_std: float | None,
    recommendations: list[str],
    small_n_warning: bool,
    has_human_labels: bool,
    eval_mode: str = "threshold",
) -> None:
    tty = sys.stdout.isatty()
    mode_label = (
        "score-threshold (research)" if eval_mode == "threshold"
        else "product decision (operational)"
    )

    _print_section(f"LOSO Evaluation Results  [{mode_label}]", tty=tty)

    if small_n_warning:
        n = cc.n
        print(_c(f"  [!] Small sample warning: n={n} (< 10)", _YELLOW, tty=tty))

    print(f"\n  Confusion Matrix:")
    print(f"    TP={cc.tp}  FN={cc.fn}  FP={cc.fp}  TN={cc.tn}  (N={cc.n})")

    _print_section("Classification Metrics", tty=tty)
    _print_metric("Accuracy", cis.get("accuracy"), tty=tty)
    _print_metric("Precision", cis.get("precision"), tty=tty)
    _print_metric("Recall / Sensitivity", cis.get("recall"), tty=tty)
    _print_metric("F1 Score", cis.get("f1"), tty=tty)
    _print_metric("AUC-ROC", cis.get("auc_roc"), tty=tty)
    _print_metric("Critical Miss Rate (FNR)", cis.get("critical_miss_rate"), tty=tty, warn_if_above=0.10)
    _print_metric("False Alarm Rate (FPR)", cis.get("false_alarm_rate"), tty=tty, warn_if_above=0.30)

    if score_mean is not None:
        _print_section("Score Statistics", tty=tty)
        print(f"  Mean score: {score_mean:.2f}  |  Std: {score_std:.2f}")

    if has_human_labels and interrater:
        _print_section("Inter-Rater Reliability", tty=tty)
        kappa = interrater.get("cohen_kappa")
        icc = interrater.get("icc")
        mae = interrater.get("score_mae")
        mcn = interrater.get("mcnemar")
        if kappa is not None:
            print(f"  Cohen's kappa:         {kappa:.4f}")
        if icc is not None:
            print(f"  ICC(2,1):              {icc:.4f}")
        if mae is not None:
            print(f"  Score MAE vs. human:   {mae:.2f}")
        if mcn and isinstance(mcn, dict) and "p_value" in mcn:
            sig_str = _c("SIGNIFICANT", _YELLOW, tty=tty) if mcn.get("significant_at_0.05") else "not significant"
            print(f"  McNemar's p-value:     {mcn['p_value']:.4f}  ({sig_str})")

    _print_section("Recommendations", tty=tty)
    for i, rec in enumerate(recommendations, 1):
        # Wrap long recommendation lines
        wrapped = textwrap.fill(rec, width=72, subsequent_indent="     ")
        print(f"  {i}. {wrapped}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone LOSO evaluation for SOPilot — reads directly from SQLite DB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        required=True,
        metavar="PATH",
        help="Path to sopilot.db SQLite database.",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        metavar="ID",
        help="Optional task_id to filter evaluation (e.g. filter_change).",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=70.0,
        metavar="SCORE",
        help="Score threshold (0-100) above which a video is predicted as PASS.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/loso_eval",
        metavar="DIR",
        help="Directory where output files are written.",
    )
    parser.add_argument(
        "--human-labels",
        default=None,
        metavar="PATH",
        help="Optional JSON file with human annotations (list of label objects).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        metavar="N",
        help="Number of bootstrap resamples for CI computation.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress the bootstrap progress bar.",
    )
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print discovered DB table schema and exit.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["threshold", "product", "both"],
        default="threshold",
        metavar="MODE",
        help=(
            "Prediction mode for evaluation. "
            "'threshold' (default): pred=pass if score >= pass-threshold — evaluates the raw "
            "scoring algorithm independently of threshold-at-scoring-time (research mode). "
            "'product': pred=pass only if stored decision field == 'pass' — evaluates the full "
            "system including critical-deviation overrides (operational mode). "
            "'both': run both modes and print a side-by-side comparison."
        ),
    )
    return parser.parse_args(argv)


def _print_dual_mode_comparison(
    all_results: list[tuple],
    *,
    pass_threshold: float,
    tty: bool,
) -> None:
    """Print a side-by-side metric comparison for threshold vs product modes."""
    _print_section("Dual-Mode Comparison", tty=tty)
    print(
        "  score-threshold  Evaluates raw scoring algorithm's discriminative power.\n"
        "                   pred=pass iff score >= pass_threshold.\n"
        "  product decision Evaluates the full operational system, including critical-\n"
        "                   deviation overrides that force fail regardless of score.\n"
        "                   pred=pass iff stored decision field == 'pass'.\n"
    )

    labels = {
        "threshold": f"threshold (score≥{pass_threshold:.0f})",
        "product":   "product  (decision field)",
    }
    header = f"  {'Mode':<32} {'Accuracy':>10} {'F1':>8} {'AUC':>8} {'Miss Rate':>12}  TP/FN/FP/TN"
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)

    for entry in all_results:
        mode, records, cc, cis, *_ = entry
        acc_ci = cis.get("accuracy")
        f1_ci = cis.get("f1")
        auc_ci = cis.get("auc_roc")
        miss_ci = cis.get("critical_miss_rate")

        label = labels.get(mode, mode)
        acc_s = f"{acc_ci.point:.2%}" if acc_ci else "N/A"
        f1_s = f"{f1_ci.point:.4f}" if f1_ci else "N/A"
        auc_s = f"{auc_ci.point:.4f}" if auc_ci else "N/A"
        miss_s = f"{miss_ci.point:.2%}" if miss_ci else "N/A"
        cm_s = f"{cc.tp}/{cc.fn}/{cc.fp}/{cc.tn}"
        print(f"  {label:<32} {acc_s:>10} {f1_s:>8} {auc_s:>8} {miss_s:>12}  {cm_s}")

    print()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    tty = sys.stdout.isatty()
    show_progress = not args.no_progress

    db_path = Path(args.db_path).resolve()
    out_dir = Path(args.output_dir).resolve()

    print(_c("\nSOPilot LOSO Evaluation", _BOLD, _CYAN, tty=tty))
    print(_c("=" * 60, _CYAN, tty=tty))
    print(f"  DB:            {db_path}")
    print(f"  Task ID:       {args.task_id or '(all)'}")
    print(f"  Pass threshold:{args.pass_threshold}")
    print(f"  Eval mode:     {args.eval_mode}")
    print(f"  Output dir:    {out_dir}")
    print(f"  Bootstrap n:   {args.n_bootstrap}")
    print(f"  Human labels:  {args.human_labels or '(none)'}")

    conn = _open_db(db_path)

    if args.print_schema:
        schema = _get_schema(conn)
        print("\n--- DB Schema ---")
        for name, sql in schema.items():
            print(f"\n[{name}]\n{sql}")
        conn.close()
        return

    # Warn if scipy not available (DeLong / chi2 will use fallback)
    if not _SCIPY_AVAILABLE:
        print(
            _c(
                "  WARNING: scipy not installed — McNemar p-value uses normal approximation.",
                _YELLOW,
                tty=tty,
            )
        )

    # Load optional human labels
    human_labels: dict[int, HumanLabel] | None = None
    if args.human_labels:
        print(f"  Loading human labels from {args.human_labels}...")
        human_labels = _load_human_labels(Path(args.human_labels))
        print(f"  Loaded {len(human_labels)} human label(s).")

    generated_at = datetime.now(timezone.utc).isoformat()

    # Determine which evaluation modes to run
    eval_modes_to_run: list[str] = (
        ["threshold", "product"] if args.eval_mode == "both" else [args.eval_mode]
    )

    # Run all evaluations before closing the DB connection
    all_eval_results: list[tuple] = []
    for mode in eval_modes_to_run:
        _print_section(f"Running Evaluation  [{mode} mode]", tty=tty)
        result = _run_evaluation(
            conn,
            task_id=args.task_id,
            pass_threshold=args.pass_threshold,
            human_labels=human_labels,
            n_bootstrap=args.n_bootstrap,
            show_progress=show_progress,
            eval_mode=mode,
        )
        all_eval_results.append((mode, *result))

    conn.close()

    out_dir.mkdir(parents=True, exist_ok=True)

    for entry in all_eval_results:
        mode, records, cc, cis, calibration, interrater, gold_count, n_total_jobs, n_labeled, n_unknown_gt = entry

        scores = [r.score for r in records]
        scores_arr = np.array(scores) if scores else np.array([])
        score_mean: float | None = float(scores_arr.mean()) if len(scores_arr) > 0 else None
        score_std: float | None = float(scores_arr.std()) if len(scores_arr) > 0 else None
        small_n_warning = n_labeled < 10
        has_human_labels = bool(human_labels)

        cohen_kappa = interrater.get("cohen_kappa") if interrater else None
        icc = interrater.get("icc") if interrater else None
        score_mae = interrater.get("score_mae") if interrater else None
        mcnemar_result = interrater.get("mcnemar") if interrater else None

        # Generate recommendations
        recommendations = _generate_recommendations(
            cc,
            cis,
            n=n_labeled,
            pass_threshold=args.pass_threshold,
            has_human_labels=has_human_labels,
            score_std=score_std,
            score_mean=score_mean,
        )

        # Print terminal summary
        _print_terminal_summary(
            cc=cc,
            cis=cis,
            interrater=interrater,
            score_mean=score_mean,
            score_std=score_std,
            recommendations=recommendations,
            small_n_warning=small_n_warning,
            has_human_labels=has_human_labels,
            eval_mode=mode,
        )

        # Assemble output artefacts
        json_report = _assemble_json_report(
            task_id=args.task_id,
            pass_threshold=args.pass_threshold,
            eval_mode=mode,
            records=records,
            cc=cc,
            cis=cis,
            interrater=interrater,
            calibration=calibration,
            recommendations=recommendations,
            gold_count=gold_count,
            n_total_jobs=n_total_jobs,
            n_labeled=n_labeled,
            n_unknown_gt=n_unknown_gt,
            generated_at=generated_at,
            db_path=str(db_path),
            n_bootstrap=args.n_bootstrap,
        )

        html_report = _build_html_report(
            task_id=args.task_id,
            pass_threshold=args.pass_threshold,
            eval_mode=mode,
            records=records,
            cc=cc,
            cis=cis,
            calibration=calibration,
            recommendations=recommendations,
            score_mean=score_mean,
            score_std=score_std,
            score_mae=score_mae,
            cohen_kappa=cohen_kappa,
            icc=icc,
            mcnemar=mcnemar_result,
            gold_count=gold_count,
            n_total_jobs=n_total_jobs,
            n_labeled=n_labeled,
            n_unknown_gt=n_unknown_gt,
            has_human_labels=has_human_labels,
            small_n_warning=small_n_warning,
            generated_at=generated_at,
        )

        cal_json = _assemble_calibration_json(
            calibration,
            pass_threshold=args.pass_threshold,
            generated_at=generated_at,
        )

        # File naming: add _threshold / _product suffix when running both modes
        suffix = f"_{mode}" if args.eval_mode == "both" else ""
        report_json_path = out_dir / f"loso_eval_report{suffix}.json"
        report_html_path = out_dir / f"loso_eval_report{suffix}.html"
        cal_json_path = out_dir / f"loso_eval_calibration{suffix}.json"

        report_json_path.write_text(json.dumps(json_report, ensure_ascii=False, indent=2), encoding="utf-8")
        report_html_path.write_text(html_report, encoding="utf-8")
        cal_json_path.write_text(json.dumps(cal_json, ensure_ascii=False, indent=2), encoding="utf-8")

        _print_section("Output Files Written", tty=tty)
        print(f"  {_c('JSON report:', _BOLD, tty=tty)}  {report_json_path}")
        print(f"  {_c('HTML report:', _BOLD, tty=tty)}  {report_html_path}")
        print(f"  {_c('Calibration:', _BOLD, tty=tty)}  {cal_json_path}")

    # For 'both' mode: print a side-by-side comparison table
    if args.eval_mode == "both" and len(all_eval_results) == 2:
        _print_dual_mode_comparison(all_eval_results, pass_threshold=args.pass_threshold, tty=tty)

    print()


if __name__ == "__main__":
    main()
