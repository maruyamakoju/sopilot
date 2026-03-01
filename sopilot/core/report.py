"""HTML audit report generation for score jobs.

Uses a Jinja2 template (``templates/report.html``) instead of inline
f-strings so that the layout is editable without touching Python code.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from sopilot import __version__

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=True,
)

# ── Decision colour map ──────────────────────────────────────────────────────
_DEC_MAP: dict[str, tuple[str, str, str, str]] = {
    "pass":         ("#10b981", "rgba(16,185,129,.12)", "rgba(16,185,129,.3)", "合格"),
    "fail":         ("#ef4444", "rgba(239,68,68,.12)",  "rgba(239,68,68,.3)",  "不合格"),
    "needs_review": ("#3b82f6", "rgba(59,130,246,.12)", "rgba(59,130,246,.3)", "要確認"),
    "retrain":      ("#f59e0b", "rgba(245,158,11,.12)", "rgba(245,158,11,.3)", "再研修"),
}

_SEV_COLOR: dict[str, str] = {"critical": "#ef4444", "quality": "#f59e0b", "efficiency": "#3b82f6"}
_SEV_JP: dict[str, str] = {"critical": "重大", "quality": "品質", "efficiency": "効率"}
_TYPE_JP: dict[str, str] = {
    "missing_step": "手順省略", "step_deviation": "品質逸脱",
    "order_swap": "手順入替", "over_time": "時間超過",
}
_VERDICT_JP: dict[str, str] = {"pass": "合格", "fail": "不合格", "needs_review": "要確認", "retrain": "再研修"}


def build_report_html(job_id: int, job: dict[str, Any]) -> str:
    """Render a self-contained HTML page suitable for ``window.print()`` -> PDF."""
    result = job.get("result") or {}
    review_raw = job.get("review") or {}

    score_raw = result.get("score")
    score_s = str(score_raw) if score_raw is not None else "\u2014"
    summary = result.get("summary") or {}
    metrics = result.get("metrics") or {}
    deviations_raw: list[dict[str, Any]] = result.get("deviations") or []
    decision = summary.get("decision", "\u2014")
    sev_counts = summary.get("severity_counts") or {}

    dec_color, dec_bg, dec_border, decision_jp = _DEC_MAP.get(
        decision, ("#94a3b8", "rgba(148,163,184,.12)", "rgba(148,163,184,.3)", decision),
    )

    # Metric helpers
    miss = int(metrics.get("miss_steps", 0) or 0)
    swap = int(metrics.get("swap_steps", 0) or 0)
    dev = int(metrics.get("deviation_steps", 0) or 0)
    ot = float(metrics.get("over_time_ratio", 0.0) or 0.0)
    dtw_v = metrics.get("dtw_normalized_cost")

    metric_rows = [
        {
            "label": "手順遺漏数 (Miss)", "value": miss,
            "color": "#ef4444" if miss > 0 else "#10b981",
            "standard": "0 が理想",
            "verdict": "\u26a0 あり" if miss > 0 else "\u2713 なし",
        },
        {
            "label": "手順入替数 (Swap)", "value": swap,
            "color": "#f59e0b" if swap > 0 else "#10b981",
            "standard": "0 が理想",
            "verdict": "\u26a0 あり" if swap > 0 else "\u2713 なし",
        },
        {
            "label": "品質逸脱数 (Dev)", "value": dev,
            "color": "#f59e0b" if dev > 0 else "#10b981",
            "standard": "0 が理想",
            "verdict": "\u26a0 あり" if dev > 0 else "\u2713 なし",
        },
        {
            "label": "時間超過率 (Over-time)", "value": f"{ot:.3f}",
            "color": "#3b82f6" if ot > 0.2 else "#10b981",
            "standard": "\u2264 0.200",
            "verdict": "\u26a0 超過" if ot > 0.2 else "\u2713 範囲内",
        },
        {
            "label": "DTW 正規化コスト",
            "value": f"{dtw_v:.4f}" if dtw_v is not None else "\u2014",
            "color": "#1e293b", "standard": "参考値", "verdict": "\u2014",
        },
    ]

    # Severity boxes
    crit_n = int(sev_counts.get("critical", 0) or 0)
    qual_n = int(sev_counts.get("quality", 0) or 0)
    eff_n = int(sev_counts.get("efficiency", 0) or 0)
    severity_boxes = [
        {
            "count": crit_n, "label": "Critical 逸脱",
            "color": "#ef4444" if crit_n > 0 else "#10b981",
            "bg": "rgba(239,68,68,.08)" if crit_n > 0 else "#f8fafc",
            "border": "rgba(239,68,68,.3)" if crit_n > 0 else "#e2e8f0",
        },
        {
            "count": qual_n, "label": "Quality 逸脱",
            "color": "#f59e0b" if qual_n > 0 else "#10b981",
            "bg": "#f8fafc", "border": "#e2e8f0",
        },
        {
            "count": eff_n, "label": "Efficiency 逸脱",
            "color": "#3b82f6" if eff_n > 0 else "#10b981",
            "bg": "#f8fafc", "border": "#e2e8f0",
        },
    ]

    # Build deviation rows
    deviations = []
    for dv in deviations_raw:
        sev = dv.get("severity", "quality")
        gt = dv.get("gold_timecode") or []
        tt = dv.get("trainee_timecode") or []
        tc = f"Gold: {gt[0]}s\u2013{gt[1]}s" if len(gt) == 2 else ""
        if len(tt) == 2:
            tc += f" / 研修生: {tt[0]}s\u2013{tt[1]}s"
        deviations.append({
            "sev_color": _SEV_COLOR.get(sev, "#94a3b8"),
            "sev_jp": _SEV_JP.get(sev, sev),
            "type_jp": _TYPE_JP.get(dv.get("type", ""), dv.get("type", "")),
            "detail": str(dv.get("detail", "") or ""),
            "timecode": tc,
        })

    # Review context
    review_ctx = None
    if review_raw:
        review_ctx = {
            "verdict_jp": _VERDICT_JP.get(review_raw.get("verdict", ""), review_raw.get("verdict", "\u2014")),
            "note": str(review_raw.get("note", "") or "\u2014"),
            "updated_at": str(review_raw.get("updated_at", "") or "\u2014"),
        }

    template = _env.get_template("report.html")
    return template.render(
        job_id=job_id,
        version=__version__,
        now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        score=score_s,
        dec_color=dec_color,
        dec_bg=dec_bg,
        dec_border=dec_border,
        decision_jp=decision_jp,
        task_id=str(result.get("task_id", "\u2014") or "\u2014"),
        dec_reason=str(summary.get("decision_reason", "") or ""),
        gold_id=result.get("gold_video_id", "\u2014"),
        trainee_id=result.get("trainee_video_id", "\u2014"),
        pass_score=summary.get("pass_score", "\u2014"),
        metric_rows=metric_rows,
        severity_boxes=severity_boxes,
        deviations=deviations,
        review=review_ctx,
    )
