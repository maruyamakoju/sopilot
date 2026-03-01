"""Japanese and English template comments for SOP deviations.

Provides localized, human-readable descriptions for each deviation
type and severity, suitable for display in the UI and PDF reports.

Design goal: the operator who reads the deviation should immediately
understand *what* went wrong and *how important* it is, without
needing to interpret raw metric values.
"""

from __future__ import annotations

# ── Deviation type → template ──────────────────────────────────────

_TYPE_TEMPLATES_JA: dict[str, str] = {
    "missing_step": "手順{step}（{name}）が実施されていません",
    "step_deviation": "手順{step}（{name}）の動作が基準と異なります",
    "order_swap": "手順{step}と前の手順の順序が入れ替わっています",
    "over_time": "作業全体の所要時間が基準を{ratio}超過しています",
}

_TYPE_TEMPLATES_EN: dict[str, str] = {
    "missing_step": "Step {step} ({name}) was not performed",
    "step_deviation": "Step {step} ({name}) deviates from the standard",
    "order_swap": "Step {step} performed out of order",
    "over_time": "Total work time exceeds the standard by {ratio}",
}


# ── Severity → labels and descriptions ────────────────────────────

_SEVERITY_JA: dict[str, str] = {
    "critical": "重大",
    "quality": "品質",
    "efficiency": "効率",
}

_SEVERITY_DESC_JA: dict[str, str] = {
    "critical": "安全・品質に直結する重大な逸脱です。即座に是正が必要です。",
    "quality": "品質に影響する可能性がある逸脱です。改善を推奨します。",
    "efficiency": "効率性に関する逸脱です。作業時間の見直しを検討してください。",
}

_SEVERITY_DESC_EN: dict[str, str] = {
    "critical": "Critical deviation affecting safety or quality. Immediate correction required.",
    "quality": "Quality-impacting deviation. Improvement recommended.",
    "efficiency": "Efficiency-related deviation. Consider reviewing work time.",
}


# ── Public API ─────────────────────────────────────────────────────

def annotate_deviation(
    deviation: dict,
    step_definitions: list[dict] | None = None,
) -> dict:
    """Enrich a single deviation dict with template comments.

    Adds to the deviation dict (in-place):
        comment_ja:              Japanese human-readable comment
        comment_en:              English human-readable comment
        severity_ja:             Japanese severity label (重大/品質/効率)
        severity_description_ja: Japanese severity explanation
        severity_description_en: English severity explanation

    Args:
        deviation: A deviation dict from ``score_alignment()``.
        step_definitions: Optional list of step definition dicts for
            resolving step names.

    Returns:
        The same deviation dict, modified in-place.
    """
    dev_type = deviation.get("type", "")
    step_idx = deviation.get("step_index")
    severity = deviation.get("severity", "quality")

    # Resolve step name from definitions
    step_name_ja = f"手順{step_idx + 1}" if step_idx is not None else "不明"
    step_name_en = f"Step {step_idx + 1}" if step_idx is not None else "Unknown"
    if step_definitions and step_idx is not None:
        for defn in step_definitions:
            if defn.get("step_index") == step_idx:
                step_name_ja = defn.get("name_ja", step_name_ja)
                step_name_en = defn.get("name_en", step_name_en)
                break

    # Format template variables
    step_num = str(step_idx + 1) if step_idx is not None else "?"
    over_time_ratio = deviation.get("over_time_ratio")
    ratio_str = f"{over_time_ratio:.0%}" if over_time_ratio else "基準超過"
    ratio_str_en = f"{over_time_ratio:.0%}" if over_time_ratio else "excess"

    template_ja = _TYPE_TEMPLATES_JA.get(dev_type, "不明な逸脱が検出されました")
    template_en = _TYPE_TEMPLATES_EN.get(dev_type, "Unknown deviation detected")

    deviation["comment_ja"] = template_ja.format(
        step=step_num, name=step_name_ja, ratio=ratio_str,
    )
    deviation["comment_en"] = template_en.format(
        step=step_num, name=step_name_en, ratio=ratio_str_en,
    )
    deviation["severity_ja"] = _SEVERITY_JA.get(severity, severity)
    deviation["severity_description_ja"] = _SEVERITY_DESC_JA.get(severity, "")
    deviation["severity_description_en"] = _SEVERITY_DESC_EN.get(severity, "")

    return deviation


def annotate_deviations(
    deviations: list[dict],
    step_definitions: list[dict] | None = None,
) -> list[dict]:
    """Annotate all deviations in a list with template comments."""
    for dev in deviations:
        annotate_deviation(dev, step_definitions)
    return deviations


def generate_summary_comment(
    score: float,
    decision: str,
    severity_counts: dict[str, int] | None = None,
) -> dict[str, str]:
    """Generate a one-line summary comment for a score result.

    Returns:
        Dict with ``ja`` and ``en`` keys containing localized summaries.
    """
    counts = severity_counts or {}
    critical = counts.get("critical", 0)
    quality = counts.get("quality", 0)
    efficiency = counts.get("efficiency", 0)

    if decision == "pass" and critical == 0 and quality == 0:
        ja = f"スコア {score:.1f}点 — 基準を満たしています。"
        en = f"Score {score:.1f} — Meets the standard."
    elif decision == "pass":
        parts_ja: list[str] = []
        parts_en: list[str] = []
        if quality > 0:
            parts_ja.append(f"品質逸脱{quality}件")
            parts_en.append(f"{quality} quality deviation(s)")
        if efficiency > 0:
            parts_ja.append(f"効率逸脱{efficiency}件")
            parts_en.append(f"{efficiency} efficiency deviation(s)")
        ja = f"スコア {score:.1f}点 — 合格ですが、{'・'.join(parts_ja)}の改善を推奨します。"
        en = f"Score {score:.1f} — Pass, but {', '.join(parts_en)} noted for improvement."
    elif decision == "fail" and critical > 0:
        ja = f"スコア {score:.1f}点 — 重大逸脱{critical}件が検出されました。再教育が必要です。"
        en = f"Score {score:.1f} — {critical} critical deviation(s) detected. Retraining required."
    elif decision == "needs_review":
        ja = f"スコア {score:.1f}点 — 合否判定基準の間です。管理者レビューが必要です。"
        en = f"Score {score:.1f} — Between pass/fail thresholds. Supervisor review required."
    elif decision == "retrain":
        ja = f"スコア {score:.1f}点 — 再訓練が必要です。"
        en = f"Score {score:.1f} — Retraining required."
    elif decision == "fail":
        ja = f"スコア {score:.1f}点 — 不合格です。"
        en = f"Score {score:.1f} — Fail."
    else:
        ja = f"スコア {score:.1f}点"
        en = f"Score {score:.1f}"

    return {"ja": ja, "en": en}
