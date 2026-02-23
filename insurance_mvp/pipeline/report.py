"""Render the multi-clip HTML report from a VideoResult.

Usage::

    from insurance_mvp.pipeline.report import render_multi_clip_report
    path = render_multi_clip_report(video_result, Path("output/report.html"))
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from insurance_mvp.pipeline.orchestrator import VideoResult

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

# Severity ordering: worst first
_SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}


def _derive_at_fault_party(fault_ratio: float) -> str:
    if fault_ratio >= 75.0:
        return "Ego Vehicle"
    elif fault_ratio <= 25.0:
        return "Other Vehicle"
    return "Shared"


def _derive_fraud_level(risk_score: float) -> str:
    if risk_score >= 0.65:
        return "HIGH"
    elif risk_score >= 0.4:
        return "MEDIUM"
    return "LOW"


def render_multi_clip_report(
    result: VideoResult,
    output_path: str | Path,
    template_name: str = "multi_clip_report.html",
    lang: str = "en",
) -> Path:
    """Render a standalone multi-clip HTML report from a VideoResult.

    Args:
        result: A ``VideoResult`` produced by ``InsurancePipeline.process_video``.
        output_path: File path where the HTML report will be written.
        template_name: Jinja2 template file inside the templates directory.
        lang: Language code ("en" or "ja").

    Returns:
        The resolved ``Path`` of the written report file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)))
    # Identity translate function (report.py doesn't do i18n)
    env.globals["_"] = lambda text: text
    template = env.get_template(template_name)

    # Build per-clip data matching the template contract
    clips_data = []
    for a in (result.assessments or []):
        clips_data.append({
            "severity": a.severity,
            "confidence": int(a.confidence * 100),
            "fault_ratio": a.fault_assessment.fault_ratio,
            "at_fault_party": _derive_at_fault_party(a.fault_assessment.fault_ratio),
            "fraud_score": f"{a.fraud_risk.risk_score:.2f}",
            "fraud_level": _derive_fraud_level(a.fraud_risk.risk_score),
            "scenario_type": a.fault_assessment.scenario_type or "unknown",
            "causal_reasoning": a.causal_reasoning or "No causal reasoning available",
            "fault_reasoning": a.fault_assessment.reasoning or "No reasoning provided",
            "red_flags": a.fraud_risk.indicators,
            "review_priority": a.review_priority,
            "review_priority_class": a.review_priority.lower(),
            "recommended_action": a.recommended_action,
            "processing_time": f"{a.processing_time_sec:.1f}",
        })

    # Sort by severity (worst first) then confidence descending
    clips_data.sort(key=lambda c: (_SEVERITY_ORDER.get(c["severity"], 4), -c["confidence"]))

    # Determine overall stats
    if clips_data:
        highest_severity = clips_data[0]["severity"]
        highest_confidence = clips_data[0]["confidence"]
    else:
        highest_severity = "NONE"
        highest_confidence = 0

    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}
    for c in clips_data:
        severity_counts[c["severity"]] = severity_counts.get(c["severity"], 0) + 1

    overall_recommendation = clips_data[0]["recommended_action"] if clips_data else "REVIEW"

    # Build summary text
    parts = [f"{v} {k}" for k, v in severity_counts.items() if v > 0]
    breakdown = ", ".join(parts)
    if highest_severity == "NONE":
        overall_summary = f"Analyzed {len(clips_data)} clips. No incidents detected. Breakdown: {breakdown}."
    else:
        overall_summary = (
            f"Analyzed {len(clips_data)} clips. "
            f"Highest severity: {highest_severity}. Breakdown: {breakdown}. "
            f"Recommended action: {overall_recommendation}."
        )

    html = template.render(
        lang=lang,
        video_id=result.video_id,
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        total_clips=len(clips_data),
        total_processing_time=f"{result.processing_time_sec:.1f}",
        highest_severity=highest_severity,
        highest_severity_confidence=highest_confidence,
        severity_counts=severity_counts,
        overall_recommendation=overall_recommendation,
        overall_summary=overall_summary,
        clips=clips_data,
    )

    output_path.write_text(html, encoding="utf-8")
    logger.info("Saved multi-clip HTML report: %s", output_path)
    return output_path
