"""Pipeline result persistence — JSON and HTML generation."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from insurance_mvp.insurance.schema import ClaimAssessment
from insurance_mvp.report_generator import ReportGenerator
from insurance_mvp.serialization import clip_to_dict, to_serializable

logger = logging.getLogger(__name__)


def save_checkpoint(
    output_dir: str,
    video_id: str,
    assessments: list[ClaimAssessment],
) -> None:
    """Save intermediate checkpoint for incremental progress."""
    out = Path(output_dir) / video_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / "checkpoint.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump([to_serializable(a) for a in assessments], f, indent=2, default=str)


def save_results(
    output_dir: str,
    video_id: str,
    danger_clips: list[Any],
    assessments: list[ClaimAssessment],
    config: Any,
) -> tuple[str, str]:
    """Save JSON and HTML results.

    Returns:
        (json_path, html_path) as strings.
    """
    out = Path(output_dir) / video_id
    out.mkdir(parents=True, exist_ok=True)

    # --- JSON ---
    json_path = out / "results.json"
    results_dict = {
        "video_id": video_id,
        "timestamp": datetime.utcnow().isoformat(),
        "config": to_serializable(config),
        "danger_clips": [clip_to_dict(c) for c in danger_clips],
        "assessments": [to_serializable(a) for a in assessments],
        "summary": generate_summary(assessments),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, default=str, ensure_ascii=False)
    logger.info("Saved JSON results: %s", json_path)

    # --- HTML ---
    html_path = out / "report.html"
    try:
        report_gen = ReportGenerator()
        if assessments:
            report_gen.generate_multi_clip_report(
                assessments=assessments, video_id=video_id, output_path=html_path
            )
            logger.info("Saved professional HTML report: %s", html_path)
        else:
            _write_fallback_html(html_path, video_id, [])
    except Exception as e:
        logger.warning("Professional report generation failed (%s), using fallback", e)
        _write_fallback_html(html_path, video_id, assessments)

    return str(json_path), str(html_path)


def generate_summary(assessments: list[ClaimAssessment]) -> dict[str, Any]:
    """Compute summary statistics from a list of assessments."""
    if not assessments:
        return {}

    severity_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    for a in assessments:
        severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1
        priority_counts[a.review_priority] = priority_counts.get(a.review_priority, 0) + 1

    return {
        "total_clips": len(assessments),
        "severity_distribution": severity_counts,
        "priority_distribution": priority_counts,
        "avg_confidence": sum(a.confidence for a in assessments) / len(assessments),
        "avg_fault_ratio": sum(a.fault_assessment.fault_ratio for a in assessments) / len(assessments),
        "avg_fraud_score": sum(a.fraud_risk.risk_score for a in assessments) / len(assessments),
    }


def _write_fallback_html(
    html_path: Path,
    video_id: str,
    assessments: list[ClaimAssessment],
) -> None:
    """Simple HTML fallback when ReportGenerator is unavailable."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Insurance Assessment Report - {video_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .HIGH {{ background-color: #ffcccc; }}
        .MEDIUM {{ background-color: #ffffcc; }}
        .LOW {{ background-color: #ccffcc; }}
        .URGENT {{ font-weight: bold; color: red; }}
        .STANDARD {{ color: orange; }}
        .LOW_PRIORITY {{ color: green; }}
    </style>
</head>
<body>
    <h1>Insurance Assessment Report</h1>
    <p><strong>Video ID:</strong> {video_id}</p>
    <p><strong>Generated:</strong> {datetime.utcnow().isoformat()}</p>
    <p><strong>Total Clips:</strong> {len(assessments)}</p>

    <h2>Assessments</h2>
    <table>
        <tr>
            <th>#</th><th>Severity</th><th>Confidence</th>
            <th>Prediction Set</th><th>Priority</th>
            <th>Fault Ratio</th><th>Fraud Score</th><th>Action</th>
        </tr>
"""
    for i, a in enumerate(assessments, 1):
        pred_set_str = ", ".join(sorted(a.prediction_set))
        html += (
            f'        <tr class="{a.severity}">'
            f"<td>{i}</td><td>{a.severity}</td>"
            f"<td>{a.confidence:.2f}</td>"
            f"<td>{{{pred_set_str}}}</td>"
            f'<td class="{a.review_priority}">{a.review_priority}</td>'
            f"<td>{a.fault_assessment.fault_ratio:.1f}%</td>"
            f"<td>{a.fraud_risk.risk_score:.2f}</td>"
            f"<td>{a.recommended_action}</td></tr>\n"
        )
    html += "    </table>\n</body>\n</html>\n"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Saved fallback HTML report: %s", html_path)
