#!/usr/bin/env python3
"""SOPilot Manufacturing Pilot - Production-Ready Evaluation.

Customer-facing tool that evaluates trainee SOP videos and generates
actionable reports with deviations, timestamps, and corrective actions.

This is the commercial product that customers receive:
- One command execution
- Clear pass/fail verdict
- Timestamped deviations with explanations
- Actionable corrective guidance
- Performance metrics (speed demonstration)

Usage:
    python scripts/sopilot_evaluate_pilot.py \
        --gold demo_videos/manufacturing/oil_change_gold.mp4 \
        --trainee demo_videos/manufacturing/oil_change_trainee_1.mp4 \
        --sop oil_change \
        --out report.json

    # PDF report (requires reportlab)
    python scripts/sopilot_evaluate_pilot.py \
        --gold demo_videos/manufacturing/oil_change_gold.mp4 \
        --trainee demo_videos/manufacturing/oil_change_trainee_1.mp4 \
        --sop oil_change \
        --out report.pdf

Output Format:
    {
        "overall": {"pass": false, "score": 65.0, "threshold": 80.0},
        "deviations": [
            {
                "type": "missing_step",
                "step": "SAFETY",
                "timestamp": "0:00-0:10",
                "severity": "critical",
                "description": "Safety equipment not worn before starting",
                "evidence": "Expected safety glasses and gloves at 0:10"
            }
        ],
        "corrective_actions": [
            "Ensure trainee wears safety glasses and gloves before starting",
            "Review safety protocol video before next attempt"
        ],
        "evaluation_time": 2.3,
        "metadata": {"sop": "oil_change", "trainee_id": "trainee_1"}
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from sopilot.embeddings import HeuristicClipEmbedder  # noqa: E402
from sopilot.step_engine import evaluate_sop  # noqa: E402
from sopilot.video import ClipWindowStream  # noqa: E402

# SOP Template Definitions
# Each SOP has: steps list, critical steps (must not skip), tool requirements, pass threshold
SOP_TEMPLATES = {
    "oil_change": {
        "name": "Oil Change Procedure",
        "steps": [
            "PARK",
            "SAFETY",
            "LIFT",
            "LOCATE",
            "DRAIN",
            "FILTER",
            "INSTALL_FILTER",
            "REINSTALL_PLUG",
            "FILL",
            "CHECK",
        ],
        "critical_steps": ["SAFETY", "CHECK"],  # Must not be skipped
        "step_descriptions": {
            "PARK": "Park vehicle on level surface",
            "SAFETY": "Put on safety glasses and gloves",
            "LIFT": "Lift vehicle with jack and stands",
            "LOCATE": "Locate oil drain plug under engine",
            "DRAIN": "Place drain pan and remove plug",
            "FILTER": "Remove old oil filter with wrench",
            "INSTALL_FILTER": "Install new filter (hand-tight)",
            "REINSTALL_PLUG": "Reinstall drain plug with torque wrench",
            "FILL": "Add new oil through filler cap",
            "CHECK": "Check oil level with dipstick",
        },
        "tool_requirements": {
            "FILTER": "wrench",
            "REINSTALL_PLUG": "torque_wrench",
        },
        "threshold": 80.0,
    },
    "brake_pads": {
        "name": "Brake Pad Replacement",
        "steps": ["SAFETY", "JACK", "WHEEL", "CALIPER", "PADS", "INSTALL", "TORQUE", "CHECK"],
        "critical_steps": ["SAFETY", "TORQUE", "CHECK"],
        "step_descriptions": {
            "SAFETY": "Don safety glasses and work gloves",
            "JACK": "Lift vehicle with hydraulic jack and secure stands",
            "WHEEL": "Remove wheel bolts and detach tire",
            "CALIPER": "Remove caliper bolts and lift assembly away",
            "PADS": "Remove old brake pads from caliper mounting",
            "INSTALL": "Install new pads and apply anti-rattle compound",
            "TORQUE": "Apply torque wrench to caliper mounting bolts (85 Nm)",
            "CHECK": "Test brake pedal and verify system response",
        },
        "tool_requirements": {
            "CALIPER": "wrench",
            "TORQUE": "torque_wrench",
        },
        "threshold": 85.0,
    },
    "ppe_check": {
        "name": "PPE (Personal Protective Equipment) Check",
        "steps": ["HELMET", "GLASSES", "GLOVES", "BOOTS", "VEST"],
        "critical_steps": ["HELMET", "GLASSES", "GLOVES"],  # Core safety items
        "step_descriptions": {
            "HELMET": "Inspect and don safety helmet securely",
            "GLASSES": "Put on safety glasses or face shield",
            "GLOVES": "Check and put on appropriate work gloves",
            "BOOTS": "Verify steel-toed safety boots are worn",
            "VEST": "Wear high-visibility safety vest properly",
        },
        "tool_requirements": {},
        "threshold": 90.0,  # PPE is strict
    },
}


@dataclass
class DeviationReport:
    """Deviation detected during evaluation."""

    deviation_type: str  # missing_step, order_swap, execution_deviation, wrong_tool
    step: str | None  # Step identifier (SAFETY, FILTER, etc.)
    timestamp: str  # Human-readable timestamp (e.g., "0:10-0:20")
    severity: str  # critical, high, medium, low
    description: str  # Human-readable description
    evidence: str  # Supporting evidence for the deviation


def _format_timestamp(start_sec: float | None, end_sec: float | None) -> str:
    """Format timestamp range as human-readable string.

    Args:
        start_sec: Start time in seconds (or None)
        end_sec: End time in seconds (or None)

    Returns:
        Formatted string like "1:23-2:34" or "N/A"
    """
    if start_sec is None or end_sec is None:
        return "N/A"

    def _fmt(sec: float) -> str:
        minutes = int(sec // 60)
        seconds = int(sec % 60)
        return f"{minutes}:{seconds:02d}"

    return f"{_fmt(start_sec)}-{_fmt(end_sec)}"


def _classify_severity(deviation_type: str, confidence: float, is_critical_step: bool) -> str:
    """Classify deviation severity for customer report.

    Args:
        deviation_type: Type of deviation
        confidence: Confidence score (0-1)
        is_critical_step: Whether this involves a critical step

    Returns:
        Severity level: critical, high, medium, low
    """
    if is_critical_step and deviation_type == "step_missing":
        return "critical"
    if deviation_type in {"step_missing", "order_swap"} and confidence > 0.8:
        return "high"
    if deviation_type == "execution_deviation" and confidence > 0.7:
        return "high"
    if confidence > 0.5:
        return "medium"
    return "low"


def _parse_deviations(
    raw_deviations: list[dict],
    sop_template: dict,
    gold_meta: list[dict],
    trainee_meta: list[dict],
) -> list[DeviationReport]:
    """Convert technical deviations to customer-facing deviation reports.

    Args:
        raw_deviations: Raw deviations from evaluate_sop()
        sop_template: SOP template with step info
        gold_meta: Gold video metadata
        trainee_meta: Trainee video metadata

    Returns:
        List of DeviationReport objects
    """
    critical_steps = set(sop_template.get("critical_steps", []))
    step_descriptions = sop_template.get("step_descriptions", {})
    steps = sop_template.get("steps", [])

    reports: list[DeviationReport] = []

    for dev in raw_deviations:
        dev_type = dev.get("type", "unknown")
        gold_step_idx = dev.get("gold_step")
        confidence = float(dev.get("confidence", 0.0))

        # Map step index to step name
        step_name = None
        if gold_step_idx is not None and 0 <= gold_step_idx < len(steps):
            step_name = steps[gold_step_idx]

        is_critical = step_name in critical_steps if step_name else False

        # Extract timestamps
        gold_time = dev.get("gold_time", {})
        trainee_time = dev.get("trainee_time", {})

        timestamp = _format_timestamp(
            trainee_time.get("start_sec"),
            trainee_time.get("end_sec"),
        )

        severity = _classify_severity(dev_type, confidence, is_critical)

        # Generate human-readable description
        if dev_type == "step_missing":
            step_desc = step_descriptions.get(step_name, step_name) if step_name else "Unknown step"
            description = f"Missing step: {step_desc}"
            evidence = f"Expected at {_format_timestamp(gold_time.get('start_sec'), gold_time.get('end_sec'))}"
            if is_critical:
                description += " (CRITICAL SAFETY STEP)"

        elif dev_type == "order_swap":
            step_desc = step_descriptions.get(step_name, step_name) if step_name else "Unknown step"
            description = f"Step performed out of order: {step_desc}"
            evidence = "Step order does not match standard procedure"

        elif dev_type == "execution_deviation":
            step_desc = step_descriptions.get(step_name, step_name) if step_name else "Unknown step"
            description = f"Execution quality issue in step: {step_desc}"
            evidence = f"Low similarity to gold standard (confidence: {confidence:.1%})"

        else:
            description = dev.get("reason", "Unknown deviation")
            evidence = str(dev)

        reports.append(
            DeviationReport(
                deviation_type=dev_type,
                step=step_name,
                timestamp=timestamp,
                severity=severity,
                description=description,
                evidence=evidence,
            )
        )

    return reports


def _generate_corrective_actions(
    deviations: list[DeviationReport],
    sop_template: dict,
) -> list[str]:
    """Generate actionable corrective actions based on deviations.

    Args:
        deviations: List of detected deviations
        sop_template: SOP template with step info

    Returns:
        List of corrective action strings
    """
    actions: list[str] = []
    critical_steps = set(sop_template.get("critical_steps", []))
    step_descriptions = sop_template.get("step_descriptions", {})

    # Collect deviation types
    missing_steps = set()
    critical_missing = set()
    has_order_issues = False
    has_quality_issues = False

    for dev in deviations:
        if dev.deviation_type == "step_missing":
            missing_steps.add(dev.step)
            if dev.step in critical_steps:
                critical_missing.add(dev.step)
        elif dev.deviation_type == "order_swap":
            has_order_issues = True
        elif dev.deviation_type == "execution_deviation":
            has_quality_issues = True

    # Generate specific actions
    if critical_missing:
        for step in sorted(critical_missing):
            step_desc = step_descriptions.get(step, step)
            actions.append(f"CRITICAL: Ensure trainee completes {step_desc} before proceeding")

    if missing_steps - critical_missing:
        actions.append(
            f"Review and complete missing steps: {', '.join(sorted(missing_steps - critical_missing))}"
        )

    if has_order_issues:
        actions.append("Review correct step sequence with trainee using gold standard video")
        actions.append("Consider step-by-step checklist for next attempt")

    if has_quality_issues:
        actions.append("Provide hands-on coaching for steps with execution quality issues")
        actions.append("Review gold standard video with trainee to highlight best practices")

    if not actions:
        actions.append("No major issues detected - trainee is ready for certification")
        actions.append("Consider minor refinements to optimize procedure time")

    return actions


def _generate_json_report(
    score: float,
    threshold: float,
    deviations: list[DeviationReport],
    corrective_actions: list[str],
    evaluation_time: float,
    sop_name: str,
    gold_path: Path,
    trainee_path: Path,
) -> dict:
    """Generate JSON report structure.

    Args:
        score: Computed SOP compliance score (0-100)
        threshold: Pass/fail threshold
        deviations: List of detected deviations
        corrective_actions: List of corrective action strings
        evaluation_time: Time taken for evaluation (seconds)
        sop_name: SOP template name
        gold_path: Gold video path
        trainee_path: Trainee video path

    Returns:
        JSON-serializable dictionary
    """
    return {
        "overall": {
            "pass": score >= threshold,
            "score": round(score, 1),
            "threshold": threshold,
            "grade": (
                "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"
            ),
        },
        "deviations": [
            {
                "type": dev.deviation_type,
                "step": dev.step,
                "timestamp": dev.timestamp,
                "severity": dev.severity,
                "description": dev.description,
                "evidence": dev.evidence,
            }
            for dev in deviations
        ],
        "corrective_actions": corrective_actions,
        "evaluation_time": round(evaluation_time, 2),
        "metadata": {
            "sop": sop_name,
            "sop_full_name": SOP_TEMPLATES[sop_name]["name"],
            "gold_video": str(gold_path.name),
            "trainee_video": str(trainee_path.name),
            "trainee_id": trainee_path.stem,  # Extract trainee ID from filename
        },
        "statistics": {
            "total_deviations": len(deviations),
            "critical_deviations": sum(1 for d in deviations if d.severity == "critical"),
            "high_severity": sum(1 for d in deviations if d.severity == "high"),
            "medium_severity": sum(1 for d in deviations if d.severity == "medium"),
            "low_severity": sum(1 for d in deviations if d.severity == "low"),
        },
    }


def _generate_pdf_report(report_data: dict, output_path: Path) -> None:
    """Generate PDF report using reportlab.

    Args:
        report_data: JSON report data
        output_path: Output PDF path
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        print("ERROR: reportlab not installed. Install with: pip install reportlab")
        print("Falling back to JSON output...")
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"Saved JSON report: {json_path}")
        return

    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=24,
        textColor=colors.HexColor("#003366"),
        spaceAfter=30,
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#003366"),
        spaceAfter=12,
    )

    # Title
    story.append(Paragraph("SOPilot Evaluation Report", title_style))
    story.append(Spacer(1, 0.2 * inch))

    # Overall Result
    overall = report_data["overall"]
    pass_fail_color = colors.green if overall["pass"] else colors.red
    pass_fail_text = "PASS ✓" if overall["pass"] else "FAIL ✗"

    result_data = [
        ["Result:", pass_fail_text],
        ["Score:", f"{overall['score']:.1f} / 100"],
        ["Grade:", overall["grade"]],
        ["Threshold:", str(overall["threshold"])],
    ]
    result_table = Table(result_data, colWidths=[2 * inch, 3 * inch])
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("TEXTCOLOR", (1, 0), (1, 0), pass_fail_color),
                ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (1, 0), (1, 0), 18),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(result_table)
    story.append(Spacer(1, 0.3 * inch))

    # Metadata
    story.append(Paragraph("Evaluation Details", heading_style))
    meta = report_data["metadata"]
    meta_data = [
        ["SOP:", meta["sop_full_name"]],
        ["Trainee:", meta["trainee_id"]],
        ["Gold Video:", meta["gold_video"]],
        ["Evaluation Time:", f"{report_data['evaluation_time']:.2f}s"],
    ]
    meta_table = Table(meta_data, colWidths=[2 * inch, 4 * inch])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 0.3 * inch))

    # Deviations
    story.append(Paragraph("Detected Deviations", heading_style))
    if report_data["deviations"]:
        for i, dev in enumerate(report_data["deviations"], 1):
            severity_color = {
                "critical": colors.red,
                "high": colors.orange,
                "medium": colors.yellow,
                "low": colors.lightblue,
            }.get(dev["severity"], colors.grey)

            dev_data = [
                [f"Deviation #{i}", ""],
                ["Severity:", dev["severity"].upper()],
                ["Type:", dev["type"]],
                ["Step:", dev["step"] or "N/A"],
                ["Timestamp:", dev["timestamp"]],
                ["Description:", dev["description"]],
                ["Evidence:", dev["evidence"]],
            ]
            dev_table = Table(dev_data, colWidths=[1.5 * inch, 4.5 * inch])
            dev_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), severity_color),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("BACKGROUND", (0, 1), (0, -1), colors.lightgrey),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ]
                )
            )
            story.append(dev_table)
            story.append(Spacer(1, 0.2 * inch))
    else:
        story.append(Paragraph("No deviations detected - excellent performance!", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    # Corrective Actions
    story.append(Paragraph("Corrective Actions", heading_style))
    for i, action in enumerate(report_data["corrective_actions"], 1):
        story.append(Paragraph(f"{i}. {action}", styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    # Statistics
    story.append(Paragraph("Statistics", heading_style))
    stats = report_data["statistics"]
    stats_data = [
        ["Total Deviations:", str(stats["total_deviations"])],
        ["Critical:", str(stats["critical_deviations"])],
        ["High Severity:", str(stats["high_severity"])],
        ["Medium Severity:", str(stats["medium_severity"])],
        ["Low Severity:", str(stats["low_severity"])],
    ]
    stats_table = Table(stats_data, colWidths=[2 * inch, 1 * inch])
    stats_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(stats_table)

    # Build PDF
    doc.build(story)
    print(f"PDF report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SOPilot Manufacturing Pilot - Production Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with JSON output
  python scripts/sopilot_evaluate_pilot.py \\
      --gold demo_videos/manufacturing/oil_change_gold.mp4 \\
      --trainee demo_videos/manufacturing/oil_change_trainee_1.mp4 \\
      --sop oil_change

  # PDF report
  python scripts/sopilot_evaluate_pilot.py \\
      --gold demo_videos/manufacturing/oil_change_gold.mp4 \\
      --trainee demo_videos/manufacturing/oil_change_trainee_3.mp4 \\
      --sop oil_change \\
      --out trainee_3_report.pdf

  # Custom threshold
  python scripts/sopilot_evaluate_pilot.py \\
      --gold demo_videos/manufacturing/brake_pads_gold.mp4 \\
      --trainee demo_videos/manufacturing/brake_pads_trainee_1.mp4 \\
      --sop brake_pads \\
      --threshold 90

Available SOPs:
  - oil_change: 10-step oil change procedure
  - brake_pads: 8-step brake pad replacement
  - ppe_check: 5-step PPE safety check
        """,
    )
    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="Gold standard video path",
    )
    parser.add_argument(
        "--trainee",
        type=Path,
        required=True,
        help="Trainee video path",
    )
    parser.add_argument(
        "--sop",
        type=str,
        required=True,
        choices=list(SOP_TEMPLATES.keys()),
        help="SOP template to use",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path (.json or .pdf). Default: report.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Pass/fail threshold (overrides SOP default)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed evaluation progress",
    )

    args = parser.parse_args()

    # Resolve output path
    if args.out is None:
        args.out = Path("report.json")

    # Validate SOP template
    sop_template = SOP_TEMPLATES[args.sop]
    threshold = args.threshold if args.threshold is not None else sop_template["threshold"]

    print("=" * 80)
    print("SOPilot Manufacturing Pilot - Production Evaluation")
    print("=" * 80)
    print(f"SOP:       {sop_template['name']}")
    print(f"Gold:      {args.gold}")
    print(f"Trainee:   {args.trainee}")
    print(f"Threshold: {threshold:.1f}")
    print(f"Output:    {args.out}")
    print()

    # Verify files exist
    if not args.gold.exists():
        print(f"ERROR: Gold video not found: {args.gold}")
        print("Generate demo videos: python scripts/generate_manufacturing_demo.py")
        return 1

    if not args.trainee.exists():
        print(f"ERROR: Trainee video not found: {args.trainee}")
        print("Generate demo videos: python scripts/generate_manufacturing_demo.py")
        return 1

    # Step 1: Extract embeddings
    print("[1/4] Extracting video embeddings...")
    start_total = time.time()
    start = time.time()

    embedder = HeuristicClipEmbedder()

    # Extract clips from gold video
    gold_stream = ClipWindowStream(
        video_path=args.gold,
        target_fps=1,
        clip_seconds=4.0,
        max_side=640,
        min_clip_coverage=0.5,
    )
    gold_clips = list(gold_stream)
    gold_emb = embedder.embed_clips(gold_clips)

    # Extract clips from trainee video
    trainee_stream = ClipWindowStream(
        video_path=args.trainee,
        target_fps=1,
        clip_seconds=4.0,
        max_side=640,
        min_clip_coverage=0.5,
    )
    trainee_clips = list(trainee_stream)
    trainee_emb = embedder.embed_clips(trainee_clips)

    elapsed = time.time() - start
    if args.verbose:
        print(f"  Gold:    {gold_emb.shape} clips")
        print(f"  Trainee: {trainee_emb.shape} clips")
    print(f"  [OK] Embeddings extracted in {elapsed:.2f}s")
    print()

    # Step 2: Evaluate with SOPilot
    print("[2/4] Evaluating SOP compliance...")
    start = time.time()

    # Generate metadata for evaluate_sop
    def _meta(n: int) -> list[dict]:
        return [{"clip_idx": i, "start_sec": float(i * 4), "end_sec": float((i + 1) * 4)} for i in range(n)]

    gold_meta = _meta(gold_emb.shape[0])
    trainee_meta = _meta(trainee_emb.shape[0])

    result = evaluate_sop(
        gold_embeddings=gold_emb,
        trainee_embeddings=trainee_emb,
        gold_meta=gold_meta,
        trainee_meta=trainee_meta,
        threshold_factor=0.5,
        min_step_clips=1,
        low_similarity_threshold=0.75,
        w_miss=12,
        w_swap=8,
        w_dev=30,
        w_time=15,
    )

    elapsed = time.time() - start
    if args.verbose:
        print(f"  Raw score:     {result['score']:.1f}")
        print(f"  Deviations:    {len(result['deviations'])}")
        print(f"  Missing steps: {result['metrics']['miss']}")
        print(f"  Order swaps:   {result['metrics']['swap']}")
    print(f"  [OK] Evaluation completed in {elapsed:.2f}s")
    print()

    # Step 3: Parse and format results
    print("[3/4] Generating customer report...")
    start = time.time()

    deviations = _parse_deviations(
        result["deviations"],
        sop_template,
        gold_meta,
        trainee_meta,
    )

    corrective_actions = _generate_corrective_actions(deviations, sop_template)

    evaluation_time = time.time() - start_total

    report_data = _generate_json_report(
        score=result["score"],
        threshold=threshold,
        deviations=deviations,
        corrective_actions=corrective_actions,
        evaluation_time=evaluation_time,
        sop_name=args.sop,
        gold_path=args.gold,
        trainee_path=args.trainee,
    )

    elapsed = time.time() - start
    print(f"  [OK] Report generated in {elapsed:.2f}s")
    print()

    # Step 4: Save output
    print("[4/4] Saving report...")
    if args.out.suffix.lower() == ".pdf":
        _generate_pdf_report(report_data, args.out)
    else:
        # Save JSON
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"JSON report saved: {args.out}")
    print()

    # Display summary
    print("=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print()
    print(f"Overall Result:  {'PASS' if report_data['overall']['pass'] else 'FAIL'}")
    print(f"Score:           {report_data['overall']['score']:.1f} / 100 (Grade: {report_data['overall']['grade']})")
    print(f"Threshold:       {threshold:.1f}")
    print(f"Deviations:      {report_data['statistics']['total_deviations']} total")
    print(f"  - Critical:    {report_data['statistics']['critical_deviations']}")
    print(f"  - High:        {report_data['statistics']['high_severity']}")
    print(f"  - Medium:      {report_data['statistics']['medium_severity']}")
    print(f"  - Low:         {report_data['statistics']['low_severity']}")
    print()

    if deviations:
        print("Top Deviations:")
        for i, dev in enumerate(deviations[:5], 1):
            print(f"  {i}. [{dev.severity.upper()}] {dev.description}")
            print(f"     @ {dev.timestamp}")
        if len(deviations) > 5:
            print(f"  ... and {len(deviations) - 5} more (see full report)")
        print()

    print("Corrective Actions:")
    for i, action in enumerate(corrective_actions, 1):
        print(f"  {i}. {action}")
    print()

    print("-" * 80)
    print(f"Evaluation Time: {evaluation_time:.2f}s")
    print(f"Output File:     {args.out.resolve()}")
    print("=" * 80)

    return 0 if report_data["overall"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
