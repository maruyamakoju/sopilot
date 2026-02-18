"""Professional HTML Report Generator for Insurance MVP

Generates contract-quality HTML reports from ClaimAssessment data.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
from jinja2 import Environment, FileSystemLoader

from .insurance.schema import ClaimAssessment


class ReportGenerator:
    """Generate professional HTML reports from assessment data."""

    def __init__(self, template_dir: Optional[Path] = None):
        """Initialize report generator.

        Args:
            template_dir: Directory containing HTML templates.
                          If None, uses built-in templates directory.
        """
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"

        self.env = Environment(loader=FileSystemLoader(str(template_dir)))

    def generate_report(
        self,
        assessment: ClaimAssessment,
        video_id: str,
        output_path: Path,
        template_name: str = "professional_report.html",
    ) -> None:
        """Generate HTML report from assessment.

        Args:
            assessment: ClaimAssessment data
            video_id: Video identifier
            output_path: Path to save HTML file
            template_name: Template filename to use
        """
        template = self.env.get_template(template_name)

        # Prepare template data
        data = {
            # Header
            "video_id": video_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time": f"{assessment.processing_time_sec:.1f}",

            # Hero cards
            "severity": assessment.severity,
            "confidence": int(assessment.confidence * 100),
            "fault_ratio": assessment.fault_assessment.fault_ratio,
            "at_fault_party": assessment.fault_assessment.at_fault_party or "N/A",
            "fraud_score": f"{assessment.fraud_risk.risk_score:.2f}",
            "fraud_level": assessment.fraud_risk.risk_level,

            # Severity section
            "prediction_set": ', '.join(sorted(assessment.prediction_set)),
            "review_priority": assessment.review_priority,
            "review_priority_class": assessment.review_priority.lower(),

            # Fault section
            "scenario_type": assessment.fault_assessment.scenario_type or "unknown",
            "applicable_rules": assessment.fault_assessment.applicable_rules,
            "fault_reasoning": assessment.fault_assessment.reasoning or "No reasoning provided",

            # Fraud section
            "red_flags": assessment.fraud_risk.red_flags,
            "fraud_reasoning": assessment.fraud_risk.reasoning or "No fraud indicators detected",

            # Causal reasoning
            "causal_reasoning": assessment.causal_reasoning or "No causal reasoning available",

            # Recommendation
            "recommended_action": assessment.recommended_action,
        }

        # Render template
        html_content = template.render(**data)

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding='utf-8')

    def generate_multi_clip_report(
        self,
        assessments: list[ClaimAssessment],
        video_id: str,
        output_path: Path,
    ) -> None:
        """Generate report for multiple clips from same video.

        Args:
            assessments: List of ClaimAssessment (one per clip)
            video_id: Video identifier
            output_path: Path to save HTML file
        """
        # For now, use the highest-severity clip
        # TODO: Create multi-clip template showing all assessments
        if not assessments:
            raise ValueError("No assessments provided")

        # Sort by severity
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
        sorted_assessments = sorted(
            assessments,
            key=lambda a: (severity_order.get(a.severity, 4), -a.confidence)
        )

        # Use most severe assessment
        primary_assessment = sorted_assessments[0]

        self.generate_report(
            assessment=primary_assessment,
            video_id=video_id,
            output_path=output_path
        )
