"""Professional HTML Report Generator for Insurance MVP

Generates contract-quality HTML reports from ClaimAssessment data.
Supports English and Japanese localization.
"""

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .insurance.schema import ClaimAssessment

# Translation table: English → Japanese
_TRANSLATIONS_JA = {
    "Insurance Claim Assessment Report": "保険請求評価レポート",
    "Video ID": "映像ID",
    "Generated": "作成日時",
    "Processing Time": "処理時間",
    "Severity Level": "重大度レベル",
    "Confidence": "確信度",
    "Fault Ratio": "過失割合",
    "At-Fault Party": "過失当事者",
    "Fraud Risk": "不正リスク",
    "Risk Level": "リスクレベル",
    "Fraud Risk Score": "不正リスクスコア",
    "Severity Assessment": "重大度評価",
    "Confidence Score": "確信度スコア",
    "Review Priority": "レビュー優先度",
    "Conformal Prediction Set (90% confidence)": "適合予測セット（信頼度90%）",
    "Fault Assessment": "過失評価",
    "Scenario Type": "シナリオ種別",
    "Applicable Traffic Rules": "適用交通規則",
    "Fault Reasoning": "過失判断理由",
    "Fraud Detection Analysis": "不正検出分析",
    "Red Flags Detected": "不正フラグ検出",
    "Fraud Analysis": "不正分析",
    "AI Causal Reasoning": "AI因果推論",
    "Recommended Action": "推奨アクション",
    "Recommended Next Step": "推奨次ステップ",
    "Executive Summary": "エグゼクティブサマリー",
    "Assessment Result": "評価結果",
    "No incident detected": "インシデント未検出",
    "Minor incident": "軽微なインシデント",
    "Moderate incident requiring review": "レビュー必要な中程度のインシデント",
    "Serious incident requiring immediate action": "即時対応が必要な重大インシデント",
    "Ego Vehicle": "契約車両",
    "Other Vehicle": "相手車両",
    "Shared": "双方",
    "APPROVE": "承認",
    "REVIEW": "要レビュー",
    "REJECT": "却下",
    "URGENT": "緊急",
    "STANDARD": "標準",
    "LOW_PRIORITY": "低優先",
    "AI-Powered Insurance Claim Assessment System": "AI搭載 保険請求評価システム",
    "Contract-Grade Professional Quality": "契約品質プロフェッショナルレポート",
    "No reasoning provided": "判断理由なし",
    "No fraud indicators detected": "不正指標は検出されませんでした",
    "No causal reasoning available": "因果推論なし",
    # Multi-clip report keys
    "Multi-Clip Assessment Report": "マルチクリップ評価レポート",
    "Total Clips": "クリップ総数",
    "Total Processing Time": "合計処理時間",
    "Highest Severity": "最高重大度",
    "Severity Breakdown": "重大度内訳",
    "Overall Recommendation": "総合推奨",
    "Clip Assessments": "クリップ評価一覧",
    "Clip": "クリップ",
    "Analyzed": "分析済み",
    "Overall Assessment": "総合評価",
}


class ReportGenerator:
    """Generate professional HTML reports from assessment data."""

    def __init__(self, template_dir: Path | None = None, lang: str = "en"):
        """Initialize report generator.

        Args:
            template_dir: Directory containing HTML templates.
                          If None, uses built-in templates directory.
            lang: Language for report labels ("en" or "ja").
        """
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"

        self.lang = lang
        self.env = Environment(loader=FileSystemLoader(str(template_dir)))
        self.env.globals["_"] = self._translate

    def _translate(self, text: str) -> str:
        """Translate text if Japanese mode is active."""
        if self.lang == "ja":
            return _TRANSLATIONS_JA.get(text, text)
        return text

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

        # Generate executive summary
        executive_summary = self._generate_executive_summary(assessment)

        # Prepare template data
        data = {
            # Language
            "lang": self.lang,
            # Executive summary
            "executive_summary": executive_summary,
            # Header
            "video_id": video_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time": f"{assessment.processing_time_sec:.1f}",
            # Hero cards
            "severity": assessment.severity,
            "confidence": int(assessment.confidence * 100),
            "fault_ratio": assessment.fault_assessment.fault_ratio,
            "at_fault_party": self._derive_at_fault_party(assessment.fault_assessment.fault_ratio),
            "fraud_score": f"{assessment.fraud_risk.risk_score:.2f}",
            "fraud_level": self._derive_fraud_level(assessment.fraud_risk.risk_score),
            # Severity section
            "prediction_set": ", ".join(sorted(assessment.prediction_set)),
            "review_priority": assessment.review_priority,
            "review_priority_class": assessment.review_priority.lower(),
            # Fault section
            "scenario_type": assessment.fault_assessment.scenario_type or "unknown",
            "applicable_rules": assessment.fault_assessment.applicable_rules,
            "fault_reasoning": assessment.fault_assessment.reasoning or "No reasoning provided",
            # Fraud section
            "red_flags": assessment.fraud_risk.indicators,
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
        output_path.write_text(html_content, encoding="utf-8")

    def _generate_executive_summary(self, assessment: ClaimAssessment) -> str:
        """Generate 1-2 sentence executive summary."""
        severity = assessment.severity
        fault = assessment.fault_assessment.fault_ratio
        fraud_level = self._derive_fraud_level(assessment.fraud_risk.risk_score)
        scenario = assessment.fault_assessment.scenario_type or "incident"
        action = assessment.recommended_action

        if self.lang == "ja":
            if severity == "NONE":
                return f"映像分析の結果、インシデントは検出されませんでした。不正リスク: {fraud_level}。推奨: {self._translate(action)}。"
            fault_party = self._derive_at_fault_party(fault)
            return (
                f"{self._translate(scenario)}シナリオを検出。重大度: {severity}、"
                f"過失割合: {fault:.0f}%（{self._translate(fault_party)}）、"
                f"不正リスク: {fraud_level}。推奨: {self._translate(action)}。"
            )
        else:
            if severity == "NONE":
                return f"Video analysis detected no incident. Fraud risk: {fraud_level}. Recommended action: {action}."
            fault_party = self._derive_at_fault_party(fault)
            return (
                f"{scenario.replace('_', ' ').title()} detected with {severity} severity. "
                f"Fault ratio: {fault:.0f}% ({fault_party}). "
                f"Fraud risk: {fraud_level}. Recommended action: {action}."
            )

    @staticmethod
    def _derive_at_fault_party(fault_ratio: float) -> str:
        if fault_ratio >= 75.0:
            return "Ego Vehicle"
        elif fault_ratio <= 25.0:
            return "Other Vehicle"
        else:
            return "Shared"

    @staticmethod
    def _derive_fraud_level(risk_score: float) -> str:
        if risk_score >= 0.65:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def generate_multi_clip_report(
        self,
        assessments: list[ClaimAssessment],
        video_id: str,
        output_path: Path,
    ) -> None:
        """Generate comprehensive HTML report for multiple clips from same video.

        Creates a report showing ALL assessments with a summary section at the top
        (highest severity, total clips, overall assessment) and individual clip cards
        sorted by severity (HIGH first) then by confidence (descending).

        Args:
            assessments: List of ClaimAssessment (one per clip)
            video_id: Video identifier
            output_path: Path to save HTML file
        """
        if not assessments:
            raise ValueError("No assessments provided")

        # Sort by severity (HIGH first) then by confidence descending
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
        sorted_assessments = sorted(
            assessments,
            key=lambda a: (severity_order.get(a.severity, 4), -a.confidence),
        )

        primary = sorted_assessments[0]

        # Compute severity counts
        severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}
        for a in sorted_assessments:
            severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1

        # Determine overall recommendation from highest-severity clip
        overall_recommendation = primary.recommended_action

        # Generate overall summary text
        overall_summary = self._generate_multi_clip_summary(sorted_assessments, severity_counts)

        # Total processing time
        total_processing_time = sum(a.processing_time_sec for a in sorted_assessments)

        # Build per-clip data
        clips_data = []
        for a in sorted_assessments:
            clips_data.append({
                "severity": a.severity,
                "confidence": int(a.confidence * 100),
                "fault_ratio": a.fault_assessment.fault_ratio,
                "at_fault_party": self._derive_at_fault_party(a.fault_assessment.fault_ratio),
                "fraud_score": f"{a.fraud_risk.risk_score:.2f}",
                "fraud_level": self._derive_fraud_level(a.fraud_risk.risk_score),
                "scenario_type": a.fault_assessment.scenario_type or "unknown",
                "causal_reasoning": a.causal_reasoning or "No causal reasoning available",
                "fault_reasoning": a.fault_assessment.reasoning or "No reasoning provided",
                "red_flags": a.fraud_risk.indicators,
                "review_priority": a.review_priority,
                "review_priority_class": a.review_priority.lower(),
                "recommended_action": a.recommended_action,
                "processing_time": f"{a.processing_time_sec:.1f}",
            })

        template = self.env.get_template("multi_clip_report.html")

        data = {
            "lang": self.lang,
            "video_id": video_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_clips": len(sorted_assessments),
            "total_processing_time": f"{total_processing_time:.1f}",
            "highest_severity": primary.severity,
            "highest_severity_confidence": int(primary.confidence * 100),
            "severity_counts": severity_counts,
            "overall_recommendation": overall_recommendation,
            "overall_summary": overall_summary,
            "clips": clips_data,
        }

        html_content = template.render(**data)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding="utf-8")

    def _generate_multi_clip_summary(
        self,
        sorted_assessments: list[ClaimAssessment],
        severity_counts: dict[str, int],
    ) -> str:
        """Generate overall summary for multi-clip report."""
        total = len(sorted_assessments)
        primary = sorted_assessments[0]
        highest = primary.severity

        # Build severity breakdown string
        parts = []
        for sev in ("HIGH", "MEDIUM", "LOW", "NONE"):
            count = severity_counts.get(sev, 0)
            if count > 0:
                parts.append(f"{count} {sev}")
        breakdown = ", ".join(parts)

        if self.lang == "ja":
            if highest == "NONE":
                return f"{total}件のクリップを分析しました。インシデントは検出されませんでした。内訳: {breakdown}。"
            return (
                f"{total}件のクリップを分析しました。"
                f"最高重大度: {highest}。内訳: {breakdown}。"
                f"推奨: {self._translate(primary.recommended_action)}。"
            )
        else:
            if highest == "NONE":
                return f"Analyzed {total} clips. No incidents detected. Breakdown: {breakdown}."
            return (
                f"Analyzed {total} clips. "
                f"Highest severity: {highest}. Breakdown: {breakdown}. "
                f"Recommended action: {primary.recommended_action}."
            )
