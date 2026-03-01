"""Training recommendation engine based on deviation pattern analysis."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from sopilot.database import Database


class RecommendationService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def get_recommendations(
        self,
        operator_id: str,
        *,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate actionable training recommendations for an operator."""
        trend = self.database.get_operator_trend(
            operator_id, task_id=task_id, limit=50
        )
        jobs = trend.get("jobs") or []
        job_count = len(jobs)

        if job_count == 0:
            return {
                "operator_id": operator_id,
                "job_count": 0,
                "total_deviations": 0,
                "recommendations": [],
                "overall_assessment": "データなし",
            }

        # Collect all deviations across all jobs
        issue_counts: dict[tuple[int, str], int] = defaultdict(int)
        total_deviations = 0

        with self.database.connect() as conn:
            for job in jobs:
                job_id = job["job_id"]
                row = conn.execute(
                    "SELECT score_json FROM score_jobs WHERE id = ?",
                    (job_id,),
                ).fetchone()
                if row is None or row["score_json"] is None:
                    continue
                try:
                    parsed = json.loads(row["score_json"])
                except (json.JSONDecodeError, TypeError):
                    continue
                deviations = parsed.get("deviations") or []
                for dv in deviations:
                    step_idx = dv.get("step_index")
                    if step_idx is None:
                        continue
                    dv_type = dv.get("type", "")
                    if not dv_type:
                        continue
                    issue_counts[(int(step_idx), dv_type)] += 1
                    total_deviations += 1

        # Build recommendations for issues with frequency >= 0.15
        recommendations: list[dict[str, Any]] = []
        for (step_idx, issue_type), count in issue_counts.items():
            frequency = count / job_count
            if frequency < 0.15:
                continue

            if frequency >= 0.5:
                priority = "critical"
            elif frequency >= 0.25:
                priority = "quality"
            else:
                priority = "efficiency"

            freq_pct = frequency
            step_label = step_idx + 1

            if issue_type == "missing_step":
                advice_ja = (
                    f"手順{step_label}が{freq_pct:.0%}の評価でスキップされています。"
                    "このステップを必ず実施してください。"
                )
            elif issue_type == "step_deviation":
                advice_ja = (
                    f"手順{step_label}の実施品質が{freq_pct:.0%}の評価で基準を下回っています。"
                    "動作の正確性を意識してください。"
                )
            elif issue_type == "order_swap":
                advice_ja = (
                    f"手順{step_label}の実施順序が{freq_pct:.0%}の評価でずれています。"
                    "手順の順番を厳守してください。"
                )
            elif issue_type == "over_time":
                advice_ja = (
                    f"手順{step_label}の実施時間が{freq_pct:.0%}の評価で超過しています。"
                    "効率を改善してください。"
                )
            else:
                advice_ja = (
                    f"手順{step_label}で{freq_pct:.0%}の評価で問題が発生しています。"
                    "確認してください。"
                )

            recommendations.append(
                {
                    "step_index": step_idx,
                    "issue_type": issue_type,
                    "frequency": round(frequency, 4),
                    "count": count,
                    "priority": priority,
                    "advice_ja": advice_ja,
                }
            )

        recommendations.sort(key=lambda r: r["frequency"], reverse=True)

        # Determine overall assessment
        priorities = {r["priority"] for r in recommendations}
        if "critical" in priorities:
            overall_assessment = "要再研修"
        elif "quality" in priorities:
            overall_assessment = "改善中"
        else:
            overall_assessment = "良好"

        return {
            "operator_id": operator_id,
            "job_count": job_count,
            "total_deviations": total_deviations,
            "recommendations": recommendations,
            "overall_assessment": overall_assessment,
        }
