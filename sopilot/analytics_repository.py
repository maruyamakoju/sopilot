from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any


class ScoreAnalyticsRepository:
    """Read-only analytics queries for scored jobs."""

    def __init__(self, connect: Callable[[], AbstractContextManager[sqlite3.Connection]]) -> None:
        self._connect = connect

    @staticmethod
    def _build_task_time_filters(
        *,
        task_id: str | None,
        days: int | None,
        task_alias: str = "gv",
        score_alias: str = "sj",
    ) -> tuple[str, str, tuple[Any, ...]]:
        task_filter = ""
        time_filter = ""
        params_list: list[Any] = []
        if task_id:
            task_filter = f" AND {task_alias}.task_id = ?"
            params_list.append(task_id)
        if days is not None:
            time_filter = f" AND {score_alias}.created_at >= datetime('now', ?)"
            params_list.append(f"-{days} days")
        return task_filter, time_filter, tuple(params_list)

    def get_analytics(self, *, task_id: str | None = None, days: int | None = None) -> dict:
        """Return aggregate scoring analytics for dashboard visualization."""
        task_filter, time_filter, params = self._build_task_time_filters(task_id=task_id, days=days)

        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*)                                                     AS total_jobs,
                    SUM(CASE WHEN sj.status = 'completed' THEN 1 ELSE 0 END)    AS completed_jobs,
                    SUM(CASE WHEN sj.status = 'completed'
                              AND json_extract(sj.score_json, '$.summary.decision') = 'pass'
                         THEN 1 ELSE 0 END)                                      AS pass_count,
                    SUM(CASE WHEN sj.status = 'completed'
                              AND json_extract(sj.score_json, '$.summary.decision') = 'fail'
                         THEN 1 ELSE 0 END)                                      AS fail_count,
                    SUM(CASE WHEN sj.status = 'completed'
                              AND json_extract(sj.score_json, '$.summary.decision') = 'needs_review'
                         THEN 1 ELSE 0 END)                                      AS needs_review_count,
                    SUM(CASE WHEN sj.status = 'completed'
                              AND json_extract(sj.score_json, '$.summary.decision') = 'retrain'
                         THEN 1 ELSE 0 END)                                      AS retrain_count,
                    AVG(CASE WHEN sj.status = 'completed'
                         THEN json_extract(sj.score_json, '$.score') END)        AS avg_score,
                    MIN(CASE WHEN sj.status = 'completed'
                         THEN json_extract(sj.score_json, '$.score') END)        AS min_score,
                    MAX(CASE WHEN sj.status = 'completed'
                         THEN json_extract(sj.score_json, '$.score') END)        AS max_score
                FROM score_jobs sj
                LEFT JOIN videos gv ON gv.id = sj.gold_video_id
                WHERE 1=1{task_filter}{time_filter}
                """,
                params,
            ).fetchone()

            total_jobs = int(row["total_jobs"]) if row and row["total_jobs"] else 0
            completed_jobs = int(row["completed_jobs"]) if row and row["completed_jobs"] else 0
            pass_count = int(row["pass_count"]) if row and row["pass_count"] else 0
            fail_count = int(row["fail_count"]) if row and row["fail_count"] else 0
            needs_review_count = int(row["needs_review_count"]) if row and row["needs_review_count"] else 0
            retrain_count = int(row["retrain_count"]) if row and row["retrain_count"] else 0
            avg_score = round(float(row["avg_score"]), 2) if row and row["avg_score"] is not None else None
            min_score = round(float(row["min_score"]), 2) if row and row["min_score"] is not None else None
            max_score = round(float(row["max_score"]), 2) if row and row["max_score"] is not None else None

            by_operator_rows = conn.execute(
                f"""
                SELECT
                    tv.operator_id_hash                                          AS operator_id,
                    COUNT(*)                                                     AS job_count,
                    AVG(json_extract(sj.score_json, '$.score'))                  AS avg_score,
                    SUM(CASE WHEN json_extract(sj.score_json, '$.summary.decision') = 'pass'
                         THEN 1 ELSE 0 END)                                      AS pass_count,
                    SUM(CASE WHEN json_extract(sj.score_json, '$.summary.decision') = 'fail'
                         THEN 1 ELSE 0 END)                                      AS fail_count
                FROM score_jobs sj
                LEFT JOIN videos gv ON gv.id = sj.gold_video_id
                LEFT JOIN videos tv ON tv.id = sj.trainee_video_id
                WHERE sj.status = 'completed' AND sj.score_json IS NOT NULL{task_filter}{time_filter}
                GROUP BY tv.operator_id_hash
                ORDER BY avg_score DESC, job_count DESC
                """,
                params,
            ).fetchall()

            by_operator = [
                {
                    "operator_id": r["operator_id"] or "unknown",
                    "job_count": int(r["job_count"]),
                    "avg_score": round(float(r["avg_score"]), 2) if r["avg_score"] is not None else 0.0,
                    "pass_count": int(r["pass_count"]),
                    "fail_count": int(r["fail_count"]),
                }
                for r in by_operator_rows
            ]

            by_site_rows = conn.execute(
                f"""
                SELECT
                    tv.site_id                                                   AS site_id,
                    COUNT(*)                                                     AS job_count,
                    AVG(json_extract(sj.score_json, '$.score'))                  AS avg_score,
                    SUM(CASE WHEN json_extract(sj.score_json, '$.summary.decision') = 'pass'
                         THEN 1 ELSE 0 END)                                      AS pass_count,
                    SUM(CASE WHEN json_extract(sj.score_json, '$.summary.decision') = 'fail'
                         THEN 1 ELSE 0 END)                                      AS fail_count
                FROM score_jobs sj
                LEFT JOIN videos gv ON gv.id = sj.gold_video_id
                LEFT JOIN videos tv ON tv.id = sj.trainee_video_id
                WHERE sj.status = 'completed' AND sj.score_json IS NOT NULL{task_filter}{time_filter}
                GROUP BY tv.site_id
                ORDER BY avg_score DESC, job_count DESC
                """,
                params,
            ).fetchall()

            by_site = [
                {
                    "site_id": r["site_id"] or "unknown",
                    "job_count": int(r["job_count"]),
                    "avg_score": round(float(r["avg_score"]), 2) if r["avg_score"] is not None else 0.0,
                    "pass_count": int(r["pass_count"]),
                    "fail_count": int(r["fail_count"]),
                }
                for r in by_site_rows
            ]

            dist_rows = conn.execute(
                f"""
                SELECT
                    CASE
                        WHEN json_extract(sj.score_json, '$.score') >= 90 THEN '90-100'
                        WHEN json_extract(sj.score_json, '$.score') >= 80 THEN '80-89'
                        WHEN json_extract(sj.score_json, '$.score') >= 70 THEN '70-79'
                        ELSE '0-69'
                    END AS bucket,
                    COUNT(*) AS cnt
                FROM score_jobs sj
                LEFT JOIN videos gv ON gv.id = sj.gold_video_id
                WHERE sj.status = 'completed' AND sj.score_json IS NOT NULL{task_filter}{time_filter}
                GROUP BY bucket
                """,
                params,
            ).fetchall()

            dist_map = {r["bucket"]: int(r["cnt"]) for r in dist_rows}
            score_distribution = [
                {"bucket": "90-100", "count": dist_map.get("90-100", 0)},
                {"bucket": "80-89", "count": dist_map.get("80-89", 0)},
                {"bucket": "70-79", "count": dist_map.get("70-79", 0)},
                {"bucket": "0-69", "count": dist_map.get("0-69", 0)},
            ]

            trend_rows = conn.execute(
                f"""
                SELECT
                    sj.id                                                        AS job_id,
                    json_extract(sj.score_json, '$.score')                       AS score,
                    json_extract(sj.score_json, '$.summary.decision')            AS decision,
                    sj.created_at
                FROM score_jobs sj
                LEFT JOIN videos gv ON gv.id = sj.gold_video_id
                WHERE sj.status = 'completed' AND sj.score_json IS NOT NULL{task_filter}{time_filter}
                ORDER BY sj.id DESC
                LIMIT 30
                """,
                params,
            ).fetchall()

            recent_trend = [
                {
                    "job_id": int(r["job_id"]),
                    "score": round(float(r["score"]), 2) if r["score"] is not None else 0.0,
                    "decision": r["decision"] or "unknown",
                    "created_at": r["created_at"],
                }
                for r in trend_rows
            ]

            review_rows = conn.execute(
                f"""
                SELECT
                    COUNT(*)                                                     AS reviewed_count,
                    SUM(CASE WHEN sr.verdict = json_extract(sj.score_json, '$.summary.decision')
                         THEN 1 ELSE 0 END)                                      AS agree_count,
                    SUM(CASE WHEN sr.verdict = 'pass' THEN 1 ELSE 0 END)        AS review_pass,
                    SUM(CASE WHEN sr.verdict = 'fail' THEN 1 ELSE 0 END)        AS review_fail,
                    SUM(CASE WHEN sr.verdict = 'retrain' THEN 1 ELSE 0 END)     AS review_retrain,
                    SUM(CASE WHEN sr.verdict = 'needs_review' THEN 1 ELSE 0 END) AS review_needs_review
                FROM score_reviews sr
                JOIN score_jobs sj ON sj.id = sr.job_id
                LEFT JOIN videos gv ON gv.id = sj.gold_video_id
                WHERE sj.status = 'completed' AND sj.score_json IS NOT NULL{task_filter}{time_filter}
                """,
                params,
            ).fetchone()

            reviewed_count = int(review_rows["reviewed_count"]) if review_rows and review_rows["reviewed_count"] else 0
            agree_count = int(review_rows["agree_count"]) if review_rows and review_rows["agree_count"] else 0
            reviewer_agreement = {
                "reviewed_count": reviewed_count,
                "agree_count": agree_count,
                "agreement_rate": round(agree_count / reviewed_count, 4) if reviewed_count > 0 else None,
                "by_verdict": {
                    "pass": int(review_rows["review_pass"]) if review_rows and review_rows["review_pass"] else 0,
                    "fail": int(review_rows["review_fail"]) if review_rows and review_rows["review_fail"] else 0,
                    "retrain": int(review_rows["review_retrain"]) if review_rows and review_rows["review_retrain"] else 0,
                    "needs_review": int(review_rows["review_needs_review"])
                    if review_rows and review_rows["review_needs_review"]
                    else 0,
                },
            }

        return {
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "needs_review_count": needs_review_count,
            "retrain_count": retrain_count,
            "avg_score": avg_score,
            "min_score": min_score,
            "max_score": max_score,
            "by_operator": by_operator,
            "by_site": by_site,
            "score_distribution": score_distribution,
            "recent_trend": recent_trend,
            "reviewer_agreement": reviewer_agreement,
        }

    def get_operator_trend(
        self,
        operator_id: str,
        *,
        task_id: str | None = None,
        limit: int = 20,
    ) -> dict:
        """Return per-job score history and trend statistics for a single operator."""
        task_filter = ""
        params_list: list[Any] = [operator_id]
        if task_id:
            task_filter = " AND gv.task_id = ?"
            params_list.append(task_id)
        params_list.append(limit)
        params: tuple[Any, ...] = tuple(params_list)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    sj.id                                                       AS job_id,
                    json_extract(sj.score_json, '$.score')                      AS score,
                    json_extract(sj.score_json, '$.summary.decision')           AS decision,
                    sj.created_at
                FROM score_jobs sj
                LEFT JOIN videos tv ON tv.id = sj.trainee_video_id
                LEFT JOIN videos gv ON gv.id = sj.gold_video_id
                WHERE sj.status = 'completed'
                  AND sj.score_json IS NOT NULL
                  AND tv.operator_id_hash = ?{task_filter}
                ORDER BY sj.created_at ASC
                LIMIT ?
                """,
                params,
            ).fetchall()

        jobs: list[dict[str, Any]] = [
            {
                "job_id": int(r["job_id"]),
                "score": round(float(r["score"]), 2) if r["score"] is not None else 0.0,
                "decision": r["decision"] or "unknown",
                "created_at": r["created_at"],
            }
            for r in rows
        ]

        job_count = len(jobs)
        if job_count == 0:
            return {
                "operator_id": operator_id,
                "job_count": 0,
                "avg_score": None,
                "latest_decision": None,
                "trend_slope": None,
                "improvement_pct": None,
                "jobs": [],
            }

        scores = [j["score"] for j in jobs]
        avg_score = round(sum(scores) / job_count, 2)
        latest_decision = jobs[-1]["decision"]

        n = job_count
        x_mean = (n - 1) / 2.0
        y_mean = avg_score
        numerator = sum((i - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        trend_slope: float | None = round(numerator / denominator, 4) if denominator != 0 else None

        first_score = scores[0]
        last_score = scores[-1]
        improvement_pct: float | None = (
            round((last_score - first_score) / first_score * 100, 2) if first_score != 0 else None
        )

        # Moving average (window=5)
        window = min(5, n)
        moving_avg: list[float | None] = []
        for i in range(n):
            if i < window - 1:
                moving_avg.append(None)
            else:
                chunk = scores[i - window + 1 : i + 1]
                moving_avg.append(round(sum(chunk) / len(chunk), 2))

        # Rolling pass rate (window=5)
        decisions = [j["decision"] for j in jobs]
        pass_rate: list[float | None] = []
        for i in range(n):
            if i < window - 1:
                pass_rate.append(None)
            else:
                chunk = decisions[i - window + 1 : i + 1]
                pass_rate.append(round(sum(1 for d in chunk if d == "pass") / len(chunk), 4))

        # Score volatility (stdev of last 5)
        recent = scores[-window:]
        if len(recent) >= 2:
            rmean = sum(recent) / len(recent)
            volatility = round((sum((s - rmean) ** 2 for s in recent) / (len(recent) - 1)) ** 0.5, 2)
        else:
            volatility = 0.0

        # Team baseline (avg score across all operators for comparison)
        team_avg: float | None = None
        team_params: tuple[Any, ...] = (task_id,) if task_id else ()
        with self._connect() as conn:
            team_row = conn.execute(
                f"""
                SELECT AVG(json_extract(sj.score_json, '$.score')) AS team_avg
                FROM score_jobs sj
                LEFT JOIN videos gv ON gv.id = sj.gold_video_id
                WHERE sj.status = 'completed' AND sj.score_json IS NOT NULL{task_filter}
                """,
                team_params,
            ).fetchone()
            if team_row and team_row["team_avg"] is not None:
                team_avg = round(float(team_row["team_avg"]), 2)

        return {
            "operator_id": operator_id,
            "job_count": job_count,
            "avg_score": avg_score,
            "latest_decision": latest_decision,
            "trend_slope": trend_slope,
            "improvement_pct": improvement_pct,
            "moving_avg": moving_avg,
            "pass_rate": pass_rate,
            "volatility": volatility,
            "team_avg": team_avg,
            "jobs": jobs,
        }

    def get_step_performance(
        self,
        *,
        task_id: str | None = None,
        days: int | None = None,
    ) -> dict:
        """Return per-step deviation frequency statistics across completed jobs."""
        task_filter, time_filter, params = self._build_task_time_filters(task_id=task_id, days=days)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT sj.score_json
                FROM score_jobs sj
                LEFT JOIN videos gv ON gv.id = sj.gold_video_id
                WHERE sj.status = 'completed'
                  AND sj.score_json IS NOT NULL{task_filter}{time_filter}
                ORDER BY sj.created_at DESC
                LIMIT 500
                """,
                params,
            ).fetchall()

        total_jobs = len(rows)
        step_counters: dict[int, dict[str, int]] = {}

        for row in rows:
            try:
                parsed = json.loads(row["score_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            deviations = parsed.get("deviations") or []
            for dv in deviations:
                step_idx = dv.get("step_index")
                if step_idx is None:
                    continue
                step_idx = int(step_idx)
                dv_type = dv.get("type", "")
                if step_idx not in step_counters:
                    step_counters[step_idx] = {
                        "miss": 0,
                        "deviation": 0,
                        "swap": 0,
                        "over_time": 0,
                        "total_issues": 0,
                    }
                if dv_type == "missing_step":
                    step_counters[step_idx]["miss"] += 1
                elif dv_type == "step_deviation":
                    step_counters[step_idx]["deviation"] += 1
                elif dv_type == "order_swap":
                    step_counters[step_idx]["swap"] += 1
                elif dv_type == "over_time":
                    step_counters[step_idx]["over_time"] += 1
                step_counters[step_idx]["total_issues"] += 1

        if total_jobs == 0:
            return {"total_jobs": 0, "steps": [], "hardest_step_index": None}

        steps = [
            {
                "step_index": idx,
                "miss_rate": round(cnt["miss"] / total_jobs, 4),
                "deviation_rate": round(cnt["deviation"] / total_jobs, 4),
                "swap_rate": round(cnt["swap"] / total_jobs, 4),
                "any_issue_rate": round(min(cnt["total_issues"] / total_jobs, 1.0), 4),
            }
            for idx, cnt in step_counters.items()
        ]
        steps.sort(key=lambda s: s["any_issue_rate"], reverse=True)
        hardest_step_index: int | None = int(steps[0]["step_index"]) if steps else None

        return {
            "total_jobs": total_jobs,
            "steps": steps,
            "hardest_step_index": hardest_step_index,
        }

    def get_compliance_overview(
        self,
        *,
        task_id: str | None = None,
        days: int | None = None,
    ) -> dict:
        """Return compliance rate overview, site breakdown, and operator rankings."""
        task_filter, time_filter, params = self._build_task_time_filters(task_id=task_id, days=days)

        with self._connect() as conn:
            overall_row = conn.execute(
                f"""
                SELECT
                    COUNT(*)                                                            AS total_jobs,
                    SUM(CASE WHEN json_extract(sj.score_json, '$.summary.decision') = 'pass'
                         THEN 1 ELSE 0 END)                                             AS pass_count
                FROM score_jobs sj
                LEFT JOIN videos gv ON gv.id = sj.gold_video_id
                WHERE sj.status = 'completed' AND sj.score_json IS NOT NULL{task_filter}{time_filter}
                """,
                params,
            ).fetchone()

            total_jobs = int(overall_row["total_jobs"]) if overall_row and overall_row["total_jobs"] else 0
            pass_count = int(overall_row["pass_count"]) if overall_row and overall_row["pass_count"] else 0
            compliance_rate = round(pass_count / total_jobs, 4) if total_jobs > 0 else 0.0

            site_rows = conn.execute(
                f"""
                SELECT
                    tv.site_id                                                          AS site_id,
                    COUNT(*)                                                            AS job_count,
                    SUM(CASE WHEN json_extract(sj.score_json, '$.summary.decision') = 'pass'
                         THEN 1 ELSE 0 END)                                             AS pass_count
                FROM score_jobs sj
                LEFT JOIN videos gv ON gv.id = sj.gold_video_id
                LEFT JOIN videos tv ON tv.id = sj.trainee_video_id
                WHERE sj.status = 'completed' AND sj.score_json IS NOT NULL{task_filter}{time_filter}
                GROUP BY tv.site_id
                ORDER BY pass_count * 1.0 / COUNT(*) ASC
                """,
                params,
            ).fetchall()

            by_site = [
                {
                    "site_id": r["site_id"] or "unknown",
                    "job_count": int(r["job_count"]),
                    "pass_count": int(r["pass_count"]) if r["pass_count"] else 0,
                    "compliance_rate": round(
                        (int(r["pass_count"]) if r["pass_count"] else 0) / int(r["job_count"]), 4
                    )
                    if int(r["job_count"]) > 0
                    else 0.0,
                }
                for r in site_rows
            ]

            op_rows = conn.execute(
                f"""
                SELECT
                    tv.operator_id_hash                                                 AS operator_id,
                    COUNT(*)                                                            AS job_count,
                    SUM(CASE WHEN json_extract(sj.score_json, '$.summary.decision') = 'pass'
                         THEN 1 ELSE 0 END)                                             AS pass_count,
                    AVG(json_extract(sj.score_json, '$.score'))                         AS avg_score
                FROM score_jobs sj
                LEFT JOIN videos gv ON gv.id = sj.gold_video_id
                LEFT JOIN videos tv ON tv.id = sj.trainee_video_id
                WHERE sj.status = 'completed' AND sj.score_json IS NOT NULL{task_filter}{time_filter}
                GROUP BY tv.operator_id_hash
                HAVING COUNT(*) >= 3
                """,
                params,
            ).fetchall()

            operator_stats = [
                {
                    "operator_id": r["operator_id"] or "unknown",
                    "job_count": int(r["job_count"]),
                    "pass_count": int(r["pass_count"]) if r["pass_count"] else 0,
                    "compliance_rate": round(
                        (int(r["pass_count"]) if r["pass_count"] else 0) / int(r["job_count"]), 4
                    )
                    if int(r["job_count"]) > 0
                    else 0.0,
                    "avg_score": round(float(r["avg_score"]), 2) if r["avg_score"] is not None else 0.0,
                }
                for r in op_rows
            ]

            operator_stats_sorted = sorted(operator_stats, key=lambda o: o["compliance_rate"], reverse=True)
            top_operators = operator_stats_sorted[:5]
            bottom_operators = list(reversed(operator_stats_sorted[-5:])) if operator_stats_sorted else []

            trend_rows = conn.execute(
                f"""
                SELECT
                    json_extract(sj.score_json, '$.summary.decision')           AS decision,
                    sj.created_at
                FROM score_jobs sj
                LEFT JOIN videos gv ON gv.id = sj.gold_video_id
                WHERE sj.status = 'completed' AND sj.score_json IS NOT NULL{task_filter}{time_filter}
                ORDER BY sj.created_at ASC
                """,
                params,
            ).fetchall()

        trend_direction = "stable"
        if trend_rows:
            n = len(trend_rows)
            mid = n // 2
            first_half = trend_rows[:mid] if mid > 0 else []
            last_half = trend_rows[mid:] if mid > 0 else trend_rows
            first_pass = sum(1 for r in first_half if r["decision"] == "pass")
            last_pass = sum(1 for r in last_half if r["decision"] == "pass")
            first_rate = first_pass / len(first_half) if first_half else 0.0
            last_rate = last_pass / len(last_half) if last_half else 0.0
            diff = last_rate - first_rate
            if diff > 0.05:
                trend_direction = "improving"
            elif diff < -0.05:
                trend_direction = "declining"

        return {
            "compliance_rate": compliance_rate,
            "total_jobs": total_jobs,
            "pass_count": pass_count,
            "trend_direction": trend_direction,
            "by_site": by_site,
            "top_operators": top_operators,
            "bottom_operators": bottom_operators,
        }
