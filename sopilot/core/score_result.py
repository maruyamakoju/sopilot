from __future__ import annotations

import json
from typing import Any, TypedDict, cast


class ScoreSummary(TypedDict, total=False):
    decision: str
    decision_reason: str
    severity_counts: dict[str, int]
    pass_score: float
    retrain_score: float


class ScoreMetrics(TypedDict, total=False):
    miss_steps: int
    swap_steps: int
    deviation_steps: int
    over_time_ratio: float
    dtw_normalized_cost: float


class ScoreResult(TypedDict, total=False):
    score: float
    summary: ScoreSummary
    metrics: ScoreMetrics
    deviations: list[dict[str, Any]]
    boundaries: dict[str, list[int]]
    alignment: dict[str, Any]
    confidence: dict[str, Any]
    step_contributions: list[dict[str, Any]]
    time_compliance: list[dict[str, Any]]
    gold_video_id: int
    trainee_video_id: int
    task_id: str


def parse_score_json(raw_score_json: str | None) -> ScoreResult | None:
    """Parse stored score JSON safely; invalid payloads return None."""
    if not raw_score_json:
        return None
    try:
        parsed = json.loads(raw_score_json)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return cast(ScoreResult, parsed)


def enrich_score_result(result: ScoreResult) -> ScoreResult:
    """Add derived display fields (decision_basis, score_band) to a score result in-place.

    Safe to call on results that were stored before these fields existed — it
    back-fills them from existing summary data without mutating the stored JSON.
    Returns the same dict (mutated) for chaining.
    """
    summary = result.get("summary")
    if not summary:
        return result
    if "decision_basis" not in summary or "score_band" not in summary:
        from sopilot.core.score_pipeline import _score_band

        decision = summary.get("decision", "")
        severity_counts = summary.get("severity_counts") or {}
        pass_score = float(summary.get("pass_score") or 60.0)
        retrain_score = float(summary.get("retrain_score") or 50.0)
        score = float(result.get("score") or 0.0)

        if "decision_basis" not in summary:
            if severity_counts.get("critical", 0) > 0:
                summary["decision_basis"] = "critical_deviation"
            elif decision == "pass":
                summary["decision_basis"] = "score_above_threshold"
            elif decision == "retrain":
                summary["decision_basis"] = "score_below_retrain"
            elif decision == "needs_review":
                summary["decision_basis"] = "score_between_thresholds"
            else:
                summary["decision_basis"] = "unknown"

        if "score_band" not in summary:
            summary["score_band"] = _score_band(
                score, pass_score=pass_score, retrain_score=retrain_score
            )
    return result


def summarize_score_result(result: ScoreResult | None) -> dict[str, Any]:
    """Extract lightweight fields used in score-job list responses."""
    if result is None:
        return {
            "score": None,
            "decision": None,
            "severity_counts": None,
            "score_band": None,
        }
    summary = result.get("summary") or {}
    score_band = summary.get("score_band")
    if score_band is None and result.get("score") is not None:
        from sopilot.core.score_pipeline import _score_band
        pass_score = float(summary.get("pass_score") or 60.0)
        retrain_score = float(summary.get("retrain_score") or 50.0)
        score_band = _score_band(float(result["score"]), pass_score=pass_score, retrain_score=retrain_score)
    return {
        "score": result.get("score"),
        "decision": summary.get("decision"),
        "severity_counts": summary.get("severity_counts"),
        "score_band": score_band,
    }
