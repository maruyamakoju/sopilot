from __future__ import annotations

from typing import Any

from sopilot.core.deviation_templates import annotate_deviations, generate_summary_comment


def clip_window_to_time(clips: list[dict], start_idx: int, end_idx: int) -> list[float] | None:
    if not clips:
        return None
    safe_start = max(0, min(start_idx, len(clips) - 1))
    safe_end = max(0, min(end_idx, len(clips) - 1))
    start = float(clips[safe_start]["start_sec"])
    end = float(clips[safe_end]["end_sec"])
    return [round(start, 3), round(end, 3)]


def attach_timecodes(
    deviations: list[dict],
    gold_clips: list[dict],
    trainee_clips: list[dict],
) -> list[dict]:
    enriched: list[dict] = []
    for item in deviations:
        enriched_item = dict(item)
        gold_range = item.get("gold_clip_range")
        trainee_range = item.get("trainee_clip_range")
        if isinstance(gold_range, list) and len(gold_range) == 2:
            enriched_item["gold_timecode"] = clip_window_to_time(gold_clips, gold_range[0], gold_range[1])
        if isinstance(trainee_range, list) and len(trainee_range) == 2:
            enriched_item["trainee_timecode"] = clip_window_to_time(trainee_clips, trainee_range[0], trainee_range[1])
        enriched.append(enriched_item)
    return enriched


def assign_severities(deviations: list[dict], policy: dict[str, str]) -> list[dict]:
    policy_map: dict[str, str] = dict(policy or {})
    for dev in deviations:
        dev_type = str(dev.get("type", "unknown"))
        dev["severity"] = policy_map.get(dev_type, "quality")
    return deviations


def detect_over_time(
    deviations: list[dict],
    *,
    over_time_ratio: float,
    efficiency_over_time_threshold: float,
    policy: dict[str, str],
) -> list[dict]:
    policy_map: dict[str, str] = dict(policy or {})
    if over_time_ratio > efficiency_over_time_threshold:
        deviations.append(
            {
                "type": "over_time",
                "severity": policy_map.get("over_time", "efficiency"),
                "detail": f"Over time ratio {over_time_ratio:.3f} exceeds threshold "
                f"{efficiency_over_time_threshold:.3f}",
            }
        )
    return deviations


def _score_band(score: float, *, pass_score: float, retrain_score: float) -> str:
    """Return a human-readable performance band relative to task thresholds."""
    margin = pass_score * 0.2  # 20% of pass threshold defines "excellent" zone
    if score >= pass_score + margin:
        return "excellent"
    if score >= pass_score:
        return "passing"
    if score >= retrain_score:
        return "needs_review"
    return "poor"


def make_decision(
    *,
    score: float,
    deviations: list[dict],
    pass_score: float,
    retrain_score: float,
) -> dict:
    counts = {"critical": 0, "quality": 0, "efficiency": 0}
    for dev in deviations:
        sev = str(dev.get("severity", "quality"))
        if sev not in counts:
            counts[sev] = 0
        counts[sev] += 1

    if counts.get("critical", 0) > 0:
        decision = "fail"
        reason = "critical deviation detected"
        decision_basis = "critical_deviation"
    elif score >= pass_score:
        decision = "pass"
        reason = f"score >= pass_score ({pass_score:.1f})"
        decision_basis = "score_above_threshold"
    elif score < retrain_score:
        decision = "retrain"
        reason = f"score < retrain_score ({retrain_score:.1f})"
        decision_basis = "score_below_retrain"
    else:
        decision = "needs_review"
        reason = "between retrain_score and pass_score"
        decision_basis = "score_between_thresholds"

    comment = generate_summary_comment(score, decision, counts)

    return {
        "decision": decision,
        "decision_reason": reason,
        "decision_basis": decision_basis,
        "score_band": _score_band(score, pass_score=pass_score, retrain_score=retrain_score),
        "severity_counts": counts,
        "pass_score": pass_score,
        "retrain_score": retrain_score,
        "comment_ja": comment["ja"],
        "comment_en": comment["en"],
    }


def apply_task_policy(
    result: dict,
    *,
    profile: dict,
    efficiency_over_time_threshold: float,
    default_pass_score: float,
    default_retrain_score: float,
) -> dict:
    deviations = list(result.get("deviations", []))
    policy: dict[str, str] = dict(profile.get("deviation_policy", {}))
    deviations = assign_severities(deviations, policy)
    over_time_ratio = float(result.get("metrics", {}).get("over_time_ratio", 0.0))
    deviations = detect_over_time(
        deviations,
        over_time_ratio=over_time_ratio,
        efficiency_over_time_threshold=efficiency_over_time_threshold,
        policy=policy,
    )
    result["deviations"] = deviations

    # Annotate deviations with localized template comments
    step_definitions = result.get("step_definitions") or profile.get("step_definitions")
    annotate_deviations(deviations, step_definitions=step_definitions)

    score = float(result.get("score", 0.0))
    pass_score = float(profile.get("pass_score", default_pass_score))
    retrain_score = float(profile.get("retrain_score", default_retrain_score))
    result["summary"] = make_decision(
        score=score,
        deviations=deviations,
        pass_score=pass_score,
        retrain_score=retrain_score,
    )
    return result


def build_step_spans_from_boundaries(boundaries: list[int], length: int) -> list[tuple[int, int]]:
    points = [0]
    points.extend(sorted(b for b in boundaries if 0 < b < length))
    points.append(length)
    spans: list[tuple[int, int]] = []
    for idx in range(len(points) - 1):
        start, end = points[idx], points[idx + 1]
        if end > start:
            spans.append((start, end))
    return spans


def build_score_timeline(result: dict, job_id: int) -> dict[str, Any]:
    """Build a UI-friendly timeline from a completed score job result."""
    deviations = result.get("deviations", [])
    boundaries = result.get("boundaries", {})
    gold_boundaries = boundaries.get("gold", [])
    alignment = result.get("alignment", {})
    path = alignment.get("path", [])

    # Build step spans from gold boundaries
    gold_len = max(len({g for g, _ in path}), 1)
    spans = build_step_spans_from_boundaries(gold_boundaries, gold_len)

    # Index deviations by step_index
    dev_by_step: dict[int, list[dict]] = {}
    for dev in deviations:
        si = dev.get("step_index")
        if si is not None:
            dev_by_step.setdefault(si, []).append(dev)

    steps: list[dict] = []
    for step_idx, (g_start, g_end) in enumerate(spans):
        step_devs = dev_by_step.get(step_idx, [])
        # Find matched trainee range from path
        trainee_indices = [t for g, t in path if g_start <= g < g_end]
        step: dict = {
            "step_index": step_idx,
            "gold_clip_range": [g_start, g_end - 1],
            "trainee_clip_range": [min(trainee_indices), max(trainee_indices)] if trainee_indices else None,
            "status": "ok",
            "deviations": step_devs,
        }
        # Determine step status from deviations
        for d in step_devs:
            dtype = d.get("type", "")
            if dtype == "missing_step":
                step["status"] = "missing"
                break
            if dtype == "step_deviation":
                step["status"] = "deviation"
            elif dtype == "order_swap" and step["status"] == "ok":
                step["status"] = "swapped"
        steps.append(step)

    summary = result.get("summary", {})
    return {
        "job_id": job_id,
        "score": result.get("score"),
        "decision": summary.get("decision"),
        "total_steps": len(spans),
        "steps": steps,
    }
