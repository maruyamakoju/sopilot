"""SOP video scoring pipeline: DTW alignment evaluation and step-level analysis.

Produces a numeric score in [0, 100] with per-step contributions, deviation
diagnostics, and optional Soft-DTW / uncertainty enrichment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from sopilot.constants import (
    COLLAPSE_COST_FACTOR,
    COLLAPSE_UNIQUE_VS_EXPECTED_RATIO,
    COLLAPSE_UNIQUE_VS_GOLD_RATIO,
    DEFAULT_WEIGHTS,
    OVER_TIME_CAP,
)
from sopilot.core.dtw import DTWAlignment

logger = logging.getLogger(__name__)

# Uncertainty module — imported lazily to avoid circular issues at collection time
try:
    from sopilot.core.uncertainty import BootstrapCI, heuristic_ci as _heuristic_ci
    _UNCERTAINTY_AVAILABLE = True
except Exception:  # pragma: no cover
    _UNCERTAINTY_AVAILABLE = False
    BootstrapCI = None  # type: ignore[assignment,misc]
    _heuristic_ci = None  # type: ignore[assignment]

# Soft-DTW module — optional enrichment only, never required for scoring
try:
    from sopilot.core.soft_dtw import soft_dtw as _soft_dtw
    _SOFT_DTW_AVAILABLE = True
except Exception:  # pragma: no cover
    _SOFT_DTW_AVAILABLE = False
    _soft_dtw = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ScoreWeights:
    """Normalized penalty weights for the four scoring dimensions."""
    w_miss: float = DEFAULT_WEIGHTS["w_miss"]
    w_swap: float = DEFAULT_WEIGHTS["w_swap"]
    w_dev: float = DEFAULT_WEIGHTS["w_dev"]
    w_time: float = DEFAULT_WEIGHTS["w_time"]

    def normalized(self) -> ScoreWeights:
        """Return a copy where weights sum to 1.0."""
        total = self.w_miss + self.w_swap + self.w_dev + self.w_time
        if total <= 1e-12:
            return ScoreWeights()
        return ScoreWeights(
            w_miss=self.w_miss / total,
            w_swap=self.w_swap / total,
            w_dev=self.w_dev / total,
            w_time=self.w_time / total,
        )


def _build_step_spans(boundaries: list[int], length: int) -> list[tuple[int, int]]:
    """Convert boundary indices into (start, end) spans."""
    points = [0]
    points.extend(sorted([b for b in boundaries if 0 < b < length]))
    points.append(length)
    spans: list[tuple[int, int]] = []
    for idx in range(len(points) - 1):
        start = points[idx]
        end = points[idx + 1]
        if end > start:
            spans.append((start, end))
    return spans


def score_alignment(
    alignment: DTWAlignment,
    gold_len: int,
    trainee_len: int,
    gold_boundaries: list[int],
    trainee_boundaries: list[int],
    weights: ScoreWeights,
    deviation_threshold: float = 0.25,
    *,
    gold_embeddings: Optional[np.ndarray] = None,
    trainee_embeddings: Optional[np.ndarray] = None,
) -> dict:
    """Evaluate a DTW alignment and produce a score in [0, 100].

    Args:
        alignment: DTWAlignment result from dtw_align().
        gold_len: Number of gold clips.
        trainee_len: Number of trainee clips.
        gold_boundaries: Step boundary indices for gold video.
        trainee_boundaries: Step boundary indices for trainee video (reserved).
        weights: Penalty weights for scoring dimensions.
        deviation_threshold: Cosine distance threshold for step deviations.
        gold_embeddings: Optional raw embeddings for Soft-DTW enrichment.
        trainee_embeddings: Optional raw embeddings for Soft-DTW enrichment.

    Returns:
        Score result dict with 'score', 'metrics', 'deviations', etc.
    """
    _ = trainee_boundaries  # kept for future task-specific score logic
    if gold_len <= 0 or trainee_len <= 0:
        raise ValueError("gold_len and trainee_len must be positive")

    norm_weights = weights.normalized()
    gold_spans = _build_step_spans(gold_boundaries, gold_len)

    deviations: list[dict] = []
    step_centers: list[float] = []
    miss = 0
    deviation = 0

    indexed_path = list(zip(alignment.path, alignment.path_costs, strict=False))
    for step_idx, (g_start, g_end) in enumerate(gold_spans):
        matched = [
            (pair, cost)
            for pair, cost in indexed_path
            if g_start <= pair[0] < g_end
        ]
        if not matched:
            miss += 1
            deviations.append(
                {
                    "type": "missing_step",
                    "step_index": step_idx,
                    "gold_clip_range": [g_start, g_end - 1],
                    "detail": "No aligned trainee segment for this gold step.",
                }
            )
            continue

        trainee_positions = [pair[1] for pair, _ in matched]
        unique_positions = sorted(set(trainee_positions))
        unique_count = len(unique_positions)
        gold_span_len = max(g_end - g_start, 1)
        expected_unique = max(1, int(round(gold_span_len * trainee_len / max(gold_len, 1))))
        center = float(np.mean(trainee_positions))
        step_centers.append(center)

        mean_cost = float(np.mean([cost for _, cost in matched]))
        unique_vs_gold = unique_count / max(gold_span_len, 1)
        unique_vs_expected = unique_count / max(expected_unique, 1)
        if (
            (unique_count <= 1 and gold_span_len >= 2)
            or (unique_vs_gold < COLLAPSE_UNIQUE_VS_GOLD_RATIO and mean_cost > (deviation_threshold * COLLAPSE_COST_FACTOR))
            or (unique_vs_expected < COLLAPSE_UNIQUE_VS_EXPECTED_RATIO and mean_cost > (deviation_threshold * COLLAPSE_COST_FACTOR))
        ):
            miss += 1
            deviations.append(
                {
                    "type": "missing_step",
                    "step_index": step_idx,
                    "gold_clip_range": [g_start, g_end - 1],
                    "trainee_clip_range": [int(min(trainee_positions)), int(max(trainee_positions))],
                    "detail": "Aligned span is overly collapsed; step likely skipped or incomplete.",
                    "unique_positions": unique_count,
                    "gold_span_len": gold_span_len,
                    "expected_span_len": expected_unique,
                    "mean_distance": round(mean_cost, 6),
                }
            )
            continue

        if mean_cost > deviation_threshold:
            deviation += 1
            deviations.append(
                {
                    "type": "step_deviation",
                    "step_index": step_idx,
                    "gold_clip_range": [g_start, g_end - 1],
                    "trainee_clip_range": [int(min(trainee_positions)), int(max(trainee_positions))],
                    "distance": mean_cost,
                }
            )

    swap = 0
    for idx in range(1, len(step_centers)):
        if step_centers[idx] + 1e-6 < step_centers[idx - 1]:
            swap += 1
            deviations.append(
                {
                    "type": "order_swap",
                    "step_index": idx,
                    "detail": "Step order appears inconsistent with gold trajectory.",
                }
            )

    over_time = max(0.0, (trainee_len - gold_len) / max(gold_len, 1))

    num_steps = max(len(gold_spans), 1)
    # Clamp ratios to [0, 1] to guarantee score stays in [0, 100]
    miss_ratio = min(miss / num_steps, 1.0)
    swap_ratio = min(swap / max(num_steps - 1, 1), 1.0)
    dev_ratio = min(deviation / num_steps, 1.0)
    over_time_ratio = float(min(over_time, OVER_TIME_CAP))

    penalty = (
        norm_weights.w_miss * miss_ratio
        + norm_weights.w_swap * swap_ratio
        + norm_weights.w_dev * dev_ratio
        + norm_weights.w_time * over_time_ratio
    )
    # Clamp penalty to [0, 1] — mathematically should hold given clamped ratios
    # and normalized weights, but enforce defensively.
    penalty = max(0.0, min(1.0, penalty))
    score = 100.0 * (1.0 - penalty)

    result_dict: dict = {
        "score": round(score, 2),
        "metrics": {
            "miss_steps": miss,
            "swap_steps": swap,
            "deviation_steps": deviation,
            "over_time_ratio": round(over_time, 4),
            "dtw_normalized_cost": round(alignment.normalized_cost, 6),
            "total_steps": num_steps,
            "penalty_breakdown": {
                "miss_penalty": round(norm_weights.w_miss * miss_ratio, 6),
                "swap_penalty": round(norm_weights.w_swap * swap_ratio, 6),
                "dev_penalty": round(norm_weights.w_dev * dev_ratio, 6),
                "time_penalty": round(norm_weights.w_time * over_time_ratio, 6),
                "total_penalty": round(penalty, 6),
            },
        },
        "boundaries": {
            "gold": gold_boundaries,
            "trainee": trainee_boundaries,
        },
        "alignment": {
            "path": [[g, t] for g, t in alignment.path],
            "normalized_cost": round(alignment.normalized_cost, 6),
        },
        "deviations": deviations,
    }

    # Optional Soft-DTW enrichment — never allowed to fail scoring
    if (
        _SOFT_DTW_AVAILABLE
        and _soft_dtw is not None
        and gold_embeddings is not None
        and trainee_embeddings is not None
        and len(gold_embeddings) >= 2
        and len(trainee_embeddings) >= 2
    ):
        try:
            sdtw_result = _soft_dtw(gold_embeddings, trainee_embeddings)
            result_dict["soft_dtw_distance"] = round(float(sdtw_result.distance), 6)
            result_dict["soft_dtw_normalized"] = round(float(sdtw_result.normalized_cost), 6)
        except Exception as exc:
            logger.warning("Soft-DTW enrichment failed: %s", exc)

    return result_dict


def compute_step_contributions(
    deviations: list[dict],
    boundaries: list[int],
    gold_len: int,
    weights: ScoreWeights,
    step_definitions: list[dict] | None = None,
) -> list[dict]:
    """Compute a per-step score breakdown as an explanatory tool.

    Each step receives an equal share of 100 points.  Deductions are applied
    based on the deviations that belong to that step.  The sum of
    ``points_earned`` values is an approximation of the overall score and need
    not match the official ``score`` field exactly, because the main scoring
    formula uses aggregate ratios rather than per-step arithmetic.

    Args:
        deviations: List of deviation dicts produced by :func:`score_alignment`
            (possibly enriched with severity by the task policy).  Each dict
            may contain ``step_index``, ``type``, and optionally ``severity``.
        boundaries: Gold boundary indices as stored in ``result["boundaries"]["gold"]``.
        gold_len: Total number of gold clips (``result["metrics"]["gold_length"]``
            or ``len(gold_embeddings)``).
        weights: The :class:`ScoreWeights` instance used for scoring.

    Returns:
        A list of dicts, one per step, with keys ``step_index``,
        ``points_possible``, ``points_earned``, and ``deductions``.
        Returns an empty list when no steps can be determined.
    """
    if gold_len <= 0:
        return []

    spans = _build_step_spans(boundaries, gold_len)
    n_steps = len(spans)
    if n_steps == 0:
        return []

    norm_w = weights.normalized()
    w_total = norm_w.w_miss + norm_w.w_swap + norm_w.w_dev + norm_w.w_time
    if w_total <= 1e-12:
        w_total = 1.0

    points_per_step: float = 100.0 / n_steps

    # Severity factor lookup for step_deviation deductions
    _severity_factor: dict[str, float] = {
        "critical": 1.0,
        "quality": 0.5,
        "efficiency": 0.25,
    }

    # Group deviations by step_index.  Over-time deviations lack a step_index
    # and are applied as a uniform deduction across all steps (handled below).
    devs_by_step: dict[int, list[dict]] = {}
    over_time_devs: list[dict] = []
    for dev in deviations:
        si = dev.get("step_index")
        dtype = str(dev.get("type", ""))
        if dtype == "over_time" or si is None:
            over_time_devs.append(dev)
        else:
            devs_by_step.setdefault(int(si), []).append(dev)

    # Per-step over_time deduction (spread evenly)
    over_time_deduction_per_step: float = 0.0
    if over_time_devs:
        over_time_deduction_per_step = (
            points_per_step * 0.2 * (norm_w.w_time / w_total) * len(over_time_devs)
        )

    contributions: list[dict] = []
    for step_idx in range(n_steps):
        possible = points_per_step
        deductions_log: list[dict] = []
        total_deduction = 0.0

        for dev in devs_by_step.get(step_idx, []):
            dtype = str(dev.get("type", ""))
            if dtype == "missing_step":
                amount = possible * (norm_w.w_miss / w_total)
                deductions_log.append({"type": "missing_step", "amount": round(amount, 4)})
                total_deduction += amount
            elif dtype == "step_deviation":
                severity = str(dev.get("severity", "quality"))
                factor = _severity_factor.get(severity, 0.5)
                amount = possible * 0.5 * factor * (norm_w.w_dev / w_total)
                deductions_log.append({"type": "step_deviation", "amount": round(amount, 4)})
                total_deduction += amount
            elif dtype == "order_swap":
                amount = possible * 0.3 * (norm_w.w_swap / w_total)
                deductions_log.append({"type": "order_swap", "amount": round(amount, 4)})
                total_deduction += amount
            elif dtype == "over_time":
                # Unlikely (over_time devs usually lack step_index) but handle it
                amount = possible * 0.2 * (norm_w.w_time / w_total)
                deductions_log.append({"type": "over_time", "amount": round(amount, 4)})
                total_deduction += amount

        # Apply the evenly-spread over_time deduction
        if over_time_deduction_per_step > 0.0:
            deductions_log.append(
                {"type": "over_time", "amount": round(over_time_deduction_per_step, 4)}
            )
            total_deduction += over_time_deduction_per_step

        earned = max(0.0, possible - total_deduction)
        contributions.append(
            {
                "step_index": step_idx,
                "points_possible": round(possible, 4),
                "points_earned": round(earned, 4),
                "deductions": deductions_log,
            }
        )

    results = contributions

    # Merge step definition metadata if provided
    if step_definitions:
        step_def_map = {s["step_index"]: s for s in step_definitions}
        for r in results:
            defn = step_def_map.get(r["step_index"], {})
            r["name_ja"] = defn.get("name_ja", f"手順{r['step_index'] + 1}")
            r["name_en"] = defn.get("name_en", f"Step {r['step_index'] + 1}")
            r["is_critical"] = bool(defn.get("is_critical", False))

    return results


def compute_score_confidence(
    result_score: float | None,
    deviations: list[dict],
    metrics: dict,
    *,
    gold_len: int,
    trainee_len: int,
) -> dict | None:
    """Compute a heuristic confidence interval for a score result.

    Delegates to :func:`sopilot.core.uncertainty.heuristic_ci` when that
    module is available.  Falls back to the original inline heuristic
    otherwise so that scoring never breaks.

    Returns:
        A dict with keys ``ci_low``, ``ci_high``, ``ci_half_width``,
        ``level``, and ``stability``, or ``None`` when the score or
        metrics are not available.
    """
    if result_score is None:
        return None
    if not metrics:
        return None

    total_clips = gold_len + trainee_len
    dtw_cost: float = float(metrics.get("dtw_normalized_cost", 0.0))

    # --- Delegate to improved heuristic_ci when uncertainty module available ---
    if _UNCERTAINTY_AVAILABLE and _heuristic_ci is not None:
        try:
            bci = _heuristic_ci(
                base_score=float(result_score),
                dtw_cost=dtw_cost,
                n_clips=total_clips,
            )
            ci_half_width = bci.width / 2.0
            return {
                "ci_low": round(bci.lower, 2),
                "ci_high": round(bci.upper, 2),
                "ci_half_width": round(ci_half_width, 2),
                "level": 0.95,
                "stability": bci.stability,
            }
        except Exception as exc:
            logger.warning("Uncertainty module heuristic_ci failed: %s", exc)

    # --- Legacy heuristic fallback (original behaviour) ---
    if total_clips >= 40:
        base_ci: float = 3.0
    elif total_clips >= 20:
        base_ci = 5.0
    elif total_clips >= 10:
        base_ci = 8.0
    else:
        base_ci = 12.0

    if dtw_cost < 0.05:
        adjustment: float = 0.0
    elif dtw_cost < 0.15:
        adjustment = 2.0
    else:
        adjustment = 4.0

    ci_half_width = base_ci + adjustment
    ci_low: float = max(0.0, result_score - ci_half_width)
    ci_high: float = min(100.0, result_score + ci_half_width)

    if ci_half_width <= 4.0:
        stability = "high"
    elif ci_half_width <= 8.0:
        stability = "medium"
    else:
        stability = "low"

    return {
        "ci_low": round(ci_low, 2),
        "ci_high": round(ci_high, 2),
        "ci_half_width": round(ci_half_width, 2),
        "level": 0.95,
        "stability": stability,
    }


def compute_time_compliance_per_step(
    boundaries: list[int],
    step_definitions: list[dict],
    *,
    clip_seconds: int = 4,
) -> list[dict]:
    """Compute per-step time compliance given gold boundaries and step definitions.

    Args:
        boundaries: Gold video clip boundary indices (length = n_steps + 1).
                    boundaries[i] to boundaries[i+1] is the clip range for step i.
        step_definitions: List of step definition dicts (from StepDefinitionService).
                          Each may have min_duration_sec / max_duration_sec.
        clip_seconds: Duration of each clip in seconds (from Settings.clip_seconds).

    Returns:
        List of per-step compliance dicts (one per step).
    """
    step_def_map = {s["step_index"]: s for s in step_definitions}
    n_steps = max(len(boundaries) - 1, 0)
    results: list[dict] = []

    for idx in range(n_steps):
        clip_count = boundaries[idx + 1] - boundaries[idx]
        actual_sec = clip_count * clip_seconds
        defn = step_def_map.get(idx, {})
        mn = defn.get("min_duration_sec")
        mx = defn.get("max_duration_sec")

        if mn is None and mx is None:
            compliance = "undefined"
        elif mn is not None and actual_sec < mn:
            compliance = "too_fast"
        elif mx is not None and actual_sec > mx:
            compliance = "too_slow"
        else:
            compliance = "ok"

        results.append(
            {
                "step_index": idx,
                "name_ja": defn.get("name_ja", f"手順{idx + 1}"),
                "name_en": defn.get("name_en", f"Step {idx + 1}"),
                "actual_duration_sec": round(float(actual_sec), 1),
                "expected_duration_sec": defn.get("expected_duration_sec"),
                "min_duration_sec": mn,
                "max_duration_sec": mx,
                "is_critical": bool(defn.get("is_critical", False)),
                "compliance": compliance,
            }
        )

    return results
