from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GateConfig:
    max_critical_miss_rate: float | None = None
    max_critical_false_positive_rate: float | None = None
    max_critical_miss_rate_ci95_high: float | None = None
    max_critical_false_positive_rate_ci95_high: float | None = None
    max_rescore_jitter: float | None = None
    max_dtw_p90: float | None = None
    max_drift_critical_score_psi: float | None = None
    max_drift_score_psi: float | None = None
    max_drift_dtw_psi: float | None = None
    max_critical_detected_rate_shift_abs: float | None = None
    min_num_completed_jobs: int | None = None
    min_labels_total_jobs: int | None = None
    min_labeled_jobs: int | None = None
    min_critical_positives: int | None = None
    min_critical_negatives: int | None = None
    min_coverage_rate: float | None = None
    min_rescore_pairs: int | None = None


_GATE_PROFILES: dict[str, GateConfig] = {
    # Legacy PoC thresholds used by the historical scripts in this repository.
    "legacy_poc": GateConfig(
        max_critical_miss_rate=0.10,
        max_critical_false_positive_rate=0.30,
        max_rescore_jitter=5.0,
        max_dtw_p90=0.60,
    ),
    # Research-focused defaults intended for partner-grade evidence quality.
    "research_v1": GateConfig(
        max_critical_miss_rate=0.05,
        max_critical_false_positive_rate=0.15,
        max_critical_miss_rate_ci95_high=0.10,
        max_critical_false_positive_rate_ci95_high=0.20,
        max_rescore_jitter=1.0,
        max_dtw_p90=0.20,
        min_num_completed_jobs=500,
        min_labels_total_jobs=200,
        min_labeled_jobs=200,
        min_critical_positives=30,
        min_critical_negatives=100,
        min_coverage_rate=0.98,
        min_rescore_pairs=100,
    ),
    # Mutable successor profile for threshold experiments.
    "research_v2": GateConfig(
        max_critical_miss_rate=0.05,
        max_critical_false_positive_rate=0.15,
        max_critical_miss_rate_ci95_high=0.10,
        max_critical_false_positive_rate_ci95_high=0.20,
        max_rescore_jitter=1.0,
        max_dtw_p90=0.20,
        min_num_completed_jobs=500,
        min_labels_total_jobs=200,
        min_labeled_jobs=200,
        min_critical_positives=30,
        min_critical_negatives=100,
        min_coverage_rate=0.98,
        min_rescore_pairs=100,
    ),
    # Operations-focused profile for unlabeled production runs.
    "ops_v1": GateConfig(
        max_rescore_jitter=1.5,
        max_dtw_p90=0.25,
        max_drift_critical_score_psi=0.25,
        max_drift_score_psi=0.25,
        max_drift_dtw_psi=0.25,
        max_critical_detected_rate_shift_abs=0.15,
        min_num_completed_jobs=200,
        min_rescore_pairs=50,
    ),
}
_LOCKED_GATE_PROFILES = {"research_v1"}


def available_gate_profiles() -> list[str]:
    return sorted(_GATE_PROFILES.keys())


def get_gate_profile(name: str) -> GateConfig:
    profile = _GATE_PROFILES.get(name)
    if profile is None:
        raise ValueError(f"unknown gate profile: {name}")
    return profile


def is_gate_profile_locked(name: str) -> bool:
    return str(name).strip().lower() in _LOCKED_GATE_PROFILES


def merge_gate_config(base: GateConfig, **overrides: Any) -> GateConfig:
    payload = asdict(base)
    for key, value in overrides.items():
        if key not in payload:
            continue
        if value is None:
            continue
        payload[key] = value
    return GateConfig(**payload)


def evaluate_gates(report: dict[str, Any], config: GateConfig) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    _add_threshold_check(
        checks=checks,
        name="critical_miss_rate",
        value=_as_float(report.get("critical_miss_rate")),
        threshold=_as_float(config.max_critical_miss_rate),
        comparator="le",
    )
    _add_threshold_check(
        checks=checks,
        name="critical_false_positive_rate",
        value=_as_float(report.get("critical_false_positive_rate")),
        threshold=_as_float(config.max_critical_false_positive_rate),
        comparator="le",
    )
    _add_threshold_check(
        checks=checks,
        name="critical_miss_rate_ci95_high",
        value=_as_float(_deep_get(report, "critical_confidence", "miss_rate", "ci95", "high")),
        threshold=_as_float(config.max_critical_miss_rate_ci95_high),
        comparator="le",
    )
    _add_threshold_check(
        checks=checks,
        name="critical_false_positive_rate_ci95_high",
        value=_as_float(_deep_get(report, "critical_confidence", "false_positive_rate", "ci95", "high")),
        threshold=_as_float(config.max_critical_false_positive_rate_ci95_high),
        comparator="le",
    )

    _add_threshold_check(
        checks=checks,
        name="rescore_jitter_max_delta",
        value=_as_float(_deep_get(report, "rescore_jitter", "max_delta")),
        threshold=_as_float(config.max_rescore_jitter),
        comparator="le",
    )
    _add_threshold_check(
        checks=checks,
        name="dtw_normalized_cost_p90",
        value=_as_float(_deep_get(report, "dtw_normalized_cost_stats", "p90")),
        threshold=_as_float(config.max_dtw_p90),
        comparator="le",
    )
    _add_threshold_check(
        checks=checks,
        name="drift_critical_score_psi",
        value=_as_float(_deep_get(report, "drift", "critical_score_psi")),
        threshold=_as_float(config.max_drift_critical_score_psi),
        comparator="le",
    )
    _add_threshold_check(
        checks=checks,
        name="drift_score_psi",
        value=_as_float(_deep_get(report, "drift", "score_psi")),
        threshold=_as_float(config.max_drift_score_psi),
        comparator="le",
    )
    _add_threshold_check(
        checks=checks,
        name="drift_dtw_psi",
        value=_as_float(_deep_get(report, "drift", "dtw_normalized_cost_psi")),
        threshold=_as_float(config.max_drift_dtw_psi),
        comparator="le",
    )
    _add_threshold_check(
        checks=checks,
        name="critical_detected_rate_shift_abs",
        value=_as_float(_deep_get(report, "drift", "critical_detected_rate_shift_abs")),
        threshold=_as_float(config.max_critical_detected_rate_shift_abs),
        comparator="le",
    )

    _add_threshold_check(
        checks=checks,
        name="num_completed_jobs",
        value=_as_float(report.get("num_completed_jobs")),
        threshold=_as_float(config.min_num_completed_jobs),
        comparator="ge",
    )
    _add_threshold_check(
        checks=checks,
        name="labels_total_jobs",
        value=_as_float(report.get("labels_total_jobs")),
        threshold=_as_float(config.min_labels_total_jobs),
        comparator="ge",
    )
    _add_threshold_check(
        checks=checks,
        name="labels_labeled_jobs",
        value=_as_float(report.get("labels_labeled_jobs")),
        threshold=_as_float(config.min_labeled_jobs),
        comparator="ge",
    )
    _add_threshold_check(
        checks=checks,
        name="critical_positives",
        value=_as_float(_critical_positives(report)),
        threshold=_as_float(config.min_critical_positives),
        comparator="ge",
    )
    _add_threshold_check(
        checks=checks,
        name="critical_negatives",
        value=_as_float(_critical_negatives(report)),
        threshold=_as_float(config.min_critical_negatives),
        comparator="ge",
    )
    _add_threshold_check(
        checks=checks,
        name="coverage_rate",
        value=_as_float(report.get("coverage_rate")),
        threshold=_as_float(config.min_coverage_rate),
        comparator="ge",
    )
    _add_threshold_check(
        checks=checks,
        name="rescore_pairs",
        value=_as_float(_deep_get(report, "rescore_jitter", "num_pairs_with_repeats")),
        threshold=_as_float(config.min_rescore_pairs),
        comparator="ge",
    )

    overall = all(item["pass"] for item in checks if item["enabled"])
    return {"overall_pass": overall, "checks": checks}


def _add_threshold_check(
    *,
    checks: list[dict[str, Any]],
    name: str,
    value: float | None,
    threshold: float | None,
    comparator: str,
) -> None:
    if threshold is None:
        checks.append(
            {
                "name": name,
                "enabled": False,
                "pass": True,
                "value": value,
                "threshold": threshold,
                "reason": "disabled",
            }
        )
        return

    if value is None:
        checks.append(
            {
                "name": name,
                "enabled": True,
                "pass": False,
                "value": value,
                "threshold": threshold,
                "reason": "value_missing",
            }
        )
        return

    passed = float(value) <= float(threshold) if comparator == "le" else float(value) >= float(threshold)
    checks.append(
        {
            "name": name,
            "enabled": True,
            "pass": bool(passed),
            "value": float(value),
            "threshold": float(threshold),
            "reason": "ok" if passed else "threshold_violation",
        }
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _deep_get(payload: Any, *path: str) -> Any:
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _critical_positives(report: dict[str, Any]) -> int | None:
    direct = report.get("critical_positives")
    if direct is not None:
        try:
            return int(direct)
        except Exception:
            pass
    tp = _as_float(_deep_get(report, "critical_confusion", "tp"))
    fn = _as_float(_deep_get(report, "critical_confusion", "fn"))
    if tp is None or fn is None:
        return None
    return int(tp + fn)


def _critical_negatives(report: dict[str, Any]) -> int | None:
    direct = report.get("critical_negatives")
    if direct is not None:
        try:
            return int(direct)
        except Exception:
            pass
    fp = _as_float(_deep_get(report, "critical_confusion", "fp"))
    tn = _as_float(_deep_get(report, "critical_confusion", "tn"))
    if fp is None or tn is None:
        return None
    return int(fp + tn)
