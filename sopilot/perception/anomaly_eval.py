"""Anomaly detection evaluation and threshold tuning.

Provides tools to:
1. Evaluate ensemble performance against labeled anomaly data
2. Find optimal sigma_threshold via grid search
3. Identify and filter false positive patterns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sopilot.perception.types import WorldState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LabeledAnomalyEvent
# ---------------------------------------------------------------------------


@dataclass
class LabeledAnomalyEvent:
    """A world state snapshot with a ground truth anomaly label.

    Used to build evaluation datasets for the AnomalyDetectorEnsemble.
    """

    world_state: WorldState
    is_anomaly: bool          # ground truth label
    frame_number: int
    timestamp: float
    note: str = ""


# ---------------------------------------------------------------------------
# AnomalyEvalResult
# ---------------------------------------------------------------------------


@dataclass
class AnomalyEvalResult:
    """Evaluation result for a single threshold setting."""

    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def precision(self) -> float:
        """TP / (TP + FP). Returns 0.0 if denominator is zero."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """TP / (TP + FN). Returns 0.0 if denominator is zero."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """(TP + TN) / total. Returns 0.0 if no samples."""
        total = (
            self.true_positives
            + self.false_positives
            + self.false_negatives
            + self.true_negatives
        )
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a JSON-friendly dict."""
        return {
            "threshold": self.threshold,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
        }


# ---------------------------------------------------------------------------
# AnomalyEvaluator
# ---------------------------------------------------------------------------


class AnomalyEvaluator:
    """Evaluates AnomalyDetectorEnsemble performance on labeled data.

    Usage::

        ensemble = AnomalyDetectorEnsemble(config)
        evaluator = AnomalyEvaluator(ensemble)
        result = evaluator.evaluate(labeled_events)
        best_thresh, best_result = evaluator.find_optimal_threshold(labeled_events)
        report = evaluator.generate_report()
    """

    def __init__(self, ensemble: Any) -> None:
        self._ensemble = ensemble
        self._results: list[AnomalyEvalResult] = []

    def evaluate(
        self,
        labeled_events: list[LabeledAnomalyEvent],
        threshold: float | None = None,
    ) -> AnomalyEvalResult:
        """Run evaluation at the given threshold (or the ensemble's current value).

        For each labeled event:
        1. Feed world_state through ensemble via observe() + check_anomalies()
        2. Compare detected anomalies vs ground truth label

        A frame is classified as "detected anomaly" if check_anomalies() returns
        at least one EntityEvent.

        Returns an AnomalyEvalResult with TP/FP/FN/TN counts and derived metrics.
        """
        # Save current threshold and optionally override
        original_threshold = self._ensemble._sigma_threshold
        if threshold is not None:
            self._ensemble._sigma_threshold = threshold
        used_threshold = self._ensemble._sigma_threshold

        # Reset ensemble state so evaluation is independent of prior history
        self._ensemble.reset()

        tp = fp = fn = tn = 0

        for event in labeled_events:
            self._ensemble.observe(event.world_state)
            detected_events = self._ensemble.check_anomalies(event.world_state)
            detected = len(detected_events) > 0

            if event.is_anomaly and detected:
                tp += 1
            elif not event.is_anomaly and detected:
                fp += 1
            elif event.is_anomaly and not detected:
                fn += 1
            else:
                tn += 1

        # Restore original threshold
        self._ensemble._sigma_threshold = original_threshold

        result = AnomalyEvalResult(
            threshold=used_threshold,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
        )
        self._results.append(result)

        logger.info(
            "Evaluation at threshold=%.2f: P=%.3f R=%.3f F1=%.3f Acc=%.3f",
            used_threshold,
            result.precision,
            result.recall,
            result.f1,
            result.accuracy,
        )
        return result

    def find_optimal_threshold(
        self,
        labeled_events: list[LabeledAnomalyEvent],
        thresholds: list[float] | None = None,
        metric: str = "f1",  # "f1" | "precision" | "recall"
    ) -> tuple[float, AnomalyEvalResult]:
        """Grid search over sigma_threshold values to find the best setting.

        Args:
            labeled_events: Labeled evaluation data.
            thresholds: Candidate threshold values to try.
                        Defaults to [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0].
            metric: Optimization target — "f1", "precision", or "recall".

        Returns:
            (best_threshold, best_result) tuple.
        """
        if thresholds is None:
            thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

        if metric not in ("f1", "precision", "recall"):
            raise ValueError(f"metric must be 'f1', 'precision', or 'recall'; got {metric!r}")

        best_result: AnomalyEvalResult | None = None
        best_score = -1.0

        for thresh in thresholds:
            result = self.evaluate(labeled_events, threshold=thresh)
            score = getattr(result, metric)
            if score > best_score:
                best_score = score
                best_result = result

        assert best_result is not None  # thresholds list is non-empty
        logger.info(
            "Optimal threshold=%.2f (%s=%.4f)",
            best_result.threshold,
            metric,
            best_score,
        )
        return best_result.threshold, best_result

    def generate_report(self) -> dict[str, Any]:
        """Return all evaluation results accumulated so far as a dict."""
        return {
            "total_evaluations": len(self._results),
            "results": [r.to_dict() for r in self._results],
        }


# ---------------------------------------------------------------------------
# AnomalyFalsePositiveFilter
# ---------------------------------------------------------------------------


class AnomalyFalsePositiveFilter:
    """Tracks FP patterns and auto-raises cooldown for noisy detectors.

    Maintains per-(detector, metric) FP rate.  When the FP rate exceeds
    ``fp_threshold``, ``apply_to_ensemble()`` multiplies the cooldown
    for that (detector, metric, entity_id) key to suppress the noisy signal.

    Usage::

        fp_filter = AnomalyFalsePositiveFilter(fp_threshold=0.7)
        for event in evaluation_events:
            fp_filter.record("behavioral", "speed_zscore", is_fp=True)
        fp_filter.apply_to_ensemble(ensemble)
    """

    def __init__(
        self,
        fp_threshold: float = 0.7,
        cooldown_multiplier: float = 3.0,
    ) -> None:
        self._fp_counts: dict[str, int] = {}    # "detector/metric" -> FP count
        self._total_counts: dict[str, int] = {}  # "detector/metric" -> total count
        self._fp_threshold = fp_threshold
        self._cooldown_multiplier = cooldown_multiplier

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(detector: str, metric: str) -> str:
        return f"{detector}/{metric}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, detector: str, metric: str, is_fp: bool) -> None:
        """Record a TP or FP detection for a (detector, metric) pair."""
        key = self._key(detector, metric)
        self._total_counts[key] = self._total_counts.get(key, 0) + 1
        if is_fp:
            self._fp_counts[key] = self._fp_counts.get(key, 0) + 1

    def get_fp_rate(self, detector: str, metric: str) -> float:
        """Return the current FP rate for a (detector, metric) pair (0.0–1.0).

        Returns 0.0 if no observations have been recorded.
        """
        key = self._key(detector, metric)
        total = self._total_counts.get(key, 0)
        if total == 0:
            return 0.0
        return self._fp_counts.get(key, 0) / total

    def get_suppressed_pairs(self) -> list[tuple[str, str]]:
        """Return (detector, metric) pairs whose FP rate exceeds fp_threshold."""
        suppressed: list[tuple[str, str]] = []
        for key, total in self._total_counts.items():
            if total == 0:
                continue
            fp_count = self._fp_counts.get(key, 0)
            rate = fp_count / total
            if rate >= self._fp_threshold:
                detector, metric = key.split("/", 1)
                suppressed.append((detector, metric))
        return suppressed

    def apply_to_ensemble(self, ensemble: Any) -> None:
        """Increase cooldown for suppressed (detector, metric) pairs.

        For every key in the ensemble's ``_cooldown_map`` whose (detector, metric)
        component is in the suppressed set, the recorded timestamp is shifted
        forward by ``(cooldown_multiplier - 1) * cooldown_seconds`` so that
        the next alert for that signal is delayed.

        Also sets a ``_fp_cooldown_overrides`` dict on the ensemble so that
        future events created by suppressed pairs use the extended cooldown.
        """
        suppressed_pairs = set(self.get_suppressed_pairs())
        if not suppressed_pairs:
            return

        base_cooldown = ensemble._cooldown_seconds
        extended_cooldown = base_cooldown * self._cooldown_multiplier

        # Patch existing cooldown map entries for suppressed pairs
        updated_keys: list[tuple[str, str, int]] = []
        for cooldown_key, last_alert_ts in list(ensemble._cooldown_map.items()):
            detector_name, metric_name, _entity_id = cooldown_key
            if (detector_name, metric_name) in suppressed_pairs:
                # Extend effective cooldown by adjusting the recorded timestamp
                extension = extended_cooldown - base_cooldown
                ensemble._cooldown_map[cooldown_key] = last_alert_ts + extension
                updated_keys.append(cooldown_key)

        # Store override so new firings also use the extended cooldown
        if not hasattr(ensemble, "_fp_cooldown_overrides"):
            ensemble._fp_cooldown_overrides = {}
        for det, met in suppressed_pairs:
            ensemble._fp_cooldown_overrides[(det, met)] = extended_cooldown

        logger.info(
            "AnomalyFalsePositiveFilter: suppressed %d pairs, patched %d cooldown entries",
            len(suppressed_pairs),
            len(updated_keys),
        )

    def get_stats(self) -> dict[str, Any]:
        """Return a stats dict suitable for API exposure."""
        pairs: list[dict[str, Any]] = []
        for key in self._total_counts:
            total = self._total_counts[key]
            fp_count = self._fp_counts.get(key, 0)
            rate = fp_count / total if total > 0 else 0.0
            detector, metric = key.split("/", 1)
            pairs.append(
                {
                    "detector": detector,
                    "metric": metric,
                    "total_detections": total,
                    "fp_count": fp_count,
                    "fp_rate": round(rate, 4),
                    "suppressed": rate >= self._fp_threshold,
                }
            )

        return {
            "fp_threshold": self._fp_threshold,
            "cooldown_multiplier": self._cooldown_multiplier,
            "pairs": pairs,
            "suppressed_count": len(self.get_suppressed_pairs()),
        }
