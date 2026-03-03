"""Phase 5 — Autonomous anomaly detection self-improvement.

Collects operator feedback (confirm / deny anomaly events) and uses it to
automatically adjust detector weights and cooldown periods in the ensemble,
reducing false positives while preserving true positive sensitivity.

Architecture::

    AnomalyTuner
      ├── record_feedback(event_key, confirmed)   — accept/reject operator label
      ├── apply_tuning(ensemble)                  — push updated weights/cooldowns
      ├── get_stats()                             — summary for API/UI
      └── save() / load()                         — persist feedback to JSON

Tuning logic:
    - Per-(detector, metric) confirmation rate tracked over a rolling window
    - If confirmed_rate < LOW_CONF  → raise that pair's ensemble weight (suppressed)
    - If confirmed_rate > HIGH_CONF → lower weight (trusted)
    - Cooldown adjustment: high FP rate → longer cooldown (up to 10x base)
    - Minimum 10 feedback items per pair before tuning is applied
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Tuning constants
_MIN_SAMPLES = 10          # require at least N feedback items before adjusting
_LOW_CONF = 0.35           # below this confirm-rate → suppress pair
_HIGH_CONF = 0.75          # above this → reward pair (reduce cooldown)
_WEIGHT_SUPPRESS = 0.5     # multiplier applied to ensemble weight when suppressed
_MAX_COOLDOWN_MULT = 10.0  # cooldown multiplied by at most this factor
_WINDOW_SIZE = 200         # rolling window for feedback events


@dataclass
class FeedbackRecord:
    """One operator feedback item."""

    event_key: str        # "{detector}/{metric}/{entity_id}"
    detector: str
    metric: str
    entity_id: int
    timestamp: float
    confirmed: bool       # True = real anomaly, False = false positive
    note: str = ""


@dataclass
class PairStats:
    """Aggregated stats for one (detector, metric) pair."""

    detector: str
    metric: str
    total: int = 0
    confirmed: int = 0
    denied: int = 0

    @property
    def confirmation_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.confirmed / self.total

    @property
    def fp_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.denied / self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "metric": self.metric,
            "total": self.total,
            "confirmed": self.confirmed,
            "denied": self.denied,
            "confirmation_rate": round(self.confirmation_rate, 3),
            "fp_rate": round(self.fp_rate, 3),
        }


class AnomalyTuner:
    """Feedback-driven self-improvement for AnomalyDetectorEnsemble.

    Usage::

        tuner = AnomalyTuner(Path("data/anomaly_feedback.json"))
        tuner.record_feedback("behavioral/speed_zscore/-1", confirmed=False)
        tuner.apply_tuning(ensemble)  # updates weights + cooldowns
    """

    def __init__(
        self,
        feedback_path: Path | str = Path("data/anomaly_feedback.json"),
    ) -> None:
        self._feedback_path = Path(feedback_path)
        self._records: list[FeedbackRecord] = []
        self._pair_stats: dict[str, PairStats] = {}
        self._last_tuning: float = 0.0
        self._load()

    # ── Public API ─────────────────────────────────────────────────────────

    def record_feedback(
        self,
        detector: str,
        metric: str,
        entity_id: int,
        confirmed: bool,
        note: str = "",
    ) -> None:
        """Record operator feedback for an anomaly detection.

        Args:
            detector: Detector name ("behavioral", "spatial", etc.)
            metric: Metric name ("speed_zscore", "rare_cell", etc.)
            entity_id: Entity ID from the event (-1 for scene-level)
            confirmed: True = genuine anomaly, False = false positive
            note: Optional operator note
        """
        event_key = f"{detector}/{metric}/{entity_id}"
        record = FeedbackRecord(
            event_key=event_key,
            detector=detector,
            metric=metric,
            entity_id=entity_id,
            timestamp=time.time(),
            confirmed=confirmed,
            note=note,
        )
        self._records.append(record)

        # Maintain rolling window
        if len(self._records) > _WINDOW_SIZE:
            self._records = self._records[-_WINDOW_SIZE:]

        # Update pair stats
        pair_key = f"{detector}/{metric}"
        if pair_key not in self._pair_stats:
            self._pair_stats[pair_key] = PairStats(detector=detector, metric=metric)
        stats = self._pair_stats[pair_key]
        stats.total += 1
        if confirmed:
            stats.confirmed += 1
        else:
            stats.denied += 1

        logger.info(
            "AnomalyTuner feedback: %s/%s entity=%d confirmed=%s (total=%d)",
            detector, metric, entity_id, confirmed, stats.total,
        )
        self._save()

    def apply_tuning(self, ensemble: Any) -> dict[str, Any]:
        """Apply tuning adjustments to ensemble weights and cooldowns.

        Returns a summary of changes made.
        """
        changes: list[dict] = []

        for pair_key, stats in self._pair_stats.items():
            if stats.total < _MIN_SAMPLES:
                continue  # not enough data yet

            confirm_rate = stats.confirmation_rate
            detector = stats.detector

            # Adjust ensemble weights
            if detector in ensemble._WEIGHTS:
                original_weight = ensemble._WEIGHTS[detector]
                if confirm_rate < _LOW_CONF:
                    # High FP rate → suppress this detector's weight
                    new_weight = max(0.1, original_weight * _WEIGHT_SUPPRESS)
                    if abs(new_weight - original_weight) > 0.01:
                        ensemble._WEIGHTS[detector] = new_weight
                        changes.append({
                            "action": "weight_suppressed",
                            "detector": detector,
                            "old_weight": round(original_weight, 3),
                            "new_weight": round(new_weight, 3),
                            "confirm_rate": round(confirm_rate, 3),
                        })
                elif confirm_rate > _HIGH_CONF:
                    # High TP rate → reward (restore toward 1.0)
                    baseline = 1.0
                    new_weight = min(baseline, original_weight + 0.1)
                    if abs(new_weight - original_weight) > 0.01:
                        ensemble._WEIGHTS[detector] = new_weight
                        changes.append({
                            "action": "weight_boosted",
                            "detector": detector,
                            "old_weight": round(original_weight, 3),
                            "new_weight": round(new_weight, 3),
                            "confirm_rate": round(confirm_rate, 3),
                        })

            # Adjust cooldown for high-FP pairs
            if stats.fp_rate > 0.5 and stats.total >= _MIN_SAMPLES:
                fp_rate = stats.fp_rate
                # Scale cooldown: 50% FP → 2x, 80% FP → 5x, 100% FP → 10x
                mult = min(_MAX_COOLDOWN_MULT, 1.0 + fp_rate * (fp_rate * 10))
                old_cd = ensemble._cooldown_seconds
                new_cd = old_cd * mult

                # Store per-pair cooldown override in ensemble
                if not hasattr(ensemble, "_pair_cooldowns"):
                    ensemble._pair_cooldowns = {}
                ensemble._pair_cooldowns[pair_key] = new_cd

                changes.append({
                    "action": "cooldown_extended",
                    "pair": pair_key,
                    "cooldown_multiplier": round(mult, 2),
                    "fp_rate": round(fp_rate, 3),
                })

        self._last_tuning = time.time()
        if changes:
            logger.info(
                "AnomalyTuner applied %d tuning adjustments to ensemble", len(changes)
            )
        return {
            "changes_applied": len(changes),
            "changes": changes,
            "pairs_evaluated": len(self._pair_stats),
        }

    def get_stats(self) -> dict[str, Any]:
        """Return comprehensive feedback statistics for API/UI."""
        total_feedback = len(self._records)
        confirmed = sum(1 for r in self._records if r.confirmed)
        denied = total_feedback - confirmed

        pair_stats_list = [
            s.to_dict()
            for s in sorted(
                self._pair_stats.values(),
                key=lambda x: x.total,
                reverse=True,
            )
        ]

        # Pairs that exceed suppression threshold
        suppressed = [
            s.to_dict()
            for s in self._pair_stats.values()
            if s.total >= _MIN_SAMPLES and s.confirmation_rate < _LOW_CONF
        ]

        # Pairs that are highly trusted
        trusted = [
            s.to_dict()
            for s in self._pair_stats.values()
            if s.total >= _MIN_SAMPLES and s.confirmation_rate > _HIGH_CONF
        ]

        return {
            "total_feedback": total_feedback,
            "confirmed": confirmed,
            "denied": denied,
            "overall_confirm_rate": round(confirmed / max(1, total_feedback), 3),
            "pairs_tracked": len(self._pair_stats),
            "pairs_suppressed": len(suppressed),
            "pairs_trusted": len(trusted),
            "last_tuning": self._last_tuning,
            "pair_stats": pair_stats_list,
            "suppressed_pairs": suppressed,
            "trusted_pairs": trusted,
            "min_samples_for_tuning": _MIN_SAMPLES,
        }

    def get_pair_stats(self, detector: str, metric: str) -> PairStats | None:
        """Get stats for a specific (detector, metric) pair."""
        return self._pair_stats.get(f"{detector}/{metric}")

    def reset(self) -> None:
        """Clear all feedback data and stats."""
        self._records.clear()
        self._pair_stats.clear()
        self._last_tuning = 0.0
        self._save()
        logger.info("AnomalyTuner reset")

    # ── Persistence ────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist feedback records to JSON (best-effort)."""
        try:
            self._feedback_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "records": [
                    {
                        "event_key": r.event_key,
                        "detector": r.detector,
                        "metric": r.metric,
                        "entity_id": r.entity_id,
                        "timestamp": r.timestamp,
                        "confirmed": r.confirmed,
                        "note": r.note,
                    }
                    for r in self._records
                ],
                "pair_stats": {
                    k: {
                        "detector": v.detector,
                        "metric": v.metric,
                        "total": v.total,
                        "confirmed": v.confirmed,
                        "denied": v.denied,
                    }
                    for k, v in self._pair_stats.items()
                },
            }
            with open(self._feedback_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.exception("AnomalyTuner: failed to save feedback")

    def _load(self) -> None:
        """Load persisted feedback data (best-effort)."""
        if not self._feedback_path.exists():
            return
        try:
            with open(self._feedback_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data.get("records", []):
                self._records.append(
                    FeedbackRecord(
                        event_key=r.get("event_key", ""),
                        detector=r.get("detector", ""),
                        metric=r.get("metric", ""),
                        entity_id=r.get("entity_id", -1),
                        timestamp=r.get("timestamp", 0.0),
                        confirmed=r.get("confirmed", False),
                        note=r.get("note", ""),
                    )
                )
            for k, v in data.get("pair_stats", {}).items():
                self._pair_stats[k] = PairStats(
                    detector=v.get("detector", ""),
                    metric=v.get("metric", ""),
                    total=v.get("total", 0),
                    confirmed=v.get("confirmed", 0),
                    denied=v.get("denied", 0),
                )
            logger.info(
                "AnomalyTuner loaded %d records, %d pairs",
                len(self._records), len(self._pair_stats),
            )
        except Exception:
            logger.exception("AnomalyTuner: failed to load feedback")
