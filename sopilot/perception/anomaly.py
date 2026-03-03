"""Autonomous Anomaly Detection — 4-detector ensemble.

Detects anomalies *without explicit rules* by learning "normal" patterns
from continuous observation and flagging deviations.  This is the core of
the "Camera OS" vision: human-like ability to sense "something is off".

Architecture::

    AnomalyDetectorEnsemble (orchestrator)
      ├── BehavioralAnomalyDetector  — speed & activity pattern anomalies
      ├── SpatialAnomalyDetector     — occupancy heatmap deviations
      ├── TemporalPatternDetector    — time-of-day density anomalies
      └── InteractionAnomalyDetector — novel entity-pair relations

All detectors share the ``observe()`` / ``check()`` interface and use EMA
(exponential moving averages) for online learning.  Pure numpy + stdlib —
no external ML libraries required.

Thread safety:
    The ensemble is always called under the WorldModel's RLock, so
    individual detectors do not need their own locks.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from sopilot.perception.types import (
    EntityEvent,
    EntityEventType,
    PerceptionConfig,
    ViolationSeverity,
    WorldState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AnomalySignal — internal signal emitted by each detector
# ---------------------------------------------------------------------------


@dataclass
class AnomalySignal:
    """Internal signal emitted by a single anomaly detector.

    Carries enough context for the ensemble to weight, deduplicate,
    and convert into EntityEvent objects.
    """

    detector: str          # detector name: "behavioral", "spatial", "temporal", "interaction"
    metric: str            # specific metric: "speed_zscore", "rare_cell", etc.
    z_score: float         # deviation magnitude (in σ units)
    description_ja: str    # human-readable Japanese explanation
    description_en: str    # human-readable English explanation
    entity_id: int = -1    # -1 for scene-level anomalies
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BehavioralAnomalyDetector
# ---------------------------------------------------------------------------


class BehavioralAnomalyDetector:
    """Detects anomalies in entity speed and activity distribution.

    Learns:
        - Mean/variance of average entity speed (EMA)
        - Activity-type frequency histogram (EMA)

    Detects:
        - Speed z-score exceeding threshold
        - Rare activity type surge (KL-divergence proxy via frequency ratio)

    Example: factory floor with 80% walking, 5% running.  Sudden frame
    with 60% running triggers an anomaly.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self._alpha = alpha
        self._speed_mean: float = 0.0
        self._speed_var: float = 0.0
        self._observations: int = 0
        # Activity frequency histogram: activity_label -> EMA count ratio
        self._activity_freq: dict[str, float] = {}
        self._activity_observations: int = 0

    def observe(self, world_state: WorldState) -> None:
        """Update running statistics from current world state."""
        tracks = world_state.active_tracks
        if not tracks:
            return

        # --- Speed statistics ---
        speeds: list[float] = []
        for track in tracks.values():
            vx, vy = track.velocity
            speed = math.hypot(vx, vy)
            speeds.append(speed)

        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            self._observations += 1

            if self._observations == 1:
                self._speed_mean = avg_speed
                self._speed_var = 0.0
            else:
                delta = avg_speed - self._speed_mean
                self._speed_mean += self._alpha * delta
                self._speed_var = (1 - self._alpha) * (
                    self._speed_var + self._alpha * delta * delta
                )

        # --- Activity distribution ---
        # Count activity types from track attributes
        activity_counts: dict[str, int] = {}
        for track in tracks.values():
            activity = track.attributes.get("activity", "unknown")
            activity_counts[activity] = activity_counts.get(activity, 0) + 1

        total = sum(activity_counts.values())
        if total > 0:
            self._activity_observations += 1
            for act, count in activity_counts.items():
                ratio = count / total
                if act not in self._activity_freq:
                    self._activity_freq[act] = ratio
                else:
                    self._activity_freq[act] += self._alpha * (
                        ratio - self._activity_freq[act]
                    )
            # Decay activities not seen this frame
            for act in list(self._activity_freq.keys()):
                if act not in activity_counts:
                    self._activity_freq[act] *= (1 - self._alpha)

    def check(self, world_state: WorldState) -> list[AnomalySignal]:
        """Check for behavioral anomalies in the current frame."""
        signals: list[AnomalySignal] = []
        tracks = world_state.active_tracks
        if not tracks:
            return signals

        # --- Speed anomaly ---
        speeds: list[float] = []
        for track in tracks.values():
            vx, vy = track.velocity
            speeds.append(math.hypot(vx, vy))

        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            speed_std = math.sqrt(max(self._speed_var, 1e-8))
            z = abs(avg_speed - self._speed_mean) / speed_std

            if z > 2.0:  # threshold checked by ensemble
                signals.append(
                    AnomalySignal(
                        detector="behavioral",
                        metric="speed_zscore",
                        z_score=z,
                        description_ja=(
                            f"平均速度の異常を検出: 現在値={avg_speed:.4f}, "
                            f"平均={self._speed_mean:.4f}, σ={speed_std:.4f}"
                        ),
                        description_en=(
                            f"Abnormal average speed: current={avg_speed:.4f}, "
                            f"mean={self._speed_mean:.4f}, std={speed_std:.4f}"
                        ),
                        details={
                            "current_speed": round(avg_speed, 6),
                            "expected_mean": round(self._speed_mean, 6),
                            "expected_std": round(speed_std, 6),
                            "z_score": round(z, 2),
                        },
                    )
                )

        # --- Activity distribution anomaly ---
        if self._activity_observations > 0 and tracks:
            activity_counts: dict[str, int] = {}
            for track in tracks.values():
                activity = track.attributes.get("activity", "unknown")
                activity_counts[activity] = activity_counts.get(activity, 0) + 1
            total = sum(activity_counts.values())

            if total > 0:
                max_divergence = 0.0
                anomalous_activity = ""
                for act, count in activity_counts.items():
                    current_ratio = count / total
                    expected_ratio = self._activity_freq.get(act, 0.0)
                    if expected_ratio < 0.01:
                        # Very rare activity appearing significantly
                        if current_ratio > 0.1:
                            divergence = current_ratio / max(expected_ratio, 0.001)
                            if divergence > max_divergence:
                                max_divergence = divergence
                                anomalous_activity = act
                    elif current_ratio > expected_ratio * 3:
                        divergence = current_ratio / expected_ratio
                        if divergence > max_divergence:
                            max_divergence = divergence
                            anomalous_activity = act

                if max_divergence > 3.0:
                    z = math.log(max_divergence + 1)  # map to z-like scale
                    signals.append(
                        AnomalySignal(
                            detector="behavioral",
                            metric="activity_distribution",
                            z_score=z,
                            description_ja=(
                                f"活動パターンの異常: '{anomalous_activity}'の"
                                f"頻度が通常の{max_divergence:.1f}倍"
                            ),
                            description_en=(
                                f"Activity pattern anomaly: '{anomalous_activity}' "
                                f"frequency is {max_divergence:.1f}x normal"
                            ),
                            details={
                                "anomalous_activity": anomalous_activity,
                                "frequency_ratio": round(max_divergence, 2),
                            },
                        )
                    )

        return signals

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore internal state from a serialised dict."""
        self._speed_mean = float(data.get("speed_mean", 0.0))
        self._speed_var = float(data.get("speed_var", 0.0))
        self._observations = int(data.get("observations", 0))
        self._activity_freq = {
            str(k): float(v) for k, v in data.get("activity_freq", {}).items()
        }
        self._activity_observations = int(data.get("activity_observations", 0))

    def reset(self) -> None:
        self._speed_mean = 0.0
        self._speed_var = 0.0
        self._observations = 0
        self._activity_freq.clear()
        self._activity_observations = 0


# ---------------------------------------------------------------------------
# SpatialAnomalyDetector
# ---------------------------------------------------------------------------


class SpatialAnomalyDetector:
    """Detects anomalies in spatial occupancy patterns.

    Learns:
        - NxN occupancy grid with EMA visit frequency per cell

    Detects:
        - Entity in a cell with very low historical frequency
          (i.e., "nobody is usually here")

    Example: Cell (3,7) has frequency 0.001 but an entity appears → anomaly.
    """

    def __init__(self, grid_size: int = 10, alpha: float = 0.05) -> None:
        self._grid_size = grid_size
        self._alpha = alpha
        # Grid of EMA occupancy frequencies: grid[row][col]
        self._grid: list[list[float]] = [
            [0.0] * grid_size for _ in range(grid_size)
        ]
        self._observations: int = 0

    def _cell_for(self, cx: float, cy: float) -> tuple[int, int]:
        """Map a normalized center coordinate to a grid cell."""
        col = min(int(cx * self._grid_size), self._grid_size - 1)
        row = min(int(cy * self._grid_size), self._grid_size - 1)
        return max(0, row), max(0, col)

    def observe(self, world_state: WorldState) -> None:
        """Update occupancy grid from current entities."""
        self._observations += 1
        # Build occupancy map for this frame
        occupied: set[tuple[int, int]] = set()
        for entity in world_state.scene_graph.entities:
            cx, cy = entity.bbox.center
            cell = self._cell_for(cx, cy)
            occupied.add(cell)

        # EMA update: occupied cells get nudged toward 1, empty toward 0
        for r in range(self._grid_size):
            for c in range(self._grid_size):
                target = 1.0 if (r, c) in occupied else 0.0
                self._grid[r][c] += self._alpha * (target - self._grid[r][c])

    def check(self, world_state: WorldState) -> list[AnomalySignal]:
        """Check for entities in unusually unoccupied cells."""
        signals: list[AnomalySignal] = []

        for entity in world_state.scene_graph.entities:
            cx, cy = entity.bbox.center
            row, col = self._cell_for(cx, cy)
            freq = self._grid[row][col]

            # Low-frequency cell: normally < 5% occupied
            if freq < 0.05 and self._observations > 10:
                # z-score analog: how rare is this cell
                z = (0.05 - freq) / max(freq, 0.001)
                signals.append(
                    AnomalySignal(
                        detector="spatial",
                        metric="rare_cell",
                        z_score=z,
                        entity_id=entity.entity_id,
                        description_ja=(
                            f"通常無人のエリアにエンティティを検出: "
                            f"セル({row},{col}), 頻度={freq:.3f}"
                        ),
                        description_en=(
                            f"Entity in rarely-occupied area: "
                            f"cell({row},{col}), frequency={freq:.3f}"
                        ),
                        details={
                            "cell_row": row,
                            "cell_col": col,
                            "cell_frequency": round(freq, 4),
                            "label": entity.label,
                        },
                    )
                )

        return signals

    def get_grid(self) -> list[list[float]]:
        """Return the current occupancy grid (for API/debug)."""
        return [row[:] for row in self._grid]

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore internal state from a serialised dict."""
        grid = data.get("grid")
        if grid and isinstance(grid, list):
            gs = len(grid)
            self._grid_size = gs
            self._grid = [
                [float(cell) for cell in row[:gs]] for row in grid[:gs]
            ]
        self._observations = int(data.get("observations", 0))

    def reset(self) -> None:
        self._grid = [
            [0.0] * self._grid_size for _ in range(self._grid_size)
        ]
        self._observations = 0


# ---------------------------------------------------------------------------
# TemporalPatternDetector
# ---------------------------------------------------------------------------


class TemporalPatternDetector:
    """Detects anomalies in time-of-day entity density patterns.

    Learns:
        - 24 hourly slots with EMA entity count mean/variance

    Detects:
        - Current entity count deviates significantly from the
          expected count for this hour

    Example: 2 AM normally has 0 people; 5 people appear → anomaly.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self._alpha = alpha
        # Per-hour statistics: [mean, variance, observation_count]
        self._hourly_mean: list[float] = [0.0] * 24
        self._hourly_var: list[float] = [0.0] * 24
        self._hourly_obs: list[int] = [0] * 24

    @staticmethod
    def _hour_from_timestamp(timestamp: float) -> int:
        """Extract hour (0-23) from a Unix timestamp."""
        import datetime
        try:
            dt = datetime.datetime.fromtimestamp(timestamp)
            return dt.hour
        except (OSError, OverflowError, ValueError):
            # Fallback for non-realistic timestamps (e.g., test seconds)
            return int(timestamp / 3600) % 24

    def observe(self, world_state: WorldState) -> None:
        """Update hourly statistics."""
        hour = self._hour_from_timestamp(world_state.timestamp)
        n = world_state.entity_count
        self._hourly_obs[hour] += 1

        if self._hourly_obs[hour] == 1:
            self._hourly_mean[hour] = float(n)
            self._hourly_var[hour] = 0.0
        else:
            delta = n - self._hourly_mean[hour]
            self._hourly_mean[hour] += self._alpha * delta
            self._hourly_var[hour] = (1 - self._alpha) * (
                self._hourly_var[hour] + self._alpha * delta * delta
            )

    def check(self, world_state: WorldState) -> list[AnomalySignal]:
        """Check for anomalous entity count at the current hour."""
        hour = self._hour_from_timestamp(world_state.timestamp)
        if self._hourly_obs[hour] < 5:
            return []

        n = world_state.entity_count
        mean = self._hourly_mean[hour]
        std = math.sqrt(max(self._hourly_var[hour], 1e-8))
        z = abs(n - mean) / std

        if z > 2.0:  # threshold checked by ensemble
            return [
                AnomalySignal(
                    detector="temporal",
                    metric="hourly_density",
                    z_score=z,
                    description_ja=(
                        f"時間帯別異常: {hour}時に通常{mean:.1f}人だが"
                        f"現在{n}人(z={z:.1f})"
                    ),
                    description_en=(
                        f"Time-of-day anomaly: hour {hour} normally has "
                        f"{mean:.1f} entities but currently has {n} (z={z:.1f})"
                    ),
                    details={
                        "hour": hour,
                        "current_count": n,
                        "expected_mean": round(mean, 2),
                        "expected_std": round(std, 2),
                        "z_score": round(z, 2),
                    },
                )
            ]

        return []

    def get_hourly_stats(self) -> list[dict[str, Any]]:
        """Return hourly statistics for API/debug."""
        return [
            {
                "hour": h,
                "mean": round(self._hourly_mean[h], 2),
                "std": round(math.sqrt(max(self._hourly_var[h], 0.0)), 2),
                "observations": self._hourly_obs[h],
            }
            for h in range(24)
        ]

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore internal state from a serialised dict."""
        hm = data.get("hourly_mean")
        hv = data.get("hourly_var")
        ho = data.get("hourly_obs")
        if hm and isinstance(hm, list) and len(hm) == 24:
            self._hourly_mean = [float(v) for v in hm]
        if hv and isinstance(hv, list) and len(hv) == 24:
            self._hourly_var = [float(v) for v in hv]
        if ho and isinstance(ho, list) and len(ho) == 24:
            self._hourly_obs = [int(v) for v in ho]

    def reset(self) -> None:
        self._hourly_mean = [0.0] * 24
        self._hourly_var = [0.0] * 24
        self._hourly_obs = [0] * 24


# ---------------------------------------------------------------------------
# InteractionAnomalyDetector
# ---------------------------------------------------------------------------


class InteractionAnomalyDetector:
    """Detects anomalies in entity-pair interaction patterns.

    Learns:
        - Frequency of (label_a, relation, label_b) triples (EMA)

    Detects:
        - Previously unseen or very rare interaction pair

    Example: "person NEAR restricted_machine" appears for the first time.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self._alpha = alpha
        self._pair_freq: dict[str, float] = {}  # "label_a|relation|label_b" -> EMA freq
        self._observations: int = 0

    @staticmethod
    def _normalize_key(label_a: str, relation: str, label_b: str) -> str:
        """Create a normalized key for the interaction triple.

        Sorts label_a/label_b alphabetically to make (A, near, B)
        equivalent to (B, near, A).
        """
        a, b = sorted([label_a.lower(), label_b.lower()])
        return f"{a}|{relation}|{b}"

    def observe(self, world_state: WorldState) -> None:
        """Update interaction frequency from current relations."""
        self._observations += 1
        sg = world_state.scene_graph

        # Build current interaction set
        current_keys: set[str] = set()
        for rel in sg.relations:
            subj = sg.get_entity(rel.subject_id)
            obj = sg.get_entity(rel.object_id)
            if subj and obj:
                key = self._normalize_key(
                    subj.label, rel.predicate.value, obj.label
                )
                current_keys.add(key)

        # EMA update
        for key in current_keys:
            if key not in self._pair_freq:
                self._pair_freq[key] = self._alpha  # start with small value
            else:
                self._pair_freq[key] += self._alpha * (1.0 - self._pair_freq[key])

        # Decay unseen pairs
        for key in list(self._pair_freq.keys()):
            if key not in current_keys:
                self._pair_freq[key] *= (1 - self._alpha)
                # Prune very small values
                if self._pair_freq[key] < 1e-6:
                    del self._pair_freq[key]

    def check(self, world_state: WorldState) -> list[AnomalySignal]:
        """Check for novel or rare interaction patterns."""
        signals: list[AnomalySignal] = []
        sg = world_state.scene_graph

        for rel in sg.relations:
            subj = sg.get_entity(rel.subject_id)
            obj = sg.get_entity(rel.object_id)
            if not subj or not obj:
                continue

            key = self._normalize_key(
                subj.label, rel.predicate.value, obj.label
            )
            freq = self._pair_freq.get(key, 0.0)

            # Novel or very rare interaction
            if freq < 0.05:
                z = (0.05 - freq) / max(freq, 0.001)
                signals.append(
                    AnomalySignal(
                        detector="interaction",
                        metric="rare_pair",
                        z_score=z,
                        entity_id=rel.subject_id,
                        description_ja=(
                            f"珍しい関係を検出: "
                            f"{subj.label}(ID:{subj.entity_id}) "
                            f"{rel.predicate.value} "
                            f"{obj.label}(ID:{obj.entity_id}), "
                            f"頻度={freq:.3f}"
                        ),
                        description_en=(
                            f"Rare interaction detected: "
                            f"{subj.label}(ID:{subj.entity_id}) "
                            f"{rel.predicate.value} "
                            f"{obj.label}(ID:{obj.entity_id}), "
                            f"frequency={freq:.3f}"
                        ),
                        details={
                            "subject_label": subj.label,
                            "relation": rel.predicate.value,
                            "object_label": obj.label,
                            "subject_id": subj.entity_id,
                            "object_id": obj.entity_id,
                            "frequency": round(freq, 4),
                        },
                    )
                )

        return signals

    def get_known_pairs(self) -> dict[str, float]:
        """Return known interaction pairs and their frequencies."""
        return dict(self._pair_freq)

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore internal state from a serialised dict."""
        pf = data.get("pair_freq")
        if pf and isinstance(pf, dict):
            self._pair_freq = {str(k): float(v) for k, v in pf.items()}
        self._observations = int(data.get("observations", 0))

    def reset(self) -> None:
        self._pair_freq.clear()
        self._observations = 0


# ---------------------------------------------------------------------------
# AnomalyDetectorEnsemble — the orchestrator
# ---------------------------------------------------------------------------


class AnomalyDetectorEnsemble:
    """Ensemble orchestrator for autonomous anomaly detection.

    Coordinates four independent detectors, applies weighted scoring,
    cooldown deduplication, and severity mapping.

    Interface is compatible with AnomalyBaseline: ``observe()`` + ``check_anomalies()``.

    Args:
        config: Perception configuration.
    """

    # Detector weights for ensemble scoring
    _WEIGHTS: dict[str, float] = {
        "behavioral": 1.0,
        "spatial": 0.8,
        "temporal": 0.9,
        "interaction": 0.7,
    }

    def __init__(self, config: PerceptionConfig | None = None) -> None:
        config = config or PerceptionConfig()

        self._warmup_frames = getattr(config, "anomaly_warmup_frames", 100)
        self._sigma_threshold = getattr(config, "anomaly_sigma_threshold", 2.0)
        self._cooldown_seconds = getattr(config, "anomaly_cooldown_seconds", 60.0)
        alpha = getattr(config, "anomaly_ema_alpha", 0.05)
        grid_size = getattr(config, "anomaly_spatial_grid_size", 10)

        self._behavioral = BehavioralAnomalyDetector(alpha=alpha)
        self._spatial = SpatialAnomalyDetector(grid_size=grid_size, alpha=alpha)
        self._temporal = TemporalPatternDetector(alpha=alpha)
        self._interaction = InteractionAnomalyDetector(alpha=alpha)

        self._observations: int = 0
        # Cooldown: (detector, metric, entity_id) -> last_alert_timestamp
        self._cooldown_map: dict[tuple[str, str, int], float] = {}

    def observe(self, world_state: WorldState) -> None:
        """Feed a new world state to all detectors for learning."""
        self._observations += 1
        self._behavioral.observe(world_state)
        self._spatial.observe(world_state)
        self._temporal.observe(world_state)
        self._interaction.observe(world_state)

    def check_anomalies(self, world_state: WorldState) -> list[EntityEvent]:
        """Run all detectors and return anomaly events (after warmup + cooldown).

        Compatible with AnomalyBaseline.check_anomalies() interface.
        """
        if self._observations < self._warmup_frames:
            return []

        # Collect signals from all detectors
        all_signals: list[AnomalySignal] = []
        all_signals.extend(self._behavioral.check(world_state))
        all_signals.extend(self._spatial.check(world_state))
        all_signals.extend(self._temporal.check(world_state))
        all_signals.extend(self._interaction.check(world_state))

        # Filter by threshold and cooldown, then convert to events
        events: list[EntityEvent] = []
        now = world_state.timestamp

        for signal in all_signals:
            # Apply ensemble weight
            weighted_z = signal.z_score * self._WEIGHTS.get(signal.detector, 1.0)
            if weighted_z < self._sigma_threshold:
                continue

            # Cooldown check
            cooldown_key = (signal.detector, signal.metric, signal.entity_id)
            last_alert = self._cooldown_map.get(cooldown_key, float("-inf"))
            if (now - last_alert) < self._cooldown_seconds:
                continue

            # Map z-score to severity
            severity = self._z_to_severity(weighted_z)

            # Record cooldown
            self._cooldown_map[cooldown_key] = now

            events.append(
                EntityEvent(
                    event_type=EntityEventType.ANOMALY,
                    entity_id=signal.entity_id,
                    timestamp=world_state.timestamp,
                    frame_number=world_state.frame_number,
                    details={
                        "detector": signal.detector,
                        "metric": signal.metric,
                        "z_score": round(weighted_z, 2),
                        "severity": severity.value,
                        "description_ja": signal.description_ja,
                        "description_en": signal.description_en,
                        **signal.details,
                    },
                    confidence=min(1.0, weighted_z / 5.0),
                )
            )

            logger.info(
                "ANOMALY [%s/%s]: %s (z=%.1f, severity=%s)",
                signal.detector,
                signal.metric,
                signal.description_en,
                weighted_z,
                severity.value,
            )

        return events

    def get_state(self) -> dict[str, Any]:
        """Return the ensemble state for API exposure."""
        return {
            "observations": self._observations,
            "warmup_frames": self._warmup_frames,
            "is_warmed_up": self._observations >= self._warmup_frames,
            "sigma_threshold": self._sigma_threshold,
            "cooldown_seconds": self._cooldown_seconds,
            "active_cooldowns": len(self._cooldown_map),
            "detectors": {
                "behavioral": {
                    "speed_mean": round(self._behavioral._speed_mean, 6),
                    "speed_std": round(
                        math.sqrt(max(self._behavioral._speed_var, 0.0)), 6
                    ),
                    "speed_observations": self._behavioral._observations,
                    "known_activities": len(self._behavioral._activity_freq),
                },
                "spatial": {
                    "grid_size": self._spatial._grid_size,
                    "observations": self._spatial._observations,
                    "grid": self._spatial.get_grid(),
                },
                "temporal": {
                    "hourly_stats": self._temporal.get_hourly_stats(),
                },
                "interaction": {
                    "known_pairs": len(self._interaction._pair_freq),
                    "observations": self._interaction._observations,
                    "pairs": self._interaction.get_known_pairs(),
                },
            },
        }

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore all detector states from a serialised dict.

        Args:
            data: Dict with keys 'behavioral', 'spatial', 'temporal',
                  'interaction', and 'observations'.
        """
        self._observations = int(data.get("observations", 0))
        if "behavioral" in data:
            self._behavioral.load_state(data["behavioral"])
        if "spatial" in data:
            self._spatial.load_state(data["spatial"])
        if "temporal" in data:
            self._temporal.load_state(data["temporal"])
        if "interaction" in data:
            self._interaction.load_state(data["interaction"])
        self._cooldown_map.clear()
        logger.info(
            "AnomalyDetectorEnsemble state loaded (%d observations)",
            self._observations,
        )

    def reset(self) -> None:
        """Reset all detectors and cooldown state."""
        self._behavioral.reset()
        self._spatial.reset()
        self._temporal.reset()
        self._interaction.reset()
        self._observations = 0
        self._cooldown_map.clear()
        logger.info("AnomalyDetectorEnsemble reset")

    @staticmethod
    def _z_to_severity(z: float) -> ViolationSeverity:
        """Map weighted z-score to severity level."""
        if z >= 4.0:
            return ViolationSeverity.CRITICAL
        if z >= 3.0:
            return ViolationSeverity.WARNING
        return ViolationSeverity.INFO
