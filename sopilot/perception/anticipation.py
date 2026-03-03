"""Predictive safety engine — anticipates hazards before they occur.

Analyzes entity trajectories and spatial relationships to compute
Time-To-Collision (TTC), convergence rates, and near-miss probability.
Designed to fire 2–5 seconds before a potential incident.

All inputs use normalized [0,1] coordinates (same as rest of perception stack).
Pure Python + numpy. No ML dependencies.
"""
from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

HAZARD_COLLISION = "collision"
HAZARD_NEAR_MISS = "near_miss"
HAZARD_ZONE_BREACH = "zone_breach"
HAZARD_CROWD_SURGE = "crowd_surge"

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"


@dataclass
class HazardAssessment:
    hazard_id: str
    entity_id_a: int
    entity_id_b: int | None   # None for single-entity hazards
    hazard_type: str           # HAZARD_* constants
    ttc_seconds: float | None  # Time-to-Collision; None if not calculable
    probability: float         # 0.0–1.0
    severity: str              # SEVERITY_* constants
    description_ja: str
    frame_number: int
    timestamp: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "hazard_id": self.hazard_id,
            "entity_id_a": self.entity_id_a,
            "entity_id_b": self.entity_id_b,
            "hazard_type": self.hazard_type,
            "ttc_seconds": self.ttc_seconds,
            "probability": self.probability,
            "severity": self.severity,
            "description_ja": self.description_ja,
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "details": self.details,
        }


class VelocityEstimator:
    """Per-entity EMA velocity estimator from position deltas."""

    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = alpha
        self._positions: dict[int, tuple[float, float]] = {}
        self._velocities: dict[int, tuple[float, float]] = {}

    def update(self, entity_id: int, cx: float, cy: float) -> tuple[float, float]:
        """Update position and return smoothed (vx, vy)."""
        prev = self._positions.get(entity_id)
        if prev is not None:
            dx = cx - prev[0]
            dy = cy - prev[1]
            prev_vx, prev_vy = self._velocities.get(entity_id, (0.0, 0.0))
            vx = self.alpha * dx + (1.0 - self.alpha) * prev_vx
            vy = self.alpha * dy + (1.0 - self.alpha) * prev_vy
        else:
            vx, vy = 0.0, 0.0
        self._positions[entity_id] = (cx, cy)
        self._velocities[entity_id] = (vx, vy)
        return vx, vy

    def get_velocity(self, entity_id: int) -> tuple[float, float]:
        return self._velocities.get(entity_id, (0.0, 0.0))

    def reset(self) -> None:
        self._positions.clear()
        self._velocities.clear()


class TrajectoryHazardAnalyzer:
    """Computes TTC and convergence rate between pairs of entities."""

    def __init__(
        self,
        collision_distance_norm: float = 0.05,
        ttc_horizon_s: float = 5.0,
        min_speed_threshold: float = 0.002,
        fps_hint: float = 10.0,
    ) -> None:
        self.collision_distance_norm = collision_distance_norm
        self.ttc_horizon_s = ttc_horizon_s
        self.min_speed_threshold = min_speed_threshold
        self.fps_hint = fps_hint

    def compute_ttc(
        self,
        pos_a: tuple[float, float],
        vel_a: tuple[float, float],
        pos_b: tuple[float, float],
        vel_b: tuple[float, float],
    ) -> float | None:
        """Compute time-to-collision. Returns None if entities are diverging or parallel."""
        dpx = pos_b[0] - pos_a[0]
        dpy = pos_b[1] - pos_a[1]
        dvx = vel_b[0] - vel_a[0]
        dvy = vel_b[1] - vel_a[1]
        dv2 = dvx * dvx + dvy * dvy
        if dv2 < 1e-12:
            return None
        # TTC = -dot(dp, dv) / dot(dv, dv)
        dot_dp_dv = dpx * dvx + dpy * dvy
        ttc = -dot_dp_dv / dv2
        if ttc <= 0:
            return None
        # Scale: velocities are per-frame deltas; multiply by fps_hint for seconds
        ttc_seconds = ttc / self.fps_hint
        return ttc_seconds if ttc_seconds <= self.ttc_horizon_s else None

    def analyze_pair(
        self,
        entity_a: Any,
        entity_b: Any,
        vel_a: tuple[float, float],
        vel_b: tuple[float, float],
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> HazardAssessment | None:
        """Assess collision risk between two entities."""
        pos_a = (_cx(entity_a), _cy(entity_a))
        pos_b = (_cx(entity_b), _cy(entity_b))

        dist = math.hypot(pos_b[0] - pos_a[0], pos_b[1] - pos_a[1])

        speed_a = math.hypot(*vel_a)
        speed_b = math.hypot(*vel_b)
        if speed_a < self.min_speed_threshold and speed_b < self.min_speed_threshold:
            return None  # both stationary

        ttc = self.compute_ttc(pos_a, vel_a, pos_b, vel_b)

        if ttc is None and dist > self.collision_distance_norm * 3:
            return None

        if ttc is not None and ttc <= self.ttc_horizon_s:
            hazard_type = HAZARD_COLLISION if ttc < 2.0 else HAZARD_NEAR_MISS
            probability = min(1.0, (self.ttc_horizon_s - ttc) / self.ttc_horizon_s)
            severity = _ttc_severity(ttc)
            desc = f"衝突リスク: {_label(entity_a)}と{_label(entity_b)} TTC={ttc:.1f}s"
        elif dist <= self.collision_distance_norm:
            hazard_type = HAZARD_NEAR_MISS
            probability = 0.6
            severity = SEVERITY_WARNING
            ttc = None
            desc = f"接近警告: {_label(entity_a)}と{_label(entity_b)} 距離={dist:.3f}"
        else:
            return None

        return HazardAssessment(
            hazard_id=str(uuid.uuid4()),
            entity_id_a=_eid(entity_a),
            entity_id_b=_eid(entity_b),
            hazard_type=hazard_type,
            ttc_seconds=ttc,
            probability=probability,
            severity=severity,
            description_ja=desc,
            frame_number=frame_number,
            timestamp=timestamp,
            details={"dist": round(dist, 4), "speed_a": round(speed_a, 4), "speed_b": round(speed_b, 4)},
        )

    def analyze_zone_boundary(
        self,
        entity: Any,
        zone_rect: tuple[float, float, float, float],  # (x1, y1, x2, y2) normalized
        vel: tuple[float, float],
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> HazardAssessment | None:
        """Assess risk of entity imminently entering a forbidden zone."""
        cx, cy = _cx(entity), _cy(entity)
        x1, y1, x2, y2 = zone_rect
        # Already inside -> not a predictive hazard
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return None
        # Project velocity toward zone edge
        vx, vy = vel
        speed = math.hypot(vx, vy)
        if speed < self.min_speed_threshold:
            return None
        # Nearest zone edge point
        nx = max(x1, min(cx, x2))
        ny = max(y1, min(cy, y2))
        dist = math.hypot(nx - cx, ny - cy)
        # Dot product of velocity with direction to zone
        direction_x = (nx - cx) / (dist + 1e-9)
        direction_y = (ny - cy) / (dist + 1e-9)
        approach = vx * direction_x + vy * direction_y
        if approach <= 0:
            return None  # moving away
        ttc_frames = dist / (approach + 1e-9)
        ttc_s = ttc_frames / self.fps_hint
        if ttc_s > self.ttc_horizon_s:
            return None

        return HazardAssessment(
            hazard_id=str(uuid.uuid4()),
            entity_id_a=_eid(entity),
            entity_id_b=None,
            hazard_type=HAZARD_ZONE_BREACH,
            ttc_seconds=ttc_s,
            probability=min(1.0, (self.ttc_horizon_s - ttc_s) / self.ttc_horizon_s),
            severity=_ttc_severity(ttc_s),
            description_ja=f"ゾーン侵入予測: {_label(entity)} {ttc_s:.1f}s後",
            frame_number=frame_number,
            timestamp=timestamp,
            details={"dist_to_zone": round(dist, 4)},
        )


class CrowdSurgeDetector:
    """Detects sudden crowd density increases (stampede/surge risk)."""

    def __init__(
        self,
        surge_rate_threshold: float = 0.3,
        window_frames: int = 10,
        cooldown_frames: int = 30,
    ) -> None:
        self.surge_rate_threshold = surge_rate_threshold
        self.window_frames = window_frames
        self.cooldown_frames = cooldown_frames
        self._history: deque[float] = deque(maxlen=window_frames)
        self._last_surge_frame: int = -999

    def update(
        self, crowd_density: float, frame_number: int = 0, timestamp: float = 0.0
    ) -> HazardAssessment | None:
        self._history.append(crowd_density)
        if len(self._history) < self.window_frames:
            return None
        if (frame_number - self._last_surge_frame) < self.cooldown_frames:
            return None
        baseline = float(np.mean(list(self._history)[: self.window_frames // 2]))
        recent = float(np.mean(list(self._history)[self.window_frames // 2 :]))
        if baseline < 1e-6:
            return None
        rate = (recent - baseline) / baseline
        if rate < self.surge_rate_threshold:
            return None
        self._last_surge_frame = frame_number
        probability = min(1.0, rate / (self.surge_rate_threshold * 2))
        severity = SEVERITY_CRITICAL if rate > 0.6 else SEVERITY_WARNING
        return HazardAssessment(
            hazard_id=str(uuid.uuid4()),
            entity_id_a=-1,
            entity_id_b=None,
            hazard_type=HAZARD_CROWD_SURGE,
            ttc_seconds=None,
            probability=probability,
            severity=severity,
            description_ja=f"群衆密度急増: {rate*100:.0f}%増加",
            frame_number=frame_number,
            timestamp=timestamp,
            details={"baseline_density": round(baseline, 3), "recent_density": round(recent, 3), "rate": round(rate, 3)},
        )

    def reset(self) -> None:
        self._history.clear()
        self._last_surge_frame = -999


class AnticipationEngine:
    """Orchestrates hazard analyzers per frame. Thread-safe via RLock."""

    MAX_HISTORY = 100

    def __init__(
        self,
        ttc_horizon_s: float = 5.0,
        collision_distance_norm: float = 0.05,
        min_probability: float = 0.2,
        cooldown_seconds: float = 10.0,
        fps_hint: float = 10.0,
        max_history: int = MAX_HISTORY,
    ) -> None:
        self.ttc_horizon_s = ttc_horizon_s
        self.collision_distance_norm = collision_distance_norm
        self.min_probability = min_probability
        self.cooldown_seconds = cooldown_seconds
        self._lock = threading.RLock()
        self._velocity_estimator = VelocityEstimator()
        self._trajectory_analyzer = TrajectoryHazardAnalyzer(
            collision_distance_norm=collision_distance_norm,
            ttc_horizon_s=ttc_horizon_s,
            fps_hint=fps_hint,
        )
        self._crowd_detector = CrowdSurgeDetector()
        self._history: deque[HazardAssessment] = deque(maxlen=max_history)
        self._active: dict[str, HazardAssessment] = {}   # cooldown key -> last assessment
        self._last_fired: dict[str, float] = {}
        self._total_hazards = 0

    def analyze(
        self,
        world_state: Any,
        depth_estimates: list | None = None,
        scene_snapshot: Any | None = None,
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> list[HazardAssessment]:
        """Run all hazard analyzers on current world state."""
        results: list[HazardAssessment] = []
        now = timestamp or time.time()

        entities = getattr(world_state, "entities", []) or []

        with self._lock:
            # Update velocity estimates
            velocities: dict[int, tuple[float, float]] = {}
            for ent in entities:
                cx, cy = _cx(ent), _cy(ent)
                # Use details vx/vy if available, else estimate from position delta
                vx = ent.details.get("vx", None) if hasattr(ent, "details") else None
                vy = ent.details.get("vy", None) if hasattr(ent, "details") else None
                if vx is None or vy is None:
                    vx, vy = self._velocity_estimator.update(_eid(ent), cx, cy)
                velocities[_eid(ent)] = (float(vx), float(vy))

            # Pairwise collision analysis (O(N^2), N typically < 50)
            for i, ea in enumerate(entities):
                for eb in entities[i + 1:]:
                    ha = self._trajectory_analyzer.analyze_pair(
                        ea, eb,
                        velocities.get(_eid(ea), (0.0, 0.0)),
                        velocities.get(_eid(eb), (0.0, 0.0)),
                        frame_number=frame_number,
                        timestamp=now,
                    )
                    if ha is not None and ha.probability >= self.min_probability:
                        key = f"pair_{min(_eid(ea), _eid(eb))}_{max(_eid(ea), _eid(eb))}_{ha.hazard_type}"
                        if self._check_cooldown(key, now):
                            results.append(ha)
                            self._record(ha, key, now)

            # Crowd surge
            crowd_density = 0.0
            if scene_snapshot is not None:
                crowd_density = getattr(scene_snapshot, "crowd_density",
                                        scene_snapshot.get("crowd_density", 0.0)
                                        if isinstance(scene_snapshot, dict) else 0.0)
            elif entities:
                crowd_density = min(1.0, len(entities) / 20.0)

            surge = self._crowd_detector.update(crowd_density, frame_number, now)
            if surge is not None and surge.probability >= self.min_probability:
                key = "crowd_surge"
                if self._check_cooldown(key, now):
                    results.append(surge)
                    self._record(surge, key, now)

        return results

    def get_active_hazards(self) -> list[HazardAssessment]:
        with self._lock:
            return list(self._active.values())

    def get_history(self, n: int = 20) -> list[HazardAssessment]:
        with self._lock:
            items = list(self._history)
            return items[-n:] if n > 0 else []

    def get_state_dict(self) -> dict:
        with self._lock:
            return {
                "total_hazards_detected": self._total_hazards,
                "active_hazards": len(self._active),
                "history_size": len(self._history),
                "ttc_horizon_s": self.ttc_horizon_s,
                "collision_distance_norm": self.collision_distance_norm,
            }

    def reset(self) -> None:
        with self._lock:
            self._velocity_estimator.reset()
            self._crowd_detector.reset()
            self._history.clear()
            self._active.clear()
            self._last_fired.clear()
            self._total_hazards = 0

    # -- Private --------------------------------------------------------------

    def _check_cooldown(self, key: str, now: float) -> bool:
        last = self._last_fired.get(key, 0.0)
        return (now - last) >= self.cooldown_seconds

    def _record(self, ha: HazardAssessment, key: str, now: float) -> None:
        self._active[key] = ha
        self._history.append(ha)
        self._last_fired[key] = now
        self._total_hazards += 1


# -- Helpers ------------------------------------------------------------------

def _cx(entity: Any) -> float:
    bbox = getattr(entity, "bbox", None)
    if bbox is None:
        return float(getattr(entity, "cx", 0.5))
    if hasattr(bbox, "__len__") and len(bbox) >= 4:
        return float((bbox[0] + bbox[2]) / 2)
    return 0.5


def _cy(entity: Any) -> float:
    bbox = getattr(entity, "bbox", None)
    if bbox is None:
        return float(getattr(entity, "cy", 0.5))
    if hasattr(bbox, "__len__") and len(bbox) >= 4:
        return float((bbox[1] + bbox[3]) / 2)
    return 0.5


def _eid(entity: Any) -> int:
    return int(getattr(entity, "entity_id", 0))


def _label(entity: Any) -> str:
    return str(getattr(entity, "label", "entity"))


def _ttc_severity(ttc_seconds: float) -> str:
    if ttc_seconds < 1.0:
        return SEVERITY_CRITICAL
    if ttc_seconds < 3.0:
        return SEVERITY_WARNING
    return SEVERITY_INFO
