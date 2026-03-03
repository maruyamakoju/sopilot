"""Holistic per-frame scene analysis.

Aggregates outputs from all perception stages into a single SceneAnalysis
data object capturing: crowd density, motion flow, risk index, active zones,
dominant activity, and anomaly density.

No external dependencies beyond stdlib + numpy.
Thread-safe under RLock.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default risk weights
_DEFAULT_RISK_WEIGHTS = {
    "crowd_density": 0.25,
    "anomaly_density": 0.45,
    "violation_rate": 0.20,
    "flow_speed": 0.10,
}

# EntityEventType names that indicate anomalies/violations
_ANOMALY_TYPES = {"ANOMALY"}
_VIOLATION_TYPES = {"RULE_VIOLATION", "PROLONGED_PRESENCE", "ZONE_VIOLATION"}


@dataclass
class SceneAnalysis:
    """Holistic scene analysis for one frame."""
    timestamp: float
    frame_number: int
    entity_count: int
    crowd_density: float           # 0-1
    flow_dx: float                 # dominant motion direction x [-1,1]
    flow_dy: float                 # dominant motion direction y [-1,1]
    flow_speed: float              # mean entity speed (magnitude of mean velocity)
    anomaly_count: int             # events of type ANOMALY this frame
    violation_count: int           # events of type RULE_VIOLATION etc.
    anomaly_density: float         # anomaly_count / max(1, entity_count)
    risk_index: float              # 0-1 composite risk score
    active_zones: list[str]        # zone names with at least 1 entity
    dominant_activity: str         # most frequent entity label or activity
    summary_ja: str
    summary_en: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "entity_count": self.entity_count,
            "crowd_density": round(self.crowd_density, 4),
            "flow_dx": round(self.flow_dx, 4),
            "flow_dy": round(self.flow_dy, 4),
            "flow_speed": round(self.flow_speed, 4),
            "anomaly_count": self.anomaly_count,
            "violation_count": self.violation_count,
            "anomaly_density": round(self.anomaly_density, 4),
            "risk_index": round(self.risk_index, 4),
            "active_zones": self.active_zones,
            "dominant_activity": self.dominant_activity,
            "summary_ja": self.summary_ja,
            "summary_en": self.summary_en,
        }


class SceneUnderstanding:
    """Holistic scene analysis engine.

    Call `analyze(world_state)` once per frame after all other perception
    stages have run. The result is stored in a rolling history.

    Parameters
    ----------
    history_size : int
        Maximum number of SceneAnalysis objects to retain. Default 60.
    risk_weights : dict | None
        Weights for crowd_density, anomaly_density, violation_rate,
        flow_speed in the risk index formula.
    high_crowd_threshold : float
        Crowd density above this -> adds to risk. Default 0.5.
    high_flow_threshold : float
        Flow speed above this (normalized) -> adds to risk. Default 0.02.
    """

    def __init__(
        self,
        history_size: int = 60,
        risk_weights: dict | None = None,
        high_crowd_threshold: float = 0.5,
        high_flow_threshold: float = 0.02,
    ) -> None:
        self._history_size = max(1, history_size)
        rw = risk_weights or {}
        self._risk_w = {k: rw.get(k, v) for k, v in _DEFAULT_RISK_WEIGHTS.items()}
        self._high_crowd = high_crowd_threshold
        self._high_flow = high_flow_threshold
        self._lock = threading.RLock()
        self._history: list[SceneAnalysis] = []

    def analyze(
        self,
        world_state: Any,
        spatial_map: Any = None,
        frame_number: int | None = None,
    ) -> SceneAnalysis:
        """Produce a SceneAnalysis from current world_state."""
        ts = getattr(world_state, "timestamp", time.time())
        fn = frame_number if frame_number is not None else getattr(world_state, "frame_number", 0)
        entity_count = getattr(world_state, "entity_count", 0)

        # Crowd density
        if spatial_map is not None:
            crowd_density = float(spatial_map.get_crowd_density())
        else:
            crowd_density = min(1.0, entity_count / 20.0)

        # Flow vector
        if spatial_map is not None:
            flow_dx, flow_dy = spatial_map.get_flow_vector()
        else:
            flow_dx, flow_dy = self._compute_flow(world_state)
        flow_speed = math.sqrt(flow_dx ** 2 + flow_dy ** 2)

        # Events
        events = list(getattr(world_state, "events", []))
        anomaly_count = 0
        violation_count = 0
        for ev in events:
            et = getattr(ev, "event_type", None)
            name = et.name if hasattr(et, "name") else str(et)
            if name in _ANOMALY_TYPES:
                anomaly_count += 1
            elif name in _VIOLATION_TYPES:
                violation_count += 1

        anomaly_density = anomaly_count / max(1, entity_count) if entity_count > 0 else 0.0
        violation_rate = violation_count / max(1, entity_count) if entity_count > 0 else 0.0

        # Risk index
        flow_factor = min(1.0, flow_speed / max(0.001, self._high_flow))
        risk_index = (
            self._risk_w["crowd_density"] * min(1.0, crowd_density / max(0.001, self._high_crowd))
            + self._risk_w["anomaly_density"] * min(1.0, anomaly_density)
            + self._risk_w["violation_rate"] * min(1.0, violation_rate)
            + self._risk_w["flow_speed"] * min(1.0, flow_factor)
        )
        risk_index = max(0.0, min(1.0, risk_index))

        # Active zones
        zone_occ = getattr(world_state, "zone_occupancy", {})
        active_zones = [z for z, ids in zone_occ.items() if ids]

        # Dominant activity (most frequent label)
        dominant_activity = self._dominant_label(world_state)

        # Summaries
        summary_ja, summary_en = self._build_summaries(
            entity_count, crowd_density, risk_index, anomaly_count, violation_count, active_zones
        )

        analysis = SceneAnalysis(
            timestamp=ts,
            frame_number=fn,
            entity_count=entity_count,
            crowd_density=round(crowd_density, 4),
            flow_dx=round(float(flow_dx), 4),
            flow_dy=round(float(flow_dy), 4),
            flow_speed=round(flow_speed, 4),
            anomaly_count=anomaly_count,
            violation_count=violation_count,
            anomaly_density=round(anomaly_density, 4),
            risk_index=round(risk_index, 4),
            active_zones=active_zones,
            dominant_activity=dominant_activity,
            summary_ja=summary_ja,
            summary_en=summary_en,
        )

        with self._lock:
            self._history.append(analysis)
            if len(self._history) > self._history_size:
                self._history = self._history[-self._history_size:]

        return analysis

    def _compute_flow(self, world_state: Any) -> tuple[float, float]:
        """Compute mean velocity from active_tracks."""
        tracks = getattr(world_state, "active_tracks", {})
        if not tracks:
            return 0.0, 0.0
        vxs, vys = [], []
        for t in tracks.values():
            vx, vy = getattr(t, "velocity", (0.0, 0.0))
            vxs.append(vx)
            vys.append(vy)
        if not vxs:
            return 0.0, 0.0
        return float(np.mean(vxs)), float(np.mean(vys))

    def _dominant_label(self, world_state: Any) -> str:
        """Return most frequent entity label."""
        sg = getattr(world_state, "scene_graph", None)
        if sg is None:
            return ""
        entities = list(getattr(sg, "entities", []))
        if not entities:
            return ""
        counts: dict[str, int] = {}
        for e in entities:
            lbl = getattr(e, "label", "")
            counts[lbl] = counts.get(lbl, 0) + 1
        return max(counts, key=counts.get)

    def _build_summaries(
        self,
        entity_count: int,
        crowd_density: float,
        risk_index: float,
        anomaly_count: int,
        violation_count: int,
        active_zones: list[str],
    ) -> tuple[str, str]:
        """Build Japanese and English scene summary strings."""
        risk_label_ja = "低" if risk_index < 0.3 else "中" if risk_index < 0.7 else "高"
        risk_label_en = "low" if risk_index < 0.3 else "medium" if risk_index < 0.7 else "high"

        parts_ja = [f"{entity_count}人/物体 検出"]
        parts_en = [f"{entity_count} entities detected"]

        if anomaly_count > 0:
            parts_ja.append(f"異常 {anomaly_count}件")
            parts_en.append(f"{anomaly_count} anomalies")
        if violation_count > 0:
            parts_ja.append(f"違反 {violation_count}件")
            parts_en.append(f"{violation_count} violations")
        if active_zones:
            parts_ja.append(f"アクティブゾーン: {', '.join(active_zones[:2])}")
            parts_en.append(f"active zones: {', '.join(active_zones[:2])}")
        parts_ja.append(f"リスク: {risk_label_ja}")
        parts_en.append(f"risk: {risk_label_en}")

        return "。".join(parts_ja), ". ".join(parts_en) + "."

    def get_history(self, n: int = 10) -> list[SceneAnalysis]:
        with self._lock:
            return list(self._history[-n:])

    def get_trend(self) -> dict:
        """Risk trend statistics over recent history."""
        with self._lock:
            if not self._history:
                return {"mean_risk": 0.0, "max_risk": 0.0, "delta_risk": 0.0, "n": 0}
            risks = [a.risk_index for a in self._history]
            mean_r = float(np.mean(risks))
            max_r = float(np.max(risks))
            delta_r = risks[-1] - risks[0] if len(risks) > 1 else 0.0
            return {
                "mean_risk": round(mean_r, 4),
                "max_risk": round(max_r, 4),
                "delta_risk": round(delta_r, 4),
                "n": len(risks),
            }

    def get_state_dict(self) -> dict:
        with self._lock:
            last = self._history[-1].to_dict() if self._history else None
            return {
                "history_size": len(self._history),
                "latest": last,
                "trend": self.get_trend(),
            }
