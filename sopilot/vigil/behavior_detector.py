"""Behavioral anomaly detection for tracked persons.

Phase 29 — Person Tracking + Behavioral Anomaly Detection.

Analyzes a TrackState's trajectory to detect:
- Loitering  : person stays in roughly the same area > N seconds
- Running    : rapid centroid displacement across recent frames
- Fall       : aspect-ratio (h/w) drops from standing (>1) to prone (<0.7)

All thresholds use normalized [0.0, 1.0] frame coordinates so they are
resolution-independent.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass

from sopilot.vigil.person_tracker import TrackState


# ── Event model ────────────────────────────────────────────────────────────────


@dataclass
class BehaviorEvent:
    event_id: str
    track_id: str
    session_id: str
    behavior_type: str      # "loitering" | "running" | "fall"
    severity: str           # "info" | "warning" | "alert"
    confidence: float       # [0.0, 1.0]
    timestamp: float
    description: str        # Japanese

    def to_dict(self) -> dict:
        return asdict(self)


# ── Detector ───────────────────────────────────────────────────────────────────


class BehaviorDetector:
    """Stateless per-frame behavior analyzer.

    Parameters
    ----------
    loitering_seconds:
        Minimum track age (seconds) before loitering can be declared.
    loitering_threshold:
        Maximum total centroid displacement (normalized) that still counts
        as "stationary".
    running_threshold:
        Average centroid displacement per frame (normalized) that triggers
        the running flag.
    running_window:
        Number of recent trajectory points used to compute average speed.
    fall_aspect_normal:
        h/w ratio above which a person is considered upright (standing).
    fall_aspect_threshold:
        h/w ratio below which a person is considered prone (fallen).
    """

    def __init__(
        self,
        loitering_seconds: float = 10.0,
        loitering_threshold: float = 0.05,
        running_threshold: float = 0.03,
        running_window: int = 5,
        fall_aspect_normal: float = 1.0,
        fall_aspect_threshold: float = 0.7,
    ) -> None:
        self.loitering_seconds = loitering_seconds
        self.loitering_threshold = loitering_threshold
        self.running_threshold = running_threshold
        self.running_window = running_window
        self.fall_aspect_normal = fall_aspect_normal
        self.fall_aspect_threshold = fall_aspect_threshold

    def analyze(self, track: TrackState, session_id: str) -> list[BehaviorEvent]:
        """Analyze one track and return any newly triggered behavior events.

        Events are only fired once per behavior episode (flag transition from
        absent → present).  Flags are cleared when the condition no longer holds.
        """
        if len(track.trajectory) < 2:
            return []

        traj = list(track.trajectory)
        events: list[BehaviorEvent] = []

        for check in (
            self._check_loitering,
            self._check_running,
            self._check_fall,
        ):
            ev = check(track, traj, session_id)
            if ev is not None:
                events.append(ev)

        return events

    # ── Individual behavior checks ─────────────────────────────────────────────

    def _check_loitering(
        self, track: TrackState, traj: list, session_id: str
    ) -> BehaviorEvent | None:
        """Loitering: track older than threshold and centroid barely moved."""
        if track.age < self.loitering_seconds:
            track.behavior_flags.discard("loitering")
            return None

        first = traj[0]
        last = traj[-1]
        dx = last["cx"] - first["cx"]
        dy = last["cy"] - first["cy"]
        displacement = (dx ** 2 + dy ** 2) ** 0.5

        if displacement < self.loitering_threshold:
            if "loitering" not in track.behavior_flags:
                track.behavior_flags.add("loitering")
                age_s = int(track.age)
                return BehaviorEvent(
                    event_id=str(uuid.uuid4()),
                    track_id=track.track_id,
                    session_id=session_id,
                    behavior_type="loitering",
                    severity="warning",
                    confidence=min(
                        1.0, track.age / (self.loitering_seconds * 2)
                    ),
                    timestamp=time.time(),
                    description=f"人物が{age_s}秒間同一エリアに滞留しています",
                )
        else:
            track.behavior_flags.discard("loitering")

        return None

    def _check_running(
        self, track: TrackState, traj: list, session_id: str
    ) -> BehaviorEvent | None:
        """Running: high average centroid displacement over recent frames."""
        window = traj[-self.running_window :]
        if len(window) < 2:
            return None

        displacements = [
            ((window[i]["cx"] - window[i - 1]["cx"]) ** 2
             + (window[i]["cy"] - window[i - 1]["cy"]) ** 2) ** 0.5
            for i in range(1, len(window))
        ]
        avg_disp = sum(displacements) / len(displacements)

        if avg_disp > self.running_threshold:
            if "running" not in track.behavior_flags:
                track.behavior_flags.add("running")
                return BehaviorEvent(
                    event_id=str(uuid.uuid4()),
                    track_id=track.track_id,
                    session_id=session_id,
                    behavior_type="running",
                    severity="info",
                    confidence=min(
                        1.0, avg_disp / (self.running_threshold * 3)
                    ),
                    timestamp=time.time(),
                    description="人物が急速に移動しています（走行検知）",
                )
        else:
            track.behavior_flags.discard("running")

        return None

    def _check_fall(
        self, track: TrackState, traj: list, session_id: str
    ) -> BehaviorEvent | None:
        """Fall: aspect-ratio h/w drops from upright (>1) to prone (<0.7)."""
        if len(traj) < 3:
            return None

        def _aspect(p: dict) -> float:
            w = max(p["x2"] - p["x1"], 1e-6)
            h = max(p["y2"] - p["y1"], 1e-6)
            return h / w

        mid = max(1, len(traj) // 2)
        early_pts = traj[:mid]
        recent_pts = traj[mid:]

        early_aspect = sum(_aspect(p) for p in early_pts) / len(early_pts)
        recent_aspect = sum(_aspect(p) for p in recent_pts) / len(recent_pts)

        if (
            early_aspect > self.fall_aspect_normal
            and recent_aspect < self.fall_aspect_threshold
        ):
            if "fall" not in track.behavior_flags:
                track.behavior_flags.add("fall")
                drop = early_aspect - recent_aspect
                return BehaviorEvent(
                    event_id=str(uuid.uuid4()),
                    track_id=track.track_id,
                    session_id=session_id,
                    behavior_type="fall",
                    severity="alert",
                    confidence=min(
                        1.0,
                        drop / (self.fall_aspect_normal + 1e-6),
                    ),
                    timestamp=time.time(),
                    description="人物の転倒が検知されました（要確認）",
                )
        else:
            track.behavior_flags.discard("fall")

        return None
