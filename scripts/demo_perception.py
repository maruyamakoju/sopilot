"""SOPilot Perception Engine — Interactive Demo Script.

Demonstrates the full perception pipeline with synthetic frames:
  - Entity detection & tracking
  - Anomaly detection (4-detector ensemble)
  - Scene understanding & spatial map
  - Predictive safety (anticipation engine)
  - NL narration (JP/EN)
  - Action execution
  - Multi-agent coordination

Usage:
    python scripts/demo_perception.py
    python scripts/demo_perception.py --frames 60 --fps 10 --verbose
    python scripts/demo_perception.py --report report.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          SOPilot Perception Engine  v2.3.0  Demo             ║
║          "カメラのOS" — Human-like perception pipeline       ║
╚══════════════════════════════════════════════════════════════╝
"""

PHASE_LABELS = {
    1: "Core Detection & Tracking",
    2: "Intelligence (Prediction & Activity)",
    3: "Deliberative Reasoning (System 2)",
    4: "Autonomous Anomaly Detection",
    5: "Predictive Safety (Phase 9)",
    6: "Spatial Intelligence (Phase 8)",
    7: "Action Loop & Multimodal Fusion",
    8: "Multi-Agent Coordination",
}


# ── Synthetic scene generator ─────────────────────────────────────────────────

@dataclass
class SyntheticEntity:
    entity_id: int
    label: str
    x: float   # center x [0,1]
    y: float   # center y [0,1]
    vx: float  # velocity x per frame
    vy: float  # velocity y per frame
    active: bool = True

    def step(self) -> None:
        self.x = float(np.clip(self.x + self.vx + random.gauss(0, 0.002), 0.05, 0.95))
        self.y = float(np.clip(self.y + self.vy + random.gauss(0, 0.002), 0.05, 0.95))
        # Bounce off walls
        if self.x <= 0.05 or self.x >= 0.95:
            self.vx *= -1
        if self.y <= 0.05 or self.y >= 0.95:
            self.vy *= -1

    def bbox(self) -> list[float]:
        w, h = 0.06, 0.12
        return [self.x - w/2, self.y - h/2, self.x + w/2, self.y + h/2]


class SyntheticScene:
    """Generates synthetic frame data to demo the perception pipeline."""

    def __init__(self, n_workers: int = 4, n_forklifts: int = 1, seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.frame_number = 0
        self.entities: list[SyntheticEntity] = []
        eid = 1
        for i in range(n_workers):
            self.entities.append(SyntheticEntity(
                entity_id=eid, label="person",
                x=random.uniform(0.1, 0.9), y=random.uniform(0.1, 0.9),
                vx=random.uniform(-0.008, 0.008), vy=random.uniform(-0.008, 0.008),
            ))
            eid += 1
        for _ in range(n_forklifts):
            self.entities.append(SyntheticEntity(
                entity_id=eid, label="forklift",
                x=random.uniform(0.2, 0.8), y=random.uniform(0.2, 0.8),
                vx=random.uniform(-0.012, 0.012), vy=random.uniform(-0.012, 0.012),
            ))
            eid += 1

    def next_frame(self) -> tuple[np.ndarray, list[SyntheticEntity]]:
        """Advance simulation one frame. Returns (frame_rgb, entities)."""
        for ent in self.entities:
            ent.step()
        self.frame_number += 1
        frame = self._render_frame()
        return frame, [e for e in self.entities if e.active]

    def inject_anomaly(self) -> None:
        """Make a worker sprint toward a forklift (collision scenario)."""
        workers = [e for e in self.entities if e.label == "person"]
        forklifts = [e for e in self.entities if e.label == "forklift"]
        if workers and forklifts:
            w, f = workers[0], forklifts[0]
            dx = f.x - w.x
            dy = f.y - w.y
            dist = math.hypot(dx, dy) + 1e-9
            w.vx = (dx / dist) * 0.025   # sprint toward forklift
            w.vy = (dy / dist) * 0.025

    def _render_frame(self) -> np.ndarray:
        frame = np.full((480, 640, 3), 30, dtype=np.uint8)
        for ent in self.entities:
            color = (100, 180, 100) if ent.label == "person" else (180, 120, 60)
            x1, y1, x2, y2 = ent.bbox()
            px1 = int(x1 * 640); py1 = int(y1 * 480)
            px2 = int(x2 * 640); py2 = int(y2 * 480)
            frame[py1:py2, px1:px2] = color
        return frame


# ── Pipeline wrapper ──────────────────────────────────────────────────────────

def build_engine_safe():
    """Build perception engine with graceful degradation."""
    try:
        from sopilot.perception.engine import build_perception_engine
        from sopilot.perception.types import PerceptionConfig
        config = PerceptionConfig(
            anomaly_enabled=True,
            warmup_frames=20,
            anomaly_sigma_threshold=2.0,
        )
        engine = build_perception_engine(config=config)
        return engine, True
    except Exception as e:
        print(f"  ⚠  Engine not available ({e}). Running in mock mode.")
        return None, False


@dataclass
class DemoStats:
    frames_processed: int = 0
    total_detections: int = 0
    total_anomalies: int = 0
    total_hazards: int = 0
    total_events: int = 0
    warmup_complete: bool = False
    elapsed_s: float = 0.0
    fps_avg: float = 0.0
    components_active: list[str] = None

    def __post_init__(self):
        if self.components_active is None:
            self.components_active = []


# ── Demo runner ───────────────────────────────────────────────────────────────

def run_demo(
    n_frames: int = 80,
    fps_hint: float = 10.0,
    verbose: bool = False,
    report_path: str | None = None,
) -> DemoStats:

    print(BANNER)
    print(f"  Frames: {n_frames}  |  FPS hint: {fps_hint}  |  Verbose: {verbose}\n")

    # Detect available components
    components = _detect_components()
    print("  Active components:")
    for name, available in components.items():
        status = "✓" if available else "○"
        print(f"    {status}  {name}")
    print()

    scene = SyntheticScene(n_workers=4, n_forklifts=1)
    engine, engine_ok = build_engine_safe()

    # Anticipation engine (Phase 9)
    anticipation = None
    if components.get("anticipation"):
        try:
            from sopilot.perception.anticipation import AnticipationEngine
            anticipation = AnticipationEngine(fps_hint=fps_hint)
        except Exception:
            pass

    # Spatial map (Phase 8)
    spatial_map = None
    if components.get("spatial_map"):
        try:
            from sopilot.perception.spatial_map import SpatialMap
            spatial_map = SpatialMap()
        except Exception:
            pass

    # Multi-agent coordinator (Phase D)
    coordinator = None
    if components.get("multi_agent"):
        try:
            from sopilot.perception.multi_agent import MultiAgentCoordinator
            coordinator = MultiAgentCoordinator()
            coordinator.register_agent("cam-01", "CAM001", "入口エリア")
            coordinator.register_agent("cam-02", "CAM002", "作業エリアA")
        except Exception:
            pass

    stats = DemoStats(components_active=[k for k, v in components.items() if v])
    anomaly_inject_done = False
    t_start = time.time()

    print(f"  {'Frame':>5}  {'Entities':>8}  {'Events':>7}  {'Hazards':>8}  {'Anomalies':>10}  Status")
    print("  " + "─" * 72)

    for frame_idx in range(n_frames):
        # Inject anomaly at frame 40 to trigger anticipation engine
        if frame_idx == 40 and not anomaly_inject_done:
            scene.inject_anomaly()
            anomaly_inject_done = True
            if verbose:
                print(f"\n  ⚡ [Frame {frame_idx}] Anomaly injected: worker sprinting toward forklift\n")

        frame_rgb, entities = scene.next_frame()
        frame_number = scene.frame_number
        timestamp = frame_number / fps_hint

        n_entities = len(entities)
        n_events = 0
        n_hazards = 0
        n_anomalies = 0
        status_flags = []

        # Run engine
        if engine_ok and engine:
            try:
                world_state, violations, timing = engine.process_frame(
                    frame_rgb, frame_number=frame_number, timestamp=timestamp
                )
                n_events = len(world_state.events)
                n_anomalies = sum(
                    1 for e in world_state.events
                    if hasattr(e.event_type, "name") and e.event_type.name == "ANOMALY"
                )
                if violations:
                    status_flags.append(f"VIOL:{len(violations)}")
                if n_anomalies:
                    status_flags.append(f"ANOM:{n_anomalies}")
                    stats.total_anomalies += n_anomalies
                stats.total_events += n_events

                # Anticipation analysis
                if anticipation is not None:
                    hazards = anticipation.analyze(
                        world_state,
                        frame_number=frame_number,
                        timestamp=timestamp,
                    )
                    n_hazards = len(hazards)
                    stats.total_hazards += n_hazards
                    if hazards:
                        status_flags.append(f"HAZ:{n_hazards}")
                        if verbose:
                            for h in hazards:
                                print(f"  ⚠  {h.description_ja}  [{h.severity}]  TTC={h.ttc_seconds}")

                # Spatial map
                if spatial_map is not None:
                    snap = spatial_map.update(world_state, timestamp=timestamp)

                # Multi-agent
                if coordinator is not None:
                    ent_dicts = [
                        {"entity_id": e.entity_id, "label": e.label,
                         "cx": (e.bbox[0]+e.bbox[2])/2, "cy": (e.bbox[1]+e.bbox[3])/2}
                        for e in world_state.entities
                    ] if hasattr(world_state, "entities") else []
                    coordinator.submit_entities("cam-01", ent_dicts, _now=timestamp)

                stats.total_detections += n_entities

                # Warmup complete check
                if not stats.warmup_complete:
                    anomaly_state = engine.get_anomaly_state()
                    if anomaly_state and anomaly_state.get("warmup_complete"):
                        stats.warmup_complete = True
                        status_flags.append("WARMUP✓")

            except Exception as exc:
                if verbose:
                    print(f"  Engine error at frame {frame_number}: {exc}")

        else:
            # Mock mode stats
            stats.total_detections += n_entities

        status_str = "  ".join(status_flags) if status_flags else "ok"

        if frame_idx % 5 == 0 or status_flags:
            print(f"  {frame_number:>5}  {n_entities:>8}  {n_events:>7}  {n_hazards:>8}  {n_anomalies:>10}  {status_str}")

        stats.frames_processed += 1

    elapsed = time.time() - t_start
    stats.elapsed_s = round(elapsed, 2)
    stats.fps_avg = round(stats.frames_processed / elapsed, 1) if elapsed > 0 else 0.0

    print("\n  " + "═" * 72)
    print("  DEMO COMPLETE")
    print(f"    Frames processed : {stats.frames_processed}")
    print(f"    Total detections : {stats.total_detections}")
    print(f"    Total events     : {stats.total_events}")
    print(f"    Total anomalies  : {stats.total_anomalies}")
    print(f"    Total hazards    : {stats.total_hazards}")
    print(f"    Warmup complete  : {stats.warmup_complete}")
    print(f"    Elapsed          : {stats.elapsed_s}s  ({stats.fps_avg} fps)")
    print(f"    Components active: {', '.join(stats.components_active)}")
    print()

    if coordinator:
        coord_state = coordinator.get_state_dict()
        print(f"  Multi-Agent: {coord_state['total_agents']} agents, "
              f"{coord_state['total_global_entities']} global entities")

    if spatial_map:
        sm_state = spatial_map.get_state_dict()
        print(f"  Spatial Map: max_occupancy={sm_state['max_occupancy']:.3f}  "
              f"updates={sm_state.get('update_count', sm_state.get('frame_count', '?'))}")

    if report_path:
        _save_report(stats, report_path)
        print(f"\n  Report saved → {report_path}")

    return stats


def _detect_components() -> dict[str, bool]:
    components = {}
    checks = [
        ("perception_engine", "sopilot.perception.engine", "build_perception_engine"),
        ("anomaly_detection", "sopilot.perception.anomaly", "AnomalyDetectorEnsemble"),
        ("anticipation", "sopilot.perception.anticipation", "AnticipationEngine"),
        ("spatial_map", "sopilot.perception.spatial_map", "SpatialMap"),
        ("adaptive_learner", "sopilot.perception.adaptive_learner", "AdaptiveLearner"),
        ("attention_broker", "sopilot.perception.attention_broker", "AttentionBroker"),
        ("scene_understanding", "sopilot.perception.scene_understanding", "SceneUnderstanding"),
        ("clip_classifier", "sopilot.perception.clip_classifier", "CLIPZeroShotClassifier"),
        ("multi_agent", "sopilot.perception.multi_agent", "MultiAgentCoordinator"),
        ("action_executor", "sopilot.perception.action_executor", "ActionExecutor"),
        ("multimodal_fusion", "sopilot.perception.multimodal", "MultimodalFusionEngine"),
    ]
    for name, module, cls in checks:
        try:
            mod = __import__(module, fromlist=[cls])
            getattr(mod, cls)
            components[name] = True
        except Exception:
            components[name] = False
    return components


def _save_report(stats: DemoStats, path: str) -> None:
    data = {
        "version": "v2.3.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frames_processed": stats.frames_processed,
        "total_detections": stats.total_detections,
        "total_events": stats.total_events,
        "total_anomalies": stats.total_anomalies,
        "total_hazards": stats.total_hazards,
        "warmup_complete": stats.warmup_complete,
        "elapsed_s": stats.elapsed_s,
        "fps_avg": stats.fps_avg,
        "components_active": stats.components_active,
    }
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOPilot Perception Engine Demo")
    parser.add_argument("--frames", type=int, default=80, help="Number of frames to simulate")
    parser.add_argument("--fps", type=float, default=10.0, help="Simulated FPS hint")
    parser.add_argument("--verbose", action="store_true", help="Print detailed event output")
    parser.add_argument("--report", type=str, default=None, help="Save JSON report to this path")
    args = parser.parse_args()

    run_demo(
        n_frames=args.frames,
        fps_hint=args.fps,
        verbose=args.verbose,
        report_path=args.report,
    )
