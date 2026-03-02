#!/usr/bin/env python3
"""Validate the Perception Engine against real video files.

Runs the full pipeline (detect → track → scene graph → world model →
predict → classify → cause → narrate → reason) on actual video frames
and reports detailed results.

Usage:
    python scripts/validate_perception.py [video_path]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from sopilot.perception.engine import PerceptionEngine, build_perception_engine
from sopilot.perception.types import (
    PerceptionConfig,
    Zone,
    TrackState,
)


def main():
    # ── Video selection ────────────────────────────────────────────
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    else:
        video_path = Path("jr23_720p.mp4")

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # ── Video info ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    print("=" * 70)
    print("  PERCEPTION ENGINE — Real Video Validation")
    print("=" * 70)
    print(f"  Video:      {video_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS:        {fps:.1f}")
    print(f"  Frames:     {total_frames}")
    print(f"  Duration:   {duration:.1f}s")
    print("=" * 70)

    # ── Configure engine ───────────────────────────────────────────
    # Use mock detector (no GPU), but process real video frames
    # The mock detector uses heuristic mode on real frames: brightness,
    # color analysis to infer objects
    config = PerceptionConfig(
        detector_backend="mock",
        track_min_hits=2,
        track_max_age=15,
        scene_near_threshold=0.15,
        temporal_memory_seconds=60.0,
        prolonged_presence_seconds=10.0,
    )

    # Define test zones
    zones = [
        Zone(
            zone_id="work_area",
            name="作業エリア",
            polygon=[(0.1, 0.2), (0.9, 0.2), (0.9, 0.9), (0.1, 0.9)],
            zone_type="work_area",
        ),
        Zone(
            zone_id="restricted_a",
            name="制限エリアA",
            polygon=[(0.0, 0.0), (0.3, 0.0), (0.3, 0.2), (0.0, 0.2)],
            zone_type="restricted",
        ),
    ]

    # Safety rules (Japanese)
    rules = [
        "ヘルメット未着用を検出",
        "制限エリアへの立入禁止",
        "作業エリアでの安全装備確認",
    ]

    # ── Build engine ───────────────────────────────────────────────
    print("\n[1] Building perception engine...")
    t0 = time.perf_counter()

    # Build engine manually to enable heuristic detection on real frames
    from sopilot.perception.detector import MockDetector
    from sopilot.perception.tracker import MultiObjectTracker
    from sopilot.perception.scene_graph import SceneGraphBuilder
    from sopilot.perception.world_model import WorldModel
    from sopilot.perception.reasoning import HybridReasoner
    from sopilot.perception.prediction import TrajectoryPredictor
    from sopilot.perception.activity import ActivityClassifier, ActivityMonitor
    from sopilot.perception.attention import SceneAttentionScorer
    from sopilot.perception.causality import CausalReasoner
    from sopilot.perception.context_memory import ContextMemory
    from sopilot.perception.narrator import SceneNarrator

    detector = MockDetector(use_heuristics=True, config=config)
    tracker = MultiObjectTracker(config)
    scene_builder = SceneGraphBuilder(config)
    world_model = WorldModel(config)
    reasoner = HybridReasoner(config=config)
    predictor = TrajectoryPredictor()
    activity_cls = ActivityClassifier()
    activity_mon = ActivityMonitor(classifier=activity_cls)
    attention = SceneAttentionScorer()
    causal = CausalReasoner()
    context_mem = ContextMemory()
    narrator = SceneNarrator()

    engine = PerceptionEngine(
        config=config,
        detector=detector,
        tracker=tracker,
        scene_builder=scene_builder,
        world_model=world_model,
        reasoner=reasoner,
        trajectory_predictor=predictor,
        activity_classifier=activity_cls,
        activity_monitor=activity_mon,
        attention_scorer=attention,
        causal_reasoner=causal,
        context_memory=context_mem,
        narrator=narrator,
    )
    engine.set_zones(zones)
    build_time = (time.perf_counter() - t0) * 1000
    print(f"    Engine built in {build_time:.0f}ms")
    print(f"    Components: detector={type(engine._detector).__name__}, "
          f"tracker={type(engine._tracker).__name__}")
    if engine._trajectory_predictor:
        print(f"    Phase 2: predictor=OK, activity=OK, attention=OK")
    if engine._causal_reasoner:
        print(f"    Phase 3: causality=OK, context_memory=OK, narrator=OK")

    # ── Process video ──────────────────────────────────────────────
    sample_fps = 1.0  # 1 frame per second
    print(f"\n[2] Processing video at {sample_fps} fps...")

    t0 = time.perf_counter()
    results = engine.process_video(
        video_path=video_path,
        rules=rules,
        sample_fps=sample_fps,
    )
    total_time = (time.perf_counter() - t0) * 1000
    print(f"    Processed {len(results)} frames in {total_time:.0f}ms")
    if results:
        avg_ms = total_time / len(results)
        print(f"    Average: {avg_ms:.1f}ms/frame ({1000/avg_ms:.1f} fps effective)")

    # ── Results summary ────────────────────────────────────────────
    print(f"\n[3] Results Summary")
    print("-" * 50)

    total_detections = sum(r.detections_count for r in results)
    total_tracks = sum(r.tracks_count for r in results)
    total_violations = sum(len(r.violations) for r in results)
    vlm_calls = sum(1 for r in results if r.vlm_called)

    print(f"    Total detections:  {total_detections}")
    print(f"    Total tracks:      {total_tracks}")
    print(f"    Total violations:  {total_violations}")
    print(f"    VLM calls:         {vlm_calls}")

    # ── Per-frame detail ───────────────────────────────────────────
    print(f"\n[4] Per-Frame Detail")
    print("-" * 70)
    print(f"    {'Frame':>6} {'Time':>6} {'Det':>4} {'Trk':>4} {'Ent':>4} "
          f"{'Rel':>4} {'Evt':>4} {'Viol':>4} {'ms':>7}")
    print("-" * 70)

    for r in results:
        ws = r.world_state
        sg = ws.scene_graph
        print(f"    {r.frame_number:>6} {r.timestamp:>5.1f}s {r.detections_count:>4} "
              f"{r.tracks_count:>4} {sg.entity_count:>4} {len(sg.relations):>4} "
              f"{len(ws.events):>4} {len(r.violations):>4} {r.processing_time_ms:>6.1f}")

    # ── Track lifecycle ────────────────────────────────────────────
    print(f"\n[5] Track Lifecycle")
    print("-" * 50)

    if results and results[-1].world_state.active_tracks:
        for tid, track in results[-1].world_state.active_tracks.items():
            print(f"    Track {tid}: label={track.label}, state={track.state.value}, "
                  f"hits={track.hits}, age={track.age}, "
                  f"vel=({track.velocity[0]:.4f}, {track.velocity[1]:.4f})")
    else:
        print("    No active tracks at final frame")

    # ── World model events ─────────────────────────────────────────
    print(f"\n[6] World Model Events (all frames)")
    print("-" * 50)

    all_events = []
    for r in results:
        for e in r.world_state.events:
            all_events.append(e)

    event_counts = {}
    for e in all_events:
        event_counts[e.event_type.value] = event_counts.get(e.event_type.value, 0) + 1

    for etype, count in sorted(event_counts.items()):
        print(f"    {etype}: {count}")

    if not event_counts:
        print("    (no events)")

    # ── Violations detail ──────────────────────────────────────────
    print(f"\n[7] Violations Detail")
    print("-" * 50)

    for r in results:
        for v in r.violations:
            print(f"    Frame {r.frame_number} | {v.severity.value:>8} | "
                  f"conf={v.confidence:.2f} | {v.description_ja}")

    if total_violations == 0:
        print("    (no violations detected)")

    # ── Scene narration (last frame) ──────────────────────────────
    print(f"\n[8] Scene Narration (last frame)")
    print("-" * 50)

    if engine._narrator and results:
        last_ws = results[-1].world_state
        last_violations = results[-1].violations
        try:
            narration = engine._narrator.narrate(
                last_ws, last_violations
            )
            print(f"    [JA] {narration.text_ja}")
            print(f"    [EN] {narration.text_en}")
        except Exception as exc:
            print(f"    Narration error: {exc}")
    else:
        print("    (narrator not available)")

    # ── Context memory query ───────────────────────────────────────
    print(f"\n[9] Context Memory Queries")
    print("-" * 50)

    if engine._context_memory:
        queries = [
            "何人いる？",
            "制限エリアに何人入った？",
            "違反は何件？",
        ]
        for q in queries:
            try:
                answer = engine._context_memory.query(q)
                print(f"    Q: {q}")
                print(f"    A: {answer}")
                print()
            except Exception as exc:
                print(f"    Q: {q} → Error: {exc}")
    else:
        print("    (context memory not available)")

    # ── Causal reasoning ───────────────────────────────────────────
    print(f"\n[10] Causal Links Found")
    print("-" * 50)

    if engine._causal_reasoner:
        links = list(engine._causal_reasoner._links) if hasattr(engine._causal_reasoner, '_links') else []
        print(f"    Total causal links: {len(links)}")
        for link in links[:5]:
            print(f"    {link.cause_type}: {link.explanation_ja} "
                  f"(conf={link.confidence:.2f}, dt={link.time_delta_seconds:.1f}s)")
    else:
        print("    (causal reasoner not available)")

    # ── Performance summary ────────────────────────────────────────
    print(f"\n[11] Performance Summary")
    print("=" * 70)

    processing_times = [r.processing_time_ms for r in results]
    if processing_times:
        print(f"    Frames processed:  {len(results)}")
        print(f"    Total time:        {total_time:.0f}ms ({total_time/1000:.1f}s)")
        print(f"    Mean:              {np.mean(processing_times):.1f}ms/frame")
        print(f"    Median:            {np.median(processing_times):.1f}ms/frame")
        print(f"    Min:               {np.min(processing_times):.1f}ms/frame")
        print(f"    Max:               {np.max(processing_times):.1f}ms/frame")
        print(f"    Effective FPS:     {1000/np.mean(processing_times):.1f}")
        print(f"    VLM calls:         {vlm_calls} (local-only)")

        # Compare with Claude VLM benchmark
        claude_mean_ms = 3710
        speedup = claude_mean_ms / np.mean(processing_times)
        print(f"\n    vs Claude VLM API ({claude_mean_ms}ms/frame):")
        print(f"    Speedup:           {speedup:.0f}x faster")

    print("\n" + "=" * 70)
    print("  Validation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
