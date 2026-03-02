"""Perception Engine — continuous visual understanding for camera systems.

Replaces stateless VLM-per-frame analysis with:
    Frame → Detect → Track → Scene Graph → World Model → Reason → Events

Phase 1 — Core Pipeline:
    types       — Shared data structures (BBox, Detection, Track, SceneGraph, etc.)
    detector    — Open-vocabulary object detection (Grounding-DINO, YOLO-World, mock)
    tracker     — Multi-object tracking with persistent identity (ByteTrack)
    scene_graph — Spatial relationship inference and scene graph construction
    world_model — Continuous world state, temporal memory, anomaly baseline
    reasoning   — Hybrid local + VLM reasoning for rule evaluation
    engine      — Main orchestrator tying all components together

Phase 2 — Intelligence:
    prediction  — Trajectory forecasting, proactive zone/collision alerts
    activity    — Activity recognition from trajectory patterns (rule-based)
    attention   — Dynamic VLM frame sampling based on scene attention scoring

Phase 3 — Deliberative Reasoning:
    causality      — Causal reasoning: "why" understanding from event sequences
    context_memory — Long-horizon session memory with NL query interface
    narrator       — Natural language scene narration (Japanese/English)
"""
