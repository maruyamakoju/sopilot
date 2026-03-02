"""Perception Engine — continuous visual understanding for camera systems.

Replaces stateless VLM-per-frame analysis with:
    Frame → Detect → Track → Scene Graph → World Model → Reason → Events

Modules:
    types       — Shared data structures (BBox, Detection, Track, SceneGraph, etc.)
    detector    — Open-vocabulary object detection (Grounding-DINO, YOLO-World, mock)
    tracker     — Multi-object tracking with persistent identity (ByteTrack)
    scene_graph — Spatial relationship inference and scene graph construction
    world_model — Continuous world state, temporal memory, anomaly baseline
    reasoning   — Hybrid local + VLM reasoning for rule evaluation
    engine      — Main orchestrator tying all components together
"""
