"""Main perception pipeline orchestrator.

Ties together the full perception pipeline:

    Frame → Detect → Track → Scene Graph → World Model → Reason → Events

Usage::

    engine = build_perception_engine()
    engine.set_zones([Zone(zone_id="restricted_1", ...)])

    for frame, ts, fn in video_stream:
        result = engine.process_frame(frame, ts, fn, rules)
        for v in result.violations:
            print(v.description_ja, v.severity)

    engine.close()

Or for batch video processing::

    results = engine.process_video(Path("input.mp4"), rules, sample_fps=1.0)
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sopilot.perception.types import (
    BBox,
    Detection,
    EntityEvent,
    EntityEventType,
    FrameResult,
    PerceptionConfig,
    SceneGraph,
    Track,
    Violation,
    ViolationSeverity,
    WorldState,
    Zone,
)
from sopilot.perception.reasoning import HybridReasoner, _OBJECT_ALIASES

logger = logging.getLogger(__name__)


# ── Detection prompt builder ──────────────────────────────────────────────

# Common safety objects that should always be detectable
_BASE_PROMPTS = ["person"]
_SAFETY_EQUIPMENT_LABELS = [
    "helmet",
    "hard hat",
    "safety vest",
    "gloves",
    "goggles",
    "safety harness",
    "mask",
    "fire extinguisher",
]


def _build_detection_prompts(rules: list[str]) -> list[str]:
    """Extract object detection prompts from rule text.

    Analyses the rule text for object references and returns a deduplicated
    list of detection prompt strings.  Always includes "person" as a base.

    Examples:
        "ヘルメット未着用の作業者を検出" → ["person", "helmet", "hard hat"]
        "立入禁止エリアへの侵入を検出" → ["person"]
        "フォークリフトの安全確認"      → ["person", "forklift"]
    """
    prompts: set[str] = set(_BASE_PROMPTS)

    combined_text = " ".join(rules).lower()
    combined_text_raw = " ".join(rules)  # for Japanese matching

    for canonical, aliases in _OBJECT_ALIASES.items():
        for alias in aliases:
            if alias.lower() in combined_text or alias in combined_text_raw:
                prompts.add(canonical)
                break

    # Always include common safety equipment if rule mentions wearing/PPE
    wearing_hint = any(
        kw in combined_text or kw in combined_text_raw
        for kw in (
            "wearing",
            "未着用",
            "着用",
            "ppe",
            "保護具",
            "安全装備",
            "equipment",
        )
    )
    if wearing_hint:
        for label in _SAFETY_EQUIPMENT_LABELS:
            prompts.add(label)

    result = sorted(prompts)
    logger.debug("Detection prompts from %d rules: %s", len(rules), result)
    return result


# ── PerceptionEngine ──────────────────────────────────────────────────────


class PerceptionEngine:
    """Main orchestrator for the perception pipeline.

    Coordinates detection, tracking, scene graph construction, world model
    updates, and hybrid rule reasoning into a single ``process_frame()``
    call.  Also provides ``process_video()`` for batch processing.

    All sub-components are injected at construction time (either directly
    or via ``build_perception_engine()``).  This makes the engine fully
    testable with mock components.
    """

    def __init__(
        self,
        config: PerceptionConfig | None = None,
        detector: Any | None = None,
        tracker: Any | None = None,
        scene_builder: Any | None = None,
        world_model: Any | None = None,
        reasoner: HybridReasoner | None = None,
        vlm_client: Any | None = None,
    ) -> None:
        """Initialize the perception engine.

        Args:
            config: Perception configuration.  Uses defaults if None.
            detector: ObjectDetector instance for frame-level detection.
            tracker: MultiObjectTracker instance for identity persistence.
            scene_builder: SceneGraphBuilder for spatial relationship inference.
            world_model: WorldModel for temporal state and event generation.
            reasoner: HybridReasoner for rule evaluation.
            vlm_client: VLM client for escalation (passed to reasoner if needed).
        """
        self._config = config or PerceptionConfig()
        self._detector = detector
        self._tracker = tracker
        self._scene_builder = scene_builder
        self._world_model = world_model
        self._vlm_client = vlm_client

        # Build reasoner if not provided
        self._reasoner = reasoner or HybridReasoner(
            config=self._config, vlm_client=vlm_client
        )

        # Zones for scene graph and world model
        self._zones: list[Zone] = list(self._config.zone_definitions)

        # Detection prompt cache
        self._cached_rules_key: tuple[str, ...] | None = None
        self._cached_prompts: list[str] = []

        # Frame counter
        self._frames_processed = 0
        self._total_processing_ms = 0.0

        logger.info(
            "PerceptionEngine initialized: detector=%s, tracker=%s, "
            "scene_builder=%s, world_model=%s, vlm=%s",
            type(self._detector).__name__ if self._detector else "None",
            type(self._tracker).__name__ if self._tracker else "None",
            type(self._scene_builder).__name__ if self._scene_builder else "None",
            type(self._world_model).__name__ if self._world_model else "None",
            type(self._vlm_client).__name__ if self._vlm_client else "None",
        )

    # ── Public API ────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_number: int,
        rules: list[str],
    ) -> FrameResult:
        """Process a single video frame through the full perception pipeline.

        Pipeline stages:
            1. Object detection (open-vocabulary)
            2. Multi-object tracking (identity persistence)
            3. Scene graph construction (spatial relationships)
            4. World model update (temporal state, events)
            5. Hybrid rule reasoning (local + VLM escalation)
            6. Event-to-violation conversion

        Args:
            frame: Video frame as numpy array (H, W, C), BGR format.
            timestamp: Frame timestamp in seconds from video start.
            frame_number: Sequential frame number.
            rules: Natural-language safety rules to evaluate.

        Returns:
            FrameResult with all violations, world state, and timing info.
        """
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        try:
            # ── Stage 1: Detect objects ───────────────────────────
            t_stage = time.perf_counter()
            prompts = self._get_detection_prompts(rules)
            detections = self._run_detection(frame, prompts)
            timings["detection_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 2: Track objects ────────────────────────────
            t_stage = time.perf_counter()
            tracks = self._run_tracking(detections, frame_number)
            timings["tracking_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 3: Build scene graph ────────────────────────
            t_stage = time.perf_counter()
            frame_shape = (frame.shape[0], frame.shape[1]) if frame.ndim >= 2 else (0, 0)
            scene_graph = self._run_scene_graph(
                tracks, frame_shape, timestamp, frame_number
            )
            timings["scene_graph_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 4: Update world model ───────────────────────
            t_stage = time.perf_counter()
            world_state = self._run_world_model(scene_graph)
            timings["world_model_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 5: Evaluate rules (hybrid reasoning) ────────
            t_stage = time.perf_counter()
            violations = self._reasoner.evaluate_rules(
                rules, scene_graph, world_state, frame
            )
            timings["reasoning_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 6: Convert world model events to violations ─
            t_stage = time.perf_counter()
            event_violations = self._events_to_violations(world_state)
            violations.extend(event_violations)
            timings["event_conversion_ms"] = (time.perf_counter() - t_stage) * 1000

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._frames_processed += 1
            self._total_processing_ms += elapsed_ms

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Frame %d processed in %.1f ms: "
                    "det=%.1f trk=%.1f sg=%.1f wm=%.1f rsn=%.1f | "
                    "%d detections, %d tracks, %d violations",
                    frame_number,
                    elapsed_ms,
                    timings.get("detection_ms", 0),
                    timings.get("tracking_ms", 0),
                    timings.get("scene_graph_ms", 0),
                    timings.get("world_model_ms", 0),
                    timings.get("reasoning_ms", 0),
                    len(detections),
                    len(tracks),
                    len(violations),
                )

            return FrameResult(
                timestamp=timestamp,
                frame_number=frame_number,
                world_state=world_state,
                violations=violations,
                processing_time_ms=elapsed_ms,
                detections_count=len(detections),
                tracks_count=len(tracks),
                vlm_called=self._reasoner._vlm_called_this_frame,
                vlm_latency_ms=self._reasoner._vlm_latency_this_frame,
            )

        except Exception:
            # Graceful degradation: if any stage fails, return empty result
            # rather than crashing the pipeline
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.exception(
                "Error processing frame %d (%.1f ms elapsed)", frame_number, elapsed_ms
            )
            return self._empty_frame_result(timestamp, frame_number, elapsed_ms)

    def process_video(
        self,
        video_path: Path,
        rules: list[str],
        sample_fps: float = 1.0,
        callback: Callable[[FrameResult], None] | None = None,
    ) -> list[FrameResult]:
        """Process a video file through the perception pipeline.

        Samples frames at the specified rate and runs process_frame() on each.

        Args:
            video_path: Path to the video file (mp4, avi, etc.).
            rules: Natural-language safety rules.
            sample_fps: Frame sampling rate (frames per second).
            callback: Optional callback invoked after each frame result.

        Returns:
            List of FrameResult objects, one per sampled frame.
        """
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV (cv2) is required for video processing.")
            return []

        video_path = Path(video_path)
        if not video_path.exists():
            logger.error("Video file not found: %s", video_path)
            return []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Failed to open video: %s", video_path)
            return []

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_interval = max(1, int(round(video_fps / sample_fps)))

            logger.info(
                "Processing video: %s | fps=%.1f, total_frames=%d, "
                "sample_fps=%.1f, frame_interval=%d",
                video_path.name,
                video_fps,
                total_frames,
                sample_fps,
                frame_interval,
            )

            results: list[FrameResult] = []
            frame_number = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_number % frame_interval == 0:
                    timestamp = frame_number / video_fps
                    result = self.process_frame(frame, timestamp, frame_number, rules)
                    results.append(result)

                    if callback is not None:
                        try:
                            callback(result)
                        except Exception:
                            logger.exception(
                                "Callback error at frame %d", frame_number
                            )

                frame_number += 1

            logger.info(
                "Video processing complete: %d frames sampled from %d total, "
                "%d violations found",
                len(results),
                frame_number,
                sum(len(r.violations) for r in results),
            )
            return results

        finally:
            cap.release()

    def set_zones(self, zones: list[Zone]) -> None:
        """Set spatial zones for the perception engine.

        Zones are used by the scene graph builder (to assign entities to zones)
        and by the world model (to generate zone entry/exit events).

        Args:
            zones: List of Zone definitions.
        """
        self._zones = list(zones)
        logger.info("Zones updated: %d zones configured", len(self._zones))

    def get_world_state(self) -> WorldState | None:
        """Return the latest world state, or None if no frames processed."""
        if self._world_model is None:
            return None
        try:
            return self._world_model.get_state()
        except (AttributeError, Exception):
            return None

    def reset(self) -> None:
        """Reset all stateful components to their initial state.

        Call this between videos or when restarting analysis.
        """
        if self._tracker is not None:
            try:
                self._tracker.reset()
            except (AttributeError, Exception):
                pass
        if self._world_model is not None:
            try:
                self._world_model.reset()
            except (AttributeError, Exception):
                pass

        self._cached_rules_key = None
        self._cached_prompts = []
        self._frames_processed = 0
        self._total_processing_ms = 0.0

        logger.info("PerceptionEngine reset")

    def close(self) -> None:
        """Release all resources held by the engine and its components."""
        if self._detector is not None:
            try:
                self._detector.close()
            except (AttributeError, Exception):
                pass
        if self._vlm_client is not None:
            try:
                self._vlm_client.close()
            except (AttributeError, Exception):
                pass

        logger.info(
            "PerceptionEngine closed after %d frames (avg %.1f ms/frame)",
            self._frames_processed,
            (
                self._total_processing_ms / self._frames_processed
                if self._frames_processed > 0
                else 0.0
            ),
        )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def frames_processed(self) -> int:
        """Total number of frames processed since last reset."""
        return self._frames_processed

    @property
    def average_processing_ms(self) -> float:
        """Average processing time per frame in milliseconds."""
        if self._frames_processed == 0:
            return 0.0
        return self._total_processing_ms / self._frames_processed

    # ── Private methods ───────────────────────────────────────────────

    def _get_detection_prompts(self, rules: list[str]) -> list[str]:
        """Get detection prompts, using cache if rules haven't changed."""
        rules_key = tuple(rules)
        if rules_key != self._cached_rules_key:
            self._cached_prompts = _build_detection_prompts(rules)
            self._cached_rules_key = rules_key
        return self._cached_prompts

    def _run_detection(
        self, frame: np.ndarray, prompts: list[str]
    ) -> list[Detection]:
        """Run object detection, gracefully handling missing detector."""
        if self._detector is None:
            return []
        try:
            detections = self._detector.detect(frame, prompts)
            # Enforce max detections limit
            if len(detections) > self._config.max_detections_per_frame:
                # Keep highest-confidence detections
                detections = sorted(
                    detections, key=lambda d: d.confidence, reverse=True
                )[: self._config.max_detections_per_frame]
            return detections
        except Exception:
            logger.exception("Detection failed")
            return []

    def _run_tracking(
        self, detections: list[Detection], frame_number: int
    ) -> list[Track]:
        """Run multi-object tracking, gracefully handling missing tracker."""
        if self._tracker is None:
            return []
        try:
            return self._tracker.update(detections, frame_number)
        except Exception:
            logger.exception("Tracking failed at frame %d", frame_number)
            return []

    def _run_scene_graph(
        self,
        tracks: list[Track],
        frame_shape: tuple[int, int],
        timestamp: float,
        frame_number: int,
    ) -> SceneGraph:
        """Build scene graph, returning empty graph on failure."""
        if self._scene_builder is not None:
            try:
                return self._scene_builder.build(
                    tracks, self._zones, frame_shape, timestamp, frame_number
                )
            except Exception:
                logger.exception(
                    "Scene graph construction failed at frame %d", frame_number
                )

        # Fallback: empty scene graph
        return SceneGraph(
            timestamp=timestamp,
            frame_number=frame_number,
            entities=[],
            relations=[],
            frame_shape=frame_shape,
        )

    def _run_world_model(self, scene_graph: SceneGraph) -> WorldState:
        """Update world model, returning minimal state on failure."""
        if self._world_model is not None:
            try:
                return self._world_model.update(scene_graph)
            except Exception:
                logger.exception(
                    "World model update failed at frame %d",
                    scene_graph.frame_number,
                )

        # Fallback: minimal world state from scene graph alone
        return WorldState(
            timestamp=scene_graph.timestamp,
            frame_number=scene_graph.frame_number,
            scene_graph=scene_graph,
            active_tracks={},
            events=[],
            zone_occupancy={},
            entity_count=scene_graph.entity_count,
            person_count=scene_graph.person_count,
        )

    def _events_to_violations(self, world_state: WorldState) -> list[Violation]:
        """Convert world model events (zone entry, prolonged presence) to violations."""
        violations: list[Violation] = []

        for event in world_state.events:
            violation = self._event_to_violation(event, world_state)
            if violation is not None:
                violations.append(violation)

        return violations

    def _event_to_violation(
        self, event: EntityEvent, world_state: WorldState
    ) -> Violation | None:
        """Convert a single world model event to a Violation if applicable."""
        if event.event_type == EntityEventType.ZONE_ENTERED:
            zone_id = event.details.get("zone_id", "unknown")
            zone_type = event.details.get("zone_type", "generic")

            # Only generate violations for restricted/hazard zones
            if zone_type not in ("restricted", "hazard", "danger"):
                return None

            entity = world_state.scene_graph.get_entity(event.entity_id)
            return Violation(
                rule=f"zone_violation:{zone_id}",
                rule_index=-1,  # not tied to a user rule
                description_ja=f"エンティティ {event.entity_id} が{zone_id}に侵入",
                severity=ViolationSeverity.CRITICAL,
                confidence=event.confidence,
                entity_ids=[event.entity_id],
                bbox=entity.bbox if entity else None,
                evidence={
                    "event_type": event.event_type.value,
                    "zone_id": zone_id,
                    "zone_type": zone_type,
                    "timestamp": event.timestamp,
                },
                source="world_model",
            )

        if event.event_type == EntityEventType.PROLONGED_PRESENCE:
            zone_id = event.details.get("zone_id", "unknown")
            duration_s = event.details.get("duration_seconds", 0)

            entity = world_state.scene_graph.get_entity(event.entity_id)
            return Violation(
                rule=f"prolonged_presence:{zone_id}",
                rule_index=-1,
                description_ja=(
                    f"エンティティ {event.entity_id} が{zone_id}に"
                    f"{duration_s:.0f}秒間滞在"
                ),
                severity=ViolationSeverity.WARNING,
                confidence=event.confidence,
                entity_ids=[event.entity_id],
                bbox=entity.bbox if entity else None,
                evidence={
                    "event_type": event.event_type.value,
                    "zone_id": zone_id,
                    "duration_seconds": duration_s,
                    "timestamp": event.timestamp,
                },
                source="world_model",
            )

        return None

    def _empty_frame_result(
        self, timestamp: float, frame_number: int, elapsed_ms: float
    ) -> FrameResult:
        """Build an empty FrameResult for error cases."""
        empty_sg = SceneGraph(
            timestamp=timestamp,
            frame_number=frame_number,
            entities=[],
            relations=[],
        )
        empty_ws = WorldState(
            timestamp=timestamp,
            frame_number=frame_number,
            scene_graph=empty_sg,
            active_tracks={},
            events=[],
            zone_occupancy={},
        )
        return FrameResult(
            timestamp=timestamp,
            frame_number=frame_number,
            world_state=empty_ws,
            violations=[],
            processing_time_ms=elapsed_ms,
        )


# ── Factory function ──────────────────────────────────────────────────────


def build_perception_engine(
    config: PerceptionConfig | None = None,
    vlm_client: Any | None = None,
) -> PerceptionEngine:
    """Factory function to build a fully configured perception engine.

    Creates all sub-components (detector, tracker, scene builder, world model,
    reasoner) based on the configuration and wires them together.

    Args:
        config: Perception configuration.  Uses defaults if None.
        vlm_client: Optional VLM client for hybrid reasoning escalation.

    Returns:
        A fully initialized PerceptionEngine ready for process_frame() calls.
    """
    config = config or PerceptionConfig()

    # ── Build detector ────────────────────────────────────────────────
    detector = None
    try:
        if config.detector_backend == "mock":
            from sopilot.perception.detector import MockDetector
            detector = MockDetector()
        elif config.detector_backend == "grounding-dino":
            from sopilot.perception.detector import GroundingDINODetector
            detector = GroundingDINODetector(
                model_id=config.detector_model_id,
                confidence_threshold=config.detection_confidence_threshold,
                nms_threshold=config.detection_nms_threshold,
                device=config.device,
            )
        else:
            logger.warning("Unknown detector backend: %s", config.detector_backend)
        if detector:
            logger.info("Detector initialized: backend=%s", config.detector_backend)
    except ImportError:
        logger.warning(
            "sopilot.perception.detector not available. "
            "Detection will be disabled."
        )
    except Exception:
        logger.exception("Failed to initialize detector")

    # ── Build tracker ─────────────────────────────────────────────────
    tracker = None
    try:
        from sopilot.perception.tracker import MultiObjectTracker

        tracker = MultiObjectTracker(config)
        logger.info("Tracker initialized")
    except ImportError:
        logger.warning(
            "sopilot.perception.tracker not available. "
            "Tracking will be disabled."
        )
    except Exception:
        logger.exception("Failed to initialize tracker")

    # ── Build scene graph builder ─────────────────────────────────────
    scene_builder = None
    try:
        from sopilot.perception.scene_graph import SceneGraphBuilder

        scene_builder = SceneGraphBuilder(config)
        logger.info("SceneGraphBuilder initialized")
    except ImportError:
        logger.warning(
            "sopilot.perception.scene_graph not available. "
            "Scene graph will be disabled."
        )
    except Exception:
        logger.exception("Failed to initialize scene graph builder")

    # ── Build world model ─────────────────────────────────────────────
    world_model = None
    try:
        from sopilot.perception.world_model import WorldModel

        world_model = WorldModel(config)
        logger.info("WorldModel initialized")
    except ImportError:
        logger.warning(
            "sopilot.perception.world_model not available. "
            "World model will be disabled."
        )
    except Exception:
        logger.exception("Failed to initialize world model")

    # ── Build hybrid reasoner ─────────────────────────────────────────
    reasoner = HybridReasoner(config=config, vlm_client=vlm_client)
    logger.info(
        "HybridReasoner initialized: escalation_threshold=%.2f, vlm=%s",
        config.vlm_escalation_threshold,
        "enabled" if vlm_client is not None else "disabled",
    )

    # ── Assemble engine ───────────────────────────────────────────────
    engine = PerceptionEngine(
        config=config,
        detector=detector,
        tracker=tracker,
        scene_builder=scene_builder,
        world_model=world_model,
        reasoner=reasoner,
        vlm_client=vlm_client,
    )

    return engine
