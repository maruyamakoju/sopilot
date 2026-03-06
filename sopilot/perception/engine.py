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
from typing import TYPE_CHECKING, Any, Callable

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

if TYPE_CHECKING:
    from sopilot.perception.pose import PoseEstimator

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
        trajectory_predictor: Any | None = None,
        activity_classifier: Any | None = None,
        activity_monitor: Any | None = None,
        attention_scorer: Any | None = None,
        causal_reasoner: Any | None = None,
        context_memory: Any | None = None,
        narrator: Any | None = None,
        pose_estimator: Any | None = None,
        anomaly_explainer: Any | None = None,
        # Phase 5: Deliberative cognition
        episodic_memory: Any | None = None,
        goal_recognizer: Any | None = None,
        deliberative_reasoner: Any | None = None,
        metacognitive_monitor: Any | None = None,
        causal_graph: Any | None = None,
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
            trajectory_predictor: TrajectoryPredictor for proactive zone/collision alerts.
            activity_classifier: ActivityClassifier for trajectory-based activity recognition.
            activity_monitor: ActivityMonitor for activity change events.
            attention_scorer: SceneAttentionScorer for dynamic frame sampling.
            causal_reasoner: CausalReasoner for "why" understanding from event sequences.
            context_memory: ContextMemory for long-horizon session understanding.
            narrator: SceneNarrator for natural language scene descriptions.
            episodic_memory: EpisodicMemoryStore for episode segmentation and retrieval.
            goal_recognizer: GoalRecognizer for entity intent inference.
            deliberative_reasoner: DeliberativeReasoner for System 2 slow reasoning.
            metacognitive_monitor: MetacognitiveMonitor for self-aware quality monitoring.
            causal_graph: CausalGraph for structural causal reasoning.
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

        # Phase 2: prediction, activity, attention
        self._trajectory_predictor = trajectory_predictor
        self._proactive_alerter = None
        if trajectory_predictor is not None:
            try:
                from sopilot.perception.prediction import ProactiveAlertGenerator
                self._proactive_alerter = ProactiveAlertGenerator(
                    predictor=trajectory_predictor
                )
            except ImportError:
                pass
        self._activity_classifier = activity_classifier
        self._activity_monitor = activity_monitor
        self._attention_scorer = attention_scorer
        self._previous_world_state: WorldState | None = None

        # Phase 3: causal reasoning, context memory, narration
        self._causal_reasoner = causal_reasoner
        self._context_memory = context_memory
        self._narrator = narrator

        # Anomaly explainer (optional, VLM-based)
        self._anomaly_explainer = anomaly_explainer

        # Phase 5: Deliberative cognition (System 2 reasoning layer)
        self._episodic_memory = episodic_memory
        self._goal_recognizer = goal_recognizer
        self._deliberative_reasoner = deliberative_reasoner
        self._metacognitive_monitor = metacognitive_monitor
        self._causal_graph = causal_graph

        # Phase 6: Real-time streaming + NL tasks + Re-ID + Long-term memory
        self._session_id: str | None = None
        self._nl_task_manager = None
        self._cross_camera_tracker = None
        self._long_term_memory = None

        # Phase 7: Action executor + Multimodal fusion
        self._action_executor = None
        self._fusion_engine = None

        # Phase 8: Depth + Spatial + Adaptive + Attention + Scene
        self._depth_estimator = None
        self._spatial_map = None
        self._adaptive_learner = None
        self._anomaly_tuner = None          # AnomalyTuner (Phase 10)
        self._auto_apply_threshold = getattr(config, "tuner_auto_apply_threshold", 20)
        self._last_tuner_feedback_count = 0
        self._attention_broker = None
        self._scene_understanding = None
        self._last_depth_estimates: list = []
        self._last_scene_analysis = None

        # Phase 9: Anticipation Engine + CLIP Classifier
        self._anticipation_engine = None
        self._clip_classifier = None
        self._last_hazards: list = []

        # Phase 11A: Active query review queue
        self._review_queue = None

        # Phase 12A: Adaptive sigma tuner
        self._sigma_tuner = None
        self._last_sigma_feedback_count = 0
        self._sigma_apply_interval = getattr(config, "sigma_apply_interval", 10)

        # Phase 15: Early Warning Engine
        self._early_warning = None

        # Phase 16: Autonomous Response
        self._early_warning_responder = None

        # Phase 19: Health History
        self._health_history = None

        # Phase 11B: Frame ring buffer (JPEG snapshots)
        from collections import deque as _deque
        _ring_size = getattr(config, "frame_ring_buffer_size", 50)
        self._frame_ring: "_deque[tuple[float, bytes]]" = _deque(maxlen=_ring_size)

        # Pose estimation (optional, opt-in via config.pose_enabled)
        self._pose_estimator: Any | None = pose_estimator

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
            "scene_builder=%s, world_model=%s, vlm=%s, "
            "predictor=%s, activity=%s, attention=%s",
            type(self._detector).__name__ if self._detector else "None",
            type(self._tracker).__name__ if self._tracker else "None",
            type(self._scene_builder).__name__ if self._scene_builder else "None",
            type(self._world_model).__name__ if self._world_model else "None",
            type(self._vlm_client).__name__ if self._vlm_client else "None",
            type(self._trajectory_predictor).__name__ if self._trajectory_predictor else "None",
            type(self._activity_classifier).__name__ if self._activity_classifier else "None",
            type(self._attention_scorer).__name__ if self._attention_scorer else "None",
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

        # ── Phase 11B: フレームをリングバッファに保存 ────────────────────────
        try:
            import io as _io
            from PIL import Image as _PILImage
            _pil_frame = _PILImage.fromarray(frame[..., ::-1])  # BGR→RGB
            _buf = _io.BytesIO()
            _pil_frame.save(_buf, format="JPEG", quality=50)
            self._frame_ring.append((timestamp, _buf.getvalue()))
        except Exception:
            pass  # PIL unavailable or frame malformed — silently skip

        try:
            # ── Stage 1: Detect objects ───────────────────────────
            t_stage = time.perf_counter()
            prompts = self._get_detection_prompts(rules)
            detections = self._run_detection(frame, prompts)
            timings["detection_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 1b: Monocular depth estimation ──────────────
            if self._depth_estimator is not None and detections:
                try:
                    self._last_depth_estimates = self._depth_estimator.estimate(detections)
                except Exception:
                    logger.debug("Depth estimation failed", exc_info=True)
                    self._last_depth_estimates = []
            else:
                self._last_depth_estimates = []

            # ── Stage 1c: CLIP zero-shot reclassification ─────────
            if self._clip_classifier is not None and detections:
                try:
                    self._clip_classifier.classify_entities(detections, frame)
                except Exception:
                    logger.debug("CLIP classification failed", exc_info=True)

            # ── Stage 2: Track objects ────────────────────────────
            t_stage = time.perf_counter()
            tracks = self._run_tracking(detections, frame_number)
            timings["tracking_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 2b: Cross-camera Re-ID ──────────────────────
            t_stage = time.perf_counter()
            self._run_reid(tracks, frame)
            timings["reid_ms"] = (time.perf_counter() - t_stage) * 1000

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

            # ── Stage 4b: Trajectory prediction (proactive alerts) ─
            t_stage = time.perf_counter()
            prediction_events = self._run_prediction(
                world_state, frame_number, timestamp
            )
            if prediction_events:
                world_state.events.extend(prediction_events)
            timings["prediction_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 4c: Activity recognition ─────────────────────
            t_stage = time.perf_counter()
            activity_events = self._run_activity_recognition(
                world_state, timestamp, frame_number
            )
            if activity_events:
                world_state.events.extend(activity_events)
            timings["activity_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 4d: Attention scoring ────────────────────────
            t_stage = time.perf_counter()
            self._run_attention_scoring(world_state)
            timings["attention_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 4e: Causal reasoning ─────────────────────────
            t_stage = time.perf_counter()
            causal_links = self._run_causal_reasoning(world_state)
            timings["causality_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 4f: Context memory update ────────────────────
            t_stage = time.perf_counter()
            self._run_context_memory(world_state)
            timings["context_memory_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 4g: Episodic memory segmentation ─────────────
            t_stage = time.perf_counter()
            self._run_episodic_memory(world_state)
            timings["episodic_memory_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 4h: Goal recognition (intent inference) ──────
            t_stage = time.perf_counter()
            self._run_goal_recognition(world_state)
            timings["goal_recognition_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 4i: NL task monitoring ───────────────────────
            t_stage = time.perf_counter()
            nl_events = self._run_nl_tasks(world_state)
            if nl_events:
                world_state.events.extend(nl_events)
            timings["nl_task_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 4j: Spatial map update ──────────────────────
            if self._spatial_map is not None:
                try:
                    self._spatial_map.update(
                        world_state, self._last_depth_estimates, timestamp
                    )
                except Exception:
                    logger.debug("Spatial map update failed", exc_info=True)

            # ── Stage 4k: Scene understanding ─────────────────────
            if self._scene_understanding is not None:
                try:
                    self._last_scene_analysis = self._scene_understanding.analyze(
                        world_state, self._spatial_map, frame_number
                    )
                except Exception:
                    logger.debug("Scene understanding failed", exc_info=True)

            # ── Stage 4l: Anticipation engine (predictive safety) ──
            if self._anticipation_engine is not None:
                try:
                    self._last_hazards = self._anticipation_engine.analyze(
                        world_state,
                        depth_estimates=self._last_depth_estimates,
                        scene_snapshot=self._last_scene_analysis,
                        frame_number=frame_number,
                        timestamp=timestamp,
                    )
                except Exception:
                    logger.debug("Anticipation engine failed", exc_info=True)
                    self._last_hazards = []
            else:
                self._last_hazards = []

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

            # ── Stage 6b: VLM explanation for ANOMALY events ──────
            t_stage = time.perf_counter()
            self._run_anomaly_explanations(world_state, frame)
            timings["anomaly_explain_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 6c: Deliberative reasoning (System 2) ────────
            t_stage = time.perf_counter()
            self._run_deliberation(world_state)
            timings["deliberation_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 6d: SSE real-time event broadcasting ─────────
            self._push_events_to_sse(world_state.events)

            # ── Stage 6e: Autonomous action executor ───────────────
            if self._action_executor is not None and world_state.events:
                try:
                    self._action_executor.evaluate(world_state.events, world_state)
                except Exception:
                    logger.debug("Action executor failed", exc_info=True)

            # ── Stage 6f: Multimodal fusion ─────────────────────────
            if self._fusion_engine is not None and world_state.events:
                try:
                    self._fusion_engine.fuse_with_visual(world_state)
                except Exception:
                    logger.debug("Multimodal fusion failed", exc_info=True)

            # ── Stage 6g: Adaptive learner (concept drift) ────────
            if self._adaptive_learner is not None and world_state.events:
                try:
                    for _ev in world_state.events:
                        if hasattr(_ev.event_type, "name") and _ev.event_type.name == "ANOMALY":
                            _score = float(_ev.details.get("z_score", 0.0))
                            self._adaptive_learner.observe(_score, timestamp)
                except Exception:
                    logger.debug("Adaptive learner failed", exc_info=True)

            # ── Stage 6h: AnomalyTuner 自動適用 ──────────────────────────────
            if self._anomaly_tuner is not None and self._world_model is not None:
                try:
                    cur = len(self._anomaly_tuner._records)
                    if (cur - self._last_tuner_feedback_count) >= self._auto_apply_threshold:
                        bl = self._world_model.get_anomaly_baseline()
                        if bl is not None:
                            self._anomaly_tuner.apply_tuning(bl)
                            self._last_tuner_feedback_count = cur
                except Exception:
                    pass

            # ── Stage 6i: 能動的クエリ (Active Query) ────────────────────────
            if self._review_queue is not None and world_state.events:
                try:
                    _frame_jpeg = self._frame_ring[-1][1] if self._frame_ring else None
                    for _ev in world_state.events:
                        if hasattr(_ev.event_type, "name") and _ev.event_type.name == "ANOMALY":
                            _z = float(_ev.details.get("z_score", 0.0)) if _ev.details else 0.0
                            self._review_queue.maybe_add(
                                _ev, _z, tuner=self._anomaly_tuner, frame_jpeg=_frame_jpeg
                            )
                            # Phase 15: Early Warning — observe anomaly burst
                            if self._early_warning is not None and _ev.details:
                                _det = _ev.details.get("detector", "")
                                if _det:
                                    self._early_warning.observe_anomaly(_det, timestamp)
                except Exception:
                    logger.debug("Active query stage failed", exc_info=True)

            # ── Stage 6j: SigmaTuner 自動σ調整 ──────────────────────────────
            if self._sigma_tuner is not None and self._anomaly_tuner is not None:
                try:
                    cur = len(self._anomaly_tuner._records)
                    if (cur - self._last_sigma_feedback_count) >= self._sigma_apply_interval:
                        stats = self._anomaly_tuner.get_stats()
                        _sigma_changes = self._sigma_tuner.compute_and_apply(stats)
                        self._last_sigma_feedback_count = cur
                        # Phase 15: Early Warning — observe sigma drift from each change
                        if _sigma_changes and self._early_warning is not None:
                            for _ch in _sigma_changes:
                                self._early_warning.observe_sigma_change(
                                    _ch["detector"],
                                    _ch["old_sigma"],
                                    _ch["new_sigma"],
                                    _ch.get("timestamp"),
                                )
                except Exception:
                    pass

            # ── Stage 6k: Early Warning 自律対応 ─────────────────────────────────
            if self._early_warning is not None and self._early_warning_responder is not None:
                try:
                    ew_state = self._early_warning.get_state()
                    _triggered = self._early_warning_responder.evaluate(
                        ew_state,
                        sigma_tuner=self._sigma_tuner,
                        review_queue=self._review_queue,
                    )
                    # Phase 17: SSE broadcast to all connected UI sessions
                    if _triggered:
                        self._push_early_warning_responses(_triggered)
                except Exception:
                    pass

            # ── Stage 7: Pose estimation + PPE check (opt-in) ─────
            t_stage = time.perf_counter()
            pose_results = []
            if self._pose_estimator is not None:
                pose_results = self._run_pose_estimation(frame)
                ppe_violations = self._pose_ppe_violations(pose_results)
                violations.extend(ppe_violations)
            timings["pose_ms"] = (time.perf_counter() - t_stage) * 1000

            # ── Stage 8: Metacognitive monitoring ─────────────────
            t_stage = time.perf_counter()
            self._run_metacognition(FrameResult(
                timestamp=timestamp, frame_number=frame_number,
                world_state=world_state, violations=violations,
                processing_time_ms=0.0,
            ))
            timings["metacognition_ms"] = (time.perf_counter() - t_stage) * 1000

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._frames_processed += 1
            self._total_processing_ms += elapsed_ms

            # Record Prometheus-compatible metrics
            try:
                from sopilot.perception.perc_metrics import get_registry
                _reg = get_registry()
                _reg.record_frame(processing_ms=elapsed_ms)
                _reg.record_detection(count=len(detections))
                if self._reasoner._vlm_called_this_frame:
                    _reg.record_vlm_call()
                for _v in violations:
                    _sev = _v.severity.value if hasattr(_v.severity, "value") else str(_v.severity).lower()
                    _reg.record_violation(severity=_sev)
                for _ev in world_state.events:
                    if hasattr(_ev.event_type, "name") and _ev.event_type.name == "ANOMALY":
                        _det = _ev.details.get("detector", "unknown")
                        _reg.record_anomaly(detector=_det)
                    _reg.record_event(event_type=_ev.event_type.name if hasattr(_ev.event_type, "name") else str(_ev.event_type))
            except Exception:
                pass

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Frame %d processed in %.1f ms: "
                    "det=%.1f trk=%.1f sg=%.1f wm=%.1f rsn=%.1f pose=%.1f | "
                    "%d detections, %d tracks, %d violations",
                    frame_number,
                    elapsed_ms,
                    timings.get("detection_ms", 0),
                    timings.get("tracking_ms", 0),
                    timings.get("scene_graph_ms", 0),
                    timings.get("world_model_ms", 0),
                    timings.get("reasoning_ms", 0),
                    timings.get("pose_ms", 0),
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
                pose_results=pose_results,
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

    def get_anomaly_state(self) -> dict | None:
        """Return the anomaly detector ensemble state, or None if unavailable."""
        if self._world_model is None:
            return None
        try:
            baseline = self._world_model.get_anomaly_baseline()
            if hasattr(baseline, "get_state"):
                return baseline.get_state()
        except (AttributeError, Exception):
            pass
        return None

    def get_deliberation_state(self) -> dict | None:
        """Return recent deliberation results for the API, or None if unavailable."""
        if self._deliberative_reasoner is None:
            return None
        try:
            return self._deliberative_reasoner.get_state_dict()
        except Exception:
            return None

    def get_goal_state(self) -> dict | None:
        """Return current goal hypothesis state for the API, or None if unavailable."""
        if self._goal_recognizer is None:
            return None
        try:
            return self._goal_recognizer.get_state_dict()
        except Exception:
            return None

    def get_metacognition_state(self) -> dict | None:
        """Return perception health report for the API, or None if unavailable."""
        if self._metacognitive_monitor is None:
            return None
        try:
            report = self._metacognitive_monitor.get_health_report()
            return report.__dict__
        except Exception:
            return None

    def get_episodes(self, n: int = 10) -> list[dict]:
        """Return recent episodes from episodic memory for the API."""
        if self._episodic_memory is None:
            return []
        try:
            episodes = self._episodic_memory.get_recent_episodes(n)
            return [e.__dict__ for e in episodes]
        except Exception:
            return []

    def get_causal_narrative(self) -> str:
        """Return a Japanese causal narrative from the structural causal graph."""
        if self._causal_graph is None:
            return ""
        try:
            return self._causal_graph.get_causal_narrative()
        except Exception:
            return ""

    # ── Phase 6 getters ───────────────────────────────────────────────

    def set_session_id(self, session_id: str) -> None:
        """Set the active session ID for SSE broadcasting. Creates event queue."""
        self._session_id = session_id
        try:
            from sopilot.perception import sse_events
            sse_events.get_or_create(session_id)
        except ImportError:
            pass

    def get_nl_task_state(self) -> dict | None:
        if self._nl_task_manager is None:
            return None
        try:
            return self._nl_task_manager.get_state_dict()
        except Exception:
            return None

    def get_reid_state(self) -> dict | None:
        if self._cross_camera_tracker is None:
            return None
        try:
            return self._cross_camera_tracker.get_state_dict()
        except Exception:
            return None

    def get_ltm_state(self) -> dict | None:
        if self._long_term_memory is None:
            return None
        try:
            return self._long_term_memory.get_state_dict()
        except Exception:
            return None

    def get_ltm_hourly(self, hour: int) -> list[dict]:
        if self._long_term_memory is None:
            return []
        try:
            facts = self._long_term_memory.get_hourly_pattern(hour)
            return [f.__dict__ for f in facts]
        except Exception:
            return []

    def get_action_state(self) -> dict | None:
        """Return action executor state dict, or None if not initialized."""
        if self._action_executor is None:
            return None
        try:
            return self._action_executor.get_state_dict()
        except Exception:
            return None

    def get_fusion_state(self) -> dict | None:
        """Return multimodal fusion engine state dict, or None if not initialized."""
        if self._fusion_engine is None:
            return None
        try:
            return self._fusion_engine.get_state_dict()
        except Exception:
            return None

    def get_depth_estimates(self) -> list:
        """Return depth estimates from the most recent frame."""
        return list(self._last_depth_estimates)

    def get_spatial_state(self) -> dict | None:
        """Return spatial map state dict, or None if not initialized."""
        if self._spatial_map is None:
            return None
        try:
            return self._spatial_map.get_state_dict()
        except Exception:
            return None

    def get_adaptive_state(self) -> dict | None:
        """Return adaptive learner state dict, or None if not initialized."""
        if self._adaptive_learner is None:
            return None
        try:
            return self._adaptive_learner.get_state_dict()
        except Exception:
            return None

    def get_adaptive_learner_state(self) -> dict:
        """Return combined AdaptiveLearner + AnomalyTuner state (Phase 10)."""
        al = self.get_adaptive_state() or {
            "total_observed": 0, "score_window_size": 0,
            "score_mean": 0.0, "score_std": 0.0,
            "drift_count": 0, "recalibration_count": 0,
            "last_recalibration": None, "ph_state": {},
        }
        tuner_state: dict = {}
        if self._anomaly_tuner is not None:
            try:
                tuner_state = self._anomaly_tuner.get_stats()
            except Exception:
                pass
        return {"adaptive_learner": al, "tuner": tuner_state or {
            "total_feedback": 0, "confirmed": 0, "denied": 0,
            "overall_confirm_rate": 0.0, "pairs_tracked": 0,
            "pairs_suppressed": 0, "pairs_trusted": 0,
            "last_tuning": 0.0, "pair_stats": [],
            "suppressed_pairs": [], "trusted_pairs": [],
            "min_samples_for_tuning": 10,
        }}

    def get_sigma_state(self) -> dict | None:
        """Return SigmaTuner state dict, or None if not initialized (Phase 12A)."""
        if self._sigma_tuner is None:
            return None
        try:
            return self._sigma_tuner.get_state()
        except Exception:
            return None

    def get_early_warning_state(
        self, tuner_stats: dict | None = None
    ) -> dict | None:
        """Return EarlyWarningEngine state dict, or None if not initialized (Phase 15)."""
        if self._early_warning is None:
            return None
        try:
            return self._early_warning.get_state(tuner_stats)
        except Exception:
            return None

    def get_early_warning_responder_state(self) -> dict | None:
        """Return EarlyWarningResponder state dict, or None if not initialized (Phase 16)."""
        if self._early_warning_responder is None:
            return None
        try:
            return self._early_warning_responder.get_state()
        except Exception:
            return None

    def get_health_score(self, *, record: bool = True) -> dict:
        """Return composite Perception Engine health score (Phase 18).

        Args:
            record: If True, auto-record the score in HealthHistoryStore (rate-limited).

        Returns:
            dict with keys: score (0-100), grade (A-F), factors, computed_at.
        """
        try:
            from sopilot.perception.perception_health import PerceptionHealthScorer
            result = PerceptionHealthScorer().compute(self)
        except Exception:
            logger.exception("PerceptionHealthScorer failed")
            return {"score": 0, "grade": "F", "factors": {}, "computed_at": 0.0}

        # Phase 19: auto-record into history (rate-limited to 1/hour by default)
        if record and self._health_history is not None:
            try:
                self._health_history.record(
                    score=result["score"],
                    grade=result["grade"],
                    factors=result.get("factors", {}),
                    total_penalty=result.get("total_penalty", 0.0),
                )
            except Exception:
                logger.debug("HealthHistoryStore.record() failed", exc_info=True)

        return result

    def get_daily_report(self) -> dict:
        """Generate a 24-hour intelligence report (Phase 20).

        Returns:
            Structured report dict with sections: summary, anomalies,
            early_warning, responses, recommendations, metadata.
        """
        try:
            from sopilot.perception.daily_report import DailyReportGenerator
            return DailyReportGenerator().generate(self)
        except Exception:
            logger.exception("DailyReportGenerator failed")
            return {"error": "Report generation failed", "metadata": {"generated_at": 0.0}}

    def get_health_history(self, days: float = 7.0) -> dict:
        """Return health score history + trend for the last N days (Phase 19).

        Returns:
            dict with keys: history (list), trend (dict), sparkline (list).
        """
        if self._health_history is None:
            return {"history": [], "trend": {}, "sparkline": []}
        try:
            return {
                "history": self._health_history.get_history(days=days),
                "trend": self._health_history.get_trend(days=days),
                "sparkline": self._health_history.get_sparkline_data(days=days),
            }
        except Exception:
            logger.debug("get_health_history failed", exc_info=True)
            return {"history": [], "trend": {}, "sparkline": []}

    def get_latest_frame_jpeg(self) -> bytes | None:
        """最新のフレーム JPEG バイト列を返す (Phase 11B)。バッファが空なら None。"""
        if not self._frame_ring:
            return None
        try:
            _ts, _jpeg = self._frame_ring[-1]
            return _jpeg
        except Exception:
            return None

    def get_attention_state(self) -> dict | None:
        """Return attention broker state dict, or None if not initialized."""
        if self._attention_broker is None:
            return None
        try:
            return self._attention_broker.get_state_dict()
        except Exception:
            return None

    def get_scene_state(self) -> dict | None:
        """Return latest scene understanding state dict, or None if not initialized."""
        if self._scene_understanding is None:
            return None
        try:
            return self._scene_understanding.get_state_dict()
        except Exception:
            return None

    def get_anticipation_state(self) -> dict | None:
        """Return anticipation engine state dict (hazard counts, history), or None."""
        if self._anticipation_engine is None:
            return None
        try:
            return self._anticipation_engine.get_state_dict()
        except Exception:
            return None

    def get_active_hazards(self) -> list:
        """Return active hazard assessments from last anticipation analysis."""
        if self._anticipation_engine is None:
            return []
        try:
            return self._anticipation_engine.get_active_hazards()
        except Exception:
            return []

    def get_clip_state(self) -> dict | None:
        """Return CLIP classifier state dict, or None if not initialized."""
        if self._clip_classifier is None:
            return None
        try:
            return self._clip_classifier.get_state_dict()
        except Exception:
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

        # Reset Phase 2 components
        if self._activity_monitor is not None:
            try:
                self._activity_monitor._previous_activities.clear()
            except (AttributeError, Exception):
                pass
        self._previous_world_state = None

        # Reset Phase 3 components
        if self._causal_reasoner is not None:
            try:
                self._causal_reasoner.reset()
            except (AttributeError, Exception):
                pass
        if self._context_memory is not None:
            try:
                self._context_memory.reset()
            except (AttributeError, Exception):
                pass

        # Reset Phase 5 components
        if self._episodic_memory is not None:
            try:
                self._episodic_memory.reset()
            except (AttributeError, Exception):
                pass
        if self._goal_recognizer is not None:
            try:
                self._goal_recognizer.reset()
            except (AttributeError, Exception):
                pass

        # Reset Phase 6 components
        if self._nl_task_manager is not None:
            try:
                self._nl_task_manager.reset()
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
        if self._pose_estimator is not None:
            try:
                self._pose_estimator.close()
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

    def _run_prediction(
        self,
        world_state: WorldState,
        frame_number: int,
        timestamp: float,
    ) -> list[EntityEvent]:
        """Run trajectory prediction and generate proactive alerts."""
        if self._proactive_alerter is None:
            return []
        try:
            return self._proactive_alerter.generate_alerts(
                tracks=world_state.active_tracks,
                zones=self._zones,
                current_frame=frame_number,
                current_timestamp=timestamp,
            )
        except Exception:
            logger.exception("Trajectory prediction failed at frame %d", frame_number)
            return []

    def _run_activity_recognition(
        self,
        world_state: WorldState,
        timestamp: float,
        frame_number: int,
    ) -> list[EntityEvent]:
        """Run activity classification and generate activity change events."""
        if self._activity_monitor is None:
            return []
        try:
            return self._activity_monitor.update(
                tracks=world_state.active_tracks,
                timestamp=timestamp,
                frame_number=frame_number,
            )
        except Exception:
            logger.exception("Activity recognition failed at frame %d", frame_number)
            return []

    def _run_attention_scoring(self, world_state: WorldState) -> None:
        """Score current scene attention and store for adaptive sampling."""
        if self._attention_scorer is None:
            return
        try:
            score = self._attention_scorer.score(
                world_state, self._previous_world_state
            )
            self._previous_world_state = world_state
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Attention score: %.3f (%s)", score.total, score.reason
                )
        except Exception:
            logger.exception("Attention scoring failed")

    def _run_causal_reasoning(self, world_state: WorldState) -> list:
        """Analyze world state events for causal relationships."""
        links: list = []
        if self._causal_reasoner is not None:
            try:
                links = self._causal_reasoner.analyze(world_state)
            except Exception:
                logger.exception("Causal reasoning failed")
        # Phase 5: feed links into the structural causal graph
        if self._causal_graph is not None and links:
            try:
                for link in links:
                    self._causal_graph.add_link(link)
                current_time = getattr(world_state, "timestamp", 0.0)
                if current_time:
                    self._causal_graph.prune_old_nodes(current_time)
            except Exception:
                logger.exception("CausalGraph update failed")
        return links

    def _run_episodic_memory(self, world_state: WorldState) -> None:
        """Segment event stream into episodes (Stage 4g). Feed closed episodes to LTM."""
        if self._episodic_memory is None:
            return
        try:
            for event in world_state.events:
                closed_ep = self._episodic_memory.push_event(event, world_state)
                if closed_ep is not None and self._long_term_memory is not None:
                    try:
                        ep_dict = {
                            "start_time": closed_ep.start_time,
                            "end_time": closed_ep.end_time,
                            "event_count": closed_ep.event_count,
                            "severity": closed_ep.severity,
                            "entity_ids": list(closed_ep.entity_ids),
                            "duration_seconds": closed_ep.duration_seconds,
                            "event_type_counts": {
                                k.name if hasattr(k, "name") else str(k): v
                                for k, v in (getattr(closed_ep, "event_type_counts", None) or {}).items()
                            },
                        }
                        self._long_term_memory.record_episode_facts(ep_dict)
                    except Exception:
                        logger.debug("LTM record_episode_facts failed", exc_info=True)
        except Exception:
            logger.exception("Episodic memory update failed")

    def _run_goal_recognition(self, world_state: WorldState) -> None:
        """Infer entity intentions from trajectory and context (Stage 4h)."""
        if self._goal_recognizer is None:
            return
        try:
            for entity in world_state.scene_graph.entities:
                if not isinstance(entity, int):  # entities is list[SceneEntity]
                    self._goal_recognizer.observe(entity, world_state)
        except Exception:
            logger.exception("Goal recognition failed")

    def _run_deliberation(self, world_state: WorldState) -> None:
        """Run System 2 deliberative reasoning on high-significance events (Stage 6c)."""
        if self._deliberative_reasoner is None:
            return
        try:
            trigger_types = {EntityEventType.ANOMALY, EntityEventType.RULE_VIOLATION,
                             EntityEventType.PROLONGED_PRESENCE, EntityEventType.COLLISION_PREDICTED}
            for event in world_state.events:
                if event.event_type in trigger_types:
                    if self._deliberative_reasoner.should_deliberate(event):
                        goal_hyps = None
                        episodes = None
                        if self._goal_recognizer is not None:
                            try:
                                goal_hyps = self._goal_recognizer.get_hypotheses(event.entity_id)
                            except Exception:
                                pass
                        if self._episodic_memory is not None:
                            try:
                                episodes = self._episodic_memory.get_recent_episodes(5)
                            except Exception:
                                pass
                        self._deliberative_reasoner.deliberate(
                            event, world_state,
                            goal_hypotheses=goal_hyps,
                            recent_episodes=episodes,
                        )
                        break  # one deliberation per frame to keep latency bounded
        except Exception:
            logger.exception("Deliberation failed")

    def _run_metacognition(self, frame_result: FrameResult) -> None:
        """Update metacognitive quality monitoring (Stage 8)."""
        if self._metacognitive_monitor is None:
            return
        try:
            self._metacognitive_monitor.observe_frame(frame_result)
        except Exception:
            logger.exception("Metacognition monitoring failed")

    def _run_context_memory(self, world_state: WorldState) -> None:
        """Update long-horizon context memory with current world state."""
        if self._context_memory is None:
            return
        try:
            self._context_memory.update(world_state)
        except Exception:
            logger.exception("Context memory update failed")

    def _run_reid(self, tracks: list, frame: np.ndarray) -> None:
        """Register tracked entities with cross-camera Re-ID tracker (Stage 2b)."""
        if self._cross_camera_tracker is None or not self._session_id:
            return
        try:
            from sopilot.perception.reid import AppearanceEncoder, ReIDFeature
            for track in tracks:
                try:
                    entity_id = getattr(track, "track_id", None) or getattr(track, "entity_id", None)
                    if entity_id is None:
                        continue
                    label = getattr(track, "label", "unknown") or "unknown"
                    bbox = getattr(track, "bbox", None)
                    if bbox is None:
                        continue
                    # bbox may be a BBox object or a tuple
                    if hasattr(bbox, "x"):
                        bbox_tuple = (bbox.x, bbox.y, bbox.w, bbox.h)
                    else:
                        bbox_tuple = tuple(bbox)[:4]
                    velocity = getattr(track, "velocity", (0.0, 0.0)) or (0.0, 0.0)
                    if not isinstance(velocity, (tuple, list)) or len(velocity) < 2:
                        velocity = (0.0, 0.0)
                    track_age = getattr(track, "age", 0) or 0
                    feat_vec = AppearanceEncoder.encode(
                        entity_id, label, bbox_tuple, tuple(velocity[:2]),
                        frame=frame, track_age=track_age,
                    )
                    feat = ReIDFeature(
                        entity_id=entity_id, session_id=self._session_id,
                        label=label, feature_vector=feat_vec,
                        bbox=bbox_tuple, track_age=track_age,
                    )
                    self._cross_camera_tracker.register(entity_id, self._session_id, feat)
                except Exception:
                    logger.debug("ReID registration failed for track", exc_info=True)
        except ImportError:
            pass
        except Exception:
            logger.debug("Re-ID stage failed", exc_info=True)

    def _run_nl_tasks(self, world_state: WorldState) -> list[EntityEvent]:
        """Check active NL tasks against current entities (Stage 4i)."""
        if self._nl_task_manager is None:
            return []
        events: list[EntityEvent] = []
        try:
            for entity in world_state.scene_graph.entities:
                if isinstance(entity, int):
                    continue
                entity_id = getattr(entity, "entity_id", None)
                label = getattr(entity, "label", "") or ""
                bbox = getattr(entity, "bbox", None)
                if entity_id is None or bbox is None:
                    continue
                if hasattr(bbox, "x"):
                    pos = (bbox.x + bbox.w / 2, bbox.y + bbox.h / 2)
                else:
                    pos = (0.5, 0.5)
                triggers = self._nl_task_manager.check_entity(
                    entity_id, label, pos, current_time=world_state.timestamp
                )
                for trig in triggers:
                    events.append(EntityEvent(
                        event_type=EntityEventType.RULE_VIOLATION,
                        entity_id=entity_id,
                        timestamp=world_state.timestamp,
                        frame_number=getattr(world_state, "frame_number", 0),
                        severity=ViolationSeverity[trig.severity.upper()]
                            if trig.severity.upper() in ViolationSeverity.__members__
                            else ViolationSeverity.WARNING,
                        details={
                            "description_ja": trig.description_ja,
                            "rule": trig.description_ja,
                            "source": "nl_task",
                            "task_id": trig.task_id,
                            "task_type": trig.task_type,
                        },
                    ))
            # Check count thresholds
            if events or world_state.scene_graph.entities:
                label_counts: dict[str, int] = {}
                for entity in world_state.scene_graph.entities:
                    if not isinstance(entity, int):
                        lbl = (getattr(entity, "label", "") or "unknown").lower()
                        label_counts[lbl] = label_counts.get(lbl, 0) + 1
                count_triggers = self._nl_task_manager.update_entity_counts(label_counts)
                for trig in count_triggers:
                    events.append(EntityEvent(
                        event_type=EntityEventType.RULE_VIOLATION,
                        entity_id=-1,
                        timestamp=world_state.timestamp,
                        frame_number=getattr(world_state, "frame_number", 0),
                        severity=ViolationSeverity[trig.severity.upper()]
                            if trig.severity.upper() in ViolationSeverity.__members__
                            else ViolationSeverity.WARNING,
                        details={
                            "description_ja": trig.description_ja,
                            "source": "nl_task",
                            "task_id": trig.task_id,
                            "task_type": trig.task_type,
                        },
                    ))
        except Exception:
            logger.debug("NL task check failed", exc_info=True)
        return events

    def _push_events_to_sse(self, events: list) -> None:
        """Broadcast new events to SSE queue (Stage 6d)."""
        if not self._session_id or not events:
            return
        try:
            from sopilot.perception import sse_events
            for event in events:
                event_type_name = (
                    event.event_type.name
                    if hasattr(event.event_type, "name")
                    else str(event.event_type)
                )
                severity_name = (
                    event.severity.name
                    if hasattr(event.severity, "name")
                    else str(event.severity)
                )
                payload: dict = {
                    "entity_id": getattr(event, "entity_id", None),
                    "severity": severity_name,
                    "frame_number": getattr(event, "frame_number", 0),
                    "description_ja": event.details.get("description_ja", ""),
                }
                # Add event-type-specific fields
                if event_type_name == "ANOMALY":
                    payload["detector"] = event.details.get("detector", "")
                    payload["z_score"] = event.details.get("z_score", 0.0)
                elif event_type_name == "RULE_VIOLATION":
                    payload["source"] = event.details.get("source", "rule")
                    payload["task_id"] = event.details.get("task_id", "")
                sse_events.push_event(self._session_id, event_type_name, payload)
        except ImportError:
            pass
        except Exception:
            logger.debug("SSE push failed", exc_info=True)

    def _push_early_warning_responses(self, actions: list) -> None:
        """Broadcast EarlyWarningResponder triggers to all SSE sessions (Phase 17).

        Unlike _push_events_to_sse (per-session), early warning alerts are
        camera-level events relevant to all connected UI clients.
        """
        try:
            from sopilot.perception import sse_events
            sessions = sse_events.list_sessions()
            if not sessions:
                return
            for action in actions:
                payload = action.to_dict()
                for sid in sessions:
                    sse_events.push_event(sid, "EARLY_WARNING_RESPONSE", payload)
        except ImportError:
            pass
        except Exception:
            logger.debug("SSE early warning broadcast failed", exc_info=True)

    def _run_pose_estimation(self, frame: np.ndarray) -> list:
        """Run pose estimation, returning empty list on failure."""
        if self._pose_estimator is None:
            return []
        try:
            return self._pose_estimator.estimate(frame)
        except Exception:
            logger.exception("Pose estimation failed")
            return []

    def _pose_ppe_violations(self, pose_results: list) -> list[Violation]:
        """Generate PPE violations from pose estimation results.

        Generates WARNING violations for:
        - Missing helmet: confidence of absence > 0.5
        - Missing safety vest: confidence of absence > 0.5

        Confidence of *absence* is defined as ``1.0 - ppe.<item>_confidence``
        when the item is inferred absent.
        """
        violations: list[Violation] = []
        for i, pr in enumerate(pose_results):
            ppe = pr.ppe

            # Missing helmet
            if not ppe.has_helmet:
                absence_conf = 1.0 - ppe.helmet_confidence
                if absence_conf > 0.5:
                    violations.append(
                        Violation(
                            rule="ヘルメット未着用を検出",
                            rule_index=-1,
                            description_ja=f"作業者#{i} ヘルメット未着用 (信頼度={absence_conf:.2f})",
                            severity=ViolationSeverity.WARNING,
                            confidence=absence_conf,
                            entity_ids=[],
                            bbox=pr.person_bbox,
                            evidence={
                                "source": "pose",
                                "helmet_confidence": ppe.helmet_confidence,
                                "vest_confidence": ppe.vest_confidence,
                            },
                            source="pose",
                        )
                    )

            # Missing vest
            if not ppe.has_vest:
                absence_conf = 1.0 - ppe.vest_confidence
                if absence_conf > 0.5:
                    violations.append(
                        Violation(
                            rule="安全ベスト未着用を検出",
                            rule_index=-1,
                            description_ja=f"作業者#{i} 安全ベスト未着用 (信頼度={absence_conf:.2f})",
                            severity=ViolationSeverity.WARNING,
                            confidence=absence_conf,
                            entity_ids=[],
                            bbox=pr.person_bbox,
                            evidence={
                                "source": "pose",
                                "helmet_confidence": ppe.helmet_confidence,
                                "vest_confidence": ppe.vest_confidence,
                            },
                            source="pose",
                        )
                    )

        return violations

    def _events_to_violations(self, world_state: WorldState) -> list[Violation]:
        """Convert world model events (zone entry, prolonged presence) to violations."""
        violations: list[Violation] = []

        for event in world_state.events:
            # Phase 12A: per-detector sigma filter for ANOMALY events
            if (
                self._sigma_tuner is not None
                and hasattr(event.event_type, "name")
                and event.event_type.name == "ANOMALY"
                and event.details
            ):
                det = event.details.get("detector", "")
                z = float(event.details.get("z_score", 0.0))
                if z < self._sigma_tuner.get_sigma(det):
                    continue  # below per-detector threshold → skip violation

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

        if event.event_type == EntityEventType.ZONE_ENTRY_PREDICTED:
            zone_id = event.details.get("zone_id", "unknown")
            seconds = event.details.get("estimated_seconds", 0)
            entity = world_state.scene_graph.get_entity(event.entity_id)
            return Violation(
                rule=f"zone_entry_predicted:{zone_id}",
                rule_index=-1,
                description_ja=(
                    f"エンティティ {event.entity_id} が約{seconds:.1f}秒後に"
                    f"{zone_id}に侵入予測"
                ),
                severity=ViolationSeverity.WARNING,
                confidence=event.confidence,
                entity_ids=[event.entity_id],
                bbox=entity.bbox if entity else None,
                evidence={
                    "event_type": event.event_type.value,
                    "zone_id": zone_id,
                    "estimated_seconds": seconds,
                    "source": "trajectory_prediction",
                },
                source="prediction",
            )

        if event.event_type == EntityEventType.COLLISION_PREDICTED:
            entity_a = event.details.get("entity_a_id", event.entity_id)
            entity_b = event.details.get("entity_b_id", -1)
            seconds = event.details.get("estimated_seconds", 0)
            return Violation(
                rule="collision_predicted",
                rule_index=-1,
                description_ja=(
                    f"エンティティ {entity_a} と {entity_b} が約{seconds:.1f}秒後に"
                    f"衝突予測"
                ),
                severity=ViolationSeverity.WARNING,
                confidence=event.confidence,
                entity_ids=[entity_a, entity_b],
                bbox=None,
                evidence={
                    "event_type": event.event_type.value,
                    "entity_a_id": entity_a,
                    "entity_b_id": entity_b,
                    "estimated_seconds": seconds,
                    "source": "trajectory_prediction",
                },
                source="prediction",
            )

        if event.event_type == EntityEventType.ANOMALY:
            detector = event.details.get("detector", "unknown")
            metric = event.details.get("metric", "unknown")
            severity_str = event.details.get("severity", "info")
            severity_map = {
                "info": ViolationSeverity.INFO,
                "warning": ViolationSeverity.WARNING,
                "critical": ViolationSeverity.CRITICAL,
            }
            severity = severity_map.get(severity_str, ViolationSeverity.INFO)
            desc_ja = event.details.get("description_ja", f"異常検出 ({detector}/{metric})")
            vlm_explanation = event.details.get("vlm_explanation")
            if vlm_explanation:
                desc_ja = f"{desc_ja}\n[VLM] {vlm_explanation}"

            entity = world_state.scene_graph.get_entity(event.entity_id)
            return Violation(
                rule=f"anomaly:{detector}/{metric}",
                rule_index=-1,
                description_ja=desc_ja,
                severity=severity,
                confidence=event.confidence,
                entity_ids=[event.entity_id] if event.entity_id >= 0 else [],
                bbox=entity.bbox if entity else None,
                evidence={
                    "event_type": event.event_type.value,
                    "detector": detector,
                    "metric": metric,
                    "z_score": event.details.get("z_score"),
                    "vlm_explanation": vlm_explanation,
                },
                source="anomaly",
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

    def _run_anomaly_explanations(
        self, world_state: WorldState, frame: np.ndarray
    ) -> None:
        """Attach VLM explanations to ANOMALY events (Stage 6b)."""
        if self._anomaly_explainer is None:
            return
        for event in world_state.events:
            if event.event_type == EntityEventType.ANOMALY:
                try:
                    explanation = self._anomaly_explainer.explain(
                        event, frame, world_state
                    )
                    if explanation:
                        event.details["vlm_explanation"] = explanation
                except Exception:
                    logger.exception("Anomaly explanation failed for event")

    def save_anomaly_profile(self, name: str) -> Path | None:
        """Save the current anomaly baseline to a named profile.

        Returns the path to the saved profile, or None if unavailable.
        """
        if self._world_model is None:
            return None
        try:
            baseline = self._world_model.get_anomaly_baseline()
            if not hasattr(baseline, "get_state"):
                return None
            from sopilot.perception.anomaly_profile import save_profile
            profile_dir = Path("data/anomaly_profiles")
            return save_profile(baseline, name, profile_dir)
        except Exception:
            logger.exception("Failed to save anomaly profile")
            return None

    def record_anomaly_feedback(
        self,
        detector: str,
        metric: str,
        entity_id: int,
        confirmed: bool,
        tuner: Any | None = None,
        note: str = "",
    ) -> bool:
        """Record operator feedback for an anomaly event.

        Args:
            detector: Detector name ("behavioral", "spatial", etc.)
            metric: Metric name ("speed_zscore", etc.)
            entity_id: Entity ID from the event (-1 for scene-level)
            confirmed: True = real anomaly, False = false positive
            tuner: AnomalyTuner instance (if None, this is a no-op)
            note: Optional operator note

        Returns:
            True if feedback was recorded, False otherwise.
        """
        if tuner is None:
            return False
        try:
            tuner.record_feedback(
                detector=detector,
                metric=metric,
                entity_id=entity_id,
                confirmed=confirmed,
                note=note,
            )
            return True
        except Exception:
            logger.exception("Failed to record anomaly feedback")
            return False

    def load_anomaly_profile(self, name: str) -> bool:
        """Load a named anomaly profile and apply it to the ensemble.

        Returns True on success, False on failure.
        """
        if self._world_model is None:
            return False
        try:
            baseline = self._world_model.get_anomaly_baseline()
            if not hasattr(baseline, "load_state"):
                return False
            from sopilot.perception.anomaly_profile import apply_profile, load_profile
            profile_path = Path("data/anomaly_profiles") / f"{name}.json"
            profile = load_profile(profile_path)
            apply_profile(baseline, profile)
            return True
        except Exception:
            logger.exception("Failed to load anomaly profile")
            return False

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
    session_id: str | None = None,
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
        elif config.detector_backend == "yolo_world":
            from sopilot.perception.detector import YOLOWorldDetector
            detector = YOLOWorldDetector(config=config)
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

    # ── Build Phase 2 components ───────────────────────────────────────
    trajectory_predictor = None
    try:
        from sopilot.perception.prediction import TrajectoryPredictor
        trajectory_predictor = TrajectoryPredictor()
        logger.info("TrajectoryPredictor initialized")
    except ImportError:
        logger.debug("Prediction module not available")
    except Exception:
        logger.exception("Failed to initialize trajectory predictor")

    activity_classifier = None
    activity_monitor = None
    try:
        from sopilot.perception.activity import ActivityClassifier, ActivityMonitor
        activity_classifier = ActivityClassifier()
        activity_monitor = ActivityMonitor(classifier=activity_classifier)
        logger.info("ActivityClassifier + ActivityMonitor initialized")
    except ImportError:
        logger.debug("Activity module not available")
    except Exception:
        logger.exception("Failed to initialize activity classifier")

    attention_scorer = None
    try:
        from sopilot.perception.attention import SceneAttentionScorer
        attention_scorer = SceneAttentionScorer()
        logger.info("SceneAttentionScorer initialized")
    except ImportError:
        logger.debug("Attention module not available")
    except Exception:
        logger.exception("Failed to initialize attention scorer")

    # ── Build Phase 3 components ───────────────────────────────────────
    causal_reasoner = None
    try:
        from sopilot.perception.causality import CausalReasoner
        causal_reasoner = CausalReasoner()
        logger.info("CausalReasoner initialized")
    except ImportError:
        logger.debug("Causality module not available")
    except Exception:
        logger.exception("Failed to initialize causal reasoner")

    context_memory = None
    try:
        from sopilot.perception.context_memory import ContextMemory
        context_memory = ContextMemory()
        logger.info("ContextMemory initialized")
    except ImportError:
        logger.debug("Context memory module not available")
    except Exception:
        logger.exception("Failed to initialize context memory")

    narrator = None
    try:
        from sopilot.perception.narrator import SceneNarrator
        narrator = SceneNarrator()
        logger.info("SceneNarrator initialized")
    except ImportError:
        logger.debug("Narrator module not available")
    except Exception:
        logger.exception("Failed to initialize scene narrator")

    # ── Build pose estimator (opt-in) ──────────────────────────────────
    pose_estimator = None
    if config.pose_enabled:
        try:
            from sopilot.perception.pose import PoseEstimator
            pose_estimator = PoseEstimator(
                model_name=config.pose_model,
                confidence_threshold=config.pose_confidence_threshold,
                keypoint_confidence=config.pose_keypoint_confidence,
            )
            logger.info(
                "PoseEstimator initialized (model=%s)", config.pose_model
            )
        except ImportError:
            logger.warning(
                "sopilot.perception.pose not available. "
                "Pose estimation will be disabled."
            )
        except Exception:
            logger.exception("Failed to initialize pose estimator")

    # ── Build anomaly explainer (VLM + anomaly_enabled) ─────────────
    anomaly_explainer = None
    if vlm_client is not None and config.anomaly_enabled:
        try:
            from sopilot.perception.anomaly_explainer import AnomalyExplainer
            anomaly_explainer = AnomalyExplainer(vlm_client=vlm_client)
            logger.info("AnomalyExplainer initialized (VLM-backed)")
        except ImportError:
            logger.debug("Anomaly explainer module not available")
        except Exception:
            logger.exception("Failed to initialize anomaly explainer")

    # ── Build Phase 5 components (deliberative cognition) ────────────
    episodic_memory = None
    try:
        from sopilot.perception.episodic_memory import EpisodicMemoryStore
        episodic_memory = EpisodicMemoryStore()
        logger.info("EpisodicMemoryStore initialized")
    except ImportError:
        logger.debug("Episodic memory module not available")
    except Exception:
        logger.exception("Failed to initialize episodic memory")

    goal_recognizer = None
    try:
        from sopilot.perception.goal_recognizer import GoalRecognizer
        goal_recognizer = GoalRecognizer()
        logger.info("GoalRecognizer initialized")
    except ImportError:
        logger.debug("Goal recognizer module not available")
    except Exception:
        logger.exception("Failed to initialize goal recognizer")

    deliberative_reasoner = None
    try:
        from sopilot.perception.deliberation import DeliberativeReasoner
        deliberative_reasoner = DeliberativeReasoner()
        logger.info("DeliberativeReasoner initialized")
    except ImportError:
        logger.debug("Deliberation module not available")
    except Exception:
        logger.exception("Failed to initialize deliberative reasoner")

    metacognitive_monitor = None
    try:
        from sopilot.perception.metacognition import MetacognitiveMonitor
        metacognitive_monitor = MetacognitiveMonitor()
        logger.info("MetacognitiveMonitor initialized")
    except ImportError:
        logger.debug("Metacognition module not available")
    except Exception:
        logger.exception("Failed to initialize metacognitive monitor")

    causal_graph = None
    try:
        from sopilot.perception.causality import CausalGraph
        causal_graph = CausalGraph()
        logger.info("CausalGraph initialized")
    except (ImportError, AttributeError):
        logger.debug("CausalGraph not available")
    except Exception:
        logger.exception("Failed to initialize causal graph")

    # ── Build Phase 6 components ──────────────────────────────────────
    nl_task_manager = None
    try:
        from sopilot.perception.nl_task import NLTaskManager
        nl_task_manager = NLTaskManager()
        logger.info("NLTaskManager initialized")
    except ImportError:
        logger.debug("NL task module not available")
    except Exception:
        logger.exception("Failed to initialize NL task manager")

    cross_camera_tracker = None
    try:
        from sopilot.perception.reid import CrossCameraTracker
        cross_camera_tracker = CrossCameraTracker()
        logger.info("CrossCameraTracker initialized")
    except ImportError:
        logger.debug("Re-ID module not available")
    except Exception:
        logger.exception("Failed to initialize cross-camera tracker")

    long_term_memory = None
    try:
        from sopilot.perception.long_term_memory import LongTermMemoryStore
        long_term_memory = LongTermMemoryStore()
        logger.info("LongTermMemoryStore initialized")
    except ImportError:
        logger.debug("Long-term memory module not available")
    except Exception:
        logger.exception("Failed to initialize long-term memory")

    # ── Assemble engine ───────────────────────────────────────────────
    engine = PerceptionEngine(
        config=config,
        detector=detector,
        tracker=tracker,
        scene_builder=scene_builder,
        world_model=world_model,
        reasoner=reasoner,
        vlm_client=vlm_client,
        trajectory_predictor=trajectory_predictor,
        activity_classifier=activity_classifier,
        activity_monitor=activity_monitor,
        attention_scorer=attention_scorer,
        causal_reasoner=causal_reasoner,
        context_memory=context_memory,
        narrator=narrator,
        pose_estimator=pose_estimator,
        anomaly_explainer=anomaly_explainer,
        episodic_memory=episodic_memory,
        goal_recognizer=goal_recognizer,
        deliberative_reasoner=deliberative_reasoner,
        metacognitive_monitor=metacognitive_monitor,
        causal_graph=causal_graph,
    )

    # Apply Phase 6 components (set after construction to keep __init__ clean)
    engine._nl_task_manager = nl_task_manager
    engine._cross_camera_tracker = cross_camera_tracker
    engine._long_term_memory = long_term_memory
    if session_id:
        engine.set_session_id(session_id)

    # ── Build Phase 7 components ──────────────────────────────────────
    try:
        from sopilot.perception.action_executor import ActionExecutor
        engine._action_executor = ActionExecutor()
        logger.info("ActionExecutor initialized")
    except ImportError:
        logger.debug("Action executor module not available")
    except Exception:
        logger.exception("Failed to initialize action executor")

    try:
        from sopilot.perception.multimodal import MultimodalFusionEngine
        engine._fusion_engine = MultimodalFusionEngine()
        logger.info("MultimodalFusionEngine initialized")
    except ImportError:
        logger.debug("Multimodal fusion module not available")
    except Exception:
        logger.exception("Failed to initialize multimodal fusion engine")

    # ── Build Phase 8 components ──────────────────────────────────────
    try:
        from sopilot.perception.depth import MonocularDepthEstimator
        engine._depth_estimator = MonocularDepthEstimator()
        logger.info("MonocularDepthEstimator initialized")
    except ImportError:
        logger.debug("Depth estimation module not available")
    except Exception:
        logger.exception("Failed to initialize depth estimator")

    try:
        from sopilot.perception.spatial_map import SpatialMap
        engine._spatial_map = SpatialMap()
        logger.info("SpatialMap initialized")
    except ImportError:
        logger.debug("Spatial map module not available")
    except Exception:
        logger.exception("Failed to initialize spatial map")

    try:
        from sopilot.perception.adaptive_learner import AdaptiveLearner
        _anomaly_ensemble = getattr(
            getattr(engine, "_world_model", None), "_anomaly_ensemble", None
        )
        engine._adaptive_learner = AdaptiveLearner(ensemble=_anomaly_ensemble)
        logger.info("AdaptiveLearner initialized")
    except ImportError:
        logger.debug("Adaptive learner module not available")
    except Exception:
        logger.exception("Failed to initialize adaptive learner")

    try:
        from sopilot.perception.attention_broker import AttentionBroker
        engine._attention_broker = AttentionBroker(
            global_cpm=config.vlm_max_calls_per_minute * 3,
            session_cpm=config.vlm_max_calls_per_minute,
        )
        logger.info("AttentionBroker initialized")
    except ImportError:
        logger.debug("Attention broker module not available")
    except Exception:
        logger.exception("Failed to initialize attention broker")

    try:
        from sopilot.perception.scene_understanding import SceneUnderstanding
        engine._scene_understanding = SceneUnderstanding()
        logger.info("SceneUnderstanding initialized")
    except ImportError:
        logger.debug("Scene understanding module not available")
    except Exception:
        logger.exception("Failed to initialize scene understanding")

    # ── Build Phase 9 components ──────────────────────────────────────────
    try:
        from sopilot.perception.anticipation import AnticipationEngine
        engine._anticipation_engine = AnticipationEngine(
            fps_hint=getattr(config, "fps_hint", 10.0),
        )
        logger.info("AnticipationEngine (Phase 9) initialized")
    except ImportError:
        logger.debug("Anticipation engine module not available")
    except Exception:
        logger.exception("Failed to initialize anticipation engine")

    try:
        from sopilot.perception.clip_classifier import build_clip_classifier
        engine._clip_classifier = build_clip_classifier(backend="mock")
        logger.info("CLIPZeroShotClassifier initialized (backend=mock)")
    except ImportError:
        logger.debug("CLIP classifier module not available")
    except Exception:
        logger.exception("Failed to initialize CLIP classifier")

    # ── Phase 10: AnomalyTuner 注入 ──────────────────────────────────────────
    try:
        from sopilot.perception.anomaly_tuner import AnomalyTuner as _AT
        engine._anomaly_tuner = _AT(Path("data/anomaly_feedback.json"))
        logger.info("AnomalyTuner (Phase 10) initialized")
    except ImportError:
        logger.debug("AnomalyTuner module not available")
    except Exception:
        logger.exception("Failed to initialize AnomalyTuner")

    # ── Phase 11A: ReviewQueue 注入 ──────────────────────────────────────────
    try:
        from sopilot.perception.active_query import ReviewQueue as _RQ
        engine._review_queue = _RQ(
            z_threshold=getattr(config, "review_z_threshold", 2.5),
            max_pending=getattr(config, "review_queue_max_pending", 50),
            dedup_seconds=getattr(config, "review_dedup_seconds", 60.0),
            persist_path=Path("data/review_queue.json"),
            frames_root=Path("data/review_frames"),
        )
        logger.info("ReviewQueue (Phase 11A) initialized")
    except ImportError:
        logger.debug("Active query module not available")
    except Exception:
        logger.exception("Failed to initialize ReviewQueue")

    # ── Phase 12A: SigmaTuner 注入 ──────────────────────────────────────────
    try:
        from sopilot.perception.sigma_tuner import SigmaTuner as _ST
        engine._sigma_tuner = _ST(
            base_sigma=getattr(config, "anomaly_sigma_threshold", 2.0),
            state_path=Path("data/sigma_state.json"),
        )
        logger.info("SigmaTuner (Phase 12A) initialized")
    except ImportError:
        logger.debug("SigmaTuner module not available")
    except Exception:
        logger.exception("Failed to initialize SigmaTuner")

    # ── Phase 15: EarlyWarningEngine 注入 ────────────────────────────────────
    try:
        from sopilot.perception.early_warning import EarlyWarningEngine as _EW
        engine._early_warning = _EW()
        logger.info("EarlyWarningEngine (Phase 15) initialized")
    except ImportError:
        logger.debug("EarlyWarning module not available")
    except Exception:
        logger.exception("Failed to initialize EarlyWarningEngine")

    # ── Phase 16: EarlyWarningResponder 注入 ─────────────────────────────────
    try:
        from sopilot.perception.early_warning_responder import EarlyWarningResponder as _EWR
        engine._early_warning_responder = _EWR()
        logger.info("EarlyWarningResponder (Phase 16) initialized")
    except ImportError:
        logger.debug("EarlyWarningResponder module not available")
    except Exception:
        logger.exception("Failed to initialize EarlyWarningResponder")

    # ── Phase 19: HealthHistoryStore 注入 ────────────────────────────────────
    try:
        from sopilot.perception.health_history import HealthHistoryStore as _HHS
        engine._health_history = _HHS(
            state_path=Path("data/health_history.json"),
        )
        logger.info("HealthHistoryStore (Phase 19) initialized")
    except ImportError:
        logger.debug("HealthHistoryStore module not available")
    except Exception:
        logger.exception("Failed to initialize HealthHistoryStore")

    return engine
