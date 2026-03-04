"""Core data types for the Perception Engine.

The Perception Engine replaces stateless VLM-per-frame analysis with a
continuous perception pipeline:

    Frame → Detect → Track → Scene Graph → World Model → Reason → Events

These types form the shared vocabulary across all perception components.
Every module in sopilot/perception/ imports from here.

Design principles:
    - Frozen dataclasses for immutable snapshots (Detection, BBox, Relation)
    - Mutable dataclasses for stateful objects (Track, WorldState)
    - Enums for finite state spaces (TrackState, SpatialRelation)
    - Normalized coordinates [0, 1] for resolution independence
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Bounding Box ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in normalized coordinates [0, 1].

    All spatial reasoning operates on normalized coords so the perception
    engine is resolution-independent.  Convert to pixel coords only at
    the rendering boundary.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    # ── Derived geometry ──────────────────────────────────────────────

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        h = self.height
        return self.width / h if h > 1e-8 else 0.0

    # ── Spatial computations ──────────────────────────────────────────

    def iou(self, other: BBox) -> float:
        """Intersection over Union."""
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)
        inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
        union = self.area + other.area - inter
        return inter / union if union > 1e-8 else 0.0

    def contains(self, other: BBox) -> bool:
        """True if *other* is fully inside *self*."""
        return (
            self.x1 <= other.x1
            and self.y1 <= other.y1
            and self.x2 >= other.x2
            and self.y2 >= other.y2
        )

    def distance_to(self, other: BBox) -> float:
        """Euclidean distance between centers (normalized coords)."""
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return math.hypot(cx1 - cx2, cy1 - cy2)

    def expanded(self, margin: float) -> BBox:
        """Return a new BBox expanded by *margin* on all sides, clamped to [0,1]."""
        return BBox(
            x1=max(0.0, self.x1 - margin),
            y1=max(0.0, self.y1 - margin),
            x2=min(1.0, self.x2 + margin),
            y2=min(1.0, self.y2 + margin),
        )

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert to pixel coordinates (x1, y1, x2, y2)."""
        return (
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height),
        )

    @classmethod
    def from_pixels(
        cls, x1: int, y1: int, x2: int, y2: int, width: int, height: int
    ) -> BBox:
        """Create from pixel coordinates."""
        return cls(
            x1=x1 / width,
            y1=y1 / height,
            x2=x2 / width,
            y2=y2 / height,
        )


# ── Detections ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Detection:
    """A single object detection in a frame.

    Produced by ObjectDetector.detect().
    Consumed by MultiObjectTracker.update().
    """

    bbox: BBox
    label: str
    confidence: float
    embedding: np.ndarray | None = None  # visual feature for re-identification
    attributes: dict[str, Any] = field(default_factory=dict)


# ── Tracking ───────────────────────────────────────────────────────────────


class TrackState(enum.Enum):
    """Lifecycle states for a tracked entity.

    State transitions:
        TENTATIVE → ACTIVE       (enough consecutive hits)
        ACTIVE    → OCCLUDED     (missed in a few frames)
        OCCLUDED  → ACTIVE       (re-detected)
        OCCLUDED  → LOST         (missed too many frames)
        ACTIVE    → LOST         (missed too many frames)
        LOST      → EXITED       (garbage collected)
    """

    TENTATIVE = "tentative"
    ACTIVE = "active"
    OCCLUDED = "occluded"
    LOST = "lost"
    EXITED = "exited"


@dataclass
class Track:
    """A tracked entity across multiple frames.

    Maintained by MultiObjectTracker.  Each Track has a unique track_id
    that persists across the entity's lifetime in the scene.
    """

    track_id: int
    label: str
    state: TrackState = TrackState.TENTATIVE
    bbox: BBox | None = None
    velocity: tuple[float, float] = (0.0, 0.0)  # dx, dy per frame
    confidence: float = 0.0
    first_frame: int = 0
    last_frame: int = 0
    age: int = 0  # total frames since creation
    hits: int = 0  # frames with successful detection match
    misses: int = 0  # consecutive frames without match
    attributes: dict[str, Any] = field(default_factory=dict)
    history: list[BBox] = field(default_factory=list)  # recent bbox history

    @property
    def is_confirmed(self) -> bool:
        return self.state in (TrackState.ACTIVE, TrackState.OCCLUDED)

    @property
    def lifetime_frames(self) -> int:
        return self.last_frame - self.first_frame + 1 if self.last_frame >= self.first_frame else 0


# ── Scene Graph ────────────────────────────────────────────────────────────


class SpatialRelation(enum.Enum):
    """Spatial relationships between entities in the scene graph."""

    NEAR = "near"
    FAR = "far"
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    INSIDE = "inside"
    CONTAINS = "contains"
    OVERLAPS = "overlaps"
    WEARING = "wearing"  # person wearing safety equipment
    HOLDING = "holding"  # person holding object
    OPERATING = "operating"  # person operating machine


@dataclass(frozen=True)
class Relation:
    """A directed relationship between two scene entities."""

    subject_id: int
    predicate: SpatialRelation
    object_id: int
    confidence: float = 1.0


@dataclass
class SceneEntity:
    """An entity in the scene graph (linked to a Track by entity_id == track_id)."""

    entity_id: int
    label: str
    bbox: BBox
    confidence: float
    attributes: dict[str, Any] = field(default_factory=dict)
    zone_ids: list[str] = field(default_factory=list)


@dataclass
class SceneGraph:
    """A snapshot of the scene at a specific moment.

    Built by SceneGraphBuilder from tracked entities.
    Consumed by WorldModel and HybridReasoner.
    """

    timestamp: float
    frame_number: int
    entities: list[SceneEntity]
    relations: list[Relation]
    frame_shape: tuple[int, int] = (0, 0)  # (height, width)

    def get_entity(self, entity_id: int) -> SceneEntity | None:
        for e in self.entities:
            if e.entity_id == entity_id:
                return e
        return None

    def get_relations_for(self, entity_id: int) -> list[Relation]:
        return [
            r
            for r in self.relations
            if r.subject_id == entity_id or r.object_id == entity_id
        ]

    def entities_with_label(self, label: str) -> list[SceneEntity]:
        return [e for e in self.entities if label.lower() in e.label.lower()]

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def person_count(self) -> int:
        return len(self.entities_with_label("person"))


# ── Zones ──────────────────────────────────────────────────────────────────


@dataclass
class Zone:
    """A named spatial region in the camera view.

    Zones enable spatial reasoning: "person inside restricted area",
    "no one in the work zone", etc.  Defined as normalized polygons.
    """

    zone_id: str
    name: str
    polygon: list[tuple[float, float]]  # normalized vertices [(x, y), ...]
    zone_type: str = "generic"  # "restricted", "hazard", "work_area", "safe"
    properties: dict[str, Any] = field(default_factory=dict)

    def contains_point(self, x: float, y: float) -> bool:
        """Ray-casting point-in-polygon test."""
        n = len(self.polygon)
        if n < 3:
            return False
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def contains_bbox(self, bbox: BBox) -> bool:
        """True if bbox center is inside the zone."""
        cx, cy = bbox.center
        return self.contains_point(cx, cy)

    def overlap_ratio(self, bbox: BBox) -> float:
        """Approximate overlap ratio of bbox with zone (sampling-based)."""
        if not self.polygon:
            return 0.0
        samples = 9  # 3x3 grid inside bbox
        hits = 0
        for i in range(3):
            for j in range(3):
                px = bbox.x1 + (i + 0.5) / 3 * bbox.width
                py = bbox.y1 + (j + 0.5) / 3 * bbox.height
                if self.contains_point(px, py):
                    hits += 1
        return hits / samples


# ── World Model Events ─────────────────────────────────────────────────────


class EntityEventType(enum.Enum):
    """Events generated by the world model's state machine."""

    ENTERED = "entered"  # new entity appeared in scene
    EXITED = "exited"  # entity left the scene
    ZONE_ENTERED = "zone_entered"  # entity entered a zone
    ZONE_EXITED = "zone_exited"  # entity left a zone
    STATE_CHANGED = "state_changed"  # entity attribute changed
    ANOMALY = "anomaly"  # unusual pattern detected
    RULE_VIOLATION = "rule_violation"  # explicit rule violated
    PROLONGED_PRESENCE = "prolonged_presence"  # entity stayed too long
    ZONE_ENTRY_PREDICTED = "zone_entry_predicted"  # predicted zone entry
    COLLISION_PREDICTED = "collision_predicted"  # predicted collision
    NEAR_MISS = "near_miss"                      # imminent near-miss hazard
    HAZARD_ZONE_BREACH = "hazard_zone_breach"    # entity about to enter forbidden zone
    CROWD_SURGE = "crowd_surge"                  # sudden crowd density increase


@dataclass
class EntityEvent:
    """An event generated by the world model."""

    event_type: EntityEventType
    entity_id: int
    timestamp: float
    frame_number: int
    details: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class WorldState:
    """The current state of the observed world.

    Produced by WorldModel.update() on every frame.
    Consumed by HybridReasoner for violation detection.
    """

    timestamp: float
    frame_number: int
    scene_graph: SceneGraph
    active_tracks: dict[int, Track]  # track_id → Track
    events: list[EntityEvent]  # events generated this frame
    zone_occupancy: dict[str, list[int]]  # zone_id → [entity_ids]
    entity_count: int = 0
    person_count: int = 0


# ── Reasoning ──────────────────────────────────────────────────────────────


class ViolationSeverity(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Violation:
    """A detected rule violation with full evidence chain.

    Contains enough information to explain *why* the violation was detected,
    *which entities* are involved, and *where* in the frame.
    """

    rule: str
    rule_index: int
    description_ja: str
    severity: ViolationSeverity
    confidence: float
    entity_ids: list[int] = field(default_factory=list)
    bbox: BBox | None = None  # primary evidence location
    evidence: dict[str, Any] = field(default_factory=dict)
    source: str = "local"  # "local" (scene graph) or "vlm" (Claude/Qwen)


@dataclass(frozen=True)
class PoseKeypoint:
    """A single COCO body keypoint with normalized coordinates."""

    x: float           # normalized [0, 1]
    y: float           # normalized [0, 1]
    confidence: float  # keypoint visibility confidence


@dataclass
class PPEStatus:
    """Per-person PPE compliance status inferred from pose + color analysis."""

    has_helmet: bool = False
    helmet_confidence: float = 0.0
    has_vest: bool = False
    vest_confidence: float = 0.0


@dataclass
class PoseResult:
    """Result for one person detected by the pose estimator."""

    person_bbox: BBox
    keypoints: list[PoseKeypoint]   # 17 COCO keypoints; confidence=0 if not visible
    ppe: PPEStatus
    pose_confidence: float          # overall person detection confidence


@dataclass
class FrameResult:
    """Complete result of processing a single frame through the perception engine.

    This is the main output type — equivalent to VLMResult but with
    much richer context from the perception pipeline.
    """

    timestamp: float
    frame_number: int
    world_state: WorldState
    violations: list[Violation]
    processing_time_ms: float
    detections_count: int = 0
    tracks_count: int = 0
    vlm_called: bool = False  # whether VLM was consulted for any rule
    vlm_latency_ms: float = 0.0
    pose_results: list["PoseResult"] = field(default_factory=list)


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass
class PerceptionConfig:
    """Configuration for the perception engine.

    All fields have sensible defaults.  Override via environment variables
    prefixed with ``PERCEPTION_`` or pass explicitly.
    """

    # Detection
    detector_backend: str = "grounding-dino"  # "grounding-dino", "yolo-world", "mock"
    detector_model_id: str = "IDEA-Research/grounding-dino-tiny"
    detection_confidence_threshold: float = 0.3
    yolo_confidence_threshold: float = 0.05  # YOLO-World uses lower threshold by default
    detection_nms_threshold: float = 0.5
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"

    # Tracking
    track_high_threshold: float = 0.5  # IoU threshold for high-confidence match
    track_low_threshold: float = 0.1  # IoU threshold for low-confidence match
    track_max_age: int = 30  # frames before track is moved to LOST
    track_min_hits: int = 3  # detections before track is confirmed
    track_history_length: int = 30  # max bbox history per track

    # Scene graph
    scene_near_threshold: float = 0.15  # normalized distance for "near" relation
    scene_overlap_threshold: float = 0.3  # IoU for "overlaps" relation
    scene_vertical_threshold: float = 0.1  # y-diff for above/below

    # World model
    temporal_memory_seconds: float = 300.0  # 5-minute rolling window
    zone_definitions: list[Zone] = field(default_factory=list)
    prolonged_presence_seconds: float = 60.0  # alert after 60s in restricted zone

    # Hybrid reasoning
    vlm_escalation_threshold: float = 0.6  # below this, ask VLM
    vlm_backend: str = "claude"  # fallback VLM backend
    vlm_api_key: str = ""
    vlm_max_calls_per_minute: int = 10  # rate-limit VLM escalation

    # Performance
    max_detections_per_frame: int = 50
    scene_graph_max_relations: int = 200
    skip_frames: int = 0  # process every Nth frame (0 = all)

    # Pose estimation
    pose_enabled: bool = False           # opt-in: disabled by default
    pose_model: str = "yolov8s-pose.pt"
    pose_confidence_threshold: float = 0.4
    pose_keypoint_confidence: float = 0.3  # minimum visibility for keypoint use

    # SAHI (Slicing Aided Hyper Inference) for small-object detection in high-res frames
    sahi_enabled: bool = True           # slice-and-infer; disable for real-time streams
    sahi_slice_height: int = 640        # tile height in pixels
    sahi_slice_width: int = 640         # tile width in pixels
    sahi_overlap_ratio: float = 0.2     # overlap fraction between adjacent tiles

    # Autonomous anomaly detection (4-detector ensemble)
    anomaly_enabled: bool = True              # enable anomaly detector ensemble
    anomaly_warmup_frames: int = 100          # frames before anomaly detection activates
    anomaly_sigma_threshold: float = 2.0      # minimum weighted z-score to fire
    anomaly_cooldown_seconds: float = 60.0    # suppress duplicate alerts per (detector, metric, entity)
    anomaly_spatial_grid_size: int = 10       # NxN spatial occupancy grid
    anomaly_ema_alpha: float = 0.05           # EMA smoothing factor (lower = slower adaptation)
    # Phase 10: Self-learning tuning parameters
    tuner_auto_apply_threshold: int = 20      # フィードバック蓄積後に apply_tuning() を自動実行
    # Phase 12A: Adaptive sigma tuning
    sigma_apply_interval: int = 10            # σ調整の最小フィードバック増分
    # Phase 11A: Active query review queue
    review_z_threshold: float = 2.5           # この z_score 以上の異常を review queue へ追加
    review_queue_max_pending: int = 50        # 最大保留件数
    review_dedup_seconds: float = 60.0        # 同一 (detector, metric) の重複抑制時間
    # Phase 11B: Frame ring buffer
    frame_ring_buffer_size: int = 50          # 保持するフレーム数 (JPEG圧縮済み)
