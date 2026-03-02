"""Scene graph construction and spatial reasoning.

Converts tracked entities and zone definitions into a rich scene graph with
spatial and semantic relations.  The scene graph is the primary data structure
consumed by the WorldModel and HybridReasoner for rule evaluation.

Pipeline position:
    Frame -> Detect -> Track -> **Scene Graph** -> World Model -> Reason

Complexity:
    - Entity creation: O(n * z) where n = entities, z = zones
    - Relation inference: O(n^2) — acceptable for n < 50 per config
    - Relation pruning: O(r log r) where r = total candidate relations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sopilot.perception.types import (
    BBox,
    PerceptionConfig,
    Relation,
    SceneEntity,
    SceneGraph,
    SpatialRelation,
    Track,
    TrackState,
    Zone,
)

logger = logging.getLogger(__name__)


# ── Equipment and machine label sets ─────────────────────────────────────────

SAFETY_EQUIPMENT: set[str] = {
    "helmet",
    "hard_hat",
    "safety_helmet",
    "vest",
    "safety_vest",
    "hi_vis_vest",
    "mask",
    "face_mask",
    "respirator",
    "goggles",
    "safety_goggles",
    "glasses",
    "gloves",
    "safety_gloves",
    "harness",
    "safety_harness",
    "safety_belt",
}

MACHINE_KEYWORDS: set[str] = {
    "forklift",
    "crane",
    "excavator",
    "conveyor",
    "machine",
    "equipment",
    "vehicle",
    "truck",
}

PERSON_LABELS: set[str] = {"person", "worker", "human", "operator", "pedestrian"}

# Equipment worn on the upper body (head, face, torso).  These are checked
# against the upper portion of a person's bounding box.
_HEAD_EQUIPMENT: set[str] = {
    "helmet",
    "hard_hat",
    "safety_helmet",
    "mask",
    "face_mask",
    "respirator",
    "goggles",
    "safety_goggles",
    "glasses",
}

# Equipment worn or held at mid-body height.
_BODY_EQUIPMENT: set[str] = {
    "vest",
    "safety_vest",
    "hi_vis_vest",
    "gloves",
    "safety_gloves",
    "harness",
    "safety_harness",
    "safety_belt",
}

# Semantic relation priorities — higher number = keep over lower.
_RELATION_PRIORITY: dict[SpatialRelation, int] = {
    SpatialRelation.WEARING: 100,
    SpatialRelation.HOLDING: 95,
    SpatialRelation.OPERATING: 90,
    SpatialRelation.INSIDE: 70,
    SpatialRelation.CONTAINS: 70,
    SpatialRelation.OVERLAPS: 60,
    SpatialRelation.NEAR: 40,
    SpatialRelation.ABOVE: 30,
    SpatialRelation.BELOW: 30,
    SpatialRelation.LEFT_OF: 20,
    SpatialRelation.RIGHT_OF: 20,
    SpatialRelation.FAR: 10,
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _is_person(label: str) -> bool:
    """Check if a label represents a person."""
    return label.lower() in PERSON_LABELS


def _is_safety_equipment(label: str) -> bool:
    """Check if a label represents safety equipment."""
    return label.lower() in SAFETY_EQUIPMENT


def _is_machine(label: str) -> bool:
    """Check if a label matches any machine/equipment keyword."""
    lower = label.lower()
    return any(kw in lower for kw in MACHINE_KEYWORDS)


def _bbox_upper_portion(bbox: BBox, fraction: float = 0.4) -> BBox:
    """Return the upper portion of a bounding box.

    For a person bbox, the upper ~40% covers head and shoulders — the
    region where helmets, goggles, masks, and vests are expected.
    """
    split_y = bbox.y1 + bbox.height * fraction
    return BBox(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=split_y)


def _bbox_middle_portion(bbox: BBox) -> BBox:
    """Return the middle third of a bounding box.

    For a person bbox, the middle ~33% covers torso to waist — the
    region where held objects, harnesses, and vests are expected.
    """
    h = bbox.height
    mid_y1 = bbox.y1 + h * 0.3
    mid_y2 = bbox.y1 + h * 0.7
    return BBox(x1=bbox.x1, y1=mid_y1, x2=bbox.x2, y2=mid_y2)


def _size_ratio(small: BBox, large: BBox) -> float:
    """Ratio of small-area to large-area.  Returns value in [0, 1]."""
    large_area = large.area
    if large_area < 1e-10:
        return 0.0
    return min(1.0, small.area / large_area)


def assign_zones(entity: SceneEntity, zones: list[Zone]) -> list[str]:
    """Return zone_ids of zones whose polygon contains this entity's bbox center.

    Parameters
    ----------
    entity:
        A scene entity with a valid bbox.
    zones:
        All zone definitions for the current camera view.

    Returns
    -------
    list[str]
        Zone IDs that contain this entity (may be empty).
    """
    return [z.zone_id for z in zones if z.contains_bbox(entity.bbox)]


def infer_person_equipment_relation(
    person: SceneEntity,
    equipment: SceneEntity,
) -> Relation | None:
    """Check if equipment is worn by or held by a person.

    Decision logic:
    1. Person must have a person-like label; equipment must be safety equipment.
    2. Equipment bbox must be significantly smaller than person bbox (area ratio < 0.6).
    3. If equipment overlaps the upper portion of the person -> WEARING.
    4. If equipment overlaps the middle portion of the person -> HOLDING.
    5. Otherwise -> None.

    Parameters
    ----------
    person:
        The candidate person entity.
    equipment:
        The candidate equipment entity.

    Returns
    -------
    Relation | None
        A WEARING or HOLDING relation, or None if no semantic match.
    """
    if not _is_person(person.label):
        return None
    if not _is_safety_equipment(equipment.label):
        return None

    # Equipment must be smaller than the person.
    ratio = _size_ratio(equipment.bbox, person.bbox)
    if ratio > 0.6:
        return None

    equip_label_lower = equipment.label.lower()

    # Check upper portion (head/shoulders) for WEARING.
    upper = _bbox_upper_portion(person.bbox)
    upper_iou = upper.iou(equipment.bbox)
    if upper_iou > 0.05 or upper.contains(equipment.bbox):
        # Head equipment strongly suggests WEARING.
        if equip_label_lower in _HEAD_EQUIPMENT:
            conf = min(1.0, person.confidence * equipment.confidence + 0.2)
            return Relation(
                subject_id=person.entity_id,
                predicate=SpatialRelation.WEARING,
                object_id=equipment.entity_id,
                confidence=round(conf, 3),
            )
        # Body equipment in upper portion -> also WEARING (e.g. vest).
        if equip_label_lower in _BODY_EQUIPMENT:
            conf = min(1.0, person.confidence * equipment.confidence + 0.1)
            return Relation(
                subject_id=person.entity_id,
                predicate=SpatialRelation.WEARING,
                object_id=equipment.entity_id,
                confidence=round(conf, 3),
            )

    # Check middle portion for HOLDING.
    middle = _bbox_middle_portion(person.bbox)
    mid_iou = middle.iou(equipment.bbox)
    if mid_iou > 0.05 or middle.contains(equipment.bbox):
        conf = min(1.0, person.confidence * equipment.confidence)
        return Relation(
            subject_id=person.entity_id,
            predicate=SpatialRelation.HOLDING,
            object_id=equipment.entity_id,
            confidence=round(conf, 3),
        )

    # Equipment overlaps the full person bbox but not specifically upper/middle.
    person_iou = person.bbox.iou(equipment.bbox)
    if person_iou > 0.1:
        conf = min(1.0, person.confidence * equipment.confidence * 0.8)
        return Relation(
            subject_id=person.entity_id,
            predicate=SpatialRelation.WEARING,
            object_id=equipment.entity_id,
            confidence=round(conf, 3),
        )

    return None


def _infer_person_machine_relation(
    person: SceneEntity,
    machine: SceneEntity,
    config: PerceptionConfig,
) -> Relation | None:
    """Check if a person is operating a machine.

    A person is considered to be operating a machine if they are NEAR it
    (within the configured near-threshold distance).

    Parameters
    ----------
    person:
        A person entity.
    machine:
        A machine/equipment entity.
    config:
        Perception configuration for threshold values.

    Returns
    -------
    Relation | None
        An OPERATING relation if the person is near the machine, else None.
    """
    if not _is_person(person.label):
        return None
    if not _is_machine(machine.label):
        return None

    distance = person.bbox.distance_to(machine.bbox)
    if distance <= config.scene_near_threshold:
        # Confidence decays with distance.
        proximity_factor = 1.0 - (distance / config.scene_near_threshold) if config.scene_near_threshold > 0 else 1.0
        conf = min(1.0, person.confidence * machine.confidence * proximity_factor)
        return Relation(
            subject_id=person.entity_id,
            predicate=SpatialRelation.OPERATING,
            object_id=machine.entity_id,
            confidence=round(conf, 3),
        )

    return None


def compute_spatial_relations(
    e1: SceneEntity,
    e2: SceneEntity,
    config: PerceptionConfig,
) -> list[Relation]:
    """Compute all spatial relations between two entities.

    Produces zero or more relations describing how *e1* relates to *e2*.
    Directional relations are emitted in one direction only (e.g. ABOVE
    from the higher entity to the lower one).

    Parameters
    ----------
    e1:
        First entity (subject for directional relations).
    e2:
        Second entity (object for directional relations).
    config:
        Perception configuration for threshold values.

    Returns
    -------
    list[Relation]
        All applicable spatial relations between the pair.
    """
    relations: list[Relation] = []
    b1, b2 = e1.bbox, e2.bbox
    pair_confidence = min(e1.confidence, e2.confidence)

    # ── Distance-based: NEAR / FAR ──────────────────────────────────────
    distance = b1.distance_to(b2)
    if distance <= config.scene_near_threshold:
        # Confidence is higher when entities are closer.
        proximity = 1.0 - (distance / config.scene_near_threshold) if config.scene_near_threshold > 0 else 1.0
        conf = round(pair_confidence * (0.5 + 0.5 * proximity), 3)
        relations.append(
            Relation(subject_id=e1.entity_id, predicate=SpatialRelation.NEAR, object_id=e2.entity_id, confidence=conf)
        )
    else:
        # FAR — lower confidence, mostly for completeness.
        relations.append(
            Relation(subject_id=e1.entity_id, predicate=SpatialRelation.FAR, object_id=e2.entity_id, confidence=round(pair_confidence * 0.5, 3))
        )

    # ── Vertical: ABOVE / BELOW ─────────────────────────────────────────
    cy1 = b1.center[1]
    cy2 = b2.center[1]
    y_diff = cy1 - cy2  # negative = e1 is above e2 (smaller y = higher in image)
    if abs(y_diff) > config.scene_vertical_threshold:
        if y_diff < 0:
            relations.append(
                Relation(subject_id=e1.entity_id, predicate=SpatialRelation.ABOVE, object_id=e2.entity_id, confidence=round(pair_confidence, 3))
            )
        else:
            relations.append(
                Relation(subject_id=e1.entity_id, predicate=SpatialRelation.BELOW, object_id=e2.entity_id, confidence=round(pair_confidence, 3))
            )

    # ── Horizontal: LEFT_OF / RIGHT_OF ──────────────────────────────────
    cx1 = b1.center[0]
    cx2 = b2.center[0]
    x_diff = cx1 - cx2
    # Use the same threshold as vertical for consistency.
    if abs(x_diff) > config.scene_vertical_threshold:
        if x_diff < 0:
            relations.append(
                Relation(subject_id=e1.entity_id, predicate=SpatialRelation.LEFT_OF, object_id=e2.entity_id, confidence=round(pair_confidence, 3))
            )
        else:
            relations.append(
                Relation(subject_id=e1.entity_id, predicate=SpatialRelation.RIGHT_OF, object_id=e2.entity_id, confidence=round(pair_confidence, 3))
            )

    # ── Containment: INSIDE / CONTAINS ──────────────────────────────────
    if b2.contains(b1):
        relations.append(
            Relation(subject_id=e1.entity_id, predicate=SpatialRelation.INSIDE, object_id=e2.entity_id, confidence=round(pair_confidence, 3))
        )
    elif b1.contains(b2):
        relations.append(
            Relation(subject_id=e1.entity_id, predicate=SpatialRelation.CONTAINS, object_id=e2.entity_id, confidence=round(pair_confidence, 3))
        )

    # ── Overlap: OVERLAPS ───────────────────────────────────────────────
    iou = b1.iou(b2)
    if iou >= config.scene_overlap_threshold:
        relations.append(
            Relation(subject_id=e1.entity_id, predicate=SpatialRelation.OVERLAPS, object_id=e2.entity_id, confidence=round(pair_confidence * iou, 3))
        )

    return relations


def _prune_relations(
    relations: list[Relation],
    max_relations: int,
) -> list[Relation]:
    """Prune relations to stay within the configured maximum.

    Priority order:
    1. Semantic relations (WEARING > HOLDING > OPERATING) always kept first.
    2. Spatial relations sorted by (priority_score, confidence) descending.

    Parameters
    ----------
    relations:
        All candidate relations.
    max_relations:
        Maximum number of relations to retain.

    Returns
    -------
    list[Relation]
        Pruned list, length <= max_relations.
    """
    if len(relations) <= max_relations:
        return relations

    def sort_key(r: Relation) -> tuple[int, float]:
        priority = _RELATION_PRIORITY.get(r.predicate, 0)
        return (priority, r.confidence)

    sorted_rels = sorted(relations, key=sort_key, reverse=True)
    return sorted_rels[:max_relations]


# ── Scene Graph Builder ──────────────────────────────────────────────────────


class SceneGraphBuilder:
    """Builds a SceneGraph from tracked entities and zone definitions.

    The builder is stateless between frames — each call to :meth:`build`
    produces an independent snapshot.  This is intentional: temporal
    reasoning belongs in the WorldModel, not the scene graph.

    Usage::

        builder = SceneGraphBuilder(config)
        graph = builder.build(tracks, zones, (720, 1280), 1.0, 30)
    """

    def __init__(self, config: PerceptionConfig) -> None:
        self._config = config

    # ── Public API ───────────────────────────────────────────────────────

    def build(
        self,
        tracks: list[Track],
        zones: list[Zone],
        frame_shape: tuple[int, int],
        timestamp: float,
        frame_number: int,
    ) -> SceneGraph:
        """Build a scene graph from the current set of tracks.

        Parameters
        ----------
        tracks:
            All current tracks (only active/confirmed ones are included in
            the graph).
        zones:
            Spatial zone definitions for the camera view.
        frame_shape:
            (height, width) of the frame in pixels.
        timestamp:
            Timestamp in seconds (e.g. from video clock).
        frame_number:
            Absolute frame index.

        Returns
        -------
        SceneGraph
            A complete scene graph with entities, relations, and metadata.
        """
        # Step 1: Create entities from confirmed tracks.
        entities = self._create_entities(tracks, zones)

        logger.debug(
            "SceneGraph frame=%d: %d entities from %d tracks",
            frame_number,
            len(entities),
            len(tracks),
        )

        # Step 2: Infer all relations (spatial + semantic).
        relations = self._infer_all_relations(entities)

        # Step 3: Prune to configured maximum.
        relations = _prune_relations(relations, self._config.scene_graph_max_relations)

        logger.debug(
            "SceneGraph frame=%d: %d relations (max=%d)",
            frame_number,
            len(relations),
            self._config.scene_graph_max_relations,
        )

        return SceneGraph(
            timestamp=timestamp,
            frame_number=frame_number,
            entities=entities,
            relations=relations,
            frame_shape=frame_shape,
        )

    # ── Entity creation ──────────────────────────────────────────────────

    def _create_entities(
        self,
        tracks: list[Track],
        zones: list[Zone],
    ) -> list[SceneEntity]:
        """Convert confirmed tracks to scene entities with zone assignment.

        Only tracks in ACTIVE or OCCLUDED state (i.e. ``is_confirmed``)
        that have a valid bbox are included.
        """
        entities: list[SceneEntity] = []
        for track in tracks:
            if not track.is_confirmed:
                continue
            if track.bbox is None:
                continue

            entity = SceneEntity(
                entity_id=track.track_id,
                label=track.label,
                bbox=track.bbox,
                confidence=track.confidence,
                attributes=dict(track.attributes),  # shallow copy
            )

            # Assign zones.
            entity.zone_ids = assign_zones(entity, zones)

            entities.append(entity)

        return entities

    # ── Relation inference ───────────────────────────────────────────────

    def _infer_all_relations(
        self,
        entities: list[SceneEntity],
    ) -> list[Relation]:
        """Infer all pairwise relations between entities.

        For each ordered pair (i, j) where i < j, we compute:
        - Spatial relations (distance, direction, containment, overlap)
        - Semantic relations (wearing, holding, operating)

        Spatial relations are computed in one direction (e1 -> e2) to avoid
        duplicates: e.g. if e1 is LEFT_OF e2, we do not also emit
        e2 RIGHT_OF e1.  The consumer can infer the inverse.

        Semantic relations are checked both ways (person-equipment and
        equipment-person) because we need to identify which entity is the
        person.
        """
        relations: list[Relation] = []
        n = len(entities)

        for i in range(n):
            for j in range(i + 1, n):
                e1 = entities[i]
                e2 = entities[j]

                # Spatial relations (e1 -> e2 direction).
                spatial = compute_spatial_relations(e1, e2, self._config)
                relations.extend(spatial)

                # Semantic: person-equipment (check both orderings).
                sem = self._infer_semantic_relations(e1, e2)
                relations.extend(sem)

        return relations

    def _infer_semantic_relations(
        self,
        e1: SceneEntity,
        e2: SceneEntity,
    ) -> list[Relation]:
        """Infer semantic relations between a pair of entities.

        Checks both orderings: (e1=person, e2=equipment) and
        (e2=person, e1=equipment).  Also checks person-machine relations.
        """
        results: list[Relation] = []

        # Person-equipment relations.
        rel = infer_person_equipment_relation(e1, e2)
        if rel is not None:
            results.append(rel)
        else:
            rel = infer_person_equipment_relation(e2, e1)
            if rel is not None:
                results.append(rel)

        # Person-machine relations.
        rel = _infer_person_machine_relation(e1, e2, self._config)
        if rel is not None:
            results.append(rel)
        else:
            rel = _infer_person_machine_relation(e2, e1, self._config)
            if rel is not None:
                results.append(rel)

        return results
