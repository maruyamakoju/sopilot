"""Goal recognition: infer entity intentions from trajectory and context.

Uses a library of goal hypotheses updated by observations:
- approach_zone   : entity moving toward a known zone
- enter_restricted: entity approaching/entering a restricted zone
- exit_area       : entity moving toward scene boundary (cx/cy near 0 or 1)
- loiter          : entity staying in small area for extended time
- interact_object : entity converging on a non-person entity
- patrol          : entity moving in large regular loops
- access_object   : entity with sustained approach toward an object

Architecture
------------
GoalRecognizer maintains a Bayesian-style belief state per entity.
Each ``observe()`` call:
  1. Records the entity's position in a rolling history buffer.
  2. Generates candidate GoalHypothesis objects from five heuristic checks.
  3. Merges candidates into the running belief state via exponential smoothing.
  4. Decays un-reinforced hypotheses and prunes those below the confidence floor.

All coordinates are normalized [0, 1].  No external dependencies beyond stdlib.
"""
from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from sopilot.perception.types import EntityEvent, EntityEventType, SceneEntity, WorldState


# ── Goal type registry ────────────────────────────────────────────────────────

GOAL_DEFINITIONS: dict[str, dict] = {
    "approach_zone": {
        "risk": 0.5,
        "description_ja": "エリアへの接近",
        "description_en": "Approaching a zone",
    },
    "enter_restricted": {
        "risk": 0.9,
        "description_ja": "立入禁止エリアへの侵入意図",
        "description_en": "Intending to enter restricted area",
    },
    "exit_area": {
        "risk": 0.2,
        "description_ja": "エリア離脱",
        "description_en": "Leaving the monitored area",
    },
    "loiter": {
        "risk": 0.45,
        "description_ja": "不審な滞留",
        "description_en": "Loitering / suspicious stationary behavior",
    },
    "interact_object": {
        "risk": 0.5,
        "description_ja": "物体への接近・操作",
        "description_en": "Approaching or interacting with an object",
    },
    "patrol": {
        "risk": 0.1,
        "description_ja": "巡回行動",
        "description_en": "Patrol / regular movement pattern",
    },
    "access_object": {
        "risk": 0.7,
        "description_ja": "対象物へのアクセス",
        "description_en": "Accessing a target object",
    },
}


@dataclass
class GoalHypothesis:
    """A probabilistic belief that entity is pursuing goal_type."""

    id: str                          # UUID
    entity_id: int
    entity_label: str
    goal_type: str                   # key in GOAL_DEFINITIONS
    confidence: float                # 0–1
    risk_score: float                # from GOAL_DEFINITIONS, modulated by confidence
    evidence: list[str]              # NL evidence descriptions (Japanese)
    target_zone: str | None          # name of target zone if applicable
    predicted_completion: float | None  # estimated unix timestamp
    created_at: float
    updated_at: float
    description_ja: str
    description_en: str


class GoalRecognizer:
    """Bayesian belief-state inference over entity goals.

    Call ``observe(entity, world_state)`` every frame for each entity.
    The system maintains a running belief state per entity, updated by:
      1. Trajectory direction (velocity vector toward zones/edges)
      2. Historical position variance (loiter detection)
      3. Zone proximity (approach / enter_restricted)
      4. Relative motion to other entities (interact / access)

    Hypotheses decay if not reinforced.  Hypotheses below
    ``_CONFIDENCE_FLOOR`` are pruned.

    Args:
        high_risk_threshold: Minimum risk_score for ``get_high_risk_intents()``.
    """

    _CONFIDENCE_FLOOR = 0.10
    _DECAY_PER_FRAME = 0.02          # confidence decay when no supporting obs
    _HISTORY_LEN = 30                # frames of position history to keep

    def __init__(self, high_risk_threshold: float = 0.55) -> None:
        self._high_risk_threshold = high_risk_threshold
        # entity_id → list[GoalHypothesis]
        self._beliefs: dict[int, list[GoalHypothesis]] = defaultdict(list)
        # entity_id → deque of (cx, cy) history
        self._pos_history: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self._HISTORY_LEN)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def observe(
        self, entity: SceneEntity, world_state: WorldState
    ) -> list[GoalHypothesis]:
        """Update beliefs for one entity given current world state.

        Records the entity's position, runs inference heuristics, merges
        candidates into the belief state, decays un-reinforced hypotheses,
        and prunes low-confidence ones.

        Returns:
            Updated list of hypotheses, sorted by confidence descending.
        """
        # Record position history
        cx, cy = entity.bbox.center
        self._pos_history[entity.entity_id].append((cx, cy))

        # Generate new candidates
        new_hyps = self._infer_new_hypotheses(entity, world_state)

        # Merge into belief state
        self._update_beliefs(entity.entity_id, new_hyps)

        # Prune below floor
        self._prune_beliefs(entity.entity_id)

        return self.get_hypotheses(entity.entity_id)

    def get_hypotheses(self, entity_id: int) -> list[GoalHypothesis]:
        """Return current hypotheses for entity_id (highest confidence first)."""
        hyps = self._beliefs.get(entity_id, [])
        return sorted(hyps, key=lambda h: h.confidence, reverse=True)

    def get_high_risk_intents(
        self, risk_threshold: float | None = None
    ) -> list[GoalHypothesis]:
        """Return all hypotheses across all entities where risk_score >= threshold."""
        threshold = risk_threshold if risk_threshold is not None else self._high_risk_threshold
        result: list[GoalHypothesis] = []
        for hyps in self._beliefs.values():
            for h in hyps:
                if h.risk_score >= threshold:
                    result.append(h)
        return result

    def reset_entity(self, entity_id: int) -> None:
        """Clear beliefs for an entity that has exited the scene."""
        self._beliefs.pop(entity_id, None)
        self._pos_history.pop(entity_id, None)

    def reset(self) -> None:
        """Clear all beliefs."""
        self._beliefs.clear()
        self._pos_history.clear()

    def get_state_dict(self) -> dict[str, Any]:
        """Serialize current state for the API (JSON-serializable)."""
        entity_hypotheses: dict[str, list[dict[str, Any]]] = {}
        for eid, hyps in self._beliefs.items():
            entity_hypotheses[str(eid)] = [
                {
                    "id": h.id,
                    "entity_id": h.entity_id,
                    "entity_label": h.entity_label,
                    "goal_type": h.goal_type,
                    "confidence": h.confidence,
                    "risk_score": h.risk_score,
                    "evidence": h.evidence,
                    "target_zone": h.target_zone,
                    "predicted_completion": h.predicted_completion,
                    "created_at": h.created_at,
                    "updated_at": h.updated_at,
                    "description_ja": h.description_ja,
                    "description_en": h.description_en,
                }
                for h in sorted(hyps, key=lambda h: h.confidence, reverse=True)
            ]
        return {
            "entity_hypotheses": entity_hypotheses,
            "high_risk_count": len(self.get_high_risk_intents()),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _infer_new_hypotheses(
        self, entity: SceneEntity, world_state: WorldState
    ) -> list[GoalHypothesis]:
        """Generate candidate hypotheses from current observation."""
        candidates: list[GoalHypothesis] = []

        # Zone proximity checks (approach_zone / enter_restricted)
        candidates.extend(self._check_zone_proximity(entity, world_state))

        # Loiter check
        loiter = self._check_loiter(entity)
        if loiter is not None:
            candidates.append(loiter)

        # Exit direction check
        exit_hyp = self._check_exit_direction(entity, world_state)
        if exit_hyp is not None:
            candidates.append(exit_hyp)

        # Patrol check
        patrol = self._check_patrol(entity)
        if patrol is not None:
            candidates.append(patrol)

        # Object approach check
        obj_hyp = self._check_object_approach(entity, world_state)
        if obj_hyp is not None:
            candidates.append(obj_hyp)

        return candidates

    def _update_beliefs(
        self, entity_id: int, new_hyps: list[GoalHypothesis]
    ) -> None:
        """Merge new hypotheses into existing belief state with Bayesian update.

        For each new hypothesis:
          - If a matching goal_type already exists: blend confidence via
            exponential smoothing (0.7 * existing + 0.3 * new).
          - Otherwise: add the hypothesis as a new belief.
        For all existing hypotheses NOT matched by any new hypothesis, apply
        decay.
        """
        now = time.time()
        existing = self._beliefs[entity_id]
        matched_goal_types: set[str] = set()

        for new_h in new_hyps:
            matched_goal_types.add(new_h.goal_type)
            found = False
            for ex in existing:
                if ex.goal_type == new_h.goal_type:
                    # Exponential smoothing blend
                    blended = 0.7 * ex.confidence + 0.3 * new_h.confidence
                    ex.confidence = min(1.0, blended)
                    ex.risk_score = GOAL_DEFINITIONS[ex.goal_type]["risk"] * ex.confidence
                    ex.evidence = new_h.evidence  # refresh evidence
                    ex.updated_at = now
                    if new_h.target_zone is not None:
                        ex.target_zone = new_h.target_zone
                    found = True
                    break
            if not found:
                existing.append(new_h)

        # Decay hypotheses not reinforced this frame
        for ex in existing:
            if ex.goal_type not in matched_goal_types:
                ex.confidence = max(0.0, ex.confidence * (1.0 - self._DECAY_PER_FRAME))
                ex.risk_score = GOAL_DEFINITIONS[ex.goal_type]["risk"] * ex.confidence
                ex.updated_at = now

    def _decay_beliefs(self, entity_id: int) -> None:
        """Reduce confidence of all hypotheses (not reinforced this frame)."""
        now = time.time()
        for h in self._beliefs.get(entity_id, []):
            h.confidence = max(0.0, h.confidence * (1.0 - self._DECAY_PER_FRAME))
            h.risk_score = GOAL_DEFINITIONS[h.goal_type]["risk"] * h.confidence
            h.updated_at = now
        self._prune_beliefs(entity_id)

    def _prune_beliefs(self, entity_id: int) -> None:
        """Remove hypotheses below _CONFIDENCE_FLOOR."""
        self._beliefs[entity_id] = [
            h for h in self._beliefs[entity_id]
            if h.confidence >= self._CONFIDENCE_FLOOR
        ]

    def _make_hypothesis(
        self,
        entity: SceneEntity,
        goal_type: str,
        confidence: float,
        evidence: list[str],
        target_zone: str | None = None,
        predicted_completion: float | None = None,
        timestamp: float = 0.0,
    ) -> GoalHypothesis:
        """Factory method for GoalHypothesis."""
        defn = GOAL_DEFINITIONS[goal_type]
        now = timestamp or time.time()
        risk_score = defn["risk"] * confidence
        return GoalHypothesis(
            id=str(uuid.uuid4()),
            entity_id=entity.entity_id,
            entity_label=entity.label,
            goal_type=goal_type,
            confidence=min(1.0, max(0.0, confidence)),
            risk_score=min(1.0, max(0.0, risk_score)),
            evidence=evidence,
            target_zone=target_zone,
            predicted_completion=predicted_completion,
            created_at=now,
            updated_at=now,
            description_ja=defn["description_ja"],
            description_en=defn["description_en"],
        )

    # ── Inference heuristics ──────────────────────────────────────────────────

    def _check_zone_proximity(
        self, entity: SceneEntity, world_state: WorldState
    ) -> list[GoalHypothesis]:
        """Detect zone approach and restricted zone entry intent.

        For each zone in world_state.scene_graph.zones (if available):
          - Compute the center of the zone polygon as the mean of its vertices.
          - If distance < 0.25:
              * Check if entity velocity points toward the zone.
              * If zone_type contains 'restricted', 'danger', or '禁止' →
                'enter_restricted' hypothesis.
              * Otherwise → 'approach_zone' hypothesis.
          - Confidence is proportional to proximity (closer → higher).
        """
        zones: dict = getattr(world_state.scene_graph, "zones", {}) or {}
        results: list[GoalHypothesis] = []
        if not zones:
            return results

        cx, cy = entity.bbox.center
        # Retrieve velocity from active_tracks
        track = world_state.active_tracks.get(entity.entity_id)
        vx, vy = track.velocity if track is not None else (0.0, 0.0)

        for zone_id, zone in zones.items():
            # Compute zone centroid from polygon
            polygon = zone.polygon
            if not polygon:
                continue
            zone_cx = sum(p[0] for p in polygon) / len(polygon)
            zone_cy = sum(p[1] for p in polygon) / len(polygon)

            dist = math.hypot(cx - zone_cx, cy - zone_cy)
            if dist >= 0.25:
                continue

            # Velocity dot product toward zone center
            to_zone_x = zone_cx - cx
            to_zone_y = zone_cy - cy
            to_zone_len = math.hypot(to_zone_x, to_zone_y) + 1e-9
            vel_len = math.hypot(vx, vy)

            # Dot product: positive means moving toward zone
            dot = (vx * to_zone_x + vy * to_zone_y) / to_zone_len

            # Only generate hypothesis if velocity has meaningful component
            # toward the zone, OR entity is very close (< 0.1)
            if dot <= 0.0 and dist >= 0.1:
                continue

            # Confidence: inversely proportional to distance, scaled 0.3–0.8
            prox_conf = max(0.3, 0.8 - dist * 2.0)

            # Boost confidence if velocity strongly points toward zone
            if vel_len > 1e-6:
                vel_alignment = dot / (vel_len + 1e-9)
                prox_conf = min(0.9, prox_conf + max(0.0, vel_alignment) * 0.2)

            zone_type = getattr(zone, "zone_type", "generic") or "generic"
            is_restricted = any(
                kw in zone_type.lower() for kw in ("restricted", "danger", "hazard")
            ) or "禁止" in zone_type

            goal_type = "enter_restricted" if is_restricted else "approach_zone"
            zone_name = getattr(zone, "name", zone_id)

            evidence = [
                f"{entity.label}(ID:{entity.entity_id})がゾーン'{zone_name}'に接近中"
                f" (距離={dist:.3f})"
            ]
            if is_restricted:
                evidence.append(f"ゾーン'{zone_name}'は立入禁止エリア ({zone_type})")

            results.append(
                self._make_hypothesis(
                    entity=entity,
                    goal_type=goal_type,
                    confidence=prox_conf,
                    evidence=evidence,
                    target_zone=zone_name,
                    timestamp=world_state.timestamp,
                )
            )

        return results

    def _check_loiter(self, entity: SceneEntity) -> GoalHypothesis | None:
        """Detect loitering: entity remaining in a small area for an extended time.

        Requires at least 15 positions in history.  If the std deviation of
        both x and y positions is < 0.05, a 'loiter' hypothesis is generated.
        Confidence increases with history length (longer stationary → higher).
        """
        history = list(self._pos_history.get(entity.entity_id, []))
        if len(history) < 15:
            return None

        xs = [p[0] for p in history]
        ys = [p[1] for p in history]

        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        var_x = sum((x - mean_x) ** 2 for x in xs) / len(xs)
        var_y = sum((y - mean_y) ** 2 for y in ys) / len(ys)
        std_x = math.sqrt(var_x)
        std_y = math.sqrt(var_y)

        if std_x >= 0.05 or std_y >= 0.05:
            return None

        # Confidence: proportional to history depth (15 frames → 0.4, 30 → 0.8)
        depth_ratio = min(1.0, len(history) / self._HISTORY_LEN)
        confidence = 0.4 + 0.4 * depth_ratio  # 0.4 – 0.8

        evidence = [
            f"{entity.label}(ID:{entity.entity_id})が同一エリアに滞留"
            f" (σx={std_x:.4f}, σy={std_y:.4f}, {len(history)}フレーム)"
        ]
        return self._make_hypothesis(
            entity=entity,
            goal_type="loiter",
            confidence=confidence,
            evidence=evidence,
        )

    def _check_exit_direction(
        self, entity: SceneEntity, world_state: WorldState
    ) -> GoalHypothesis | None:
        """Detect exit intent: entity near a scene boundary and moving outward.

        An entity is considered to be exiting if:
          - Its center is within 0.1 of any edge (cx < 0.1, cx > 0.9,
            cy < 0.1, or cy > 0.9).
          - Its velocity vector has a component pointing toward that edge
            (i.e., moving outward).

        Confidence is set to 0.6 (fixed) when the condition is met.
        """
        cx, cy = entity.bbox.center
        track = world_state.active_tracks.get(entity.entity_id)
        vx, vy = track.velocity if track is not None else (0.0, 0.0)

        near_left = cx < 0.1
        near_right = cx > 0.9
        near_top = cy < 0.1
        near_bottom = cy > 0.9

        near_edge = near_left or near_right or near_top or near_bottom
        if not near_edge:
            return None

        # Check velocity component toward the nearby edge
        moving_out = (
            (near_left and vx < 0.0)
            or (near_right and vx > 0.0)
            or (near_top and vy < 0.0)
            or (near_bottom and vy > 0.0)
        )

        # Allow static entities very close to the edge (< 0.05) even without
        # explicit outward velocity — they may be paused before exit.
        very_close = (
            cx < 0.05 or cx > 0.95 or cy < 0.05 or cy > 0.95
        )

        if not moving_out and not very_close:
            return None

        evidence = [
            f"{entity.label}(ID:{entity.entity_id})がエリア境界に接近"
            f" (cx={cx:.3f}, cy={cy:.3f}, vx={vx:.4f}, vy={vy:.4f})"
        ]
        return self._make_hypothesis(
            entity=entity,
            goal_type="exit_area",
            confidence=0.6,
            evidence=evidence,
            timestamp=world_state.timestamp,
        )

    def _check_patrol(self, entity: SceneEntity) -> GoalHypothesis | None:
        """Detect patrol behavior: large regular movement over a wide area.

        Requires ≥ 20 positions.  Conditions:
          - Total path length (sum of consecutive distances) > 0.8.
          - Spatial variance (mean of var_x and var_y) > 0.03 (covers a
            large area rather than a tight loop).

        Confidence is fixed at 0.4 (weak signal — needs more context).
        """
        history = list(self._pos_history.get(entity.entity_id, []))
        if len(history) < 20:
            return None

        # Total path length
        path_len = 0.0
        for i in range(1, len(history)):
            dx = history[i][0] - history[i - 1][0]
            dy = history[i][1] - history[i - 1][1]
            path_len += math.hypot(dx, dy)

        if path_len <= 0.8:
            return None

        xs = [p[0] for p in history]
        ys = [p[1] for p in history]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        var_x = sum((x - mean_x) ** 2 for x in xs) / len(xs)
        var_y = sum((y - mean_y) ** 2 for y in ys) / len(ys)
        spatial_var = (var_x + var_y) / 2.0

        if spatial_var <= 0.03:
            return None

        evidence = [
            f"{entity.label}(ID:{entity.entity_id})が広範囲を移動中 "
            f"(総移動距離={path_len:.3f}, 空間分散={spatial_var:.4f})"
        ]
        return self._make_hypothesis(
            entity=entity,
            goal_type="patrol",
            confidence=0.4,
            evidence=evidence,
        )

    def _check_object_approach(
        self, entity: SceneEntity, world_state: WorldState
    ) -> GoalHypothesis | None:
        """Detect entity converging on a non-person object.

        For each non-person entity in the scene graph:
          - If the current distance is < 0.2:
              * Check whether distance has been decreasing over the last
                5 frames by comparing current distance to the distance 5
                frames ago (using position history).
              * If converging: generate 'interact_object'.
              * If the entity is near a restricted zone (zone_ids contains a
                restricted zone): upgrade to 'access_object' (higher risk).

        Returns only the highest-confidence candidate (closest object).
        """
        sg = world_state.scene_graph
        entities_list = sg.entities if sg else []
        zones_map: dict = getattr(sg, "zones", {}) or {}

        best: GoalHypothesis | None = None
        best_dist = 1.0

        cx, cy = entity.bbox.center

        for other in entities_list:
            if other.entity_id == entity.entity_id:
                continue
            # Skip other persons
            if "person" in other.label.lower():
                continue

            other_cx, other_cy = other.bbox.center
            dist = math.hypot(cx - other_cx, cy - other_cy)

            if dist >= 0.2:
                continue

            # Check if distance has been decreasing (converging)
            history = list(self._pos_history.get(entity.entity_id, []))
            converging = False
            if len(history) >= 5:
                old_cx, old_cy = history[-5]
                old_dist = math.hypot(old_cx - other_cx, old_cy - other_cy)
                converging = old_dist > dist  # distance decreased → converging
            elif len(history) >= 2:
                # Shorter history: use earliest available
                old_cx, old_cy = history[0]
                old_dist = math.hypot(old_cx - other_cx, old_cy - other_cy)
                converging = old_dist > dist

            if not converging and dist >= 0.15:
                # Not converging and not very close → skip
                continue

            # Determine if near a restricted zone
            is_restricted_adj = False
            for zone_id in (entity.zone_ids or []):
                zone = zones_map.get(zone_id)
                if zone is None:
                    continue
                zone_type = getattr(zone, "zone_type", "generic") or "generic"
                if any(
                    kw in zone_type.lower()
                    for kw in ("restricted", "danger", "hazard")
                ) or "禁止" in zone_type:
                    is_restricted_adj = True
                    break

            goal_type = "access_object" if is_restricted_adj else "interact_object"
            confidence = max(0.3, 0.7 - dist * 2.0)  # closer → more confident

            if dist < best_dist:
                best_dist = dist
                evidence = [
                    f"{entity.label}(ID:{entity.entity_id})が"
                    f"{other.label}(ID:{other.entity_id})に接近中 (距離={dist:.3f})"
                ]
                if is_restricted_adj:
                    evidence.append("制限エリア付近での対象物へのアクセス")
                best = self._make_hypothesis(
                    entity=entity,
                    goal_type=goal_type,
                    confidence=confidence,
                    evidence=evidence,
                )

        return best
