"""Natural Language Scene Narration.

Generates human-readable descriptions of the current scene from structured
WorldState data.  No VLM is needed -- narration is built entirely from
template-based generation over the scene graph, zone occupancy, tracks,
events, and violations.

The narrator produces professional Japanese text by default, with parallel
English output.  Three verbosity levels are available:

    BRIEF    -- 1-2 sentences (entity count + violation count)
    STANDARD -- 3-5 sentences (entities, zones, activities, violations)
    DETAILED -- comprehensive paragraph (per-entity descriptions,
                predictions, changes)

Design decisions:
    - Template-based, deterministic generation (no LLM dependency)
    - Japanese counters (1名, 2件, 3箇所) for natural output
    - Severity markers: warning for warnings, critical emoji for critical
    - Frozen SceneNarration dataclass for immutable snapshots
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any

from sopilot.perception.types import (
    EntityEvent,
    EntityEventType,
    SceneEntity,
    SpatialRelation,
    Track,
    TrackState,
    Violation,
    ViolationSeverity,
    WorldState,
    Zone,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

_LABEL_JA: dict[str, str] = {
    "person": "作業員",
    "worker": "作業員",
    "helmet": "ヘルメット",
    "hard_hat": "ヘルメット",
    "safety_vest": "安全ベスト",
    "vest": "安全ベスト",
    "vehicle": "車両",
    "car": "車両",
    "truck": "トラック",
    "forklift": "フォークリフト",
    "crane": "クレーン",
    "tool": "工具",
    "cone": "コーン",
    "barrier": "バリケード",
    "fire_extinguisher": "消火器",
    "ladder": "はしご",
    "scaffold": "足場",
    "gloves": "手袋",
    "goggles": "保護メガネ",
    "mask": "マスク",
    "harness": "安全帯",
}

_LABEL_EN: dict[str, str] = {
    "person": "worker",
    "worker": "worker",
    "helmet": "helmet",
    "hard_hat": "hard hat",
    "safety_vest": "safety vest",
    "vest": "safety vest",
    "vehicle": "vehicle",
    "car": "vehicle",
    "truck": "truck",
    "forklift": "forklift",
    "crane": "crane",
    "tool": "tool",
    "cone": "cone",
    "barrier": "barrier",
    "fire_extinguisher": "fire extinguisher",
    "ladder": "ladder",
    "scaffold": "scaffold",
    "gloves": "gloves",
    "goggles": "goggles",
    "mask": "mask",
    "harness": "harness",
}

_ZONE_TYPE_JA: dict[str, str] = {
    "restricted": "制限エリア",
    "hazard": "危険エリア",
    "work_area": "作業エリア",
    "safe": "安全エリア",
    "generic": "エリア",
}

_ZONE_TYPE_EN: dict[str, str] = {
    "restricted": "restricted area",
    "hazard": "hazardous area",
    "work_area": "work area",
    "safe": "safe area",
    "generic": "area",
}

_EVENT_TYPE_JA: dict[EntityEventType, str] = {
    EntityEventType.ENTERED: "入場しました",
    EntityEventType.EXITED: "退場しました",
    EntityEventType.ZONE_ENTERED: "に入りました",
    EntityEventType.ZONE_EXITED: "から出ました",
    EntityEventType.STATE_CHANGED: "の状態が変化しました",
    EntityEventType.ANOMALY: "異常が検出されました",
    EntityEventType.RULE_VIOLATION: "規則違反が検出されました",
    EntityEventType.PROLONGED_PRESENCE: "長時間滞在しています",
    EntityEventType.ZONE_ENTRY_PREDICTED: "がエリアに到達する予測です",
    EntityEventType.COLLISION_PREDICTED: "衝突が予測されます",
}

_ACTIVITY_JA: dict[str, str] = {
    "stationary": "静止中",
    "walking": "歩行中",
    "running": "走行中",
    "loitering": "徘徊中",
    "erratic": "不規則な動き",
    "approaching": "接近中",
    "departing": "離脱中",
    "unknown": "不明",
}

_ACTIVITY_EN: dict[str, str] = {
    "stationary": "stationary",
    "walking": "walking",
    "running": "running",
    "loitering": "loitering",
    "erratic": "moving erratically",
    "approaching": "approaching",
    "departing": "departing",
    "unknown": "unknown activity",
}

_SEVERITY_MARKER: dict[ViolationSeverity, str] = {
    ViolationSeverity.INFO: "",
    ViolationSeverity.WARNING: "\u26a0 ",
    ViolationSeverity.CRITICAL: "\U0001f6a8 ",
}


# ---------------------------------------------------------------------------
# NarrationStyle enum
# ---------------------------------------------------------------------------


class NarrationStyle(enum.Enum):
    """Verbosity level for scene narration."""

    BRIEF = "brief"        # 1-2 sentences
    STANDARD = "standard"  # 3-5 sentences
    DETAILED = "detailed"  # comprehensive paragraph


# ---------------------------------------------------------------------------
# SceneNarration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneNarration:
    """A generated scene description."""

    text_ja: str                    # Japanese narration
    text_en: str                    # English narration
    style: NarrationStyle
    timestamp: float
    frame_number: int
    key_facts: list[str]            # bullet-point facts used in narration
    entity_mentions: list[int]      # entity IDs mentioned


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _label_ja(label: str) -> str:
    """Map a detection label to a Japanese term."""
    return _LABEL_JA.get(label.lower(), label)


def _label_en(label: str) -> str:
    """Map a detection label to an English term."""
    return _LABEL_EN.get(label.lower(), label)


def _zone_name_ja(zone_id: str, zone_type: str = "generic") -> str:
    """Create a Japanese zone name from zone_id and type."""
    type_ja = _ZONE_TYPE_JA.get(zone_type, "エリア")
    return f"{type_ja}{zone_id}"


def _is_person(label: str) -> bool:
    """Check if a label refers to a person."""
    return label.lower() in ("person", "worker")


def _count_by_label(entities: list[SceneEntity]) -> dict[str, int]:
    """Count entities grouped by label."""
    counts: dict[str, int] = {}
    for e in entities:
        key = e.label.lower()
        counts[key] = counts.get(key, 0) + 1
    return counts


def _velocity_description_ja(track: Track) -> str:
    """Describe a track's movement in Japanese."""
    vx, vy = track.velocity
    import math
    speed = math.hypot(vx, vy)
    if speed < 0.002:
        return "静止しています"
    if speed < 0.01:
        return "ゆっくり移動中です"
    if speed < 0.03:
        return "移動中です"
    return "高速で移動中です"


def _velocity_description_en(track: Track) -> str:
    """Describe a track's movement in English."""
    vx, vy = track.velocity
    import math
    speed = math.hypot(vx, vy)
    if speed < 0.002:
        return "is stationary"
    if speed < 0.01:
        return "is moving slowly"
    if speed < 0.03:
        return "is moving"
    return "is moving quickly"


# ---------------------------------------------------------------------------
# SceneNarrator class
# ---------------------------------------------------------------------------


class SceneNarrator:
    """Template-based natural language scene narrator.

    Generates human-readable descriptions from structured WorldState data.
    No VLM dependency -- purely rule-based template filling.

    Args:
        style: Default narration verbosity.
        language: Primary language (``"ja"`` or ``"en"``).
    """

    def __init__(
        self,
        style: NarrationStyle = NarrationStyle.STANDARD,
        language: str = "ja",
    ) -> None:
        self._default_style = style
        self._language = language

    # -- public API --------------------------------------------------------

    def narrate(
        self,
        world_state: WorldState,
        violations: list[Violation] | None = None,
        style: NarrationStyle | None = None,
    ) -> SceneNarration:
        """Generate a natural language description of the current scene.

        No VLM needed -- purely from structured world state.

        Args:
            world_state: Current world state snapshot.
            violations: Optional list of violations detected this frame.
            style: Narration style override (uses default if *None*).

        Returns:
            A SceneNarration with Japanese and English text.
        """
        active_style = style or self._default_style
        violations = violations or []
        sg = world_state.scene_graph
        entities = sg.entities
        entity_mentions: list[int] = []
        key_facts: list[str] = []

        # --- Gather facts ---
        persons = [e for e in entities if _is_person(e.label)]
        non_persons = [e for e in entities if not _is_person(e.label)]
        person_count = len(persons)
        label_counts = _count_by_label(non_persons)

        # --- BRIEF ---
        lines_ja: list[str] = []
        lines_en: list[str] = []

        # Overview sentence
        if person_count == 0 and not non_persons:
            lines_ja.append("現在、シーンにエンティティは検出されていません。")
            lines_en.append("No entities are currently detected in the scene.")
            key_facts.append("No entities in scene")
        else:
            # Person count
            if person_count > 0:
                lines_ja.append(f"現在、{person_count}名の作業員がいます。")
                lines_en.append(
                    f"Currently, {person_count} worker{'s' if person_count != 1 else ''} "
                    f"{'are' if person_count != 1 else 'is'} present."
                )
                key_facts.append(f"{person_count} workers present")
                entity_mentions.extend(e.entity_id for e in persons)
            elif non_persons:
                # Only non-persons
                total = sum(label_counts.values())
                lines_ja.append(f"現在、{total}個のオブジェクトが検出されています。")
                lines_en.append(
                    f"Currently, {total} object{'s' if total != 1 else ''} "
                    f"{'are' if total != 1 else 'is'} detected."
                )
                key_facts.append(f"{total} objects detected")
                entity_mentions.extend(e.entity_id for e in non_persons)

        # Violation summary (BRIEF level)
        if violations:
            v_count = len(violations)
            critical_count = sum(
                1 for v in violations if v.severity == ViolationSeverity.CRITICAL
            )
            warning_count = sum(
                1 for v in violations if v.severity == ViolationSeverity.WARNING
            )
            lines_ja.append(f"{v_count}件の違反が検出されています。")
            lines_en.append(
                f"{v_count} violation{'s' if v_count != 1 else ''} detected."
            )
            key_facts.append(f"{v_count} violations detected")

        if active_style == NarrationStyle.BRIEF:
            return SceneNarration(
                text_ja="".join(lines_ja),
                text_en=" ".join(lines_en),
                style=active_style,
                timestamp=world_state.timestamp,
                frame_number=world_state.frame_number,
                key_facts=key_facts,
                entity_mentions=sorted(set(entity_mentions)),
            )

        # --- STANDARD: add zone status, equipment, activities, violation details ---

        # Equipment / non-person entities
        if label_counts:
            equip_parts_ja: list[str] = []
            equip_parts_en: list[str] = []
            for lbl, cnt in sorted(label_counts.items()):
                equip_parts_ja.append(f"{_label_ja(lbl)}{cnt}個")
                equip_parts_en.append(f"{cnt} {_label_en(lbl)}{'s' if cnt != 1 else ''}")
                entity_mentions.extend(
                    e.entity_id for e in non_persons if e.label.lower() == lbl
                )
            lines_ja.append("検出機器: " + "、".join(equip_parts_ja) + "。")
            lines_en.append("Detected equipment: " + ", ".join(equip_parts_en) + ".")
            key_facts.append("Equipment: " + ", ".join(equip_parts_en))

        # Zone occupancy
        if world_state.zone_occupancy:
            for zone_id, occupant_ids in sorted(world_state.zone_occupancy.items()):
                n = len(occupant_ids)
                if n == 0:
                    lines_ja.append(f"{zone_id}には誰もいません。")
                    lines_en.append(f"No one is in {zone_id}.")
                else:
                    lines_ja.append(f"{zone_id}に{n}名がいます。")
                    lines_en.append(
                        f"{n} {'person is' if n == 1 else 'people are'} in {zone_id}."
                    )
                    entity_mentions.extend(occupant_ids)
                key_facts.append(f"{zone_id}: {n} occupants")

        # Activities from tracks
        activity_lines_ja: list[str] = []
        activity_lines_en: list[str] = []
        for tid, track in sorted(world_state.active_tracks.items()):
            if not _is_person(track.label):
                continue
            if track.state not in (TrackState.ACTIVE, TrackState.OCCLUDED):
                continue
            desc_ja = _velocity_description_ja(track)
            desc_en = _velocity_description_en(track)
            activity_lines_ja.append(f"作業員(ID: {tid})は{desc_ja}。")
            activity_lines_en.append(f"Worker (ID: {tid}) {desc_en}.")
            entity_mentions.append(tid)

        if activity_lines_ja:
            lines_ja.extend(activity_lines_ja)
            lines_en.extend(activity_lines_en)

        # Violation details (STANDARD level)
        for v in violations:
            marker = _SEVERITY_MARKER.get(v.severity, "")
            lines_ja.append(f"{marker}{v.description_ja}")
            # Build simple English violation line
            ids_str = ", ".join(str(eid) for eid in v.entity_ids) if v.entity_ids else "N/A"
            lines_en.append(
                f"{marker}Violation ({v.severity.value}): {v.rule} "
                f"[entities: {ids_str}]"
            )
            entity_mentions.extend(v.entity_ids)

        if active_style == NarrationStyle.STANDARD:
            return SceneNarration(
                text_ja="".join(lines_ja),
                text_en=" ".join(lines_en),
                style=active_style,
                timestamp=world_state.timestamp,
                frame_number=world_state.frame_number,
                key_facts=key_facts,
                entity_mentions=sorted(set(entity_mentions)),
            )

        # --- DETAILED: per-entity descriptions, predictions, relations ---

        # Per-entity detailed description
        for entity in entities:
            eid = entity.entity_id
            lbl_ja = _label_ja(entity.label)
            lbl_en = _label_en(entity.label)
            zones_str_ja = ""
            zones_str_en = ""
            if entity.zone_ids:
                zones_str_ja = "（" + "、".join(entity.zone_ids) + "内）"
                zones_str_en = " (in " + ", ".join(entity.zone_ids) + ")"

            attrs = entity.attributes
            attr_parts_ja: list[str] = []
            attr_parts_en: list[str] = []
            for k, v in sorted(attrs.items()):
                attr_parts_ja.append(f"{k}={v}")
                attr_parts_en.append(f"{k}={v}")

            attr_str_ja = "、属性: " + ", ".join(attr_parts_ja) if attr_parts_ja else ""
            attr_str_en = ", attributes: " + ", ".join(attr_parts_en) if attr_parts_en else ""

            track = world_state.active_tracks.get(eid)
            movement_ja = ""
            movement_en = ""
            if track:
                movement_ja = "、" + _velocity_description_ja(track)
                movement_en = ", " + _velocity_description_en(track)

            lines_ja.append(
                f"{lbl_ja}(ID: {eid}){zones_str_ja}: "
                f"信頼度{entity.confidence:.0%}{movement_ja}{attr_str_ja}。"
            )
            lines_en.append(
                f"{lbl_en} (ID: {eid}){zones_str_en}: "
                f"confidence {entity.confidence:.0%}{movement_en}{attr_str_en}."
            )
            entity_mentions.append(eid)
            key_facts.append(f"Entity {eid}: {lbl_en}{zones_str_en}")

        # Relations
        relations = world_state.scene_graph.relations
        if relations:
            rel_parts_ja: list[str] = []
            rel_parts_en: list[str] = []
            for rel in relations:
                pred_ja = _relation_ja(rel.predicate)
                pred_en = rel.predicate.value.replace("_", " ")
                rel_parts_ja.append(
                    f"ID {rel.subject_id}は{pred_ja}ID {rel.object_id}"
                )
                rel_parts_en.append(
                    f"ID {rel.subject_id} is {pred_en} ID {rel.object_id}"
                )
            lines_ja.append("関係: " + "、".join(rel_parts_ja) + "。")
            lines_en.append("Relations: " + "; ".join(rel_parts_en) + ".")

        # Events (predictions, anomalies)
        for evt in world_state.events:
            evt_ja = self.narrate_event(evt, world_state)
            lines_ja.append(evt_ja)
            lines_en.append(self._narrate_event_en(evt, world_state))
            entity_mentions.append(evt.entity_id)

        return SceneNarration(
            text_ja="".join(lines_ja),
            text_en=" ".join(lines_en),
            style=active_style,
            timestamp=world_state.timestamp,
            frame_number=world_state.frame_number,
            key_facts=key_facts,
            entity_mentions=sorted(set(entity_mentions)),
        )

    def narrate_event(self, event: EntityEvent, world_state: WorldState) -> str:
        """Describe a single event in natural language (Japanese).

        Args:
            event: The event to describe.
            world_state: Current world state for context.

        Returns:
            A Japanese sentence describing the event.
        """
        eid = event.entity_id
        entity = world_state.scene_graph.get_entity(eid)
        label_ja = _label_ja(entity.label) if entity else "エンティティ"
        details = event.details

        if event.event_type == EntityEventType.ENTERED:
            return f"{label_ja}(ID: {eid})が入場しました。"

        if event.event_type == EntityEventType.EXITED:
            return f"{label_ja}(ID: {eid})が退場しました。"

        if event.event_type == EntityEventType.ZONE_ENTERED:
            zone = details.get("zone_id", "不明エリア")
            return f"{label_ja}(ID: {eid})が{zone}に入りました。"

        if event.event_type == EntityEventType.ZONE_EXITED:
            zone = details.get("zone_id", "不明エリア")
            return f"{label_ja}(ID: {eid})が{zone}から出ました。"

        if event.event_type == EntityEventType.STATE_CHANGED:
            change_type = details.get("change_type", "")
            if change_type == "activity":
                old_act = details.get("old_activity", "不明")
                new_act = details.get("new_activity", "不明")
                old_ja = _ACTIVITY_JA.get(old_act, old_act)
                new_ja = _ACTIVITY_JA.get(new_act, new_act)
                return (
                    f"{label_ja}(ID: {eid})の活動が{old_ja}から{new_ja}に変化しました。"
                )
            return f"{label_ja}(ID: {eid})の状態が変化しました。"

        if event.event_type == EntityEventType.ANOMALY:
            return f"{label_ja}(ID: {eid})に異常が検出されました。"

        if event.event_type == EntityEventType.RULE_VIOLATION:
            rule = details.get("rule", "不明")
            return f"{label_ja}(ID: {eid})が規則「{rule}」に違反しました。"

        if event.event_type == EntityEventType.PROLONGED_PRESENCE:
            zone = details.get("zone_id", "エリア")
            duration = details.get("duration_seconds", 0)
            return (
                f"\u26a0 {label_ja}(ID: {eid})が{zone}に"
                f"{duration:.0f}秒以上滞在しています。"
            )

        if event.event_type == EntityEventType.ZONE_ENTRY_PREDICTED:
            zone = details.get("zone_id", "エリア")
            eta = details.get("eta_seconds", 0)
            return (
                f"\u26a0 {label_ja}(ID: {eid})が約{eta:.0f}秒後に"
                f"{zone}に到達する予測です。"
            )

        if event.event_type == EntityEventType.COLLISION_PREDICTED:
            other_id = details.get("other_entity_id", "不明")
            eta = details.get("eta_seconds", 0)
            return (
                f"\u26a0 {label_ja}(ID: {eid})とID {other_id}の"
                f"衝突が約{eta:.0f}秒後に予測されます。"
            )

        # Fallback
        return f"{label_ja}(ID: {eid})に関するイベントが発生しました。"

    def narrate_entity(self, entity_id: int, world_state: WorldState) -> str:
        """Describe what a specific entity is doing (Japanese).

        Args:
            entity_id: The entity to describe.
            world_state: Current world state for context.

        Returns:
            A Japanese sentence describing the entity's current state.
        """
        entity = world_state.scene_graph.get_entity(entity_id)
        if entity is None:
            return f"ID {entity_id}のエンティティは現在検出されていません。"

        label_ja = _label_ja(entity.label)
        track = world_state.active_tracks.get(entity_id)

        parts: list[str] = []
        parts.append(f"{label_ja}(ID: {entity_id})")

        # Zone info
        if entity.zone_ids:
            parts.append("は" + "、".join(entity.zone_ids) + "内で")
        else:
            parts.append("は")

        # Movement
        if track:
            parts.append(_velocity_description_ja(track))
        else:
            parts.append("追跡データなし")

        return "".join(parts) + "。"

    def generate_alert(self, violation: Violation, world_state: WorldState) -> str:
        """Generate a human-readable alert message for a violation (Japanese).

        Args:
            violation: The violation to describe.
            world_state: Current world state for context.

        Returns:
            A Japanese alert string with severity marker.
        """
        marker = _SEVERITY_MARKER.get(violation.severity, "")
        desc = violation.description_ja

        # Add entity info if available
        if violation.entity_ids:
            entity_descs: list[str] = []
            for eid in violation.entity_ids:
                entity = world_state.scene_graph.get_entity(eid)
                if entity:
                    lbl = _label_ja(entity.label)
                    entity_descs.append(f"{lbl}(ID: {eid})")
                else:
                    entity_descs.append(f"ID: {eid}")
            entities_str = "、".join(entity_descs)
            return f"{marker}{desc}（関連: {entities_str}）"

        return f"{marker}{desc}"

    def summarize_changes(
        self, current: WorldState, previous: WorldState
    ) -> str:
        """Describe what changed between two world states (Japanese).

        Args:
            current: The newer world state.
            previous: The older world state.

        Returns:
            A Japanese summary of changes.
        """
        prev_ids = set(previous.active_tracks.keys())
        curr_ids = set(current.active_tracks.keys())
        entered = curr_ids - prev_ids
        exited = prev_ids - curr_ids

        # Count persons
        entered_persons = sum(
            1 for eid in entered
            if eid in current.active_tracks and _is_person(current.active_tracks[eid].label)
        )
        exited_persons = sum(
            1 for eid in exited
            if eid in previous.active_tracks and _is_person(previous.active_tracks[eid].label)
        )

        # Time span
        dt = current.timestamp - previous.timestamp
        if dt < 60:
            time_str = f"{dt:.0f}秒間"
        else:
            minutes = dt / 60
            time_str = f"{minutes:.0f}分間"

        # Violation count from current events
        violation_events = [
            e for e in current.events
            if e.event_type == EntityEventType.RULE_VIOLATION
        ]

        parts: list[str] = []
        parts.append(f"この{time_str}で: ")

        changes: list[str] = []
        if entered_persons > 0:
            changes.append(f"{entered_persons}名の作業員が入場")
        if exited_persons > 0:
            changes.append(f"{exited_persons}名の作業員が退場")

        entered_objects = len(entered) - entered_persons
        exited_objects = len(exited) - exited_persons
        if entered_objects > 0:
            changes.append(f"{entered_objects}個のオブジェクトが出現")
        if exited_objects > 0:
            changes.append(f"{exited_objects}個のオブジェクトが消失")

        if violation_events:
            changes.append(f"{len(violation_events)}件の違反が検出されました")

        # Person count change
        person_diff = current.person_count - previous.person_count
        if person_diff != 0 and not entered_persons and not exited_persons:
            if person_diff > 0:
                changes.append(f"作業員が{abs(person_diff)}名増加")
            else:
                changes.append(f"作業員が{abs(person_diff)}名減少")

        if not changes:
            return f"この{time_str}で変化はありませんでした。"

        return parts[0] + "、".join(changes) + "。"

    # -- internal helpers --------------------------------------------------

    def _narrate_event_en(self, event: EntityEvent, world_state: WorldState) -> str:
        """Describe a single event in English (used for parallel output)."""
        eid = event.entity_id
        entity = world_state.scene_graph.get_entity(eid)
        label_en = _label_en(entity.label) if entity else "entity"
        details = event.details

        if event.event_type == EntityEventType.ENTERED:
            return f"{label_en} (ID: {eid}) entered the scene."

        if event.event_type == EntityEventType.EXITED:
            return f"{label_en} (ID: {eid}) exited the scene."

        if event.event_type == EntityEventType.ZONE_ENTERED:
            zone = details.get("zone_id", "unknown zone")
            return f"{label_en} (ID: {eid}) entered {zone}."

        if event.event_type == EntityEventType.ZONE_EXITED:
            zone = details.get("zone_id", "unknown zone")
            return f"{label_en} (ID: {eid}) exited {zone}."

        if event.event_type == EntityEventType.STATE_CHANGED:
            change_type = details.get("change_type", "")
            if change_type == "activity":
                old_act = details.get("old_activity", "unknown")
                new_act = details.get("new_activity", "unknown")
                old_en = _ACTIVITY_EN.get(old_act, old_act)
                new_en = _ACTIVITY_EN.get(new_act, new_act)
                return (
                    f"{label_en} (ID: {eid}) changed from {old_en} to {new_en}."
                )
            return f"{label_en} (ID: {eid}) changed state."

        if event.event_type == EntityEventType.ANOMALY:
            return f"Anomaly detected for {label_en} (ID: {eid})."

        if event.event_type == EntityEventType.RULE_VIOLATION:
            rule = details.get("rule", "unknown")
            return f"{label_en} (ID: {eid}) violated rule: {rule}."

        if event.event_type == EntityEventType.PROLONGED_PRESENCE:
            zone = details.get("zone_id", "area")
            duration = details.get("duration_seconds", 0)
            return (
                f"Warning: {label_en} (ID: {eid}) has been in {zone} "
                f"for {duration:.0f}+ seconds."
            )

        if event.event_type == EntityEventType.ZONE_ENTRY_PREDICTED:
            zone = details.get("zone_id", "area")
            eta = details.get("eta_seconds", 0)
            return (
                f"Warning: {label_en} (ID: {eid}) predicted to reach {zone} "
                f"in ~{eta:.0f} seconds."
            )

        if event.event_type == EntityEventType.COLLISION_PREDICTED:
            other_id = details.get("other_entity_id", "unknown")
            eta = details.get("eta_seconds", 0)
            return (
                f"Warning: collision predicted between {label_en} (ID: {eid}) "
                f"and ID {other_id} in ~{eta:.0f} seconds."
            )

        return f"Event occurred for {label_en} (ID: {eid})."


# ---------------------------------------------------------------------------
# Relation label mapping
# ---------------------------------------------------------------------------


def _relation_ja(relation: SpatialRelation) -> str:
    """Map a SpatialRelation to Japanese text."""
    _map: dict[SpatialRelation, str] = {
        SpatialRelation.NEAR: "の近くにいる",
        SpatialRelation.FAR: "から離れている",
        SpatialRelation.ABOVE: "の上にいる",
        SpatialRelation.BELOW: "の下にいる",
        SpatialRelation.LEFT_OF: "の左にいる",
        SpatialRelation.RIGHT_OF: "の右にいる",
        SpatialRelation.INSIDE: "の中にいる",
        SpatialRelation.CONTAINS: "を含んでいる",
        SpatialRelation.OVERLAPS: "と重なっている",
        SpatialRelation.WEARING: "を着用している",
        SpatialRelation.HOLDING: "を持っている",
        SpatialRelation.OPERATING: "を操作している",
    }
    return _map.get(relation, "と関係がある")
