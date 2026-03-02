"""Hybrid local + VLM reasoning for rule evaluation.

The key innovation: evaluate safety rules locally against the scene graph
first, and only escalate to VLM when the local reasoner is uncertain.
This makes the perception engine a true reasoning system rather than a
VLM wrapper.

Pipeline per rule:
    1. RuleParser   — parse natural-language rule into structured check
    2. LocalReasoner — evaluate against scene graph (fast, free)
    3. VLMEscalator  — call VLM only when confidence < threshold (slow, costly)
    4. HybridReasoner — orchestrate the above

Typical VLM escalation rate: ~5-15% of rules per frame (behavioral rules,
ambiguous scenes).  The rest resolve locally in <1 ms.
"""

from __future__ import annotations

import enum
import logging
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sopilot.perception.types import (
    BBox,
    EntityEventType,
    PerceptionConfig,
    Relation,
    SceneEntity,
    SceneGraph,
    SpatialRelation,
    Violation,
    ViolationSeverity,
    WorldState,
    Zone,
)

logger = logging.getLogger(__name__)

# ── Rule check type ───────────────────────────────────────────────────────


class RuleCheckType(enum.Enum):
    """The type of check a parsed rule requires."""

    PRESENCE = "presence"  # object must be present
    ABSENCE = "absence"  # object must not be present
    SPATIAL = "spatial"  # spatial relationship check
    WEARING = "wearing"  # person wearing / not wearing equipment
    BEHAVIORAL = "behavioral"  # complex behavior (VLM required)
    ZONE_VIOLATION = "zone"  # entity in wrong zone


# ── Parsed rule ───────────────────────────────────────────────────────────


@dataclass
class ParsedRule:
    """A structured representation of a natural-language rule."""

    original: str
    rule_index: int
    target_object: str  # what to look for ("ヘルメット", "helmet", "person")
    check_type: RuleCheckType
    negated: bool  # True if checking for absence ("未着用", "not wearing")
    zone_restriction: str | None  # restrict to a zone ("立入禁止エリア")
    related_object: str | None  # secondary object ("作業者" in "ヘルメット未着用の作業者")
    severity_hint: ViolationSeverity  # inferred from keywords


# ── Japanese / English keyword tables ─────────────────────────────────────

# Wearing / PPE patterns
_WEARING_KEYWORDS_JA = (
    "未着用",
    "未使用",
    "未装着",
    "着用していない",
    "装着していない",
    "使用していない",
    "つけていない",
    "被っていない",
    "なしで",
)
_WEARING_KEYWORDS_EN = (
    "not wearing",
    "without",
    "missing",
    "no helmet",
    "no hardhat",
    "no hard hat",
    "no vest",
    "no gloves",
    "no goggles",
    "no safety",
)

# Zone / restricted-area patterns
_ZONE_KEYWORDS_JA = ("立入禁止", "侵入禁止", "禁止エリア", "制限区域", "危険区域", "進入禁止")
_ZONE_KEYWORDS_EN = (
    "restricted area",
    "restricted zone",
    "no-go zone",
    "forbidden area",
    "prohibited area",
    "exclusion zone",
)

# Spatial patterns (height, distance)
_SPATIAL_KEYWORDS_JA = ("高所作業", "高所", "足場", "梯子", "はしご")
_SPATIAL_KEYWORDS_EN = ("elevated", "height", "ladder", "scaffold", "above")

# Presence / detection patterns
_PRESENCE_KEYWORDS_JA = ("検出", "存在", "確認", "発見")
_PRESENCE_KEYWORDS_EN = ("detect", "find", "identify", "presence", "locate")

# Severity inference keywords
_CRITICAL_KEYWORDS_JA = (
    "危険",
    "重大",
    "緊急",
    "転倒",
    "落下",
    "感電",
    "爆発",
    "火災",
    "致命",
    "立入禁止",
)
_CRITICAL_KEYWORDS_EN = (
    "danger",
    "critical",
    "fatal",
    "fall",
    "electrocution",
    "explosion",
    "fire",
    "restricted",
    "emergency",
)

# Object label normalization (Japanese → detection label)
_OBJECT_ALIASES: dict[str, list[str]] = {
    "person": [
        "作業者",
        "作業員",
        "人",
        "人物",
        "worker",
        "person",
        "people",
        "operator",
    ],
    "helmet": [
        "ヘルメット",
        "保護帽",
        "安全帽",
        "ハードハット",
        "helmet",
        "hard hat",
        "hardhat",
    ],
    "vest": [
        "安全ベスト",
        "反射ベスト",
        "ベスト",
        "vest",
        "safety vest",
        "hi-vis vest",
        "high visibility vest",
    ],
    "safety_harness": [
        "安全帯",
        "安全ベルト",
        "ハーネス",
        "harness",
        "safety belt",
        "safety harness",
    ],
    "gloves": ["手袋", "安全手袋", "グローブ", "gloves", "safety gloves"],
    "goggles": [
        "ゴーグル",
        "保護メガネ",
        "安全メガネ",
        "goggles",
        "safety glasses",
        "protective eyewear",
    ],
    "mask": ["マスク", "防塵マスク", "防毒マスク", "mask", "respirator", "face mask"],
    "forklift": [
        "フォークリフト",
        "forklift",
        "fork lift",
    ],
    "vehicle": ["車両", "車", "トラック", "vehicle", "truck", "car"],
    "fire_extinguisher": [
        "消火器",
        "fire extinguisher",
    ],
}


def _normalize_label(text: str) -> str:
    """Extract the canonical label from rule text."""
    text_lower = text.lower()
    for canonical, aliases in _OBJECT_ALIASES.items():
        for alias in aliases:
            if alias.lower() in text_lower or alias in text:
                return canonical
    return "object"


def _extract_related_object(text: str) -> str | None:
    """Extract a secondary/related object from rule text."""
    text_lower = text.lower()
    # Look for person-related terms when the primary object is equipment
    person_terms = ["作業者", "作業員", "人", "worker", "person", "people", "operator"]
    for term in person_terms:
        if term.lower() in text_lower or term in text:
            return "person"
    return None


def _infer_severity(text: str) -> ViolationSeverity:
    """Infer violation severity from rule text keywords."""
    text_lower = text.lower()
    for kw in _CRITICAL_KEYWORDS_JA:
        if kw in text:
            return ViolationSeverity.CRITICAL
    for kw in _CRITICAL_KEYWORDS_EN:
        if kw in text_lower:
            return ViolationSeverity.CRITICAL
    return ViolationSeverity.WARNING


def _extract_zone_name(text: str) -> str | None:
    """Extract zone restriction name from rule text."""
    # Zone-related suffix characters (エリア, ゾーン, 区域, etc.)
    # Limit to common zone-noun suffixes to avoid greedy matching.
    _ZONE_SUFFIXES = r"(?:エリア|ゾーン|区域|区画|範囲|場所)?"

    # Look for zone names in Japanese
    for kw in _ZONE_KEYWORDS_JA:
        if kw in text:
            match = re.search(rf"({re.escape(kw)}{_ZONE_SUFFIXES})", text)
            if match:
                return match.group(1)
            return kw
    for kw in _ZONE_KEYWORDS_EN:
        if kw in text.lower():
            return kw
    return None


# ── RuleParser ────────────────────────────────────────────────────────────


class RuleParser:
    """Parses natural-language Japanese/English safety rules into structured checks.

    Parsing strategy:
        1. Match Japanese/English keyword patterns to determine check type
        2. Extract the target object label (with alias normalization)
        3. Identify zone restrictions, related objects, negation
        4. Infer severity from keyword hints
        5. Rules that defy pattern matching → BEHAVIORAL (will escalate to VLM)
    """

    def parse(self, rules: list[str]) -> list[ParsedRule]:
        """Parse a list of natural-language rules into ParsedRule objects."""
        return [self._parse_single(rule, i) for i, rule in enumerate(rules)]

    def _parse_single(self, rule: str, index: int) -> ParsedRule:
        """Parse a single rule string."""
        rule_stripped = rule.strip()
        rule_lower = rule_stripped.lower()

        # Determine check type, negation, zone restriction
        check_type = RuleCheckType.BEHAVIORAL  # default: escalate to VLM
        negated = False
        zone_restriction: str | None = None

        # 1) WEARING check (PPE compliance)
        is_wearing = False
        for kw in _WEARING_KEYWORDS_JA:
            if kw in rule_stripped:
                is_wearing = True
                negated = True
                break
        if not is_wearing:
            for kw in _WEARING_KEYWORDS_EN:
                if kw in rule_lower:
                    is_wearing = True
                    negated = True
                    break
        if is_wearing:
            check_type = RuleCheckType.WEARING

        # 2) ZONE_VIOLATION check
        if check_type == RuleCheckType.BEHAVIORAL:
            zone_restriction = _extract_zone_name(rule_stripped)
            if zone_restriction is not None:
                check_type = RuleCheckType.ZONE_VIOLATION

        # 3) SPATIAL check (height, ladder, scaffold)
        if check_type == RuleCheckType.BEHAVIORAL:
            for kw in _SPATIAL_KEYWORDS_JA:
                if kw in rule_stripped:
                    check_type = RuleCheckType.SPATIAL
                    break
            if check_type == RuleCheckType.BEHAVIORAL:
                for kw in _SPATIAL_KEYWORDS_EN:
                    if kw in rule_lower:
                        check_type = RuleCheckType.SPATIAL
                        break

        # 4) PRESENCE / ABSENCE check
        if check_type == RuleCheckType.BEHAVIORAL:
            for kw in _PRESENCE_KEYWORDS_JA:
                if kw in rule_stripped:
                    check_type = RuleCheckType.PRESENCE
                    break
            if check_type == RuleCheckType.BEHAVIORAL:
                for kw in _PRESENCE_KEYWORDS_EN:
                    if kw in rule_lower:
                        check_type = RuleCheckType.PRESENCE
                        break

        # Determine target object
        target_object = _normalize_label(rule_stripped)

        # For wearing checks, if target is person, try to find the equipment
        if check_type == RuleCheckType.WEARING and target_object == "person":
            # The equipment is probably what's being checked
            for canonical, aliases in _OBJECT_ALIASES.items():
                if canonical == "person":
                    continue
                for alias in aliases:
                    if alias.lower() in rule_lower or alias in rule_stripped:
                        target_object = canonical
                        break
                if target_object != "person":
                    break

        # Extract related object
        related_object = _extract_related_object(rule_stripped)
        # If the target is equipment and related is person, that's correct
        # If the target is person, related should be None
        if target_object == "person":
            related_object = None

        # Infer severity
        severity_hint = _infer_severity(rule_stripped)

        # Zone violation is always at least WARNING severity
        if check_type == RuleCheckType.ZONE_VIOLATION:
            severity_hint = max(severity_hint, ViolationSeverity.CRITICAL,
                                key=lambda s: list(ViolationSeverity).index(s))

        logger.debug(
            "Parsed rule %d: type=%s target=%s negated=%s zone=%s severity=%s | %r",
            index,
            check_type.value,
            target_object,
            negated,
            zone_restriction,
            severity_hint.value,
            rule_stripped[:60],
        )

        return ParsedRule(
            original=rule_stripped,
            rule_index=index,
            target_object=target_object,
            check_type=check_type,
            negated=negated,
            zone_restriction=zone_restriction,
            related_object=related_object,
            severity_hint=severity_hint,
        )


# ── LocalReasoner ─────────────────────────────────────────────────────────


class LocalReasoner:
    """Evaluates parsed rules against the scene graph without calling VLM.

    Returns (violations, confidence) where confidence indicates how certain
    the local reasoner is.  When confidence < vlm_escalation_threshold,
    the HybridReasoner will escalate to VLM.
    """

    def evaluate(
        self,
        rule: ParsedRule,
        scene_graph: SceneGraph,
        world_state: WorldState,
    ) -> tuple[list[Violation], float]:
        """Evaluate a single parsed rule against the current scene.

        Returns:
            A tuple of (violations_found, confidence).
            confidence in [0, 1] — lower values trigger VLM escalation.
        """
        if rule.check_type == RuleCheckType.WEARING:
            return self._check_wearing(rule, scene_graph, world_state)
        if rule.check_type == RuleCheckType.PRESENCE:
            return self._check_presence(rule, scene_graph, world_state)
        if rule.check_type == RuleCheckType.ABSENCE:
            return self._check_absence(rule, scene_graph, world_state)
        if rule.check_type == RuleCheckType.ZONE_VIOLATION:
            return self._check_zone_violation(rule, scene_graph, world_state)
        if rule.check_type == RuleCheckType.SPATIAL:
            return self._check_spatial(rule, scene_graph, world_state)
        if rule.check_type == RuleCheckType.BEHAVIORAL:
            # Cannot evaluate behavioral rules locally
            return [], 0.0

        return [], 0.0

    # ── Check: WEARING ────────────────────────────────────────────────

    def _check_wearing(
        self,
        rule: ParsedRule,
        scene_graph: SceneGraph,
        world_state: WorldState,
    ) -> tuple[list[Violation], float]:
        """Check PPE wearing compliance.

        For negated rules (e.g., "ヘルメット未着用"):
            Find persons who do NOT have a WEARING relation to the target equipment.
        For non-negated rules (e.g., "ヘルメット着用を確認"):
            Find persons who DO have a WEARING relation.
        """
        persons = scene_graph.entities_with_label("person")
        if not persons:
            # No persons in scene — no violation possible, but medium confidence
            # (maybe the detector missed them)
            return [], 0.6

        equipment_entities = scene_graph.entities_with_label(rule.target_object)

        # Build a set of person entity_ids that have a WEARING relation
        # to any entity matching the target equipment
        equipment_ids = {e.entity_id for e in equipment_entities}
        persons_wearing: set[int] = set()

        for rel in scene_graph.relations:
            if rel.predicate == SpatialRelation.WEARING:
                if rel.object_id in equipment_ids:
                    persons_wearing.add(rel.subject_id)
                elif rel.subject_id in equipment_ids:
                    # Handle reverse direction just in case
                    persons_wearing.add(rel.object_id)

        violations: list[Violation] = []

        if rule.negated:
            # Find persons NOT wearing the equipment
            persons_without = [
                p for p in persons if p.entity_id not in persons_wearing
            ]
            for person in persons_without:
                violations.append(
                    Violation(
                        rule=rule.original,
                        rule_index=rule.rule_index,
                        description_ja=f"{rule.target_object}未着用の作業者を検出（entity {person.entity_id}）",
                        severity=rule.severity_hint,
                        confidence=person.confidence * 0.9,
                        entity_ids=[person.entity_id],
                        bbox=person.bbox,
                        evidence={
                            "check_type": "wearing",
                            "target": rule.target_object,
                            "negated": True,
                            "equipment_detected": len(equipment_entities),
                            "persons_total": len(persons),
                            "persons_without": len(persons_without),
                        },
                        source="local",
                    )
                )
        else:
            # Non-negated: find persons WEARING the equipment (informational)
            for person in persons:
                if person.entity_id in persons_wearing:
                    violations.append(
                        Violation(
                            rule=rule.original,
                            rule_index=rule.rule_index,
                            description_ja=f"{rule.target_object}着用を確認（entity {person.entity_id}）",
                            severity=ViolationSeverity.INFO,
                            confidence=person.confidence * 0.9,
                            entity_ids=[person.entity_id],
                            bbox=person.bbox,
                            evidence={
                                "check_type": "wearing",
                                "target": rule.target_object,
                                "negated": False,
                            },
                            source="local",
                        )
                    )

        # Confidence depends on whether we have equipment detections
        if equipment_entities:
            # We can see both persons and equipment — high confidence
            confidence = 0.85
        elif persons and rule.negated:
            # Persons visible but no equipment detected at all.
            # Could mean everyone is non-compliant, OR the detector
            # missed the equipment.  Medium-high confidence.
            confidence = 0.65
        else:
            confidence = 0.5

        return violations, confidence

    # ── Check: PRESENCE ───────────────────────────────────────────────

    def _check_presence(
        self,
        rule: ParsedRule,
        scene_graph: SceneGraph,
        world_state: WorldState,
    ) -> tuple[list[Violation], float]:
        """Check for the presence of target objects."""
        matching = scene_graph.entities_with_label(rule.target_object)
        violations: list[Violation] = []

        if matching:
            for entity in matching:
                violations.append(
                    Violation(
                        rule=rule.original,
                        rule_index=rule.rule_index,
                        description_ja=f"{rule.target_object}を検出（entity {entity.entity_id}）",
                        severity=rule.severity_hint,
                        confidence=entity.confidence,
                        entity_ids=[entity.entity_id],
                        bbox=entity.bbox,
                        evidence={
                            "check_type": "presence",
                            "target": rule.target_object,
                            "count": len(matching),
                        },
                        source="local",
                    )
                )
            confidence = 0.9
        else:
            # Nothing found — fairly confident if scene has entities
            confidence = 0.8 if scene_graph.entities else 0.5

        return violations, confidence

    # ── Check: ABSENCE ────────────────────────────────────────────────

    def _check_absence(
        self,
        rule: ParsedRule,
        scene_graph: SceneGraph,
        world_state: WorldState,
    ) -> tuple[list[Violation], float]:
        """Check that a target object is NOT present."""
        matching = scene_graph.entities_with_label(rule.target_object)

        if matching:
            # Object is present when it shouldn't be — violation
            violations = [
                Violation(
                    rule=rule.original,
                    rule_index=rule.rule_index,
                    description_ja=f"{rule.target_object}が検出されました（禁止）",
                    severity=rule.severity_hint,
                    confidence=max(e.confidence for e in matching),
                    entity_ids=[e.entity_id for e in matching],
                    bbox=matching[0].bbox,
                    evidence={
                        "check_type": "absence",
                        "target": rule.target_object,
                        "count": len(matching),
                    },
                    source="local",
                )
            ]
            confidence = 0.85
        else:
            violations = []
            confidence = 0.8 if scene_graph.entities else 0.5

        return violations, confidence

    # ── Check: ZONE_VIOLATION ─────────────────────────────────────────

    def _check_zone_violation(
        self,
        rule: ParsedRule,
        scene_graph: SceneGraph,
        world_state: WorldState,
    ) -> tuple[list[Violation], float]:
        """Check for persons inside restricted zones."""
        violations: list[Violation] = []

        if not world_state.zone_occupancy:
            # No zone data available — low confidence
            return [], 0.3

        # Find matching zone(s) by name
        matching_zones: list[str] = []
        for zone_id, entity_ids in world_state.zone_occupancy.items():
            # Match zone by zone_id or by rule's zone restriction text
            if rule.zone_restriction and (
                rule.zone_restriction.lower() in zone_id.lower()
                or zone_id.lower() in rule.zone_restriction.lower()
            ):
                matching_zones.append(zone_id)

        if not matching_zones:
            # If no zones matched by name, check all restricted-type zones
            # from world state events
            for zone_id in world_state.zone_occupancy:
                # Zone naming convention: if it contains "restricted" or "禁止"
                if any(
                    kw in zone_id.lower()
                    for kw in ("restricted", "禁止", "hazard", "danger", "危険")
                ):
                    matching_zones.append(zone_id)

        for zone_id in matching_zones:
            entity_ids = world_state.zone_occupancy.get(zone_id, [])
            for eid in entity_ids:
                entity = scene_graph.get_entity(eid)
                if entity is None:
                    continue
                # Only flag persons (or the target if specified)
                target_label = rule.target_object if rule.target_object != "object" else "person"
                if target_label.lower() not in entity.label.lower():
                    continue

                violations.append(
                    Violation(
                        rule=rule.original,
                        rule_index=rule.rule_index,
                        description_ja=f"{zone_id}に{entity.label}が侵入（entity {eid}）",
                        severity=rule.severity_hint,
                        confidence=entity.confidence * 0.95,
                        entity_ids=[eid],
                        bbox=entity.bbox,
                        evidence={
                            "check_type": "zone_violation",
                            "zone_id": zone_id,
                            "zone_restriction": rule.zone_restriction,
                            "entity_label": entity.label,
                        },
                        source="local",
                    )
                )

        confidence = 0.9 if matching_zones else 0.4
        return violations, confidence

    # ── Check: SPATIAL ────────────────────────────────────────────────

    def _check_spatial(
        self,
        rule: ParsedRule,
        scene_graph: SceneGraph,
        world_state: WorldState,
    ) -> tuple[list[Violation], float]:
        """Check spatial relationships (e.g., person at height without harness)."""
        violations: list[Violation] = []

        # Spatial checks are inherently harder to resolve locally.
        # We look for ABOVE relations and high-position entities.
        persons = scene_graph.entities_with_label("person")
        if not persons:
            return [], 0.5

        for person in persons:
            # Check if person is in an elevated position
            # Heuristic: person bbox with low y1 (top of frame = high position)
            # This is camera-dependent so confidence is moderate.
            relations = scene_graph.get_relations_for(person.entity_id)

            is_elevated = False
            for rel in relations:
                if rel.predicate == SpatialRelation.ABOVE:
                    is_elevated = True
                    break

            # Also check if person is in the upper portion of the frame
            if person.bbox and person.bbox.y1 < 0.3:
                is_elevated = True

            if is_elevated:
                violations.append(
                    Violation(
                        rule=rule.original,
                        rule_index=rule.rule_index,
                        description_ja=f"高所作業中の作業者を検出（entity {person.entity_id}）",
                        severity=rule.severity_hint,
                        confidence=0.5,  # spatial checks are uncertain
                        entity_ids=[person.entity_id],
                        bbox=person.bbox,
                        evidence={
                            "check_type": "spatial",
                            "is_elevated": is_elevated,
                            "bbox_y1": person.bbox.y1 if person.bbox else None,
                        },
                        source="local",
                    )
                )

        # Spatial checks are inherently low-confidence locally
        confidence = 0.45
        return violations, confidence


# ── VLMEscalator ──────────────────────────────────────────────────────────


class VLMEscalator:
    """Escalates uncertain rule evaluations to VLM.

    Rate-limited to prevent excessive API costs.  Includes scene graph
    context in the VLM prompt so the VLM can focus on what the local
    reasoner couldn't determine.
    """

    def __init__(self, vlm_client: Any, rate_limit: int = 10) -> None:
        """Initialize the VLM escalator.

        Args:
            vlm_client: A VLMClient instance (from sopilot.vigil.vlm).
            rate_limit: Maximum VLM calls per minute.
        """
        self._vlm_client = vlm_client
        self._rate_limit = rate_limit
        self._call_timestamps: list[float] = []
        self._total_calls = 0
        self._total_latency_ms = 0.0

    @property
    def calls_this_minute(self) -> int:
        """Number of VLM calls in the current sliding-window minute."""
        now = time.monotonic()
        self._call_timestamps = [
            t for t in self._call_timestamps if now - t < 60.0
        ]
        return len(self._call_timestamps)

    @property
    def average_latency_ms(self) -> float:
        if self._total_calls == 0:
            return 0.0
        return self._total_latency_ms / self._total_calls

    def can_call(self) -> bool:
        """Check if we're within the rate limit."""
        return self.calls_this_minute < self._rate_limit

    def escalate(
        self,
        frame: np.ndarray,
        rule: ParsedRule,
        context: dict[str, Any],
    ) -> list[Violation]:
        """Call VLM to evaluate a rule that the local reasoner couldn't resolve.

        Args:
            frame: The current video frame as a numpy array (H, W, C).
            rule: The parsed rule to evaluate.
            context: Scene graph context dict for prompt enrichment.

        Returns:
            List of Violation objects with source="vlm".
        """
        if not self.can_call():
            logger.warning(
                "VLM rate limit reached (%d/%d per minute). Skipping escalation for rule %d.",
                self.calls_this_minute,
                self._rate_limit,
                rule.rule_index,
            )
            return []

        if self._vlm_client is None:
            logger.debug("No VLM client available; cannot escalate rule %d.", rule.rule_index)
            return []

        # Write frame to a temporary file for the VLM client
        try:
            import cv2
        except ImportError:
            logger.warning("cv2 not available; cannot escalate to VLM.")
            return []

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
                cv2.imwrite(str(tmp_path), frame)

            # Build an enriched prompt that includes scene context
            enriched_rule = self._build_enriched_rule(rule, context)

            t0 = time.perf_counter()
            self._call_timestamps.append(time.monotonic())
            result = self._vlm_client.analyze_frame(tmp_path, [enriched_rule])
            elapsed_ms = (time.perf_counter() - t0) * 1000

            self._total_calls += 1
            self._total_latency_ms += elapsed_ms

            logger.info(
                "VLM escalation for rule %d completed in %.0f ms: %d violations",
                rule.rule_index,
                elapsed_ms,
                len(result.violations),
            )

            # Convert VLM result to Violation objects
            violations: list[Violation] = []
            for v in result.violations:
                severity_str = v.get("severity", "warning")
                try:
                    severity = ViolationSeverity(severity_str)
                except ValueError:
                    severity = ViolationSeverity.WARNING

                violations.append(
                    Violation(
                        rule=rule.original,
                        rule_index=rule.rule_index,
                        description_ja=v.get("description_ja", rule.original),
                        severity=severity,
                        confidence=float(v.get("confidence", 0.5)),
                        entity_ids=[],
                        bbox=None,
                        evidence={
                            "check_type": rule.check_type.value,
                            "source_detail": "vlm_escalation",
                            "vlm_latency_ms": elapsed_ms,
                            "vlm_raw": v,
                        },
                        source="vlm",
                    )
                )
            return violations

        except Exception:
            logger.exception("VLM escalation failed for rule %d", rule.rule_index)
            return []
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _build_enriched_rule(
        self, rule: ParsedRule, context: dict[str, Any]
    ) -> str:
        """Build an enriched rule description with scene graph context.

        This gives the VLM better focus: instead of analyzing the whole scene,
        it knows what the local reasoner already found and what's uncertain.
        """
        parts = [rule.original]

        # Add scene context
        entity_count = context.get("entity_count", 0)
        person_count = context.get("person_count", 0)
        if entity_count > 0:
            parts.append(
                f"（シーンコンテキスト: 検出済みエンティティ {entity_count}個、"
                f"うち作業者 {person_count}名）"
            )

        entity_labels = context.get("entity_labels", [])
        if entity_labels:
            parts.append(f"検出済みオブジェクト: {', '.join(entity_labels)}")

        reason = context.get("escalation_reason", "")
        if reason:
            parts.append(f"エスカレーション理由: {reason}")

        return " ".join(parts)


# ── HybridReasoner (main entry point) ─────────────────────────────────────


class HybridReasoner:
    """Orchestrates local reasoning with VLM escalation.

    For each rule:
        1. Try LocalReasoner (fast, free)
        2. If confident enough → use local result
        3. If uncertain AND VLM available → escalate to VLM
        4. Otherwise → use local result with lower confidence

    This hybrid approach typically resolves 85-95% of rules locally,
    calling VLM only for genuinely ambiguous situations.
    """

    def __init__(
        self,
        config: PerceptionConfig,
        vlm_client: Any = None,
    ) -> None:
        self._config = config
        self._parser = RuleParser()
        self._local = LocalReasoner()
        self._vlm = VLMEscalator(
            vlm_client=vlm_client,
            rate_limit=config.vlm_max_calls_per_minute,
        ) if vlm_client is not None else None
        self._escalation_threshold = config.vlm_escalation_threshold

        # Cache parsed rules to avoid re-parsing on every frame
        self._cached_rules_key: tuple[str, ...] | None = None
        self._cached_parsed: list[ParsedRule] = []

        # Per-frame tracking flag
        self._vlm_called_this_frame: bool = False
        self._vlm_latency_this_frame: float = 0.0

    def evaluate_rules(
        self,
        rules: list[str],
        scene_graph: SceneGraph,
        world_state: WorldState,
        frame: np.ndarray | None = None,
    ) -> list[Violation]:
        """Evaluate all rules against the current scene.

        Args:
            rules: Natural-language rule strings.
            scene_graph: Current scene graph snapshot.
            world_state: Current world model state.
            frame: Raw video frame (needed for VLM escalation).

        Returns:
            All detected violations, sorted by severity (critical first).
        """
        self._vlm_called_this_frame = False
        self._vlm_latency_this_frame = 0.0

        # Parse rules (cached if unchanged)
        parsed_rules = self._get_parsed_rules(rules)

        # Build scene context for VLM enrichment
        scene_context = self._build_scene_context(scene_graph, world_state)

        all_violations: list[Violation] = []

        for parsed_rule in parsed_rules:
            # Step 1: Try local reasoning
            local_violations, confidence = self._local.evaluate(
                parsed_rule, scene_graph, world_state
            )

            logger.debug(
                "Rule %d [%s] local: %d violations, confidence=%.2f (threshold=%.2f)",
                parsed_rule.rule_index,
                parsed_rule.check_type.value,
                len(local_violations),
                confidence,
                self._escalation_threshold,
            )

            # Step 2: Decide whether to escalate
            if confidence >= self._escalation_threshold:
                # Local result is confident enough
                all_violations.extend(local_violations)
            elif (
                confidence < self._escalation_threshold
                and frame is not None
                and self._vlm is not None
            ):
                # Escalate to VLM
                self._vlm_called_this_frame = True
                vlm_context = {
                    **scene_context,
                    "local_confidence": confidence,
                    "local_violation_count": len(local_violations),
                    "escalation_reason": (
                        f"local confidence {confidence:.2f} < threshold "
                        f"{self._escalation_threshold:.2f}"
                    ),
                }
                t0 = time.perf_counter()
                vlm_violations = self._vlm.escalate(frame, parsed_rule, vlm_context)
                self._vlm_latency_this_frame += (time.perf_counter() - t0) * 1000

                if vlm_violations:
                    all_violations.extend(vlm_violations)
                else:
                    # VLM returned nothing — still use local results
                    # (they might have low-confidence violations)
                    all_violations.extend(local_violations)
            else:
                # No VLM available — use local results with a caveat
                for v in local_violations:
                    v.evidence["low_confidence_note"] = (
                        "Local reasoner confidence below threshold; "
                        "VLM escalation was not available."
                    )
                all_violations.extend(local_violations)

        # Sort by severity (critical first), then by confidence (descending)
        severity_order = {
            ViolationSeverity.CRITICAL: 0,
            ViolationSeverity.WARNING: 1,
            ViolationSeverity.INFO: 2,
        }
        all_violations.sort(
            key=lambda v: (severity_order.get(v.severity, 9), -v.confidence)
        )

        logger.debug(
            "HybridReasoner: %d rules → %d violations (VLM called: %s)",
            len(parsed_rules),
            len(all_violations),
            self._vlm_called_this_frame,
        )

        return all_violations

    def _get_parsed_rules(self, rules: list[str]) -> list[ParsedRule]:
        """Return parsed rules, using cache if rules haven't changed."""
        rules_key = tuple(rules)
        if rules_key != self._cached_rules_key:
            self._cached_parsed = self._parser.parse(rules)
            self._cached_rules_key = rules_key
        return self._cached_parsed

    def _build_scene_context(
        self,
        scene_graph: SceneGraph,
        world_state: WorldState,
    ) -> dict[str, Any]:
        """Build a context dict summarizing the current scene for VLM prompts."""
        entity_labels = list({e.label for e in scene_graph.entities})
        return {
            "entity_count": scene_graph.entity_count,
            "person_count": scene_graph.person_count,
            "entity_labels": entity_labels,
            "relation_count": len(scene_graph.relations),
            "zone_occupancy": {
                k: len(v) for k, v in world_state.zone_occupancy.items()
            },
            "active_events": len(world_state.events),
        }
