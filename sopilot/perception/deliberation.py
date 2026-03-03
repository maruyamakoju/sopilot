"""Deliberative reasoning: System 2 slow-thinking layer.

When triggered by significant events (ANOMALY, high-severity violations),
the DeliberativeReasoner:
1. Generates competing hypotheses about what is happening and why
2. Gathers evidence from available sources (world state, recent events)
3. Ranks hypotheses using a Dempster-Shafer-inspired evidence accumulation
4. Recommends an appropriate operator action in Japanese and English

This is intentionally slower and more thorough than the fast reactive
reasoning in reasoning.py.  It runs at most every `cooldown_seconds`.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from sopilot.perception.types import EntityEvent, EntityEventType, WorldState


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class Evidence:
    source: str        # "world_state" | "event_history" | "goal_recognizer" | "episodic_memory"
    content_ja: str    # NL description in Japanese
    content_en: str    # NL description in English
    confidence: float  # 0–1 how strong this evidence is
    timestamp: float
    supports: bool     # True = supports hypothesis, False = contradicts


@dataclass
class Hypothesis:
    id: str
    claim_ja: str          # NL claim in Japanese
    claim_en: str          # NL claim in English
    belief: float          # 0–1 (Dempster-Shafer mass)
    plausibility: float    # 0–1 (upper probability bound)
    evidence_for: list[Evidence]
    evidence_against: list[Evidence]
    entity_ids: list[int]
    event_type: str        # EntityEventType.name that triggered this
    alternative_ja: str    # alternative explanation in Japanese
    created_at: float


@dataclass
class DeliberationResult:
    id: str
    trigger_event: EntityEvent
    hypotheses: list[Hypothesis]      # sorted by belief desc
    best_hypothesis: Hypothesis | None
    action_ja: str                    # recommended operator action in Japanese
    action_en: str                    # in English
    urgency: str                      # "low" | "medium" | "high" | "critical"
    overall_confidence: float         # 0–1
    duration_ms: float
    created_at: float


# ── Main class ────────────────────────────────────────────────────────────────


class DeliberativeReasoner:
    """System 2 reasoning engine: slow, thorough, evidence-based.

    Usage::

        reasoner = DeliberativeReasoner()
        if reasoner.should_deliberate(event):
            result = reasoner.deliberate(event, world_state)
            # result.best_hypothesis.claim_ja — what the engine concluded
            # result.action_ja — what to do
    """

    # Event types that trigger deliberation
    _TRIGGER_TYPES: frozenset[EntityEventType] = frozenset({
        EntityEventType.ANOMALY,
        EntityEventType.RULE_VIOLATION,
        EntityEventType.PROLONGED_PRESENCE,
        EntityEventType.COLLISION_PREDICTED,
    })

    def __init__(
        self,
        cooldown_seconds: float = 20.0,
        max_history: int = 50,
        max_hypotheses: int = 5,
    ) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._max_history = max_history
        self._max_hypotheses = max_hypotheses
        self._results: deque[DeliberationResult] = deque(maxlen=max_history)
        self._last_deliberation_time: float = 0.0
        self._total_deliberations: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def should_deliberate(self, event: EntityEvent) -> bool:
        """Return True if this event warrants deliberation (type + cooldown check)."""
        if event.event_type not in self._TRIGGER_TYPES:
            return False
        now = time.time()
        if now - self._last_deliberation_time < self._cooldown_seconds:
            return False
        return True

    def deliberate(
        self,
        trigger: EntityEvent,
        world_state: WorldState,
        *,
        goal_hypotheses: list | None = None,   # list of GoalHypothesis (duck-typed)
        recent_episodes: list | None = None,    # list of Episode (duck-typed)
    ) -> DeliberationResult:
        """Run the full deliberation cycle. Returns a DeliberationResult.

        Steps:
        1. Generate 2–5 competing hypotheses based on trigger event type
        2. For each hypothesis, gather evidence from world_state and recent events
        3. If goal_hypotheses provided, use them as additional evidence
        4. If recent_episodes provided, search for similar past episodes
        5. Rank hypotheses by accumulated belief
        6. Generate action recommendation based on best hypothesis
        7. Compute urgency from event severity and hypothesis belief
        """
        t_start = time.perf_counter()

        # Step 1: Generate hypotheses
        hypotheses = self._generate_hypotheses(trigger, world_state)
        # Limit to max_hypotheses
        hypotheses = hypotheses[: self._max_hypotheses]

        # Step 2–4: Gather evidence for each hypothesis
        for hyp in hypotheses:
            self._gather_evidence(
                hyp, trigger, world_state, goal_hypotheses, recent_episodes
            )

        # Step 5: Rank by accumulated belief
        hypotheses = self._rank_hypotheses(hypotheses)

        # Best hypothesis
        best = hypotheses[0] if hypotheses else None

        # Step 6: Generate action recommendation
        urgency = self._compute_urgency(trigger, best)
        action_ja, action_en = self._generate_action(best, urgency, trigger)

        # Step 7: Overall confidence
        overall_conf = self._compute_overall_confidence(hypotheses)

        duration_ms = (time.perf_counter() - t_start) * 1000.0
        now = time.time()

        result = DeliberationResult(
            id=str(uuid.uuid4()),
            trigger_event=trigger,
            hypotheses=hypotheses,
            best_hypothesis=best,
            action_ja=action_ja,
            action_en=action_en,
            urgency=urgency,
            overall_confidence=overall_conf,
            duration_ms=duration_ms,
            created_at=now,
        )

        self._results.appendleft(result)
        self._last_deliberation_time = now
        self._total_deliberations += 1

        return result

    def get_recent_deliberations(self, n: int = 10) -> list[DeliberationResult]:
        """Return the n most-recent DeliberationResults (newest first)."""
        results = list(self._results)
        return results[:n]

    def get_state_dict(self) -> dict[str, Any]:
        """JSON-serializable state for the API."""
        recent = self.get_recent_deliberations(5)
        return {
            "total_deliberations": self._total_deliberations,
            "last_deliberation_time": self._last_deliberation_time,
            "cooldown_seconds": self._cooldown_seconds,
            "recent_deliberations": [
                {
                    "id": r.id,
                    "urgency": r.urgency,
                    "action_ja": r.action_ja,
                    "action_en": r.action_en,
                    "overall_confidence": r.overall_confidence,
                    "duration_ms": r.duration_ms,
                    "created_at": r.created_at,
                    "best_hypothesis": (
                        {
                            "claim_ja": r.best_hypothesis.claim_ja,
                            "claim_en": r.best_hypothesis.claim_en,
                            "belief": r.best_hypothesis.belief,
                        }
                        if r.best_hypothesis
                        else None
                    ),
                }
                for r in recent
            ],
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _generate_hypotheses(
        self, event: EntityEvent, world_state: WorldState
    ) -> list[Hypothesis]:
        """Generate competing hypotheses based on event type.

        Hypothesis templates by event type:
        - ANOMALY:
            H1: "通常と異なる行動パターンが検出された" (behavioral anomaly is genuine)
            H2: "センサーノイズまたは誤検知の可能性がある" (false positive)
            H3: "複数の要因が組み合わさった複合的異常" (compound anomaly)
        - RULE_VIOLATION:
            H1: "意図的なルール違反が発生した" (intentional violation)
            H2: "ルールの認識不足による過失" (unintentional / ignorance)
            H3: "緊急事態による止むを得ない行動" (emergency / unavoidable)
        - PROLONGED_PRESENCE:
            H1: "不審者が区域に滞留している" (suspicious loitering)
            H2: "作業員が通常業務を行っている" (legitimate work)
            H3: "体調不良または緊急事態の可能性" (medical / emergency)
        - COLLISION_PREDICTED:
            H1: "危険な衝突が予測される" (genuine collision risk)
            H2: "軌道が交差するが互いに認識している" (aware, will avoid)
        - Default:
            H1: "セキュリティ上の懸念事象が発生した" (security concern)
            H2: "誤検知または正常な業務活動" (false positive / normal)
        """
        now = time.time()
        etype = event.event_type
        etype_name = etype.name
        entity_ids = [event.entity_id]

        if etype == EntityEventType.ANOMALY:
            hyps = [
                Hypothesis(
                    id=str(uuid.uuid4()),
                    claim_ja="通常と異なる行動パターンが検出された",
                    claim_en="An abnormal behavior pattern was detected",
                    belief=0.5,
                    plausibility=0.8,
                    evidence_for=[],
                    evidence_against=[],
                    entity_ids=list(entity_ids),
                    event_type=etype_name,
                    alternative_ja="センサーのキャリブレーションが必要な可能性がある",
                    created_at=now,
                ),
                Hypothesis(
                    id=str(uuid.uuid4()),
                    claim_ja="センサーノイズまたは誤検知の可能性がある",
                    claim_en="This may be sensor noise or a false positive detection",
                    belief=0.3,
                    plausibility=0.6,
                    evidence_for=[],
                    evidence_against=[],
                    entity_ids=list(entity_ids),
                    event_type=etype_name,
                    alternative_ja="環境変化による一時的なノイズの可能性",
                    created_at=now,
                ),
                Hypothesis(
                    id=str(uuid.uuid4()),
                    claim_ja="複数の要因が組み合わさった複合的異常",
                    claim_en="A compound anomaly arising from multiple interacting factors",
                    belief=0.2,
                    plausibility=0.5,
                    evidence_for=[],
                    evidence_against=[],
                    entity_ids=list(entity_ids),
                    event_type=etype_name,
                    alternative_ja="単一原因ではなく複合的な要因による異常",
                    created_at=now,
                ),
            ]
        elif etype == EntityEventType.RULE_VIOLATION:
            hyps = [
                Hypothesis(
                    id=str(uuid.uuid4()),
                    claim_ja="意図的なルール違反が発生した",
                    claim_en="An intentional rule violation has occurred",
                    belief=0.5,
                    plausibility=0.8,
                    evidence_for=[],
                    evidence_against=[],
                    entity_ids=list(entity_ids),
                    event_type=etype_name,
                    alternative_ja="過失または認識不足の可能性がある",
                    created_at=now,
                ),
                Hypothesis(
                    id=str(uuid.uuid4()),
                    claim_ja="ルールの認識不足による過失",
                    claim_en="An unintentional violation due to lack of rule awareness",
                    belief=0.3,
                    plausibility=0.6,
                    evidence_for=[],
                    evidence_against=[],
                    entity_ids=list(entity_ids),
                    event_type=etype_name,
                    alternative_ja="教育・訓練の不足が原因の可能性",
                    created_at=now,
                ),
                Hypothesis(
                    id=str(uuid.uuid4()),
                    claim_ja="緊急事態による止むを得ない行動",
                    claim_en="An unavoidable action taken in response to an emergency",
                    belief=0.2,
                    plausibility=0.4,
                    evidence_for=[],
                    evidence_against=[],
                    entity_ids=list(entity_ids),
                    event_type=etype_name,
                    alternative_ja="緊急対応のためルールを逸脱した可能性",
                    created_at=now,
                ),
            ]
        elif etype == EntityEventType.PROLONGED_PRESENCE:
            hyps = [
                Hypothesis(
                    id=str(uuid.uuid4()),
                    claim_ja="不審者が区域に滞留している",
                    claim_en="A suspicious individual is loitering in the area",
                    belief=0.5,
                    plausibility=0.7,
                    evidence_for=[],
                    evidence_against=[],
                    entity_ids=list(entity_ids),
                    event_type=etype_name,
                    alternative_ja="不審者が意図的に滞留している可能性",
                    created_at=now,
                ),
                Hypothesis(
                    id=str(uuid.uuid4()),
                    claim_ja="作業員が通常業務を行っている",
                    claim_en="A worker is performing normal operational duties",
                    belief=0.3,
                    plausibility=0.7,
                    evidence_for=[],
                    evidence_against=[],
                    entity_ids=list(entity_ids),
                    event_type=etype_name,
                    alternative_ja="通常の業務活動の一部である可能性",
                    created_at=now,
                ),
                Hypothesis(
                    id=str(uuid.uuid4()),
                    claim_ja="体調不良または緊急事態の可能性",
                    claim_en="A medical emergency or physical incapacitation may have occurred",
                    belief=0.2,
                    plausibility=0.4,
                    evidence_for=[],
                    evidence_against=[],
                    entity_ids=list(entity_ids),
                    event_type=etype_name,
                    alternative_ja="要救助状態の可能性を除外できない",
                    created_at=now,
                ),
            ]
        elif etype == EntityEventType.COLLISION_PREDICTED:
            hyps = [
                Hypothesis(
                    id=str(uuid.uuid4()),
                    claim_ja="危険な衝突が予測される",
                    claim_en="A dangerous collision is predicted to occur",
                    belief=0.5,
                    plausibility=0.8,
                    evidence_for=[],
                    evidence_against=[],
                    entity_ids=list(entity_ids),
                    event_type=etype_name,
                    alternative_ja="即時の介入が必要な衝突リスク",
                    created_at=now,
                ),
                Hypothesis(
                    id=str(uuid.uuid4()),
                    claim_ja="軌道が交差するが互いに認識している",
                    claim_en="Trajectories cross but both parties are mutually aware",
                    belief=0.3,
                    plausibility=0.6,
                    evidence_for=[],
                    evidence_against=[],
                    entity_ids=list(entity_ids),
                    event_type=etype_name,
                    alternative_ja="双方が認識しており自然に回避する可能性",
                    created_at=now,
                ),
            ]
        else:
            # Default hypotheses for any other trigger type
            hyps = [
                Hypothesis(
                    id=str(uuid.uuid4()),
                    claim_ja="セキュリティ上の懸念事象が発生した",
                    claim_en="A security-relevant incident has occurred",
                    belief=0.5,
                    plausibility=0.7,
                    evidence_for=[],
                    evidence_against=[],
                    entity_ids=list(entity_ids),
                    event_type=etype_name,
                    alternative_ja="調査が必要なセキュリティ上の問題",
                    created_at=now,
                ),
                Hypothesis(
                    id=str(uuid.uuid4()),
                    claim_ja="誤検知または正常な業務活動",
                    claim_en="A false positive or routine operational activity",
                    belief=0.3,
                    plausibility=0.6,
                    evidence_for=[],
                    evidence_against=[],
                    entity_ids=list(entity_ids),
                    event_type=etype_name,
                    alternative_ja="正常な業務の一環である可能性",
                    created_at=now,
                ),
            ]

        return hyps

    def _gather_evidence(
        self,
        hypothesis: Hypothesis,
        event: EntityEvent,
        world_state: WorldState,
        goal_hypotheses: list | None,
        recent_episodes: list | None,
    ) -> None:
        """Fill hypothesis.evidence_for and evidence_against in place.

        Evidence sources:
        1. World state: count entities, check entity labels in scene
        2. Recent events in world_state.events: look for corroborating events
        3. Event details: use event.details dict fields (severity, detector, z_score)
        4. Goal hypotheses: if any entity has high-risk goal matching this hypothesis
        5. Recent episodes: if similar events occurred in past episodes
        """
        now = time.time()
        # Determine hypothesis index (0=H1 genuine/suspicious, 1=H2 false positive, 2=H3 compound)
        # We infer the "role" of each hypothesis from its position in the list context.
        # Since this method is called per-hypothesis, we check claim keywords to determine role.
        claim_lower = hypothesis.claim_ja.lower() + hypothesis.claim_en.lower()

        # Classify the hypothesis role from its content
        is_h1 = any(
            kw in claim_lower
            for kw in [
                "abnormal", "intentional", "dangerous", "suspicious",
                "通常と異なる", "意図的", "危険", "不審者", "セキュリティ上の懸念",
            ]
        )
        is_h2 = any(
            kw in claim_lower
            for kw in [
                "false positive", "noise", "unintentional", "aware", "normal operational",
                "誤検知", "センサーノイズ", "過失", "認識している", "通常業務",
            ]
        )
        is_h3 = any(
            kw in claim_lower
            for kw in [
                "compound", "emergency", "medical", "multiple",
                "複合的", "緊急事態", "体調不良",
            ]
        )

        # ── Source 1: World state entity count ────────────────────────────────
        try:
            scene_entities = world_state.scene_graph.entities
            if isinstance(scene_entities, dict):
                entity_count = len(scene_entities)
                entity_list = list(scene_entities.values())
            elif isinstance(scene_entities, list):
                entity_count = len(scene_entities)
                entity_list = scene_entities
            else:
                entity_count = 0
                entity_list = []
        except (AttributeError, TypeError):
            entity_count = 0
            entity_list = []

        if is_h1 and entity_count > 2:
            hypothesis.evidence_for.append(
                Evidence(
                    source="world_state",
                    content_ja=f"シーン内に{entity_count}個のエンティティが存在し、異常の真正性を支持する",
                    content_en=f"Scene contains {entity_count} entities, supporting genuine anomaly",
                    confidence=0.5,
                    timestamp=now,
                    supports=True,
                )
            )

        if is_h2 and entity_count <= 1:
            hypothesis.evidence_for.append(
                Evidence(
                    source="world_state",
                    content_ja="シーン内のエンティティが1つのみ — 誤検知の可能性が高い",
                    content_en="Only 1 entity in scene — increased likelihood of false positive",
                    confidence=0.4,
                    timestamp=now,
                    supports=True,
                )
            )

        # Check entity labels for "person" (common false positive signal for H2)
        if is_h2:
            for ent in entity_list:
                label = ""
                try:
                    label = str(ent.label).lower()
                except AttributeError:
                    try:
                        label = str(ent.get("label", "")).lower()
                    except (AttributeError, TypeError):
                        pass
                if "person" in label:
                    hypothesis.evidence_for.append(
                        Evidence(
                            source="world_state",
                            content_ja="エンティティのラベルが 'person' — 誤検知が多い対象",
                            content_en="Entity label is 'person' — commonly misdetected class",
                            confidence=0.3,
                            timestamp=now,
                            supports=True,
                        )
                    )
                    break

        # ── Source 2: Event confidence ─────────────────────────────────────────
        if is_h1 and event.confidence > 0.7:
            hypothesis.evidence_for.append(
                Evidence(
                    source="event_history",
                    content_ja=f"イベント信頼度 {event.confidence:.2f} — 高信頼度で異常を支持",
                    content_en=f"Event confidence {event.confidence:.2f} — high confidence supports genuine anomaly",
                    confidence=event.confidence,
                    timestamp=now,
                    supports=True,
                )
            )

        if is_h2 and event.confidence < 0.5:
            hypothesis.evidence_for.append(
                Evidence(
                    source="event_history",
                    content_ja=f"イベント信頼度 {event.confidence:.2f} — 低信頼度は誤検知を示唆",
                    content_en=f"Event confidence {event.confidence:.2f} — low confidence suggests false positive",
                    confidence=1.0 - event.confidence,
                    timestamp=now,
                    supports=True,
                )
            )

        # ── Source 3: Event details (z_score, severity) ────────────────────────
        z_score = float(event.details.get("z_score", 0.0)) if event.details else 0.0
        severity = str(event.details.get("severity", "")).lower() if event.details else ""

        if is_h1 and z_score > 2.0:
            hypothesis.evidence_for.append(
                Evidence(
                    source="event_history",
                    content_ja=f"z-score {z_score:.2f} — 統計的に有意な異常偏差",
                    content_en=f"z-score {z_score:.2f} — statistically significant deviation",
                    confidence=min(0.9, 0.5 + (z_score - 2.0) * 0.1),
                    timestamp=now,
                    supports=True,
                )
            )

        if is_h1 and severity == "critical":
            hypothesis.evidence_for.append(
                Evidence(
                    source="event_history",
                    content_ja="イベント重大度が critical — 深刻な異常を示す",
                    content_en="Event severity is critical — indicates a serious anomaly",
                    confidence=0.8,
                    timestamp=now,
                    supports=True,
                )
            )

        # ── Source 4: Goal hypotheses ─────────────────────────────────────────
        if goal_hypotheses is not None:
            for gh in goal_hypotheses:
                try:
                    risk = float(getattr(gh, "risk_score", 0.0))
                except (TypeError, ValueError):
                    risk = 0.0

                if is_h1 and risk > 0.6:
                    hypothesis.evidence_for.append(
                        Evidence(
                            source="goal_recognizer",
                            content_ja=f"目標推定リスクスコア {risk:.2f} — 危険な意図の可能性",
                            content_en=f"Goal hypothesis risk score {risk:.2f} — indicates high-risk intent",
                            confidence=risk,
                            timestamp=now,
                            supports=True,
                        )
                    )

        # ── Source 5: Recent events in world_state (H3 compound) ──────────────
        try:
            ws_events = list(world_state.events) if world_state.events else []
        except (AttributeError, TypeError):
            ws_events = []

        recent_threshold = now - 30.0
        recent_ws_events = [
            e for e in ws_events
            if getattr(e, "timestamp", 0.0) >= recent_threshold
        ]

        if is_h3 and len(recent_ws_events) > 5:
            hypothesis.evidence_for.append(
                Evidence(
                    source="event_history",
                    content_ja=f"直近30秒に{len(recent_ws_events)}件のイベント — 複合異常の可能性",
                    content_en=f"{len(recent_ws_events)} events in the last 30s — potential compound anomaly",
                    confidence=0.6,
                    timestamp=now,
                    supports=True,
                )
            )

        # ── Source 6: Recent episodes (episodic memory) ────────────────────────
        if recent_episodes is not None:
            for ep in recent_episodes:
                ep_severity = ""
                try:
                    ep_severity = str(getattr(ep, "severity", "")).lower()
                except (TypeError, AttributeError):
                    pass

                if is_h3 and ep_severity == "critical":
                    hypothesis.evidence_for.append(
                        Evidence(
                            source="episodic_memory",
                            content_ja="過去のエピソードで同様の重大異常が記録されている",
                            content_en="A similar critical anomaly was recorded in a past episode",
                            confidence=0.5,
                            timestamp=now,
                            supports=True,
                        )
                    )
                    break  # one episode evidence is enough

        # ── Update belief based on accumulated evidence ────────────────────────
        for ev in hypothesis.evidence_for:
            hypothesis.belief += ev.confidence * 0.15

        for ev in hypothesis.evidence_against:
            hypothesis.belief -= ev.confidence * 0.10

        # Clamp before normalization step
        hypothesis.belief = max(0.01, min(0.99, hypothesis.belief))

        # Update plausibility as belief + epistemic uncertainty margin
        hypothesis.plausibility = min(1.0, hypothesis.belief + 0.2)

    def _rank_hypotheses(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Sort hypotheses by belief descending. Normalise beliefs to sum <= 1."""
        if not hypotheses:
            return hypotheses

        total = sum(h.belief for h in hypotheses)
        if total > 0:
            for h in hypotheses:
                h.belief = h.belief / total
                # Clamp normalized belief to valid range
                h.belief = max(0.0, min(1.0, h.belief))

        hypotheses.sort(key=lambda h: h.belief, reverse=True)
        return hypotheses

    def _compute_urgency(
        self, event: EntityEvent, best: Hypothesis | None
    ) -> str:
        """Compute urgency from event type + best hypothesis belief.

        ANOMALY + belief > 0.7 -> "critical"
        RULE_VIOLATION + belief > 0.6 -> "high"
        PROLONGED_PRESENCE -> "medium"
        belief < 0.3 -> "low"
        default -> "medium"
        """
        belief = best.belief if best is not None else 0.0
        etype = event.event_type

        if belief < 0.3:
            return "low"

        if etype == EntityEventType.ANOMALY and belief > 0.55:
            return "critical"

        if etype == EntityEventType.RULE_VIOLATION and belief > 0.5:
            return "high"

        if etype == EntityEventType.PROLONGED_PRESENCE:
            return "medium"

        if etype == EntityEventType.COLLISION_PREDICTED:
            if belief > 0.6:
                return "high"
            return "medium"

        return "medium"

    def _generate_action(
        self, best: Hypothesis | None, urgency: str, event: EntityEvent
    ) -> tuple[str, str]:
        """Return (action_ja, action_en) based on best hypothesis and urgency.

        Templates:
        "critical": "直ちに現場を確認し、必要に応じてセキュリティ要員を派遣してください。"
        "high": "映像を確認し、状況を記録してください。必要に応じて担当者に連絡してください。"
        "medium": "状況を注意深く監視してください。異常が継続する場合は対応を検討してください。"
        "low": "記録として保存します。特別な対応は不要と判断されます。"
        """
        templates_ja = {
            "critical": "直ちに現場を確認し、必要に応じてセキュリティ要員を派遣してください。",
            "high": "映像を確認し、状況を記録してください。必要に応じて担当者に連絡してください。",
            "medium": "状況を注意深く監視してください。異常が継続する場合は対応を検討してください。",
            "low": "記録として保存します。特別な対応は不要と判断されます。",
        }
        templates_en = {
            "critical": "Immediately verify the scene and dispatch security personnel if necessary.",
            "high": "Review footage and log the incident. Contact the relevant supervisor if required.",
            "medium": "Monitor the situation carefully. Consider intervention if the anomaly persists.",
            "low": "Save as a record. No immediate action is judged to be necessary.",
        }

        action_ja = templates_ja.get(urgency, templates_ja["medium"])
        action_en = templates_en.get(urgency, templates_en["medium"])

        # Optionally augment with the best hypothesis conclusion
        if best is not None:
            action_ja = f"【{best.claim_ja}】 " + action_ja
            action_en = f"[{best.claim_en}] " + action_en

        return action_ja, action_en

    def _compute_overall_confidence(self, hypotheses: list[Hypothesis]) -> float:
        """Return belief of best hypothesis (or 0.0 if none)."""
        if not hypotheses:
            return 0.0
        return hypotheses[0].belief
