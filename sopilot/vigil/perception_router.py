"""Perception Engine API endpoints for VigilPilot.

Exposes the advanced capabilities of the Perception Engine:
- Scene narration (Japanese/English)
- Context memory queries (NL question answering)
- Activity classification for tracked entities
- Prediction alerts (zone entry, collision)
- Causal reasoning explanations
- Full perception state snapshot

All endpoints are mounted under ``/vigil/perception/``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


# ── Request / Response schemas ─────────────────────────────────────────────


class NarrationRequest(BaseModel):
    style: str = Field(
        default="standard",
        pattern="^(brief|standard|detailed)$",
        description="ナレーション詳細度 (brief / standard / detailed)",
    )
    language: str = Field(
        default="ja",
        pattern="^(ja|en)$",
        description="言語 (ja / en)",
    )


class NarrationResponse(BaseModel):
    text_ja: str
    text_en: str
    style: str
    key_facts: list[str]
    entity_mentions: list[int]
    timestamp: float
    frame_number: int


class ContextQueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="質問（日本語または英語）",
        examples=["何人いる？", "制限エリアに何人入った？", "違反は何件？"],
    )


class ContextQueryResponse(BaseModel):
    question: str
    answer: str
    session_id: int


class SessionSummaryResponse(BaseModel):
    start_time: float
    current_time: float
    duration_seconds: float
    total_frames_processed: int
    unique_entities_seen: int
    current_entity_count: int
    total_violations: int
    violations_by_severity: dict[str, int]
    violations_by_rule: dict[str, int]
    notable_events: list[str]


class EntitySummaryResponse(BaseModel):
    entity_id: int
    label: str
    first_seen: float
    last_seen: float
    total_frames: int
    zones_visited: list[str]
    activities: list[str]
    current_activity: str
    current_zone: str | None
    total_distance: float


class ActivityResponse(BaseModel):
    entity_id: int
    activity: str
    confidence: float
    secondary_activity: str | None = None
    secondary_confidence: float = 0.0
    features: dict[str, float]


class PredictionResponse(BaseModel):
    prediction_type: str  # "zone_entry" | "collision"
    entity_id: int
    details: dict[str, Any]
    confidence: float
    estimated_seconds: float


class CausalLinkResponse(BaseModel):
    cause_type: str
    explanation_ja: str
    explanation_en: str
    confidence: float
    time_delta_seconds: float


class PerceptionStateResponse(BaseModel):
    frames_processed: int
    average_processing_ms: float
    active_tracks: int
    total_entities_seen: int
    total_violations: int
    components: dict[str, bool]  # which components are active
    latest_narration_ja: str | None = None
    latest_narration_en: str | None = None


class TimelineEvent(BaseModel):
    event_type: str
    entity_id: int
    timestamp: float
    frame_number: int
    details: dict[str, Any]


class TimelineRequest(BaseModel):
    entity_id: int | None = None
    zone_id: str | None = None
    event_types: list[str] | None = None
    last_n_minutes: float | None = None


# ── Helpers ────────────────────────────────────────────────────────────────


def _get_perception_engine(request: Request):
    """Extract the PerceptionEngine from the VLM client if perception backend is active."""
    pipeline = request.app.state.vigil_pipeline
    vlm = pipeline._vlm

    # Check if it's a PerceptionVLMClient
    if hasattr(vlm, '_engine'):
        return vlm._engine
    raise HTTPException(
        status_code=400,
        detail="Perception engine not active. Set VIGIL_VLM_BACKEND=perception",
    )


def _require_component(engine, attr_name: str, component_name: str):
    """Raise 404 if the requested component is not initialized."""
    component = getattr(engine, attr_name, None)
    if component is None:
        raise HTTPException(
            status_code=404,
            detail=f"{component_name} is not initialized in the perception engine",
        )
    return component


# ── Router ─────────────────────────────────────────────────────────────────


def build_perception_router() -> APIRouter:
    """Build the perception API router."""
    router = APIRouter(prefix="/vigil/perception", tags=["perception"])

    # ── Scene Narration ────────────────────────────────────────────

    @router.post("/narration", response_model=NarrationResponse)
    async def generate_narration(
        body: NarrationRequest, request: Request
    ) -> NarrationResponse:
        """現在のシーンの自然言語ナレーションを生成。

        VLM不要 — 構造化されたワールドステートからテンプレートベースで生成。
        """
        engine = _get_perception_engine(request)
        narrator = _require_component(engine, "_narrator", "SceneNarrator")

        world_state = engine.get_world_state()
        if world_state is None:
            raise HTTPException(status_code=404, detail="No frames processed yet")

        from sopilot.perception.narrator import NarrationStyle

        style_map = {
            "brief": NarrationStyle.BRIEF,
            "standard": NarrationStyle.STANDARD,
            "detailed": NarrationStyle.DETAILED,
        }
        style = style_map.get(body.style, NarrationStyle.STANDARD)

        def _narrate():
            return narrator.narrate(world_state, style=style)

        narration = await run_in_threadpool(_narrate)

        return NarrationResponse(
            text_ja=narration.text_ja,
            text_en=narration.text_en,
            style=narration.style.value,
            key_facts=narration.key_facts,
            entity_mentions=narration.entity_mentions,
            timestamp=narration.timestamp,
            frame_number=narration.frame_number,
        )

    # ── Context Memory Query ───────────────────────────────────────

    @router.post("/query", response_model=ContextQueryResponse)
    async def query_context(
        body: ContextQueryRequest, request: Request
    ) -> ContextQueryResponse:
        """コンテキストメモリに自然言語で質問。

        例: "何人いる？", "制限エリアに何人入った？", "違反は何件？"
        """
        engine = _get_perception_engine(request)
        context_mem = _require_component(engine, "_context_memory", "ContextMemory")

        def _query():
            return context_mem.query(body.question)

        answer = await run_in_threadpool(_query)

        return ContextQueryResponse(
            question=body.question,
            answer=answer,
            session_id=0,  # Perception engine is session-agnostic
        )

    # ── Session Summary ────────────────────────────────────────────

    @router.get("/summary", response_model=SessionSummaryResponse)
    async def get_session_summary(request: Request) -> SessionSummaryResponse:
        """知覚エンジンのセッションサマリー（エンティティ統計、違反集計）。"""
        engine = _get_perception_engine(request)
        context_mem = _require_component(engine, "_context_memory", "ContextMemory")

        def _summarize():
            return context_mem.get_session_summary()

        summary = await run_in_threadpool(_summarize)

        return SessionSummaryResponse(
            start_time=summary.start_time,
            current_time=summary.current_time,
            duration_seconds=summary.duration_seconds,
            total_frames_processed=summary.total_frames_processed,
            unique_entities_seen=summary.unique_entities_seen,
            current_entity_count=summary.current_entity_count,
            total_violations=summary.total_violations,
            violations_by_severity=summary.violations_by_severity,
            violations_by_rule=summary.violations_by_rule,
            notable_events=summary.notable_events,
        )

    # ── Entity Summary ─────────────────────────────────────────────

    @router.get("/entities/{entity_id}", response_model=EntitySummaryResponse)
    async def get_entity_summary(
        entity_id: int, request: Request
    ) -> EntitySummaryResponse:
        """特定エンティティの行動履歴サマリー。"""
        engine = _get_perception_engine(request)
        context_mem = _require_component(engine, "_context_memory", "ContextMemory")

        def _get():
            return context_mem.get_entity_summary(entity_id)

        summary = await run_in_threadpool(_get)

        if summary is None:
            raise HTTPException(
                status_code=404,
                detail=f"Entity {entity_id} not found in context memory",
            )

        return EntitySummaryResponse(
            entity_id=summary.entity_id,
            label=summary.label,
            first_seen=summary.first_seen,
            last_seen=summary.last_seen,
            total_frames=summary.total_frames,
            zones_visited=summary.zones_visited,
            activities=summary.activities,
            current_activity=summary.current_activity,
            current_zone=summary.current_zone,
            total_distance=summary.total_distance,
        )

    # ── Activity Classification ────────────────────────────────────

    @router.get("/activities", response_model=list[ActivityResponse])
    async def get_activities(request: Request) -> list[ActivityResponse]:
        """全アクティブトラックの活動分類結果。"""
        engine = _get_perception_engine(request)
        classifier = _require_component(
            engine, "_activity_classifier", "ActivityClassifier"
        )

        world_state = engine.get_world_state()
        if world_state is None:
            return []

        def _classify():
            return classifier.classify_batch(world_state.active_tracks)

        results = await run_in_threadpool(_classify)

        return [
            ActivityResponse(
                entity_id=eid,
                activity=cls.activity.value,
                confidence=cls.confidence,
                secondary_activity=(
                    cls.secondary_activity.value if cls.secondary_activity else None
                ),
                secondary_confidence=cls.secondary_confidence,
                features={
                    "mean_speed": cls.features.mean_speed,
                    "max_speed": cls.features.max_speed,
                    "speed_variance": cls.features.speed_variance,
                    "direction_change_rate": cls.features.direction_change_rate,
                    "displacement_ratio": cls.features.displacement_ratio,
                    "bounding_area": cls.features.bounding_area,
                    "duration_frames": float(cls.features.duration_frames),
                },
            )
            for eid, cls in results.items()
        ]

    # ── Predictions ────────────────────────────────────────────────

    @router.get("/predictions", response_model=list[PredictionResponse])
    async def get_predictions(request: Request) -> list[PredictionResponse]:
        """軌跡予測アラート（ゾーン侵入予測、衝突予測）。"""
        engine = _get_perception_engine(request)
        predictor = _require_component(
            engine, "_trajectory_predictor", "TrajectoryPredictor"
        )

        world_state = engine.get_world_state()
        if world_state is None:
            return []

        def _predict():
            predictions = []
            zones = engine._zones

            for tid, track in world_state.active_tracks.items():
                # Zone entry predictions
                zone_preds = predictor.predict_zone_entry(track, zones)
                for zp in zone_preds:
                    predictions.append(
                        PredictionResponse(
                            prediction_type="zone_entry",
                            entity_id=tid,
                            details={
                                "zone_id": zp.zone_id,
                                "zone_name": zp.zone_name,
                                "predicted_entry_point": list(zp.predicted_entry_point),
                            },
                            confidence=zp.confidence,
                            estimated_seconds=zp.estimated_seconds,
                        )
                    )

            # Collision predictions (pairwise)
            track_ids = [
                tid for tid, t in world_state.active_tracks.items()
                if t.state.value in ("active", "tentative")
            ]
            for i, tid_a in enumerate(track_ids):
                for tid_b in track_ids[i + 1:]:
                    cp = predictor.predict_collision(
                        world_state.active_tracks[tid_a],
                        world_state.active_tracks[tid_b],
                    )
                    if cp is not None:
                        predictions.append(
                            PredictionResponse(
                                prediction_type="collision",
                                entity_id=cp.entity_a_id,
                                details={
                                    "entity_b_id": cp.entity_b_id,
                                    "collision_point": list(cp.collision_point),
                                },
                                confidence=cp.confidence,
                                estimated_seconds=cp.estimated_seconds,
                            )
                        )

            return predictions

        return await run_in_threadpool(_predict)

    # ── Causal Reasoning ───────────────────────────────────────────

    @router.get("/causality", response_model=list[CausalLinkResponse])
    async def get_causal_links(request: Request) -> list[CausalLinkResponse]:
        """検出された因果関係リンク。"""
        engine = _get_perception_engine(request)
        causal = _require_component(engine, "_causal_reasoner", "CausalReasoner")

        links = list(causal._links) if hasattr(causal, '_links') else []

        return [
            CausalLinkResponse(
                cause_type=link.cause_type,
                explanation_ja=link.explanation_ja,
                explanation_en=link.explanation_en,
                confidence=link.confidence,
                time_delta_seconds=link.time_delta_seconds,
            )
            for link in links
        ]

    # ── Event Timeline ─────────────────────────────────────────────

    @router.post("/timeline", response_model=list[TimelineEvent])
    async def get_timeline(
        body: TimelineRequest, request: Request
    ) -> list[TimelineEvent]:
        """フィルタ付きイベントタイムライン。"""
        engine = _get_perception_engine(request)
        context_mem = _require_component(engine, "_context_memory", "ContextMemory")

        from sopilot.perception.types import EntityEventType

        event_types = None
        if body.event_types:
            event_types = []
            for et_str in body.event_types:
                try:
                    event_types.append(EntityEventType(et_str))
                except ValueError:
                    pass

        def _timeline():
            return context_mem.get_timeline(
                entity_id=body.entity_id,
                zone_id=body.zone_id,
                event_types=event_types,
                last_n_minutes=body.last_n_minutes,
            )

        events = await run_in_threadpool(_timeline)

        return [
            TimelineEvent(
                event_type=e.get("event_type", "unknown"),
                entity_id=e.get("entity_id", -1),
                timestamp=e.get("timestamp", 0.0),
                frame_number=e.get("frame_number", 0),
                details=e.get("details", {}),
            )
            for e in events
        ]

    # ── Perception State ───────────────────────────────────────────

    @router.get("/state", response_model=PerceptionStateResponse)
    async def get_perception_state(request: Request) -> PerceptionStateResponse:
        """知覚エンジンの現在の状態スナップショット。"""
        engine = _get_perception_engine(request)

        world_state = engine.get_world_state()
        active_tracks = len(world_state.active_tracks) if world_state else 0

        # Get narration if available
        narration_ja = None
        narration_en = None
        if engine._narrator and world_state:
            try:
                narration = engine._narrator.narrate(world_state)
                narration_ja = narration.text_ja
                narration_en = narration.text_en
            except Exception:
                pass

        # Count unique entities from context memory
        total_entities = 0
        total_violations = 0
        if engine._context_memory:
            try:
                summary = engine._context_memory.get_session_summary()
                total_entities = summary.unique_entities_seen
                total_violations = summary.total_violations
            except Exception:
                pass

        return PerceptionStateResponse(
            frames_processed=engine.frames_processed,
            average_processing_ms=engine.average_processing_ms,
            active_tracks=active_tracks,
            total_entities_seen=total_entities,
            total_violations=total_violations,
            components={
                "detector": engine._detector is not None,
                "tracker": engine._tracker is not None,
                "scene_builder": engine._scene_builder is not None,
                "world_model": engine._world_model is not None,
                "trajectory_predictor": engine._trajectory_predictor is not None,
                "activity_classifier": engine._activity_classifier is not None,
                "attention_scorer": engine._attention_scorer is not None,
                "causal_reasoner": engine._causal_reasoner is not None,
                "context_memory": engine._context_memory is not None,
                "narrator": engine._narrator is not None,
            },
            latest_narration_ja=narration_ja,
            latest_narration_en=narration_en,
        )

    return router
