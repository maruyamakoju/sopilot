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

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
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
    features: dict[str, float] = Field(default_factory=dict)


class SOPStepSchema(BaseModel):
    step_id: str
    name: str
    description: str
    order: int
    is_completed: bool
    is_current: bool


class SOPStateResponse(BaseModel):
    enabled: bool
    current_step_id: str | None = None
    percent_complete: float = 0.0
    steps: list[SOPStepSchema] = Field(default_factory=list)
    action_suggestion: str | None = None


class FloorInfo(BaseModel):
    width_m: float
    height_m: float


class MapPoint(BaseModel):
    point_id: str
    camera_id: str
    person_id: str
    label: str
    world_x: float
    world_y: float
    timestamp: float
    confidence: float
    age_seconds: float


class OccupancyGrid(BaseModel):
    cols: int
    rows: int
    data: list[list[float]]


class SpatialMapResponse(BaseModel):
    floor: FloorInfo
    cameras: list[str]
    points: list[MapPoint]
    occupancy_grid: OccupancyGrid
    stats: dict[str, Any]
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


class AnomalyEventResponse(BaseModel):
    timestamp: float
    frame_number: int
    detector: str
    metric: str
    severity: str
    description_ja: str
    z_score: float
    entity_id: int
    vlm_explanation: str | None = None


class AnomalyStateResponse(BaseModel):
    observations: int
    warmup_frames: int
    is_warmed_up: bool
    sigma_threshold: float
    cooldown_seconds: float
    active_cooldowns: int
    detectors: dict[str, Any]
    recent_events: list[AnomalyEventResponse] = []


class ProfileSaveRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, pattern="^[a-zA-Z0-9_-]+$")


class ProfileLoadRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, pattern="^[a-zA-Z0-9_-]+$")


class ProfileListItem(BaseModel):
    name: str
    created_at: str
    observations: int


class AnomalyFeedbackRequest(BaseModel):
    detector: str = Field(..., min_length=1)
    metric: str = Field(..., min_length=1)
    entity_id: int = -1
    confirmed: bool
    note: str = ""


class TuningApplyResponse(BaseModel):
    changes_applied: int
    changes: list[dict]
    pairs_evaluated: int


# ── Phase 10: Self-learning state schemas ─────────────────────────────────────


class AdaptiveLearnerStateSchema(BaseModel):
    total_observed: int = 0
    score_window_size: int = 0
    score_mean: float = 0.0
    score_std: float = 0.0
    drift_count: int = 0
    recalibration_count: int = 0
    last_recalibration: dict | None = None
    ph_state: dict = Field(default_factory=dict)


class PairStatSchema(BaseModel):
    detector: str
    metric: str
    total: int
    confirmed: int
    denied: int
    confirmation_rate: float
    fp_rate: float


class TunerStateSchema(BaseModel):
    total_feedback: int = 0
    confirmed: int = 0
    denied: int = 0
    overall_confirm_rate: float = 0.0
    pairs_tracked: int = 0
    pairs_suppressed: int = 0
    pairs_trusted: int = 0
    last_tuning: float = 0.0
    pair_stats: list[PairStatSchema] = Field(default_factory=list)
    suppressed_pairs: list[PairStatSchema] = Field(default_factory=list)
    trusted_pairs: list[PairStatSchema] = Field(default_factory=list)
    min_samples_for_tuning: int = 10


class AnomalyLearningStateResponse(BaseModel):
    adaptive_learner: AdaptiveLearnerStateSchema
    tuner: TunerStateSchema


# ── Phase 11A: Active Query schemas ───────────────────────────────────────────


class ReviewItemSchema(BaseModel):
    review_id: str
    detector: str
    metric: str
    entity_id: int
    z_score: float
    timestamp: float
    frame_number: int
    description_ja: str
    priority: float
    created_at: float
    reviewed: bool = False
    confirmed: bool | None = None
    note: str = ""
    frame_path: str = ""


class ReviewQueueResponse(BaseModel):
    items: list[ReviewItemSchema]
    pending_count: int


class ReviewSubmitRequest(BaseModel):
    confirmed: bool
    note: str = ""


class ReviewSubmitResponse(BaseModel):
    review_id: str
    confirmed: bool
    detector: str
    metric: str
    note: str
    tuner_updated: bool = False


class ReviewStatsResponse(BaseModel):
    pending_count: int
    reviewed_count: int
    confirmed_count: int
    denied_count: int
    confirm_rate: float
    detector_counts: dict[str, int] = Field(default_factory=dict)
    z_threshold: float
    max_pending: int


# ── Phase 12A: Sigma Tuner schemas ────────────────────────────────────────────


class DetectorSigmaInfo(BaseModel):
    current_sigma: float
    base_sigma: float
    delta: float
    adjusted: bool


class SigmaStateResponse(BaseModel):
    base_sigma: float
    target_fp_rate: float
    total_adjustments: int
    detector_sigmas: dict[str, DetectorSigmaInfo]
    recent_adjustments: list[dict] = Field(default_factory=list)


# ── Phase 14A: Learning Dashboard schemas ─────────────────────────────────────


class LearningHealthStatus(BaseModel):
    status: str = "ok"  # "ok" | "warning" | "degraded"
    issues: list[str] = Field(default_factory=list)
    sigma_clamped_detectors: list[str] = Field(default_factory=list)
    pending_backlog: int = 0
    overall_fp_rate: float = 0.0
    total_feedback: int = 0


class LearningDashboardResponse(BaseModel):
    health: LearningHealthStatus
    sigma_history: list[dict] = Field(default_factory=list)
    confirm_timeline: list[dict] = Field(default_factory=list)
    current_sigmas: dict[str, Any] = Field(default_factory=dict)
    suppressed_pairs: list[dict] = Field(default_factory=list)
    trusted_pairs: list[dict] = Field(default_factory=list)
    queue_stats: dict[str, Any] = Field(default_factory=dict)
    tuner_stats: dict[str, Any] = Field(default_factory=dict)


# ── Phase 12B: Shift Report schemas ───────────────────────────────────────────


class DetectorSummarySchema(BaseModel):
    detector: str
    event_count: int
    fp_count: int = 0
    tp_count: int = 0
    fp_rate: float = 0.0


class ShiftReportResponse(BaseModel):
    session_id: int
    session_name: str
    generated_at: float
    duration_seconds: float | None = None
    status: str
    total_events: int
    events_by_severity: dict[str, int] = Field(default_factory=dict)
    top_rules: list[dict] = Field(default_factory=list)
    top_detectors: list[DetectorSummarySchema] = Field(default_factory=list)
    tuner_feedback_total: int = 0
    tuner_confirm_rate: float = 0.0
    sigma_adjustments_total: int = 0
    detector_sigmas: dict[str, float] = Field(default_factory=dict)
    drift_events: int = 0
    recalibrations: int = 0
    review_pending: int = 0
    review_confirmed: int = 0
    review_denied: int = 0
    recommendations: list[str] = Field(default_factory=list)


class PerceptionStateResponse(BaseModel):
    frames_processed: int
    average_processing_ms: float
    active_tracks: int
    total_entities_seen: int
    total_violations: int
    components: dict[str, bool]  # which components are active
    latest_narration_ja: str | None = None
    latest_narration_en: str | None = None


# ── Phase 5 schemas ───────────────────────────────────────────────────────────


class GoalHypothesisResponse(BaseModel):
    entity_id: int
    entity_label: str
    goal_type: str
    confidence: float
    risk_score: float
    evidence: list[str]
    target_zone: str | None = None
    description_ja: str
    description_en: str


class GoalStateResponse(BaseModel):
    high_risk_count: int
    entity_hypotheses: dict[str, list[GoalHypothesisResponse]]


class EpisodeResponse(BaseModel):
    id: str
    start_time: float
    end_time: float | None
    duration_seconds: float
    entity_count: int
    event_count: int
    severity: str
    summary_ja: str
    summary_en: str
    tags: list[str]


class TemporalPatternResponse(BaseModel):
    pattern_id: str
    description_ja: str
    description_en: str
    occurrences: int
    confidence: float
    typical_hours: list[float]


class EpisodesResponse(BaseModel):
    episodes: list[EpisodeResponse]
    patterns: list[TemporalPatternResponse]
    cross_summary: str


class DeliberationHypothesisResponse(BaseModel):
    claim_ja: str
    claim_en: str
    belief: float
    plausibility: float
    evidence_count_for: int
    evidence_count_against: int
    alternative_ja: str


class DeliberationResultResponse(BaseModel):
    trigger_event_type: str
    urgency: str
    overall_confidence: float
    action_ja: str
    action_en: str
    duration_ms: float
    hypotheses: list[DeliberationHypothesisResponse]


class DeliberationStateResponse(BaseModel):
    total_deliberations: int
    recent: list[DeliberationResultResponse]


class HealthReportResponse(BaseModel):
    frames_observed: int
    detection_confidence_avg: float
    tracking_stability: float
    event_rate_per_minute: float
    fp_rate_estimate: float
    coverage: float
    quality_grade: str
    quality_score: float
    issues: list[str]
    recommendations: list[str]
    auto_adjustments: list[str]


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


# ── Phase 36: Online Learner schemas ──────────────────────────────────────


class ExemplarAddRequest(BaseModel):
    category: str = Field(..., min_length=1, description="カテゴリ名")
    features: list[float] = Field(..., description="768次元特徴ベクトル (DINOv2 CLS)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="operator", pattern="^(operator|auto|feedback)$")


class ExemplarAddResponse(BaseModel):
    exemplar_id: str
    category: str
    exemplar_count: int


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

    # ── Spatial Mapping ────────────────────────────────────────────

    @router.get("/spatial/map", response_model=SpatialMapResponse)
    async def get_spatial_map(request: Request) -> SpatialMapResponse:
        """2Dフロアマップの現状（物体位置、混雑度グリッド）を取得。"""
        engine = _get_perception_engine(request)
        mapper = _require_component(engine, "_spatial_mapper", "SpatialMapper")

        def _get_map():
            return mapper.to_map_data()

        data = await run_in_threadpool(_get_map)
        return SpatialMapResponse(**data)

    # ── SOP Analysis ───────────────────────────────────────────────

    @router.get("/sop/state", response_model=SOPStateResponse)
    async def get_sop_state(request: Request) -> SOPStateResponse:
        """現在のSOP進捗状態を取得。"""
        engine = _get_perception_engine(request)
        
        if engine._sop_engine is None:
            return SOPStateResponse(enabled=False, steps=[], action_suggestion=None)
            
        sop = engine._sop_engine
        
        def _get_state():
            steps_schema = []
            for s in sop._steps:
                is_comp = s.step_id in sop._completed_steps
                is_curr = False
                if sop._current_step_idx < len(sop._steps):
                    is_curr = sop._steps[sop._current_step_idx].step_id == s.step_id
                
                steps_schema.append(SOPStepSchema(
                    step_id=s.step_id,
                    name=s.name,
                    description=s.description,
                    order=s.order,
                    is_completed=is_comp,
                    is_current=is_curr
                ))
            
            percent = (sop._current_step_idx / len(sop._steps)) * 100 if sop._steps else 0
            curr_step_id = sop._steps[sop._current_step_idx].step_id if sop._current_step_idx < len(sop._steps) else "completed"
            
            # Simple suggestion logic
            suggestion = f"次の手順: {sop._steps[sop._current_step_idx].name}" if sop._current_step_idx < len(sop._steps) else "全手順完了"
            
            return SOPStateResponse(
                enabled=True,
                current_step_id=curr_step_id,
                percent_complete=percent,
                steps=steps_schema,
                action_suggestion=suggestion
            )

        return await run_in_threadpool(_get_state)

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

    # ── Anomaly Detection State ───────────────────────────────────

    @router.get("/anomalies", response_model=AnomalyStateResponse)
    async def get_anomaly_state(
        request: Request,
        lookback_seconds: float = 300.0,
    ) -> AnomalyStateResponse:
        """自律型異常検知アンサンブルの状態スナップショット + 直近異常イベント。"""
        engine = _get_perception_engine(request)

        def _get_state():
            return engine.get_anomaly_state()

        state = await run_in_threadpool(_get_state)

        if state is None:
            raise HTTPException(
                status_code=404,
                detail="Anomaly detector not available",
            )

        # Collect recent ANOMALY events from context memory
        recent_events: list[AnomalyEventResponse] = []
        context_mem = getattr(engine, "_context_memory", None)
        if context_mem is not None:
            try:
                from sopilot.perception.types import EntityEventType
                import time as _time

                now = _time.time()
                cutoff = now - lookback_seconds
                events_list = getattr(context_mem, "_events", [])
                for evt in events_list:
                    if (
                        evt.event_type == EntityEventType.ANOMALY
                        and evt.timestamp >= cutoff
                    ):
                        recent_events.append(
                            AnomalyEventResponse(
                                timestamp=evt.timestamp,
                                frame_number=evt.frame_number,
                                detector=evt.details.get("detector", "unknown"),
                                metric=evt.details.get("metric", "unknown"),
                                severity=evt.details.get("severity", "info"),
                                description_ja=evt.details.get(
                                    "description_ja", ""
                                ),
                                z_score=evt.details.get("z_score", 0.0),
                                entity_id=evt.entity_id,
                                vlm_explanation=evt.details.get(
                                    "vlm_explanation"
                                ),
                            )
                        )
            except Exception:
                logger.exception("Failed to collect recent anomaly events")

        return AnomalyStateResponse(
            observations=state.get("observations", 0),
            warmup_frames=state.get("warmup_frames", 100),
            is_warmed_up=state.get("is_warmed_up", False),
            sigma_threshold=state.get("sigma_threshold", 2.0),
            cooldown_seconds=state.get("cooldown_seconds", 60.0),
            active_cooldowns=state.get("active_cooldowns", 0),
            detectors=state.get("detectors", {}),
            recent_events=recent_events,
        )

    # ── Anomaly Profile Management ────────────────────────────────

    @router.post("/anomaly-profile/save")
    async def save_anomaly_profile(
        body: ProfileSaveRequest, request: Request
    ) -> dict:
        """学習済み異常検知ベースラインをプロファイルとして保存。"""
        engine = _get_perception_engine(request)

        def _save():
            return engine.save_anomaly_profile(body.name)

        path = await run_in_threadpool(_save)

        if path is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to save anomaly profile. Anomaly detector may not be available.",
            )

        return {"status": "ok", "name": body.name, "path": str(path)}

    @router.post("/anomaly-profile/load")
    async def load_anomaly_profile(
        body: ProfileLoadRequest, request: Request
    ) -> dict:
        """保存済みプロファイルを読み込んでアンサンブルに適用。"""
        engine = _get_perception_engine(request)

        def _load():
            return engine.load_anomaly_profile(body.name)

        success = await run_in_threadpool(_load)

        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load anomaly profile '{body.name}'. File may not exist.",
            )

        return {"status": "ok", "name": body.name}

    @router.get("/anomaly-profiles", response_model=list[ProfileListItem])
    async def list_anomaly_profiles(request: Request) -> list[ProfileListItem]:
        """保存済み異常検知プロファイル一覧を取得。"""
        from pathlib import Path as _Path

        from sopilot.perception.anomaly_profile import list_profiles

        def _list():
            return list_profiles(_Path("data/anomaly_profiles"))

        profiles = await run_in_threadpool(_list)

        return [
            ProfileListItem(
                name=p["name"],
                created_at=p.get("created_at", ""),
                observations=p.get("observations", 0),
            )
            for p in profiles
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

    # ── Anomaly Feedback & Tuning ──────────────────────────────────

    def _get_tuner(request: Request):
        """Get AnomalyTuner — prefer engine._anomaly_tuner to avoid two-object problem."""
        try:
            eng = _get_perception_engine(request)
            t = getattr(eng, "_anomaly_tuner", None)
            if t is not None:
                return t
        except Exception:
            pass
        if not hasattr(request.app.state, "anomaly_tuner"):
            from pathlib import Path as _Path
            from sopilot.perception.anomaly_tuner import AnomalyTuner
            request.app.state.anomaly_tuner = AnomalyTuner(
                _Path("data/anomaly_feedback.json")
            )
        return request.app.state.anomaly_tuner

    @router.post("/anomaly-feedback")
    async def record_anomaly_feedback(
        body: AnomalyFeedbackRequest, request: Request
    ) -> dict:
        """オペレーターが異常検知の結果を確認/否定 (フィードバック)。

        confirmed=true → 本物の異常（正解）
        confirmed=false → 誤検知（FP）
        """
        tuner = _get_tuner(request)

        def _record():
            tuner.record_feedback(
                detector=body.detector,
                metric=body.metric,
                entity_id=body.entity_id,
                confirmed=body.confirmed,
                note=body.note,
            )

        await run_in_threadpool(_record)
        return {"status": "ok", "detector": body.detector, "metric": body.metric,
                "confirmed": body.confirmed}

    @router.post("/anomaly-tuning/apply", response_model=TuningApplyResponse)
    async def apply_anomaly_tuning(request: Request) -> TuningApplyResponse:
        """蓄積されたフィードバックをアンサンブルに適用してチューニング。"""
        engine = _get_perception_engine(request)
        tuner = _get_tuner(request)

        if engine._world_model is None:
            raise HTTPException(status_code=400, detail="World model not available")

        def _apply():
            try:
                baseline = engine._world_model.get_anomaly_baseline()
                return tuner.apply_tuning(baseline)
            except Exception as exc:
                raise RuntimeError(str(exc)) from exc

        result = await run_in_threadpool(_apply)
        return TuningApplyResponse(**result)

    @router.get("/anomaly-tuning/stats")
    async def get_anomaly_tuning_stats(request: Request) -> dict:
        """異常検知チューニングの統計情報（フィードバック集計）。"""
        tuner = _get_tuner(request)

        def _stats():
            return tuner.get_stats()

        return await run_in_threadpool(_stats)

    @router.delete("/anomaly-tuning/reset")
    async def reset_anomaly_tuning(request: Request) -> dict:
        """フィードバックデータとチューニング履歴をリセット。"""
        tuner = _get_tuner(request)
        tuner.reset()
        return {"status": "ok"}

    @router.get("/anomaly-learning-state", response_model=AnomalyLearningStateResponse)
    async def get_anomaly_learning_state(request: Request) -> AnomalyLearningStateResponse:
        """自己学習状態取得 — AdaptiveLearner + AnomalyTuner を統合 (Phase 10)。"""
        engine = _get_perception_engine(request)
        tuner = _get_tuner(request)

        def _get():
            raw = engine.get_adaptive_learner_state()
            if not raw["tuner"]:
                raw["tuner"] = tuner.get_stats()
            return raw

        data = await run_in_threadpool(_get)
        return AnomalyLearningStateResponse(**data)

    # ── Phase 11A: Active Query Review Queue ───────────────────────

    def _get_review_queue(request: Request):
        """engine._review_queue を返す。なければ 400。"""
        try:
            eng = _get_perception_engine(request)
            rq = getattr(eng, "_review_queue", None)
            if rq is not None:
                return rq
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Review queue not available")

    @router.get("/review-queue", response_model=ReviewQueueResponse)
    async def get_review_queue(
        request: Request, n: int = 20
    ) -> ReviewQueueResponse:
        """保留中のオペレーターレビューリストを取得 (Phase 11A)。

        priority 降順 (z_score 高い順) で最大 n 件を返す。
        """
        rq = _get_review_queue(request)

        def _get():
            items = rq.get_pending(n=n)
            return [ReviewItemSchema(**item.to_dict()) for item in items]

        items = await run_in_threadpool(_get)
        return ReviewQueueResponse(items=items, pending_count=len(items))

    @router.post("/review/{review_id}", response_model=ReviewSubmitResponse)
    async def submit_review(
        review_id: str, body: ReviewSubmitRequest, request: Request
    ) -> ReviewSubmitResponse:
        """オペレーターのラベル付け結果を登録し、AnomalyTuner へ自動ルーティング。

        confirmed=true → 本物の異常 / confirmed=false → 誤検知
        """
        rq = _get_review_queue(request)
        tuner = _get_tuner(request)

        def _submit():
            item = rq.record_review(review_id, confirmed=body.confirmed, note=body.note)
            if item is None:
                return None
            # AnomalyTuner に自動ルーティング
            tuner_updated = False
            try:
                tuner.record_feedback(
                    detector=item.detector,
                    metric=item.metric,
                    entity_id=item.entity_id,
                    confirmed=body.confirmed,
                    note=body.note,
                )
                tuner_updated = True
            except Exception:
                pass
            return item, tuner_updated

        result = await run_in_threadpool(_submit)
        if result is None:
            raise HTTPException(status_code=404, detail=f"review_id '{review_id}' not found")
        item, tuner_updated = result
        return ReviewSubmitResponse(
            review_id=item.review_id,
            confirmed=body.confirmed,
            detector=item.detector,
            metric=item.metric,
            note=body.note,
            tuner_updated=tuner_updated,
        )

    @router.get("/review-stats", response_model=ReviewStatsResponse)
    async def get_review_stats(request: Request) -> ReviewStatsResponse:
        """能動的クエリキューの統計情報。"""
        rq = _get_review_queue(request)

        def _get():
            return rq.get_stats()

        stats = await run_in_threadpool(_get)
        return ReviewStatsResponse(**stats)

    @router.get("/review/{review_id}/frame")
    async def get_review_frame(review_id: str, request: Request):
        """レビューアイテムの証拠フレーム JPEG を返す (Phase 13B)。

        frame_path が空または存在しない場合は 404 を返す。
        """
        rq = _get_review_queue(request)

        def _find():
            with rq._lock:
                all_items = list(rq._pending) + list(rq._reviewed)
            for it in all_items:
                if it.review_id == review_id:
                    return it
            return None

        item = await run_in_threadpool(_find)
        if item is None:
            raise HTTPException(status_code=404, detail=f"review_id '{review_id}' not found")

        from pathlib import Path as _P
        fp = item.frame_path
        if not fp:
            raise HTTPException(status_code=404, detail="No frame captured for this review item")
        p = _P(fp)
        if not p.exists():
            raise HTTPException(status_code=404, detail="Frame file not found on disk")

        return Response(content=p.read_bytes(), media_type="image/jpeg")

    @router.get("/frame-snapshot")
    async def get_frame_snapshot(request: Request):
        """最新フレームのJPEGスナップショットを返す (Phase 11B)。

        フレームがない場合は 204 No Content を返す。
        """
        from fastapi.responses import Response
        engine = _get_perception_engine(request)

        def _get():
            return engine.get_latest_frame_jpeg()

        jpeg = await run_in_threadpool(_get)
        if jpeg is None:
            return Response(status_code=204)
        return Response(content=jpeg, media_type="image/jpeg")

    # ── Phase 12A: Adaptive Sigma Tuner ────────────────────────────

    @router.get("/sigma-state", response_model=SigmaStateResponse)
    async def get_sigma_state(request: Request) -> SigmaStateResponse:
        """detector ごとの現在 sigma_threshold 状態を返す (Phase 12A)。"""
        engine = _get_perception_engine(request)

        def _get() -> dict:
            state = engine.get_sigma_state()
            if state is None:
                base = getattr(engine._config if hasattr(engine, "_config") else object(), "anomaly_sigma_threshold", 2.0)
                state = {
                    "base_sigma": base,
                    "target_fp_rate": 0.30,
                    "total_adjustments": 0,
                    "detector_sigmas": {
                        det: {"current_sigma": base, "base_sigma": base, "delta": 0.0, "adjusted": False}
                        for det in ("behavioral", "spatial", "temporal", "interaction")
                    },
                    "recent_adjustments": [],
                }
            return state

        data = await run_in_threadpool(_get)
        return SigmaStateResponse(**data)

    @router.post("/sigma-reset")
    async def reset_sigma(request: Request) -> dict:
        """全 detector の sigma をベースラインにリセット (Phase 12A)。"""
        engine = _get_perception_engine(request)

        def _reset():
            sigma_tuner = getattr(engine, "_sigma_tuner", None)
            if sigma_tuner is not None:
                sigma_tuner.reset()
                return True
            return False

        ok = await run_in_threadpool(_reset)
        return {"status": "ok" if ok else "no_sigma_tuner"}

    # ── Phase 14A: Learning Dashboard ─────────────────────────────
    SIGMA_CLAMP_THRESHOLD = 5.8  # near SIGMA_MAX(6.0)
    HIGH_FP_THRESHOLD = 0.60
    MIN_FEEDBACK_FOR_FP_ALERT = 10
    BACKLOG_ALERT_THRESHOLD = 20

    @router.get("/learning-dashboard", response_model=LearningDashboardResponse)
    async def get_learning_dashboard(
        sigma_limit: int = 100,
        review_limit: int = 50,
        request: Request = ...,
    ):
        """自己学習モニタリングダッシュボード (Phase 14A)。

        σ調整履歴・確認率タイムライン・ヘルスステータスを一括返却する。
        """
        engine = _get_perception_engine(request)

        def _build():
            # ── sigma history & current state ──
            sigma_history: list[dict] = []
            current_sigmas: dict[str, Any] = {}
            sigma_clamped: list[str] = []
            sigma_tuner = getattr(engine, "_sigma_tuner", None)
            if sigma_tuner is not None:
                sigma_history = sigma_tuner.get_history(limit=sigma_limit)
                state = sigma_tuner.get_state()
                current_sigmas = state.get("detector_sigmas", {})
                for det, info in current_sigmas.items():
                    if info.get("current_sigma", 0) >= SIGMA_CLAMP_THRESHOLD:
                        sigma_clamped.append(det)

            # ── review timeline & queue stats ──
            confirm_timeline: list[dict] = []
            queue_stats: dict[str, Any] = {}
            review_queue = getattr(engine, "_review_queue", None)
            if review_queue is not None:
                confirm_timeline = review_queue.get_review_history(limit=review_limit)
                queue_stats = review_queue.get_stats()

            # ── tuner stats (suppressed/trusted pairs) ──
            suppressed_pairs: list[dict] = []
            trusted_pairs: list[dict] = []
            tuner_stats: dict[str, Any] = {}
            overall_fp_rate = 0.0
            tuner = getattr(engine, "_anomaly_tuner", None)
            if tuner is not None:
                tuner_stats = tuner.get_stats()
                suppressed_pairs = tuner_stats.get("suppressed_pairs", [])
                trusted_pairs = tuner_stats.get("trusted_pairs", [])
                total_fb = tuner_stats.get("total_feedback", 0)
                confirmed_fb = tuner_stats.get("confirmed", 0)
                overall_fp_rate = (
                    1.0 - confirmed_fb / total_fb if total_fb > 0 else 0.0
                )

            # ── health ──
            issues: list[str] = []
            if sigma_clamped:
                issues.append(f"σ上限付近: {', '.join(sigma_clamped)}")
            total_fb = tuner_stats.get("total_feedback", 0)
            if overall_fp_rate > HIGH_FP_THRESHOLD and total_fb >= MIN_FEEDBACK_FOR_FP_ALERT:
                issues.append(f"高FP率: {overall_fp_rate:.0%}")
            pending_backlog = queue_stats.get("pending_count", 0)
            if pending_backlog > BACKLOG_ALERT_THRESHOLD:
                issues.append(f"レビュー積滞: {pending_backlog}件")

            h_status = (
                "ok" if not issues
                else ("warning" if len(issues) == 1 else "degraded")
            )
            health = LearningHealthStatus(
                status=h_status,
                issues=issues,
                sigma_clamped_detectors=sigma_clamped,
                pending_backlog=pending_backlog,
                overall_fp_rate=round(overall_fp_rate, 4),
                total_feedback=total_fb,
            )

            return LearningDashboardResponse(
                health=health,
                sigma_history=sigma_history,
                confirm_timeline=confirm_timeline,
                current_sigmas=current_sigmas,
                suppressed_pairs=suppressed_pairs,
                trusted_pairs=trusted_pairs,
                queue_stats=queue_stats,
                tuner_stats=tuner_stats,
            )

        return await run_in_threadpool(_build)

    # ── Phase 15: Early Warning Risk Score ────────────────────────

    @router.get("/early-warning")
    async def get_early_warning(request: Request) -> dict:
        """予兆検知リスクスコアを返す (Phase 15)。

        detector ごとに σドリフト速度 + FP率 + 異常バースト頻度から
        リスクスコア [0, 1] を計算する。エンジン未起動時はゼロ状態を返す。
        """
        engine = _get_perception_engine(request)

        def _get() -> dict:
            tuner_stats = None
            tuner = getattr(engine, "_anomaly_tuner", None)
            if tuner is not None:
                try:
                    tuner_stats = tuner.get_stats()
                except Exception:
                    pass
            state = engine.get_early_warning_state(tuner_stats)
            if state is None:
                return {
                    "overall_risk": 0.0,
                    "overall_level": "low",
                    "detectors": {},
                    "note": "EarlyWarningEngine not available",
                }
            return state

        return await run_in_threadpool(_get)

    # ── Phase 16: Autonomous Response ─────────────────────────────

    @router.post("/early-warning/respond")
    async def trigger_early_warning_response(request: Request) -> dict:
        """HIGH リスク detector に対して自律対応を実行する (Phase 16)。

        EarlyWarningEngine の現在状態を評価し、リスク閾値超過 detector に対して
        - 日本語説明生成 (なぜリスクが高いか)
        - 推奨アクションリスト生成
        - クールダウン管理 (同一 detector の連続対応を防止)
        を実行し、発火した ResponseAction 一覧を返す。
        """
        engine = _get_perception_engine(request)

        def _respond() -> dict:
            responder = getattr(engine, "_early_warning_responder", None)
            ew = getattr(engine, "_early_warning", None)
            if responder is None or ew is None:
                return {"triggered": [], "note": "EarlyWarningResponder not available"}

            tuner_stats = None
            tuner = getattr(engine, "_anomaly_tuner", None)
            if tuner is not None:
                try:
                    tuner_stats = tuner.get_stats()
                except Exception:
                    pass

            ew_state = ew.get_state(tuner_stats)
            actions = responder.evaluate(
                ew_state,
                sigma_tuner=getattr(engine, "_sigma_tuner", None),
                review_queue=getattr(engine, "_review_queue", None),
            )
            return {
                "triggered": [a.to_dict() for a in actions],
                "total_triggered": len(actions),
            }

        return await run_in_threadpool(_respond)

    @router.get("/early-warning/response-history")
    async def get_early_warning_response_history(
        request: Request, n: int = 20
    ) -> dict:
        """自律対応の実行履歴を返す (Phase 16)。"""
        engine = _get_perception_engine(request)

        def _get() -> dict:
            state = engine.get_early_warning_responder_state()
            if state is None:
                return {"total_responses": 0, "recent_responses": []}
            return state

        return await run_in_threadpool(_get)

    # ── Phase 18: Perception Health Score ─────────────────────────

    @router.get("/health-score")
    async def get_perception_health_score(request: Request) -> dict:
        """Perception Engine 総合ヘルススコアを返す (Phase 18).

        Returns:
            score (0-100), grade (A-F), factors (per-signal breakdown),
            computed_at (Unix timestamp).
        """
        engine = _get_perception_engine(request)

        def _get() -> dict:
            return engine.get_health_score()

        return await run_in_threadpool(_get)

    # ── Phase 19: Health Score History ────────────────────────────

    @router.get("/health-score/history")
    async def get_health_score_history(
        request: Request, days: float = 7.0
    ) -> dict:
        """ヘルスヒストリと 30 日トレンドを返す (Phase 19).

        Args:
            days: 過去何日分を返すか (デフォルト 7 日、最大 30 日).

        Returns:
            history (list), trend (dict), sparkline (list).
        """
        engine = _get_perception_engine(request)
        days = max(0.1, min(float(days), 30.0))

        def _get() -> dict:
            return engine.get_health_history(days=days)

        return await run_in_threadpool(_get)

    # ── Phase 20: Daily Intelligence Report ───────────────────────

    @router.get("/daily-report")
    async def get_daily_report(request: Request) -> dict:
        """過去 24 時間の知覚パイプライン活動サマリーを返す (Phase 20).

        Returns:
            summary, anomalies, early_warning, responses,
            recommendations, metadata セクションからなる構造化レポート。
        """
        engine = _get_perception_engine(request)

        def _get() -> dict:
            return engine.get_daily_report()

        return await run_in_threadpool(_get)

    # ── Phase 12B: Shift Report ────────────────────────────────────

    @router.get("/shift-report/{session_id}", response_model=ShiftReportResponse)
    async def get_shift_report(
        session_id: int, request: Request
    ) -> ShiftReportResponse:
        """セッションのシフトレポートを生成して返す (Phase 12B)。

        vigil_events + 学習状態 (AnomalyTuner + SigmaTuner + ReviewQueue) を統合し
        構造化レポートと推奨アクションを返す。
        """
        from sopilot.vigil.shift_report import ShiftReportGenerator

        vigil_repo = getattr(request.app.state, "vigil_repo", None)
        if vigil_repo is None:
            raise HTTPException(status_code=503, detail="Vigil repository not available")

        engine = None
        try:
            engine = _get_perception_engine(request)
        except Exception:
            pass

        def _generate():
            gen = ShiftReportGenerator()
            return gen.generate(session_id, vigil_repo, engine=engine)

        try:
            report = await run_in_threadpool(_generate)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

        return ShiftReportResponse(**report.to_dict())

    # ── Phase 5: Goal Recognition ──────────────────────────────────

    @router.get("/goals", response_model=GoalStateResponse)
    async def get_goal_state(request: Request) -> GoalStateResponse:
        """エンティティの意図推定（GoalRecognizer）状態を取得。

        エンティティごとの目標仮説を返す。high_risk_count は
        risk_threshold 以上のリスクスコアを持つ仮説の数。
        """
        engine = _get_perception_engine(request)

        def _get():
            state = engine.get_goal_state()
            if state is None:
                return {"high_risk_count": 0, "entity_hypotheses": {}}
            return state

        raw = await run_in_threadpool(_get)

        entity_hyps: dict[str, list[GoalHypothesisResponse]] = {}
        for eid_str, hyps in raw.get("entity_hypotheses", {}).items():
            entity_hyps[eid_str] = [
                GoalHypothesisResponse(
                    entity_id=h.get("entity_id", -1),
                    entity_label=h.get("entity_label", "unknown"),
                    goal_type=h.get("goal_type", ""),
                    confidence=h.get("confidence", 0.0),
                    risk_score=h.get("risk_score", 0.0),
                    evidence=h.get("evidence", []),
                    target_zone=h.get("target_zone"),
                    description_ja=h.get("description_ja", ""),
                    description_en=h.get("description_en", ""),
                )
                for h in hyps
            ]

        return GoalStateResponse(
            high_risk_count=raw.get("high_risk_count", 0),
            entity_hypotheses=entity_hyps,
        )

    # ── Phase 5: Episodic Memory ───────────────────────────────────

    @router.get("/episodes", response_model=EpisodesResponse)
    async def get_episodes(request: Request, n: int = 10) -> EpisodesResponse:
        """最近のエピソード一覧と時間帯パターンを取得。

        エピソード: イベントストリームを意味のある「シーン」に
        セグメント化した単位。quiet period (120秒以上イベントなし) で区切られる。
        """
        engine = _get_perception_engine(request)

        def _get():
            episodes = engine.get_episodes(n)
            patterns: list = []
            cross: str = ""
            if engine._episodic_memory is not None:
                try:
                    patterns = [p.__dict__ for p in engine._episodic_memory.get_temporal_patterns()]
                    cross = engine._episodic_memory.get_cross_episode_summary(24)
                except Exception:
                    pass
            return episodes, patterns, cross

        raw_eps, raw_pats, cross_summary = await run_in_threadpool(_get)

        episodes = [
            EpisodeResponse(
                id=e.get("id", ""),
                start_time=e.get("start_time", 0.0),
                end_time=e.get("end_time"),
                duration_seconds=e.get("duration_seconds", 0.0),
                entity_count=len(e.get("entity_ids", [])),
                event_count=e.get("event_count", 0),
                severity=e.get("severity", "normal"),
                summary_ja=e.get("summary_ja", ""),
                summary_en=e.get("summary_en", ""),
                tags=e.get("tags", []),
            )
            for e in raw_eps
        ]
        patterns = [
            TemporalPatternResponse(
                pattern_id=p.get("pattern_id", ""),
                description_ja=p.get("description_ja", ""),
                description_en=p.get("description_en", ""),
                occurrences=p.get("occurrences", 0),
                confidence=p.get("confidence", 0.0),
                typical_hours=p.get("typical_hours", []),
            )
            for p in raw_pats
        ]

        return EpisodesResponse(
            episodes=episodes,
            patterns=patterns,
            cross_summary=cross_summary,
        )

    # ── Phase 5: Deliberation ──────────────────────────────────────

    @router.get("/deliberation", response_model=DeliberationStateResponse)
    async def get_deliberation_state(
        request: Request, n: int = 5
    ) -> DeliberationStateResponse:
        """System 2 熟慮推論の最近の結果を取得。

        各結果には:
        - 競合する仮説リスト (信頼度順)
        - 最善仮説の主張 (日本語・英語)
        - 推奨オペレーターアクション
        - 緊急度 (low / medium / high / critical)
        """
        engine = _get_perception_engine(request)

        def _get():
            state = engine.get_deliberation_state()
            if state is None:
                return {"total_deliberations": 0, "recent": []}
            return state

        raw = await run_in_threadpool(_get)

        recent = []
        for r in raw.get("recent", [])[:n]:
            hyps = [
                DeliberationHypothesisResponse(
                    claim_ja=h.get("claim_ja", ""),
                    claim_en=h.get("claim_en", ""),
                    belief=h.get("belief", 0.0),
                    plausibility=h.get("plausibility", 0.0),
                    evidence_count_for=len(h.get("evidence_for", [])),
                    evidence_count_against=len(h.get("evidence_against", [])),
                    alternative_ja=h.get("alternative_ja", ""),
                )
                for h in r.get("hypotheses", [])
            ]
            recent.append(
                DeliberationResultResponse(
                    trigger_event_type=r.get("trigger_event_type", ""),
                    urgency=r.get("urgency", "low"),
                    overall_confidence=r.get("overall_confidence", 0.0),
                    action_ja=r.get("action_ja", ""),
                    action_en=r.get("action_en", ""),
                    duration_ms=r.get("duration_ms", 0.0),
                    hypotheses=hyps,
                )
            )

        return DeliberationStateResponse(
            total_deliberations=raw.get("total_deliberations", 0),
            recent=recent,
        )

    # ── Phase 5: Metacognition ─────────────────────────────────────

    @router.get("/health", response_model=HealthReportResponse)
    async def get_perception_health(request: Request) -> HealthReportResponse:
        """知覚エンジンの自己診断レポート。

        品質グレード (A-F)、信頼度トレンド、推定誤検知率、
        自動キャリブレーション推奨事項を返す。
        """
        engine = _get_perception_engine(request)

        def _get():
            state = engine.get_metacognition_state()
            if state is None:
                return None
            return state

        raw = await run_in_threadpool(_get)

        if raw is None:
            raise HTTPException(
                status_code=404,
                detail="MetacognitiveMonitor not initialized in perception engine",
            )

        return HealthReportResponse(
            frames_observed=raw.get("frames_observed", 0),
            detection_confidence_avg=raw.get("detection_confidence_avg", 0.0),
            tracking_stability=raw.get("tracking_stability", 0.0),
            event_rate_per_minute=raw.get("event_rate_per_minute", 0.0),
            fp_rate_estimate=raw.get("fp_rate_estimate", 0.0),
            coverage=raw.get("coverage", 0.0),
            quality_grade=raw.get("quality_grade", "?"),
            quality_score=raw.get("quality_score", 0.0),
            issues=raw.get("issues", []),
            recommendations=raw.get("recommendations", []),
            auto_adjustments=raw.get("auto_adjustments", []),
        )

    @router.post("/metacognition/record-feedback")
    async def record_metacognition_feedback(
        confirmed: bool, request: Request
    ) -> dict:
        """オペレーターフィードバックをメタ認知モニターに記録。

        confirmed=true: 検出は正しかった
        confirmed=false: 誤検知だった
        """
        engine = _get_perception_engine(request)
        monitor = engine._metacognitive_monitor
        if monitor is None:
            raise HTTPException(
                status_code=404,
                detail="MetacognitiveMonitor not initialized",
            )

        def _record():
            monitor.record_feedback(confirmed)

        await run_in_threadpool(_record)
        return {"status": "ok", "confirmed": confirmed}

    # ── Phase 6: SSE Real-time stream ──────────────────────────────────────

    @router.get("/events/stream")
    async def stream_perception_events(request: Request, session_id: str = "default"):
        """SSE stream of real-time perception events for a session.

        Browser usage:
            const es = new EventSource('/vigil/perception/events/stream?session_id=123');
            es.onmessage = (e) => { const evt = JSON.parse(e.data); ... };
        """
        try:
            from sopilot.perception import sse_events
        except ImportError:
            raise HTTPException(status_code=503, detail="SSE event module not available")

        async def _generator():
            async for chunk in sse_events.event_generator(session_id):
                yield chunk

        return StreamingResponse(
            _generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @router.get("/events/stream/status")
    async def get_stream_status(request: Request, session_id: str = "default"):
        """Return SSE queue stats for a session."""
        try:
            from sopilot.perception import sse_events
            eq = sse_events.get(session_id)
            if eq is None:
                return {"session_id": session_id, "registered": False}
            return {"registered": True, **eq.get_stats()}
        except ImportError:
            return {"registered": False, "error": "SSE module not available"}

    # ── Phase 6: NL Task Specification ─────────────────────────────────────

    class NLTaskRequest(BaseModel):
        text: str = Field(..., description="Natural language monitoring rule in Japanese or English")

    class NLTaskResponse(BaseModel):
        id: str
        raw_text: str
        task_type: str
        parameters: dict
        active: bool
        description_ja: str
        description_en: str
        severity: str

    @router.post("/tasks", response_model=NLTaskResponse)
    async def add_nl_task(body: NLTaskRequest, request: Request):
        """Parse a natural-language monitoring rule and add it as an active task."""
        engine = _get_perception_engine(request)
        if engine._nl_task_manager is None:
            raise HTTPException(status_code=503, detail="NLTaskManager not initialized")

        def _add():
            return engine._nl_task_manager.parse_and_add(body.text)

        task = await run_in_threadpool(_add)
        return NLTaskResponse(**{k: v for k, v in task.to_dict().items() if k != "created_at"})

    @router.get("/tasks")
    async def list_nl_tasks(request: Request):
        """List all active NL monitoring tasks."""
        engine = _get_perception_engine(request)
        if engine._nl_task_manager is None:
            return {"tasks": [], "total": 0}

        def _list():
            return engine._nl_task_manager.get_tasks()

        tasks = await run_in_threadpool(_list)
        return {"tasks": [t.to_dict() for t in tasks], "total": len(tasks)}

    @router.delete("/tasks/{task_id}")
    async def remove_nl_task(task_id: str, request: Request):
        """Remove a NL monitoring task by ID."""
        engine = _get_perception_engine(request)
        if engine._nl_task_manager is None:
            raise HTTPException(status_code=503, detail="NLTaskManager not initialized")

        def _remove():
            return engine._nl_task_manager.remove_task(task_id)

        removed = await run_in_threadpool(_remove)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return {"status": "deleted", "task_id": task_id}

    # ── Phase 6: Cross-Camera Re-ID ─────────────────────────────────────────

    @router.get("/reid/tracks")
    async def get_reid_tracks(request: Request, cross_camera_only: bool = False):
        """Return global Re-ID tracks. cross_camera_only=true filters to multi-session tracks."""
        engine = _get_perception_engine(request)
        if engine._cross_camera_tracker is None:
            return {"tracks": [], "total": 0, "cross_camera_count": 0}

        def _get():
            if cross_camera_only:
                tracks = engine._cross_camera_tracker.get_cross_camera_global_tracks()
            else:
                tracks = engine._cross_camera_tracker.get_global_tracks()
            state = engine._cross_camera_tracker.get_state_dict()
            return tracks, state

        tracks, state = await run_in_threadpool(_get)
        return {
            "tracks": [t.to_dict() for t in tracks],
            "total": state.get("total_global_tracks", 0),
            "cross_camera_count": state.get("cross_camera_tracks", 0),
        }

    # ── Phase 6: Long-term Memory ───────────────────────────────────────────

    @router.get("/long-term-memory")
    async def get_ltm_state(request: Request):
        """Return long-term memory state dict and Japanese summary."""
        engine = _get_perception_engine(request)
        if engine._long_term_memory is None:
            return {"total_facts": 0, "by_type": {}, "summary_ja": "長期記憶は無効です"}

        def _get():
            state = engine._long_term_memory.get_state_dict()
            summary = engine._long_term_memory.generate_summary_ja()
            return state, summary

        state, summary = await run_in_threadpool(_get)
        return {**state, "summary_ja": summary}

    @router.get("/long-term-memory/hourly")
    async def get_ltm_hourly(request: Request, hour: int = 9):
        """Return long-term memory facts for a specific hour (0-23)."""
        if not (0 <= hour <= 23):
            raise HTTPException(status_code=400, detail="hour must be 0-23")
        engine = _get_perception_engine(request)
        if engine._long_term_memory is None:
            return {"hour": hour, "facts": []}

        def _get():
            return engine._long_term_memory.get_hourly_pattern(hour)

        facts = await run_in_threadpool(_get)
        return {
            "hour": hour,
            "facts": [f.__dict__ for f in facts],
        }

    # ── Phase 7: Action Executor ────────────────────────────────────────────

    class ActionPlanRequest(BaseModel):
        action_type: str = Field(..., description="alert/webhook/escalate/record/suppress")
        trigger_event_type: str = Field(..., description="EntityEventType name or '*' for any")
        trigger_severity_min: str = "warning"
        cooldown_seconds: float = 60.0
        parameters: dict = Field(default_factory=dict)
        description_ja: str = ""
        description_en: str = ""

    @router.get("/actions/state")
    async def get_action_state(request: Request) -> dict:
        """アクションエグゼキューターの状態 (登録プラン数、実行ログ)。"""
        engine = _get_perception_engine(request)

        def _get():
            state = engine.get_action_state()
            if state is None:
                return {"total_plans": 0, "enabled_plans": 0, "total_executions": 0, "plans": []}
            return state

        return await run_in_threadpool(_get)

    @router.post("/actions/plans")
    async def add_action_plan(body: ActionPlanRequest, request: Request) -> dict:
        """新しいアクションプランを登録。"""
        engine = _get_perception_engine(request)
        if engine._action_executor is None:
            raise HTTPException(status_code=503, detail="ActionExecutor not initialized")

        def _add():
            try:
                from sopilot.perception.action_executor import ActionType
                action_type = ActionType(body.action_type)
            except (ValueError, ImportError):
                raise ValueError(f"Invalid action_type: {body.action_type}")
            plan = engine._action_executor.create_plan(
                action_type=action_type,
                trigger_event_type=body.trigger_event_type,
                trigger_severity_min=body.trigger_severity_min,
                cooldown_seconds=body.cooldown_seconds,
                parameters=body.parameters,
                description_ja=body.description_ja,
                description_en=body.description_en,
            )
            return plan.to_dict()

        try:
            plan_dict = await run_in_threadpool(_add)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return plan_dict

    @router.delete("/actions/plans/{plan_id}")
    async def remove_action_plan(plan_id: str, request: Request) -> dict:
        """アクションプランを削除。"""
        engine = _get_perception_engine(request)
        if engine._action_executor is None:
            raise HTTPException(status_code=503, detail="ActionExecutor not initialized")

        def _remove():
            return engine._action_executor.remove_plan(plan_id)

        removed = await run_in_threadpool(_remove)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
        return {"status": "deleted", "plan_id": plan_id}

    @router.get("/actions/log")
    async def get_action_log(request: Request, n: int = 20) -> dict:
        """最近のアクション実行ログを取得。"""
        engine = _get_perception_engine(request)
        if engine._action_executor is None:
            return {"log": [], "total": 0}

        def _get():
            return engine._action_executor.get_log(n)

        log = await run_in_threadpool(_get)
        return {"log": [r.to_dict() for r in log], "total": len(log)}

    # ── Phase 7: Multimodal Fusion ──────────────────────────────────────────

    class MultimodalSignalRequest(BaseModel):
        source: str = Field(..., description="audio/iot/access/custom")
        signal_type: str = Field(..., description="e.g. 'gunshot', 'door_open', 'motion_pir'")
        value: float = Field(default=0.5, ge=0.0, le=1.0)
        metadata: dict = Field(default_factory=dict)

    @router.post("/multimodal/signals")
    async def ingest_multimodal_signal(
        body: MultimodalSignalRequest, request: Request
    ) -> dict:
        """外部マルチモーダル信号（音声/IoT/アクセスログ）を取り込む。"""
        engine = _get_perception_engine(request)
        if engine._fusion_engine is None:
            raise HTTPException(status_code=503, detail="MultimodalFusionEngine not initialized")

        def _ingest():
            try:
                from sopilot.perception.multimodal import SignalSource
                src = SignalSource(body.source)
            except (ValueError, ImportError):
                raise ValueError(f"Invalid source: {body.source}")
            signal = engine._fusion_engine.create_signal(
                source=src,
                signal_type=body.signal_type,
                value=body.value,
                metadata=body.metadata,
            )
            return signal.to_dict()

        try:
            result = await run_in_threadpool(_ingest)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return result

    @router.get("/multimodal/signals")
    async def get_multimodal_signals(
        request: Request,
        source: str | None = None,
        lookback_seconds: float = 60.0,
    ) -> dict:
        """バッファ内のマルチモーダル信号を取得。"""
        engine = _get_perception_engine(request)
        if engine._fusion_engine is None:
            return {"signals": [], "total": 0}

        def _get():
            signals = engine._fusion_engine.get_signals(
                source=source, lookback_seconds=lookback_seconds
            )
            return signals

        signals = await run_in_threadpool(_get)
        return {"signals": [s.to_dict() for s in signals], "total": len(signals)}

    @router.get("/multimodal/fusion-log")
    async def get_fusion_log(request: Request, n: int = 20) -> dict:
        """最近の融合イベントログを取得。"""
        engine = _get_perception_engine(request)
        if engine._fusion_engine is None:
            return {"log": [], "total": 0}

        def _get():
            return engine._fusion_engine.get_fusion_log(n)

        log = await run_in_threadpool(_get)
        return {"log": [e.to_dict() for e in log], "total": len(log)}

    @router.get("/multimodal/state")
    async def get_multimodal_state(request: Request) -> dict:
        """マルチモーダル融合エンジンの状態サマリー。"""
        engine = _get_perception_engine(request)

        def _get():
            state = engine.get_fusion_state()
            if state is None:
                return {"buffered_signals": 0, "total_ingested": 0, "total_fused": 0}
            return state

        return await run_in_threadpool(_get)

    # ── Phase 8: Depth Estimation ───────────────────────────────────────────

    @router.get("/depth")
    async def get_depth_estimates(request: Request) -> dict:
        """直近フレームのエンティティ深度推定結果。"""
        engine = _get_perception_engine(request)

        def _get():
            estimates = engine.get_depth_estimates()
            return {
                "estimates": [e.to_dict() if hasattr(e, "to_dict") else vars(e) for e in estimates],
                "count": len(estimates),
            }

        return await run_in_threadpool(_get)

    # ── Phase 8: Spatial Map ────────────────────────────────────────────────

    @router.get("/spatial-map")
    async def get_spatial_map(request: Request) -> dict:
        """空間占有グリッドの現在状態。"""
        engine = _get_perception_engine(request)

        def _get():
            state = engine.get_spatial_state()
            if state is None:
                return {"grid_w": 0, "grid_h": 0, "crowd_density": 0.0, "hotspots": []}
            return state

        return await run_in_threadpool(_get)

    # ── Phase 8: Scene Understanding ────────────────────────────────────────

    @router.get("/scene")
    async def get_scene(request: Request) -> dict:
        """最新のホリスティックシーン解析結果。"""
        engine = _get_perception_engine(request)

        def _get():
            state = engine.get_scene_state()
            if state is None:
                return {"history_size": 0, "latest": None, "trend": {}}
            return state

        return await run_in_threadpool(_get)

    @router.get("/scene/history")
    async def get_scene_history(request: Request, n: int = 10) -> dict:
        """シーン解析の履歴 (直近 n 件)。"""
        engine = _get_perception_engine(request)

        def _get():
            if engine._scene_understanding is None:
                return {"history": [], "total": 0}
            history = engine._scene_understanding.get_history(n)
            return {
                "history": [h.to_dict() for h in history],
                "total": len(history),
            }

        return await run_in_threadpool(_get)

    # ── Phase 8: Adaptive Learner ───────────────────────────────────────────

    @router.get("/adaptive-learner/state")
    async def get_adaptive_learner_state(request: Request) -> dict:
        """コンセプトドリフト検出 + 自動再キャリブレーションの状態。"""
        engine = _get_perception_engine(request)

        def _get():
            state = engine.get_adaptive_state()
            if state is None:
                return {"total_observed": 0, "recalibration_count": 0, "drift_count": 0}
            return state

        return await run_in_threadpool(_get)

    @router.post("/adaptive-learner/recalibrate")
    async def force_recalibrate(request: Request) -> dict:
        """手動で閾値再キャリブレーションをトリガー。"""
        engine = _get_perception_engine(request)
        if engine._adaptive_learner is None:
            raise HTTPException(status_code=503, detail="AdaptiveLearner not initialized")

        def _do():
            record = engine._adaptive_learner.force_recalibrate()
            if record is None:
                return {"status": "skipped", "reason": "insufficient_observations"}
            return {"status": "applied", "record": record.to_dict()}

        return await run_in_threadpool(_do)

    # ── Phase 8: Attention Broker ───────────────────────────────────────────

    @router.get("/attention/state")
    async def get_attention_state(request: Request) -> dict:
        """マルチカメラ VLM コールバジェット管理の状態。"""
        engine = _get_perception_engine(request)

        def _get():
            state = engine.get_attention_state()
            if state is None:
                return {"global_cpm": 0, "session_count": 0, "global_allowed_total": 0}
            return state

        return await run_in_threadpool(_get)

    # ── Phase 9: Anticipation Engine ─────────────────────────────────────────

    @router.get("/anticipation/hazards")
    async def get_anticipation_hazards(request: Request) -> dict:
        """予測安全エンジン: アクティブ危険アセスメント一覧。"""
        engine = _get_perception_engine(request)

        def _get():
            hazards = engine.get_active_hazards()
            return {
                "hazards": [h.to_dict() if hasattr(h, "to_dict") else h for h in hazards],
                "count": len(hazards),
            }

        return await run_in_threadpool(_get)

    @router.get("/anticipation/state")
    async def get_anticipation_state(request: Request) -> dict:
        """予測安全エンジンの統計・設定。"""
        engine = _get_perception_engine(request)

        def _get():
            state = engine.get_anticipation_state()
            if state is None:
                return {"enabled": False, "total_hazards_detected": 0}
            return {"enabled": True, **state}

        return await run_in_threadpool(_get)

    # ── CLIP Zero-shot Classifier ─────────────────────────────────────────────

    @router.get("/clip/state")
    async def get_clip_state(request: Request) -> dict:
        """CLIP ゼロショット分類器の状態・ラベル一覧。"""
        engine = _get_perception_engine(request)

        def _get():
            state = engine.get_clip_state()
            if state is None:
                return {"enabled": False}
            return {"enabled": True, **state}

        return await run_in_threadpool(_get)

    # ── Multi-Agent Coordinator ───────────────────────────────────────────────

    @router.get("/multi-agent/state")
    async def get_multi_agent_state() -> dict:
        """マルチエージェント協調の状態 (グローバルシングルトン)。"""
        try:
            from sopilot.perception.multi_agent import get_coordinator
            coordinator = get_coordinator()
            return coordinator.get_state_dict()
        except ImportError:
            return {"enabled": False, "total_agents": 0}

    @router.post("/multi-agent/agents")
    async def register_multi_agent(request: Request) -> dict:
        """新しいカメラエージェントを登録。"""
        body = await request.json()
        agent_id = str(body.get("agent_id", ""))
        camera_id = str(body.get("camera_id", ""))
        location = str(body.get("location", ""))
        if not agent_id:
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail="agent_id required")
        try:
            from sopilot.perception.multi_agent import get_coordinator
            info = get_coordinator().register_agent(agent_id, camera_id, location)
            return info.to_dict()
        except ImportError:
            return {"error": "multi_agent module not available"}

    @router.get("/multi-agent/agents")
    async def list_multi_agents(active_only: bool = False) -> dict:
        """登録済みエージェント一覧。active_only=trueで活動中のみ。"""
        try:
            from sopilot.perception.multi_agent import get_coordinator
            coord = get_coordinator()
            if active_only:
                agents = coord.get_active_agents()
            else:
                agents = list(coord._agents.values())
            return {"agents": [a.to_dict() for a in agents], "count": len(agents)}
        except ImportError:
            return {"agents": [], "count": 0}

    @router.post("/multi-agent/agents/{agent_id}/heartbeat")
    async def multi_agent_heartbeat(agent_id: str) -> dict:
        """エージェントのハートビート更新。"""
        try:
            from sopilot.perception.multi_agent import get_coordinator
            ok = get_coordinator().heartbeat(agent_id)
            return {"ok": ok, "agent_id": agent_id}
        except ImportError:
            return {"ok": False}

    @router.delete("/multi-agent/agents/{agent_id}")
    async def unregister_multi_agent(agent_id: str) -> dict:
        """エージェントを登録解除。"""
        try:
            from sopilot.perception.multi_agent import get_coordinator
            ok = get_coordinator().unregister_agent(agent_id)
            return {"ok": ok, "agent_id": agent_id}
        except ImportError:
            return {"ok": False}

    @router.post("/multi-agent/agents/{agent_id}/spatial")
    async def submit_agent_spatial(agent_id: str, request: Request) -> dict:
        """エージェントの占有グリッドを共有マップに反映。"""
        body = await request.json()
        grid = body.get("occupancy_grid", [])
        if not grid:
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail="occupancy_grid required")
        try:
            from sopilot.perception.multi_agent import get_coordinator
            get_coordinator().submit_spatial_update(agent_id, grid)
            return {"ok": True, "agent_id": agent_id, "grid_rows": len(grid)}
        except ImportError:
            return {"ok": False}

    @router.post("/multi-agent/agents/{agent_id}/entities")
    async def submit_agent_entities(agent_id: str, request: Request) -> dict:
        """エージェントのローカルエンティティをグローバルIDに解決。"""
        body = await request.json()
        entities = body.get("entities", [])
        try:
            from sopilot.perception.multi_agent import get_coordinator
            global_ids = get_coordinator().submit_entities(agent_id, entities)
            return {"agent_id": agent_id, "global_ids": global_ids, "count": len(global_ids)}
        except ImportError:
            return {"agent_id": agent_id, "global_ids": [], "count": 0}

    @router.get("/multi-agent/entities")
    async def list_global_entities() -> dict:
        """グローバルエンティティ一覧 (クロスカメラ解決済み)。"""
        try:
            from sopilot.perception.multi_agent import get_coordinator
            entities = get_coordinator().get_global_entities()
            return {"entities": [e.to_dict() for e in entities], "count": len(entities)}
        except ImportError:
            return {"entities": [], "count": 0}

    @router.get("/multi-agent/entities/{global_id}")
    async def get_global_entity(global_id: str) -> dict:
        """特定グローバルエンティティの詳細。"""
        try:
            from sopilot.perception.multi_agent import get_coordinator
            e = get_coordinator().get_entity_by_global_id(global_id)
            if e is None:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="entity not found")
            return e.to_dict()
        except ImportError:
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail="multi_agent module not available")

    @router.post("/multi-agent/events")
    async def broadcast_multi_agent_event(request: Request) -> dict:
        """エージェントイベントをブロードキャスト。"""
        body = await request.json()
        agent_id = str(body.get("agent_id", "unknown"))
        event = {k: v for k, v in body.items() if k != "agent_id"}
        try:
            from sopilot.perception.multi_agent import get_coordinator
            get_coordinator().broadcast_event(agent_id, event)
            return {"ok": True, "agent_id": agent_id}
        except ImportError:
            return {"ok": False}

    @router.get("/multi-agent/events")
    async def get_multi_agent_events(n: int = 20) -> dict:
        """最近のクロスカメライベント一覧。"""
        try:
            from sopilot.perception.multi_agent import get_coordinator
            events = get_coordinator().get_recent_events(n)
            return {"events": events, "count": len(events)}
        except ImportError:
            return {"events": [], "count": 0}

    @router.get("/multi-agent/map/hotspots")
    async def get_multi_agent_hotspots(top_n: int = 10) -> dict:
        """共有空間マップのホットスポット (高占有セル)。"""
        try:
            from sopilot.perception.multi_agent import get_coordinator
            hotspots = get_coordinator().get_shared_map().get_hotspots(top_n=top_n)
            grid_state = get_coordinator().get_shared_map().get_state_dict()
            return {"hotspots": hotspots, "map_stats": grid_state}
        except ImportError:
            return {"hotspots": [], "map_stats": {}}

    @router.post("/multi-agent/reset")
    async def reset_multi_agent_coordinator() -> dict:
        """マルチエージェントコーディネーターをリセット。"""
        try:
            from sopilot.perception.multi_agent import get_coordinator
            get_coordinator().reset()
            return {"ok": True, "message": "coordinator reset"}
        except ImportError:
            return {"ok": False}

    # ── Phase 53: Causal Inference Engine ──────────────────────────────────

    @router.get("/causal/inference/stats")
    async def get_causal_inference_stats(request: Request) -> dict:
        """因果推論エンジン — バッファ統計・発見済みエッジ数 (Phase 53)。"""
        engine = _get_perception_engine(request)
        state = engine.get_causal_inference_state()
        if state is None:
            return {"enabled": False, "message": "CausalInferenceEngine not initialized (causal_inference_enabled=False)"}
        return {"enabled": True, **state}

    @router.get("/causal/inference/graph")
    async def get_causal_graph(request: Request) -> dict:
        """因果グラフ — Granger因果エッジ一覧 (Phase 53)。"""
        engine = _get_perception_engine(request)
        cie = getattr(engine, "_causal_inference_engine", None)
        if cie is None:
            return {"enabled": False, "edges": []}
        def _get():
            edges = cie.compute_causal_graph(force=False)
            return {
                "enabled": True,
                "edge_count": len(edges),
                "edges": [e.to_dict() for e in edges],
                "event_types": cie.get_event_types(),
            }
        return await run_in_threadpool(_get)

    @router.post("/causal/inference/intervention")
    async def query_causal_intervention(request: Request) -> dict:
        """介入シミュレーション — 指定イベント防止時の連鎖効果を推定 (Phase 53)。

        Request body: {"prevent_event": "PPE未着用", "camera_id": "cam01"}
        """
        engine = _get_perception_engine(request)
        cie = getattr(engine, "_causal_inference_engine", None)
        if cie is None:
            return {"enabled": False, "message": "CausalInferenceEngine not initialized"}
        body = await request.json()
        prevent_event = body.get("prevent_event", "")
        camera_id = body.get("camera_id")
        if not prevent_event:
            return {"error": "prevent_event is required"}
        def _query():
            result = cie.query_intervention(prevent_event=prevent_event, camera_id=camera_id)
            return {"enabled": True, "result": result.to_dict()}
        return await run_in_threadpool(_query)

    @router.post("/causal/inference/observe")
    async def observe_causal_event(request: Request) -> dict:
        """イベントを因果推論バッファに追加 (Phase 53)。

        Request body: {"event_type": "PPE未着用", "camera_id": "cam01", "confidence": 0.9}
        """
        engine = _get_perception_engine(request)
        cie = getattr(engine, "_causal_inference_engine", None)
        if cie is None:
            return {"enabled": False, "message": "CausalInferenceEngine not initialized"}
        body = await request.json()
        event_type = body.get("event_type", "")
        camera_id = body.get("camera_id", "api")
        confidence = float(body.get("confidence", 1.0))
        if not event_type:
            return {"error": "event_type is required"}
        cie.observe(event_type=event_type, camera_id=camera_id, confidence=confidence)
        return {"ok": True, "event_type": event_type, "camera_id": camera_id}

    # ── Phase 54: Long-Horizon Safety Planner ──────────────────────────────

    @router.get("/planner/stats")
    async def get_planner_stats(request: Request) -> dict:
        """安全プランナー統計 (Phase 54)。"""
        engine = _get_perception_engine(request)
        state = engine.get_safety_planner_state()
        if state is None:
            return {"enabled": False, "message": "SafetyPlanner not initialized (safety_planner_enabled=False)"}
        return {"enabled": True, **state}

    @router.post("/planner/goals")
    async def submit_planner_goal(request: Request) -> dict:
        """安全目標を提出してプランを生成 (Phase 54)。

        Request body::

            {
              "description": "倉庫エリアのPPEコンプライアンスを確保",
              "priority": 2,
              "camera_ids": ["cam01", "cam02"],
              "horizon_seconds": 3600
            }
        """
        engine = _get_perception_engine(request)
        planner = getattr(engine, "_safety_planner", None)
        if planner is None:
            return {"enabled": False, "message": "SafetyPlanner not initialized"}
        body = await request.json()
        description = body.get("description", "")
        if not description:
            return {"error": "description is required"}
        priority = int(body.get("priority", 2))
        camera_ids = body.get("camera_ids") or None
        horizon_seconds = body.get("horizon_seconds")
        def _submit():
            plan_id = planner.submit_goal(
                description=description,
                priority=priority,
                camera_ids=camera_ids,
                horizon_seconds=float(horizon_seconds) if horizon_seconds else None,
            )
            plan = planner.get_plan(plan_id)
            return {"ok": True, "plan_id": plan_id, "plan": plan.to_dict() if plan else None}
        return await run_in_threadpool(_submit)

    @router.get("/planner/plans")
    async def get_planner_plans(request: Request, status: str = "all", limit: int = 20) -> dict:
        """プラン一覧 (Phase 54)。status: all | running | completed | expired"""
        engine = _get_perception_engine(request)
        planner = getattr(engine, "_safety_planner", None)
        if planner is None:
            return {"enabled": False, "plans": []}
        def _get():
            if status == "running":
                plans = planner.get_active_plans()
            else:
                plans = planner.get_all_plans(limit=limit)
            return {"enabled": True, "count": len(plans), "plans": [p.to_dict() for p in plans]}
        return await run_in_threadpool(_get)

    @router.get("/planner/plans/{plan_id}")
    async def get_planner_plan(request: Request, plan_id: str) -> dict:
        """特定プランの詳細 (Phase 54)。"""
        engine = _get_perception_engine(request)
        planner = getattr(engine, "_safety_planner", None)
        if planner is None:
            return {"enabled": False, "message": "SafetyPlanner not initialized"}
        plan = planner.get_plan(plan_id)
        if plan is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
        return {"enabled": True, "plan": plan.to_dict()}

    @router.post("/planner/plans/{plan_id}/cancel")
    async def cancel_planner_plan(request: Request, plan_id: str) -> dict:
        """プランをキャンセル (Phase 54)。"""
        engine = _get_perception_engine(request)
        planner = getattr(engine, "_safety_planner", None)
        if planner is None:
            return {"enabled": False, "message": "SafetyPlanner not initialized"}
        def _cancel():
            ok = planner.cancel_plan(plan_id)
            return {"ok": ok, "plan_id": plan_id}
        return await run_in_threadpool(_cancel)

    @router.get("/planner/tasks")
    async def get_planner_tasks(request: Request, camera_id: str | None = None) -> dict:
        """保留中・実行中のサブタスク一覧 (Phase 54)。"""
        engine = _get_perception_engine(request)
        planner = getattr(engine, "_safety_planner", None)
        if planner is None:
            return {"enabled": False, "tasks": []}
        def _get():
            tasks = planner.get_pending_tasks(camera_id=camera_id)
            return {"enabled": True, "count": len(tasks), "tasks": [t.to_dict() for t in tasks]}
        return await run_in_threadpool(_get)

    @router.post("/planner/tasks/{task_id}/complete")
    async def complete_planner_task(request: Request, task_id: str) -> dict:
        """タスク完了を報告 (Phase 54)。

        Request body: {"success": true, "result": {…}}
        """
        engine = _get_perception_engine(request)
        planner = getattr(engine, "_safety_planner", None)
        if planner is None:
            return {"enabled": False, "message": "SafetyPlanner not initialized"}
        body = await request.json()
        success = bool(body.get("success", True))
        result = body.get("result")
        def _complete():
            ok = planner.report_task_complete(task_id=task_id, success=success, result=result)
            return {"ok": ok, "task_id": task_id, "success": success}
        return await run_in_threadpool(_complete)

    # ── Phase 31: Model Orchestrator Status ────────────────────────────

    @router.get("/orchestrator/status")
    async def get_orchestrator_status(request: Request) -> dict:
        """モデルオーケストレーター — VRAM使用状況・ロード済みモデル一覧。"""
        engine = _get_perception_engine(request)
        orchestrator = getattr(engine, "_orchestrator", None)
        if orchestrator is None:
            return {"enabled": False, "message": "Model orchestrator not initialized"}
        def _get():
            return orchestrator.get_status()
        return await run_in_threadpool(_get)

    # ── Phase 33: Cascade Detector Stats ───────────────────────────────

    @router.get("/cascade/stats")
    async def get_cascade_stats(request: Request) -> dict:
        """カスケード検出器 — 段階別ヒット率・平均レイテンシ。"""
        engine = _get_perception_engine(request)
        detector = getattr(engine, "_detector", None)
        if detector is None or not hasattr(detector, "get_stats"):
            return {"enabled": False, "message": "Cascade detector not initialized"}
        def _get():
            return detector.get_stats()
        return await run_in_threadpool(_get)

    # ── Phase 34: SAM Segmenter State ──────────────────────────────────

    @router.get("/sam/state")
    async def get_sam_state(request: Request) -> dict:
        """SAMセグメンター — セグメンテーション統計。"""
        engine = _get_perception_engine(request)
        state = engine.get_sam_state()
        if state is None:
            return {"enabled": False, "message": "SAM segmenter not initialized (sam_enabled=False)"}
        return state

    # ── Phase 35: Temporal Fusion State ────────────────────────────────

    @router.get("/temporal/state")
    async def get_temporal_state(request: Request) -> dict:
        """時系列フュージョン — 行動分類・シーン変化・時系列異常。"""
        engine = _get_perception_engine(request)
        state = engine.get_temporal_state()
        if state is None:
            return {"enabled": False, "message": "Temporal fusion not initialized (temporal_fusion_enabled=False)"}
        def _get():
            tf = engine._temporal_fusion
            action = tf.classify_action()
            scene_changed, scene_sim = tf.detect_scene_change()
            anomalous, anomaly_score = tf.detect_temporal_anomaly()
            trend = tf.get_activity_trend()
            return {
                **state,
                "current_action": action,
                "scene_changed": scene_changed,
                "scene_similarity": round(scene_sim, 4),
                "temporal_anomaly": anomalous,
                "temporal_anomaly_score": round(anomaly_score, 4),
                "activity_trend": trend,
            }
        return await run_in_threadpool(_get)

    # ── Phase 36: Online Learner Management ────────────────────────────

    @router.get("/online-learner/categories")
    async def get_online_learner_categories(request: Request) -> dict:
        """オンライン学習 — 登録カテゴリ一覧・精度統計。"""
        engine = _get_perception_engine(request)
        state = engine.get_online_learner_state()
        if state is None:
            return {"enabled": False, "message": "Online learner not initialized (online_learning_enabled=False)"}
        ol = engine._online_learner
        def _get():
            return {
                **state,
                "categories": ol.get_categories(),
                "exemplar_count": ol.get_exemplar_count(),
            }
        return await run_in_threadpool(_get)

    @router.post("/online-learner/exemplars", response_model=ExemplarAddResponse)
    async def add_exemplar(body: ExemplarAddRequest, request: Request) -> ExemplarAddResponse:
        """教師サンプルを登録 — オンライン学習に新カテゴリ特徴を追加。"""
        engine = _get_perception_engine(request)
        if engine._online_learner is None:
            raise HTTPException(status_code=404, detail="Online learner not initialized")
        features = np.array(body.features, dtype=np.float32)
        def _add():
            eid = engine._online_learner.add_exemplar(
                category=body.category,
                features=features,
                confidence=body.confidence,
                source=body.source,
            )
            return eid, engine._online_learner.get_exemplar_count()
        eid, count = await run_in_threadpool(_add)
        return ExemplarAddResponse(exemplar_id=eid, category=body.category, exemplar_count=count)

    @router.delete("/online-learner/exemplars/{exemplar_id}")
    async def remove_exemplar(exemplar_id: str, request: Request) -> dict:
        """教師サンプルを削除。"""
        engine = _get_perception_engine(request)
        if engine._online_learner is None:
            raise HTTPException(status_code=404, detail="Online learner not initialized")
        def _remove():
            return engine._online_learner.remove_exemplar(exemplar_id)
        ok = await run_in_threadpool(_remove)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Exemplar {exemplar_id!r} not found")
        return {"status": "deleted", "exemplar_id": exemplar_id}

    # ── Phase 38: LLaVA-NeXT VLM ───────────────────────────────────────

    @router.get("/vlm/state")
    async def get_vlm_state(request: Request) -> dict:
        """LLaVA-NeXT VLM — ロード状況・使用統計。"""
        engine = _get_perception_engine(request)
        state = engine.get_llava_state()
        if state is None:
            return {"enabled": False, "message": "LLaVA client not initialized (llava_enabled=False)"}
        return {"enabled": True, **state}

    @router.post("/vlm/describe")
    async def vlm_describe_scene(request: Request) -> dict:
        """LLaVA-NeXT でシーンを日本語・英語で説明。

        直近フレームに対してホリスティックなシーン理解を実行します。
        """
        engine = _get_perception_engine(request)
        if engine._llava_client is None:
            raise HTTPException(status_code=404, detail="LLaVA client not initialized")
        def _describe():
            # Use a blank synthetic frame as placeholder when no live frame is available
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return engine._llava_client.describe_scene(frame, language="both")
        return await run_in_threadpool(_describe)

    @router.post("/vlm/ask")
    async def vlm_ask_question(request: Request) -> dict:
        """LLaVA-NeXT にビジュアルQ&Aを実行。

        Body: {"question": "ヘルメットを着用していない人は何人？"}
        """
        engine = _get_perception_engine(request)
        if engine._llava_client is None:
            raise HTTPException(status_code=404, detail="LLaVA client not initialized")
        body = await request.json()
        question = body.get("question", "")
        if not question:
            raise HTTPException(status_code=422, detail="question field required")
        def _ask():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return {"answer": engine._llava_client.answer_question(frame, question)}
        return await run_in_threadpool(_ask)

    # ── Phase 39: Grounding DINO 1.5 ───────────────────────────────────

    @router.get("/grounding-dino/state")
    async def get_grounding_dino_state(request: Request) -> dict:
        """Grounding DINO 1.5 — ロード状況・検出統計。"""
        engine = _get_perception_engine(request)
        state = engine.get_grounding_dino_15_state()
        if state is None:
            return {"enabled": False, "message": "Grounding DINO 1.5 not initialized (grounding_dino_15_enabled=False)"}
        return {"enabled": True, **state}

    # ── Phase 40: Incident Intelligence ────────────────────────────────

    @router.get("/incidents")
    async def get_incidents(request: Request, limit: int = 20) -> dict:
        """インシデント一覧 — 直近の自動生成インシデントレポート。"""
        engine = _get_perception_engine(request)
        state = engine.get_incident_reporter_state()
        if state is None:
            return {"enabled": False, "message": "Incident reporter not initialized (incident_reporting_enabled=False)"}
        def _get():
            incidents = engine._incident_reporter.get_recent_incidents(limit=limit)
            return {"enabled": True, **state, "incidents": incidents}
        return await run_in_threadpool(_get)

    @router.post("/incidents/webhook/test")
    async def test_webhook(request: Request) -> dict:
        """Webhook疎通確認 — 設定済みWebhookにテスト送信。"""
        engine = _get_perception_engine(request)
        if engine._incident_reporter is None:
            raise HTTPException(status_code=404, detail="Incident reporter not initialized")
        webhook = engine._incident_reporter._webhook
        if webhook is None:
            return {"status": "no_webhook", "message": "Webhook not configured"}
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        url = body.get("url", "")
        def _test():
            if url:
                return webhook.test_webhook(url)
            return {"status": "ok", "message": "No URL provided for test"}
        return await run_in_threadpool(_test)

    # ── Phase 41: Federated Re-ID ───────────────────────────────────────

    @router.get("/federated-reid/stats")
    async def get_federated_reid_stats(request: Request) -> dict:
        """連合Re-ID — プライバシー予算・カメラ間統計。"""
        engine = _get_perception_engine(request)
        state = engine.get_federated_reid_state()
        if state is None:
            return {"enabled": False, "message": "Federated Re-ID not initialized (federated_reid_enabled=False)"}
        return {"enabled": True, **state}

    @router.post("/federated-reid/export")
    async def export_federated_embeddings(request: Request) -> dict:
        """差分プライバシー付き埋め込みをエクスポート — 他カメラへの共有用。"""
        engine = _get_perception_engine(request)
        if engine._federated_reid is None:
            raise HTTPException(status_code=404, detail="Federated Re-ID not initialized")
        def _export():
            try:
                return engine._federated_reid.export_private()
            except RuntimeError as e:
                raise HTTPException(status_code=429, detail=str(e))
        return await run_in_threadpool(_export)

    @router.post("/federated-reid/import")
    async def import_federated_embeddings(request: Request) -> dict:
        """他カメラからエクスポートされた埋め込みをインポート。"""
        engine = _get_perception_engine(request)
        if engine._federated_reid is None:
            raise HTTPException(status_code=404, detail="Federated Re-ID not initialized")
        body = await request.json()
        source_camera = body.get("source_camera", "unknown")
        def _import():
            count = engine._federated_reid.import_from_camera(source_camera, body)
            return {"imported": count, "source_camera": source_camera}
        return await run_in_threadpool(_import)

    # ── Phase 42: Edge Optimizer ────────────────────────────────────────

    @router.get("/edge/profile")
    async def get_edge_profile(request: Request) -> dict:
        """エッジ最適化 — デバイスプロファイル・最適化推奨。"""
        engine = _get_perception_engine(request)
        state = engine.get_edge_optimizer_state()
        if state is None:
            return {"enabled": False, "message": "Edge optimizer not initialized (edge_optimization_enabled=False)"}
        optimizer = engine._edge_optimizer
        def _get():
            return {
                "enabled": True,
                **state,
                "device_profile": optimizer._profile.to_dict(),
                "quantization_int8": optimizer.generate_quantization_config("int8"),
                "quantization_fp16": optimizer.generate_quantization_config("fp16"),
            }
        return await run_in_threadpool(_get)

    # ── Phase 43: Video Memory + RAG ────────────────────────────────────────

    @router.get("/memory/stats")
    async def get_memory_stats(request: Request) -> dict:
        """ビデオメモリ統計 — インデックス済みフレーム数・語彙サイズ。"""
        engine = _get_perception_engine(request)
        state = engine.get_video_memory_state()
        if state is None:
            return {"enabled": False, "message": "Video memory not initialized (video_memory_enabled=False)"}
        rag_state = engine.get_rag_engine_state() or {}
        return {"enabled": True, "memory": state, "rag": rag_state}

    @router.post("/memory/query")
    async def query_memory(request: Request) -> dict:
        """RAGクエリ — 自然言語でビデオメモリを検索し、LLaVA回答を生成。

        Request body: {"query": "ヘルメット未着用を探して", "top_k": 5,
                        "camera_filter": null, "time_after": null, "time_before": null}
        """
        engine = _get_perception_engine(request)
        if engine._rag_engine is None:
            return {"enabled": False, "message": "RAG engine not initialized (rag_enabled=False)"}
        body = await request.json()
        query_text = body.get("query", "")
        top_k = int(body.get("top_k", 10))
        camera_filter = body.get("camera_filter")
        time_after = body.get("time_after")
        time_before = body.get("time_before")
        if not query_text:
            return {"error": "query must not be empty"}
        rag = engine._rag_engine
        def _query():
            return rag.full_query(
                query_text=query_text,
                top_k=top_k,
                camera_filter=camera_filter,
                time_after=float(time_after) if time_after else None,
                time_before=float(time_before) if time_before else None,
            )
        return await run_in_threadpool(_query)

    # ── Phase 44: Causal Chain Analysis ─────────────────────────────────────

    @router.get("/causal/stats")
    async def get_causal_stats(request: Request) -> dict:
        """因果連鎖分析 — バッファ統計・最近の因果チェーン。"""
        engine = _get_perception_engine(request)
        state = engine.get_causal_analyzer_state()
        if state is None:
            return {"enabled": False, "message": "Causal analyzer not initialized (causal_analysis_enabled=False)"}
        return {"enabled": True, **state}

    @router.get("/causal/chains")
    async def get_causal_chains(request: Request, limit: int = 20) -> dict:
        """最近の因果連鎖チェーン一覧。"""
        engine = _get_perception_engine(request)
        if engine._causal_analyzer is None:
            return {"enabled": False, "chains": []}
        def _get():
            return {
                "enabled": True,
                "chains": engine._causal_analyzer.get_recent_chains(limit=limit),
            }
        return await run_in_threadpool(_get)

    @router.post("/causal/analyze")
    async def analyze_causal_chain(request: Request) -> dict:
        """指定インシデントの因果連鎖を即時分析。

        Request body: {"incident_type": "転倒", "camera_id": "cam01",
                        "lookback_seconds": 30}
        """
        engine = _get_perception_engine(request)
        if engine._causal_analyzer is None:
            return {"enabled": False, "message": "Causal analyzer not initialized"}
        body = await request.json()
        incident_type = body.get("incident_type", "unknown")
        camera_id = body.get("camera_id")
        lookback = float(body.get("lookback_seconds", 30.0))
        def _analyze():
            chain = engine._causal_analyzer.analyze(
                incident_type=incident_type,
                camera_id=camera_id,
                lookback_seconds=lookback,
            )
            return {"enabled": True, "chain": chain.to_dict()}
        return await run_in_threadpool(_analyze)

    # ── Phase 45: Active Learning Loop ──────────────────────────────────────

    @router.get("/active-learning/stats")
    async def get_active_learning_stats(request: Request) -> dict:
        """能動学習ループ統計 — ラベル数・学習更新数・ラベル分布。"""
        engine = _get_perception_engine(request)
        state = engine.get_active_loop_state()
        if state is None:
            return {"enabled": False, "message": "Active learning loop not initialized (active_loop_enabled=False)"}
        return {"enabled": True, **state}

    @router.get("/active-learning/queue")
    async def get_active_learning_queue(request: Request, limit: int = 20) -> dict:
        """ReviewQueueのラベル待ちフレーム一覧。"""
        engine = _get_perception_engine(request)
        if engine._active_loop is None:
            return {"enabled": False, "pending": []}
        def _get():
            return {
                "enabled": True,
                "pending": engine._active_loop.get_pending_frames(limit=limit),
            }
        return await run_in_threadpool(_get)

    @router.post("/active-learning/label")
    async def submit_active_label(request: Request) -> dict:
        """オペレーターラベル提出 → OnlineLearner即時更新。

        Request body: {"frame_id": "uuid", "label": "helmet_violation",
                        "confidence": 0.95, "operator_id": "op01"}
        """
        engine = _get_perception_engine(request)
        if engine._active_loop is None:
            return {"enabled": False, "message": "Active learning loop not initialized"}
        body = await request.json()
        frame_id = body.get("frame_id", "")
        label = body.get("label", "")
        confidence = float(body.get("confidence", 1.0))
        operator_id = body.get("operator_id", "operator")
        if not frame_id or not label:
            return {"error": "frame_id and label are required"}
        def _submit():
            return engine._active_loop.submit_label(
                frame_id=frame_id,
                label=label,
                confidence=confidence,
                operator_id=operator_id,
            )
        return await run_in_threadpool(_submit)

    @router.get("/active-learning/labels")
    async def get_recent_labels(request: Request, limit: int = 20) -> dict:
        """最近のオペレーターラベル提出履歴。"""
        engine = _get_perception_engine(request)
        if engine._active_loop is None:
            return {"enabled": False, "labels": []}
        def _get():
            return {
                "enabled": True,
                "labels": engine._active_loop.get_recent_labels(limit=limit),
            }
        return await run_in_threadpool(_get)

    # ── Phase 48: SLA Observatory endpoints ──────────────────────────────

    @router.get("/sla/stats")
    async def get_sla_stats(request: Request) -> dict:
        """SLA監視統計 — 全ステージP50/P95/P99 + SLO違反数。"""
        engine = _get_perception_engine(request)
        state = engine.get_sla_state()
        if state is None:
            return {"enabled": False, "message": "SLAMonitor not initialized (sla_monitor_enabled=False)"}
        return {"enabled": True, **state}

    @router.get("/sla/slos")
    async def get_sla_slos(request: Request) -> dict:
        """SLO遵守状況一覧 (compliant / violated)。"""
        engine = _get_perception_engine(request)
        if engine._sla_monitor is None:
            return {"enabled": False, "slos": []}
        def _get():
            return {
                "enabled": True,
                "slos": [s.to_dict() for s in engine._sla_monitor.check_slos()],
            }
        return await run_in_threadpool(_get)

    @router.get("/sla/stages")
    async def get_sla_stages(request: Request) -> dict:
        """ステージ別レイテンシ統計 (P50/P95/P99/mean/max)。"""
        engine = _get_perception_engine(request)
        if engine._sla_monitor is None:
            # Fallback: PercMetricsRegistry always captures stage latency
            try:
                from sopilot.perception.perc_metrics import get_registry
                return {"enabled": True, "source": "perc_metrics", "stages": get_registry().get_stage_percentiles()}
            except Exception:
                return {"enabled": False, "stages": {}}
        def _get():
            return {"enabled": True, "source": "sla_monitor", "stages": engine._sla_monitor.get_all_stage_stats()}
        return await run_in_threadpool(_get)

    # ── Phase 46: Spatial Intelligence endpoints ──────────────────────────

    @router.get("/spatial/stats")
    async def get_spatial_stats(request: Request) -> dict:
        """空間マッパー統計 — カメラ数・点群サイズ・フロアサイズ。"""
        engine = _get_perception_engine(request)
        state = engine.get_spatial_mapper_state()
        if state is None:
            return {"enabled": False, "message": "SpatialMapper not initialized (spatial_mapper_enabled=False)"}
        return {"enabled": True, **state}

    @router.get("/spatial/map")
    async def get_spatial_map(request: Request, cols: int = 40, rows: int = 30, max_age: float = 60.0) -> dict:
        """フロアマップデータ — 点群・占有グリッド・カメラ一覧。"""
        engine = _get_perception_engine(request)
        if engine._spatial_mapper is None:
            return {"enabled": False, "message": "SpatialMapper not initialized"}
        def _get():
            return {
                "enabled": True,
                **engine._spatial_mapper.to_map_data(
                    grid_cols=cols,
                    grid_rows=rows,
                    max_age_seconds=max_age,
                ),
            }
        return await run_in_threadpool(_get)

    @router.post("/spatial/cameras/register")
    async def register_spatial_camera(request: Request) -> dict:
        """カメラをフロアホモグラフィで登録。

        Request body: {"camera_id": "cam01", "homography": [[...],[...],[...]],
                        "image_size": [1920, 1080]}
        homography は 3×3 フラット行列 (リスト of リスト).
        """
        engine = _get_perception_engine(request)
        if engine._spatial_mapper is None:
            return {"success": False, "message": "SpatialMapper not initialized"}
        body = await request.json()
        camera_id = body.get("camera_id", "")
        H_raw = body.get("homography", None)
        image_size = tuple(body.get("image_size", [1920, 1080]))
        if not camera_id or H_raw is None:
            return {"success": False, "message": "camera_id and homography are required"}
        import numpy as np
        H = np.array(H_raw, dtype=np.float64).reshape(3, 3)
        def _reg():
            engine._spatial_mapper.register_camera(camera_id, H, image_size)
            return {"success": True, "camera_id": camera_id, "image_size": list(image_size)}
        return await run_in_threadpool(_reg)

    @router.get("/spatial/zones")
    async def list_spatial_zones(request: Request) -> dict:
        """登録済みゾーン一覧。"""
        engine = _get_perception_engine(request)
        if engine._zone_analyzer is None:
            return {"enabled": False, "zones": []}
        def _get():
            return {
                "enabled": True,
                "zones": [z.to_dict() for z in engine._zone_analyzer.get_zones()],
            }
        return await run_in_threadpool(_get)

    @router.post("/spatial/zones")
    async def add_spatial_zone(request: Request) -> dict:
        """ゾーン登録 (polygon, zone_type, name).

        Request body: {"zone_id": "restricted_A", "zone_type": "restricted",
                        "polygon": [{"x": 0, "y": 0}, ...], "name": "立入禁止区域A"}
        """
        engine = _get_perception_engine(request)
        if engine._zone_analyzer is None:
            return {"success": False, "message": "ZoneAnalyzer not initialized"}
        body = await request.json()
        zone_id = body.get("zone_id", "")
        zone_type = body.get("zone_type", "normal")
        polygon_raw = body.get("polygon", [])
        name = body.get("name", zone_id)
        if not zone_id or len(polygon_raw) < 3:
            return {"success": False, "message": "zone_id and polygon (≥3 points) are required"}
        polygon = [(float(p["x"]), float(p["y"])) for p in polygon_raw]
        def _add():
            z = engine._zone_analyzer.add_zone(zone_id, polygon, zone_type, name=name)
            return {"success": True, "zone": z.to_dict()}
        return await run_in_threadpool(_add)

    @router.delete("/spatial/zones/{zone_id}")
    async def remove_spatial_zone(request: Request, zone_id: str) -> dict:
        """ゾーン削除。"""
        engine = _get_perception_engine(request)
        if engine._zone_analyzer is None:
            return {"success": False, "message": "ZoneAnalyzer not initialized"}
        removed = engine._zone_analyzer.remove_zone(zone_id)
        return {"success": removed, "zone_id": zone_id}

    @router.get("/spatial/zones/events")
    async def get_zone_events(request: Request, limit: int = 50, event_type: str | None = None) -> dict:
        """ゾーンイベント履歴 (enter/exit/dwell)。"""
        engine = _get_perception_engine(request)
        if engine._zone_analyzer is None:
            return {"enabled": False, "events": []}
        def _get():
            return {
                "enabled": True,
                "events": engine._zone_analyzer.get_recent_events(limit=limit, event_type=event_type),
            }
        return await run_in_threadpool(_get)

    @router.get("/spatial/zones/dwell")
    async def get_zone_dwell_stats(request: Request) -> dict:
        """ゾーン別滞在時間統計 (平均・最大・現在占有人数)。"""
        engine = _get_perception_engine(request)
        if engine._zone_analyzer is None:
            return {"enabled": False, "stats": {}}
        def _get():
            return {
                "enabled": True,
                "stats": engine._zone_analyzer.get_dwell_stats(),
            }
        return await run_in_threadpool(_get)

    @router.get("/spatial/zones/flow")
    async def get_zone_flow(request: Request) -> dict:
        """ゾーン間移動フロー (from_zone → to_zone)。"""
        engine = _get_perception_engine(request)
        if engine._zone_analyzer is None:
            return {"enabled": False, "flows": []}
        def _get():
            return {
                "enabled": True,
                "flows": engine._zone_analyzer.get_flow_vectors(),
                "zone_stats": engine._zone_analyzer.get_stats(),
            }
        return await run_in_threadpool(_get)

    @router.get("/spatial/zone-analyzer/stats")
    async def get_zone_analyzer_stats(request: Request) -> dict:
        """ゾーン解析エンジン統計。"""
        engine = _get_perception_engine(request)
        state = engine.get_zone_analyzer_state()
        if state is None:
            return {"enabled": False, "message": "ZoneAnalyzer not initialized"}
        return {"enabled": True, **state}

    # ── Phase 50: Crowd Flow Analytics endpoints ──────────────────────────

    @router.get("/crowd/stats")
    async def get_crowd_stats(request: Request) -> dict:
        """群衆解析統計 — 人数・密度・輻輳セル数・フローベクトル数。"""
        engine = _get_perception_engine(request)
        state = engine.get_crowd_analytics_state()
        if state is None:
            return {"enabled": False, "message": "CrowdAnalytics not initialized (crowd_analytics_enabled=False)"}
        return state

    @router.get("/crowd/state")
    async def get_crowd_state(request: Request) -> dict:
        """群衆解析フルスナップショット — 密度グリッド・フローベクトル・輻輳ゾーン。"""
        engine = _get_perception_engine(request)
        if engine._crowd_analytics is None:
            return {"enabled": False}
        def _get():
            return {"enabled": True, **engine._crowd_analytics.get_state()}
        return await run_in_threadpool(_get)

    @router.get("/crowd/flow")
    async def get_crowd_flow(request: Request) -> dict:
        """群衆移動方向サマリー — 支配方向・平均速度・最大速度。"""
        engine = _get_perception_engine(request)
        if engine._crowd_analytics is None:
            return {"enabled": False}
        def _get():
            return {"enabled": True, **engine._crowd_analytics.get_flow_summary()}
        return await run_in_threadpool(_get)

    @router.get("/crowd/alerts")
    async def get_crowd_alerts(request: Request, active_only: bool = True) -> dict:
        """ゾーン収容量アラート (警告 / 危険)。"""
        engine = _get_perception_engine(request)
        if engine._crowd_analytics is None:
            return {"enabled": False, "alerts": []}
        def _get():
            return {
                "enabled": True,
                "alerts": engine._crowd_analytics.get_capacity_alerts(active_only=active_only),
            }
        return await run_in_threadpool(_get)

    @router.post("/crowd/capacity")
    async def set_zone_capacity(request: Request) -> dict:
        """ゾーン収容人数上限を設定 (zone_id, max_persons, zone_name?)。"""
        engine = _get_perception_engine(request)
        if engine._crowd_analytics is None:
            return {"success": False, "message": "CrowdAnalytics not initialized"}
        body = await request.json()
        zone_id = body.get("zone_id", "")
        max_persons = int(body.get("max_persons", 10))
        zone_name = body.get("zone_name", zone_id)
        if not zone_id:
            return {"success": False, "message": "zone_id required"}
        def _set():
            engine._crowd_analytics.set_zone_capacity(zone_id, max_persons, zone_name)
            return {"success": True, "zone_id": zone_id, "max_persons": max_persons}
        return await run_in_threadpool(_set)

    # ── Phase 57: Predictive Maintenance ───────────────────────────────────

    @router.get("/forecast/stats")
    async def get_forecast_stats(request: Request) -> dict:
        """予知保全フォーキャスター統計 (Phase 57)。"""
        engine = _get_perception_engine(request)
        state = engine.get_anomaly_forecaster_state()
        if state is None:
            return {"enabled": False}
        return state

    @router.get("/forecast/cameras")
    async def get_forecast_cameras(request: Request) -> dict:
        """カメラ別予測状態一覧。"""
        engine = _get_perception_engine(request)
        if engine._anomaly_forecaster is None:
            return {"enabled": False, "cameras": []}
        def _get():
            return {"enabled": True, "cameras": engine._anomaly_forecaster.get_camera_states()}
        return await run_in_threadpool(_get)

    @router.get("/forecast/predict")
    async def get_forecast_predict(
        request: Request,
        camera_id: str = "cam",
        horizon: int = 10,
    ) -> dict:
        """指定カメラの異常スコア予測 (Holt-Winters)。"""
        engine = _get_perception_engine(request)
        if engine._anomaly_forecaster is None:
            return {"enabled": False}
        def _predict():
            r = engine._anomaly_forecaster.forecast(camera_id=camera_id, horizon_steps=horizon)
            return {
                "enabled": True,
                "camera_id": r.camera_id,
                "horizon_steps": r.horizon_steps,
                "current_level": r.current_level,
                "current_trend": r.current_trend,
                "forecast_values": r.forecast_values,
                "failure_probabilities": r.failure_probabilities,
                "peak_failure_prob": r.peak_failure_prob,
                "alert_level": r.alert_level,
                "samples_used": r.samples_used,
            }
        return await run_in_threadpool(_predict)

    @router.get("/forecast/trend")
    async def get_forecast_trend(request: Request, camera_id: str = "cam") -> dict:
        """指定カメラの異常スコアトレンド (上昇/下降/安定)。"""
        engine = _get_perception_engine(request)
        if engine._anomaly_forecaster is None:
            return {"enabled": False}
        def _trend():
            t = engine._anomaly_forecaster.get_trend(camera_id)
            return {
                "enabled": True,
                "camera_id": t.camera_id,
                "direction": t.direction,
                "trend_per_step": t.trend_per_step,
                "current_score": t.current_score,
                "smoothed_score": t.smoothed_score,
            }
        return await run_in_threadpool(_trend)

    @router.get("/forecast/alerts")
    async def get_forecast_alerts(request: Request, camera_id: str | None = None) -> dict:
        """予知保全アラート一覧。"""
        engine = _get_perception_engine(request)
        if engine._anomaly_forecaster is None:
            return {"enabled": False, "alerts": []}
        def _get():
            alerts = engine._anomaly_forecaster.get_maintenance_alerts(camera_id)
            return {"enabled": True, "alerts": alerts}
        return await run_in_threadpool(_get)

    @router.post("/forecast/observe")
    async def forecast_observe(request: Request) -> dict:
        """異常スコアを手動登録 (テスト・外部センサー連携用)。

        Body: { anomaly_score: float, violation_count: int, camera_id: str }
        """
        engine = _get_perception_engine(request)
        if engine._anomaly_forecaster is None:
            return {"success": False, "message": "AnomalyForecaster not initialized"}
        body = await request.json()
        score = float(body.get("anomaly_score", 0.0))
        viols = int(body.get("violation_count", 0))
        camera_id = body.get("camera_id", "cam")
        def _obs():
            engine._anomaly_forecaster.observe(score, viols, camera_id=camera_id)
            return {"success": True, "camera_id": camera_id, "anomaly_score": score}
        return await run_in_threadpool(_obs)

    @router.delete("/forecast/reset")
    async def reset_forecast(request: Request, camera_id: str | None = None) -> dict:
        """フォーキャスターをリセット (全体または指定カメラ)。"""
        engine = _get_perception_engine(request)
        if engine._anomaly_forecaster is None:
            return {"success": False, "message": "AnomalyForecaster not initialized"}
        def _reset():
            engine._anomaly_forecaster.reset(camera_id)
            return {"success": True, "camera_id": camera_id}
        return await run_in_threadpool(_reset)

    # ── Phase 56: XAI Explainability ───────────────────────────────────────

    @router.get("/explainability/stats")
    async def get_explainability_stats(request: Request) -> dict:
        """XAI エンジン統計 (Phase 56)。"""
        engine = _get_perception_engine(request)
        state = engine.get_explainability_state()
        if state is None:
            return {"enabled": False}
        return state

    @router.get("/explainability/latest")
    async def get_latest_saliency(request: Request) -> dict:
        """最新フレームのサリエンシーマップを返す。"""
        engine = _get_perception_engine(request)
        if engine._explainability_engine is None:
            return {"enabled": False}
        sal = getattr(engine, "_last_saliency", None)
        if sal is None:
            return {"enabled": True, "available": False, "message": "まだフレームが処理されていません"}
        def _serialize(s) -> dict:
            return {
                "enabled": True,
                "available": True,
                "heatmap_jpeg_b64": s.heatmap_jpeg_b64,
                "peak_saliency": s.peak_saliency,
                "explanation_ja": s.explanation_ja,
                "explanation_en": s.explanation_en,
                "frame_id": s.frame_id,
                "processing_ms": s.processing_ms,
                "region_count": len(s.region_scores),
                "violation_count": len(s.violation_explanations),
            }
        return await run_in_threadpool(lambda: _serialize(sal))

    @router.post("/explainability/saliency")
    async def compute_saliency(request: Request) -> dict:
        """JPEG フレーム (base64) + 検出リストからサリエンシーマップを計算。

        Body: { frame_jpeg_b64: str, detections: list, violations: list }
        """
        engine = _get_perception_engine(request)
        if engine._explainability_engine is None:
            return {"enabled": False}
        body = await request.json()
        frame_b64 = body.get("frame_jpeg_b64", "")
        raw_detections = body.get("detections", [])
        raw_violations = body.get("violations", [])
        import base64 as _b64mod

        # Build lightweight proxy objects
        class _Det:
            def __init__(self, d):
                from sopilot.perception.types import BBox
                b = d.get("bbox", {})
                self.bbox = BBox(b.get("x1", 0), b.get("y1", 0), b.get("x2", 1), b.get("y2", 1))
                self.label = d.get("label", "")
                self.confidence = float(d.get("confidence", 1.0))

        class _Viol:
            def __init__(self, v):
                self.rule = v.get("rule", str(v))
                self.confidence = float(v.get("confidence", 1.0))

        dets = [_Det(d) for d in raw_detections]
        viols = [_Viol(v) for v in raw_violations]
        frame_bytes = _b64mod.b64decode(frame_b64) if frame_b64 else None

        def _compute():
            sal = engine._explainability_engine.compute_saliency(
                frame_jpeg=frame_bytes,
                detections=dets,
                violations=viols,
                frame_id="api_request",
            )
            return {
                "enabled": True,
                "heatmap_jpeg_b64": sal.heatmap_jpeg_b64,
                "peak_saliency": sal.peak_saliency,
                "explanation_ja": sal.explanation_ja,
                "explanation_en": sal.explanation_en,
                "processing_ms": sal.processing_ms,
                "regions": [
                    {
                        "label": r.label, "score": r.score,
                        "is_violation": r.is_violation,
                        "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
                    }
                    for r in sal.region_scores
                ],
                "violation_explanations": [
                    {
                        "rule": e.rule, "confidence": e.confidence,
                        "explanation_ja": e.explanation_ja,
                        "explanation_en": e.explanation_en,
                    }
                    for e in sal.violation_explanations
                ],
            }
        return await run_in_threadpool(_compute)

    @router.post("/explainability/explain")
    async def explain_violations(request: Request) -> dict:
        """違反リストの自然言語説明を生成 (ヒートマップなし)。

        Body: { violations: list[{rule: str, confidence: float}] }
        """
        engine = _get_perception_engine(request)
        if engine._explainability_engine is None:
            return {"enabled": False, "explanations": []}
        body = await request.json()
        raw_viols = body.get("violations", [])

        class _Viol:
            def __init__(self, v):
                self.rule = v.get("rule", str(v))
                self.confidence = float(v.get("confidence", 1.0))

        viols = [_Viol(v) for v in raw_viols]

        def _explain():
            expls = engine._explainability_engine.explain_violations(viols)
            return {
                "enabled": True,
                "explanations": [
                    {
                        "rule": e.rule, "confidence": e.confidence,
                        "explanation_ja": e.explanation_ja,
                        "explanation_en": e.explanation_en,
                    }
                    for e in expls
                ],
            }
        return await run_in_threadpool(_explain)

    @router.delete("/explainability/reset")
    async def reset_explainability(request: Request) -> dict:
        """XAI エンジン統計をリセット。"""
        engine = _get_perception_engine(request)
        if engine._explainability_engine is None:
            return {"success": False, "message": "ExplainabilityEngine not initialized"}
        def _reset():
            engine._explainability_engine.reset()
            engine._last_saliency = None
            return {"success": True}
        return await run_in_threadpool(_reset)

    # ── Phase 55: Self-Supervised Domain Adaptation ────────────────────────

    @router.get("/domain/stats")
    async def get_domain_stats(request: Request) -> dict:
        """ドメイン適応全体統計 (Phase 55)。"""
        engine = _get_perception_engine(request)
        state = engine.get_domain_adapter_state()
        if state is None:
            return {"enabled": False}
        return state

    @router.get("/domain/cameras")
    async def get_domain_cameras(request: Request) -> dict:
        """カメラ別ドメイン適応状態一覧。"""
        engine = _get_perception_engine(request)
        if engine._domain_adapter is None:
            return {"enabled": False, "cameras": []}
        def _get():
            return {"enabled": True, "cameras": engine._domain_adapter.get_camera_states()}
        return await run_in_threadpool(_get)

    @router.get("/domain/shift")
    async def get_domain_shift(request: Request, camera_id: str = "cam") -> dict:
        """指定カメラの MMD ドメインシフトスコアを計算。"""
        engine = _get_perception_engine(request)
        if engine._domain_adapter is None:
            return {"enabled": False, "camera_id": camera_id}
        def _get():
            return engine._domain_adapter.detect_domain_shift(camera_id)
        return await run_in_threadpool(_get)

    @router.get("/domain/shift/history")
    async def get_domain_shift_history(request: Request, camera_id: str = "cam") -> dict:
        """ドメインシフト学習曲線 (時系列)。"""
        engine = _get_perception_engine(request)
        if engine._domain_adapter is None:
            return {"enabled": False, "camera_id": camera_id, "history": []}
        def _get():
            return {
                "enabled": True,
                "camera_id": camera_id,
                "history": engine._domain_adapter.get_shift_history(camera_id),
            }
        return await run_in_threadpool(_get)

    @router.post("/domain/adapt")
    async def trigger_domain_adapt(request: Request) -> dict:
        """指定カメラのドメイン適応ステップを手動実行。"""
        engine = _get_perception_engine(request)
        if engine._domain_adapter is None:
            return {"enabled": False, "adapted": False}
        body = await request.json()
        camera_id = body.get("camera_id", "cam")
        def _adapt():
            return engine._domain_adapter.adapt(camera_id)
        return await run_in_threadpool(_adapt)

    @router.post("/domain/observe")
    async def domain_observe_frame(request: Request) -> dict:
        """特徴ベクトル (base64 float32) をドメインバッファに追加。"""
        engine = _get_perception_engine(request)
        if engine._domain_adapter is None:
            return {"success": False, "message": "DomainAdapter not initialized"}
        body = await request.json()
        camera_id = body.get("camera_id", "cam")
        features_b64 = body.get("features_b64", "")
        if not features_b64:
            raise HTTPException(status_code=400, detail="features_b64 required")
        import base64
        features = np.frombuffer(base64.b64decode(features_b64), dtype=np.float32)
        def _obs():
            engine._domain_adapter.observe_frame(features, camera_id=camera_id)
            return {"success": True, "camera_id": camera_id, "feature_dim": len(features)}
        return await run_in_threadpool(_obs)

    @router.post("/domain/label")
    async def domain_apply_label(request: Request) -> dict:
        """ラベル付きサンプルをドメイン適応に統合 (Phase 45 連携)。"""
        engine = _get_perception_engine(request)
        if engine._domain_adapter is None:
            return {"success": False, "message": "DomainAdapter not initialized"}
        body = await request.json()
        camera_id = body.get("camera_id", "cam")
        category = body.get("category", "")
        features_b64 = body.get("features_b64", "")
        if not category:
            raise HTTPException(status_code=400, detail="category required")
        if not features_b64:
            raise HTTPException(status_code=400, detail="features_b64 required")
        import base64
        features = np.frombuffer(base64.b64decode(features_b64), dtype=np.float32)
        def _label():
            engine._domain_adapter.apply_label(features, category, camera_id=camera_id)
            return {"success": True, "camera_id": camera_id, "category": category}
        return await run_in_threadpool(_label)

    @router.delete("/domain/reset")
    async def reset_domain_adapter(request: Request, camera_id: str | None = None) -> dict:
        """ドメインアダプタをリセット (全体または指定カメラ)。"""
        engine = _get_perception_engine(request)
        if engine._domain_adapter is None:
            return {"success": False, "message": "DomainAdapter not initialized"}
        def _reset():
            engine._domain_adapter.reset(camera_id)
            return {"success": True, "camera_id": camera_id}
        return await run_in_threadpool(_reset)

    # ── Phase 59: Agentic Closed-Loop (OODA) ─────────────────────────────

    @router.get("/agent/stats")
    async def get_agent_stats(request: Request) -> dict:
        """OODAエージェント統計 (Phase 59)。"""
        engine = _get_perception_engine(request)
        state = engine.get_agent_loop_state()
        if state is None:
            return {"enabled": False}
        return state

    @router.get("/agent/decisions")
    async def get_agent_decisions(request: Request, limit: int = 10) -> dict:
        """意思決定履歴。"""
        engine = _get_perception_engine(request)
        if engine._agent_loop is None:
            return {"enabled": False, "decisions": []}
        def _get():
            return {"enabled": True, "decisions": engine._agent_loop.get_recent_decisions(limit)}
        return await run_in_threadpool(_get)

    @router.get("/agent/events")
    async def get_agent_events(request: Request) -> dict:
        """保留イベントバッファ。"""
        engine = _get_perception_engine(request)
        if engine._agent_loop is None:
            return {"enabled": False, "events": []}
        def _get():
            return {"enabled": True, "events": engine._agent_loop.get_pending_events()}
        return await run_in_threadpool(_get)

    @router.post("/agent/think")
    async def trigger_agent_think(request: Request) -> dict:
        """手動think()実行。"""
        engine = _get_perception_engine(request)
        if engine._agent_loop is None:
            return {"enabled": False, "decision": None}
        def _think():
            decision = engine._agent_loop.think()
            return {
                "enabled": True,
                "decision": decision.to_dict() if decision else None,
            }
        return await run_in_threadpool(_think)

    @router.post("/agent/inject")
    async def inject_agent_event(request: Request) -> dict:
        """テスト用イベント注入。"""
        engine = _get_perception_engine(request)
        if engine._agent_loop is None:
            return {"success": False, "message": "AgentLoop not initialized"}
        body = await request.json()
        from sopilot.perception.agent_loop import AgentEvent
        ev = AgentEvent(
            event_type=body.get("event_type", "violation"),
            camera_id=body.get("camera_id", "cam"),
            severity=body.get("severity", "info"),
            violation_rule=body.get("violation_rule", ""),
        )
        import time as _time
        ev.timestamp = _time.time()
        def _inject():
            engine._agent_loop.inject_event(ev)
            return {"success": True, "pending_events": len(engine._agent_loop._pending_events)}
        return await run_in_threadpool(_inject)

    @router.get("/agent/actions")
    async def get_agent_actions(request: Request) -> dict:
        """アクション別カウント。"""
        engine = _get_perception_engine(request)
        if engine._agent_loop is None:
            return {"enabled": False, "action_counts": {}}
        return {"enabled": True, "action_counts": engine._agent_loop._action_counts}

    @router.delete("/agent/reset")
    async def reset_agent_loop(request: Request) -> dict:
        """エージェントループリセット。"""
        engine = _get_perception_engine(request)
        if engine._agent_loop is None:
            return {"success": False, "message": "AgentLoop not initialized"}
        def _reset():
            engine._agent_loop.reset()
            return {"success": True}
        return await run_in_threadpool(_reset)

    # ── Phase 60: Closed-Loop Verification ───────────────────────────────

    @router.get("/verifier/stats")
    async def get_verifier_stats(request: Request) -> dict:
        """検証エンジン統計 (Phase 60)。"""
        engine = _get_perception_engine(request)
        state = engine.get_closed_loop_verifier_state()
        if state is None:
            return {"enabled": False}
        return state

    @router.get("/verifier/records")
    async def get_verifier_records(request: Request, limit: int = 10) -> dict:
        """最近の検証結果一覧。"""
        engine = _get_perception_engine(request)
        if engine._closed_loop_verifier is None:
            return {"enabled": False, "records": []}
        def _get():
            return {
                "enabled": True,
                "records": engine._closed_loop_verifier.get_recent_records(limit),
            }
        return await run_in_threadpool(_get)

    @router.get("/verifier/pending")
    async def get_verifier_pending(request: Request) -> dict:
        """保留中の検証タスク一覧。"""
        engine = _get_perception_engine(request)
        if engine._closed_loop_verifier is None:
            return {"enabled": False, "tasks": []}
        def _get():
            return {
                "enabled": True,
                "tasks": engine._closed_loop_verifier.get_pending_tasks(),
            }
        return await run_in_threadpool(_get)

    @router.get("/verifier/meta")
    async def get_verifier_meta(request: Request) -> dict:
        """メタ学習状態 (閾値調整・成功率)。"""
        engine = _get_perception_engine(request)
        if engine._closed_loop_verifier is None:
            return {"enabled": False}
        def _get():
            return {"enabled": True, **engine._closed_loop_verifier.get_meta_state()}
        return await run_in_threadpool(_get)

    @router.post("/verifier/check")
    async def trigger_verifier_check(request: Request) -> dict:
        """指定カメラの保留タスクを手動で検証実行。"""
        engine = _get_perception_engine(request)
        if engine._closed_loop_verifier is None:
            return {"enabled": False, "completed": 0}
        body = await request.json()
        camera_id = body.get("camera_id", "cam")
        violation_count = body.get("violation_count", 0)

        class _MockResult:
            violations = [None] * violation_count

        def _check():
            completed = engine._closed_loop_verifier.check_pending(_MockResult(), camera_id)
            return {"enabled": True, "completed": len(completed),
                    "records": [r.to_dict() for r in completed]}
        return await run_in_threadpool(_check)

    # ── Phase 61: Agent Intelligence Report ─────────────────────────────

    @router.get("/agent/report")
    async def get_agent_report(request: Request, window_hours: float = 24.0) -> dict:
        """AIエージェント活動レポートを生成 (Phase 61)。"""
        engine = _get_perception_engine(request)
        if engine._agent_report_generator is None:
            return {"enabled": False}
        def _generate():
            report = engine._agent_report_generator.generate(window_hours=window_hours)
            return {"enabled": True, **report.to_dict()}
        return await run_in_threadpool(_generate)

    @router.get("/agent/report/summary")
    async def get_agent_report_summary(request: Request, window_hours: float = 24.0) -> dict:
        """軽量サマリー (think回数・成功率・トレンド)。"""
        engine = _get_perception_engine(request)
        if engine._agent_report_generator is None:
            return {"enabled": False}
        def _get():
            return {"enabled": True,
                    **engine._agent_report_generator.get_summary(window_hours)}
        return await run_in_threadpool(_get)

    @router.get("/agent/report/insights")
    async def get_agent_report_insights(request: Request, window_hours: float = 24.0) -> dict:
        """パターン検出インサイト一覧。"""
        engine = _get_perception_engine(request)
        if engine._agent_report_generator is None:
            return {"enabled": False, "insights": []}
        def _get():
            return {"enabled": True,
                    "insights": engine._agent_report_generator.get_insights(window_hours)}
        return await run_in_threadpool(_get)

    @router.delete("/verifier/reset")
    async def reset_verifier(request: Request) -> dict:
        """検証エンジンリセット。"""
        engine = _get_perception_engine(request)
        if engine._closed_loop_verifier is None:
            return {"success": False, "message": "ClosedLoopVerifier not initialized"}
        def _reset():
            engine._closed_loop_verifier.reset()
            return {"success": True}
        return await run_in_threadpool(_reset)

    # ── Phase 62: Agent Episodic Memory ───────────────────────────────────

    @router.get("/agent/memory/stats")
    async def get_episode_stats(request: Request) -> dict:
        """エピソードメモリ統計。"""
        engine = _get_perception_engine(request)
        state = engine.get_agent_episode_state()
        if state is None:
            return {"enabled": False}
        return state

    @router.get("/agent/memory/episodes")
    async def get_recent_episodes(request: Request, limit: int = 10) -> dict:
        """最近のエピソード一覧。"""
        engine = _get_perception_engine(request)
        if engine._agent_episode_store is None:
            return {"enabled": False, "episodes": []}
        def _get():
            return {
                "enabled": True,
                "episodes": engine._agent_episode_store.get_recent_episodes(limit),
            }
        return await run_in_threadpool(_get)

    @router.get("/agent/memory/search")
    async def search_episodes_by_action(
        request: Request, action: str = "escalate", limit: int = 10
    ) -> dict:
        """アクション別エピソード検索。"""
        engine = _get_perception_engine(request)
        if engine._agent_episode_store is None:
            return {"enabled": False, "episodes": []}
        def _get():
            return {
                "enabled": True,
                "action": action,
                "episodes": engine._agent_episode_store.search_by_action(action, limit),
            }
        return await run_in_threadpool(_get)

    @router.delete("/agent/memory/reset")
    async def reset_episode_store(request: Request) -> dict:
        """エピソードメモリリセット。"""
        engine = _get_perception_engine(request)
        if engine._agent_episode_store is None:
            return {"success": False, "message": "AgentEpisodeStore not initialized"}
        def _reset():
            engine._agent_episode_store.reset()
            return {"success": True}
        return await run_in_threadpool(_reset)

    return router
