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
from fastapi.responses import StreamingResponse
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
        """Get or lazily create an AnomalyTuner attached to app.state."""
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

    return router
