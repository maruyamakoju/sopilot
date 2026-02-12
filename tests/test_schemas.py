"""Tests for Pydantic schema models in sopilot.schemas."""

from __future__ import annotations

import pytest

from sopilot.schemas import (
    AlignmentPreviewItem,
    AuditExportResponse,
    AuditSignature,
    AuditTrailItem,
    AuditTrailResponse,
    ClipCount,
    DeviationItem,
    HealthResponse,
    NightlyStatusResponse,
    QueueBackendMetrics,
    QueueMetricsResponse,
    QueueStatsItem,
    ReindexStats,
    ScoreMetrics,
    ScoreRequest,
    ScoreResult,
    ScoreResultResponse,
    SearchResponse,
    SearchResultItem,
    StepBoundaries,
    StepMapPreview,
    TimeRange,
    TrainingResult,
    TrainingResultResponse,
    VideoDeleteResponse,
    VideoInfoResponse,
    VideoIngestCreateResponse,
    VideoIngestResultResponse,
    VideoListResponse,
)

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealthResponse:
    def test_ok(self) -> None:
        h = HealthResponse(status="ok", db=True)
        assert h.status == "ok"
        assert h.db is True

    def test_degraded(self) -> None:
        h = HealthResponse(status="degraded", db=False)
        assert h.status == "degraded"
        assert h.db is False


# ---------------------------------------------------------------------------
# Score result nested models
# ---------------------------------------------------------------------------


def _make_score_metrics(**overrides) -> ScoreMetrics:
    defaults = dict(
        miss=1,
        swap=0,
        deviation=0.05,
        over_time=0.1,
        temporal_warp=0.02,
        path_stretch=0.01,
        duplicate_ratio=0.0,
        order_violation_ratio=0.0,
        temporal_drift=0.0,
        confidence_loss=0.0,
        local_similarity_gap=0.0,
        adaptive_low_similarity_threshold=0.75,
        effective_low_similarity_threshold=0.75,
        hard_miss_ratio=0.0,
        mean_alignment_cost=0.5,
    )
    defaults.update(overrides)
    return ScoreMetrics(**defaults)


def _make_score_result(**overrides) -> ScoreResult:
    defaults = dict(
        score=85.0,
        metrics=_make_score_metrics(),
        step_boundaries=StepBoundaries(gold=[0, 5, 10], trainee=[0, 4, 9]),
        deviations=[],
        alignment_preview=[
            AlignmentPreviewItem(gold_clip=0, trainee_clip=0, similarity=0.95),
        ],
        clip_count=ClipCount(gold=10, trainee=9),
        step_map_preview=StepMapPreview(gold=[0, 0, 1, 1, 2], trainee=[0, 0, 1, 1]),
    )
    defaults.update(overrides)
    return ScoreResult(**defaults)


class TestScoreMetrics:
    def test_round_trip(self) -> None:
        m = _make_score_metrics()
        assert m.miss == 1
        assert m.swap == 0
        d = m.model_dump()
        m2 = ScoreMetrics(**d)
        assert m2 == m

    def test_all_fields_present(self) -> None:
        m = _make_score_metrics()
        d = m.model_dump()
        assert "mean_alignment_cost" in d
        assert "effective_low_similarity_threshold" in d


class TestDeviationItem:
    def test_step_missing(self) -> None:
        d = DeviationItem(
            type="step_missing",
            gold_step=2,
            gold_time=TimeRange(start_sec=4.0, end_sec=8.0),
            trainee_time=TimeRange(),
            confidence=1.0,
            reason="no aligned trainee clips",
        )
        assert d.type == "step_missing"
        assert d.trainee_time.start_sec is None

    def test_confidence_bounds(self) -> None:
        with pytest.raises(Exception):
            DeviationItem(
                type="test",
                gold_step=0,
                gold_time=TimeRange(),
                trainee_time=TimeRange(),
                confidence=1.5,
                reason="out of range",
            )


class TestScoreResult:
    def test_basic(self) -> None:
        r = _make_score_result()
        assert r.score == 85.0
        assert r.metrics.miss == 1
        assert len(r.step_boundaries.gold) == 3

    def test_extra_fields_allowed(self) -> None:
        r = _make_score_result(custom_field="hello")
        assert r.model_dump()["custom_field"] == "hello"

    def test_optional_video_fields(self) -> None:
        r = _make_score_result(gold_video_id=1, trainee_video_id=2, task_id="t1")
        assert r.gold_video_id == 1
        assert r.task_id == "t1"

    def test_score_bounds(self) -> None:
        with pytest.raises(Exception):
            _make_score_result(score=101.0)
        with pytest.raises(Exception):
            _make_score_result(score=-1.0)

    def test_from_dict(self) -> None:
        """ScoreResult can be built from a dict (as returned by step_engine)."""
        raw = {
            "score": 90.0,
            "metrics": _make_score_metrics().model_dump(),
            "step_boundaries": {"gold": [0, 5], "trainee": [0, 4]},
            "deviations": [
                {
                    "type": "step_missing",
                    "gold_step": 1,
                    "gold_time": {"start_sec": 1.0, "end_sec": 2.0},
                    "trainee_time": {"start_sec": None, "end_sec": None},
                    "confidence": 1.0,
                    "reason": "test",
                }
            ],
            "alignment_preview": [{"gold_clip": 0, "trainee_clip": 0, "similarity": 0.9}],
            "clip_count": {"gold": 5, "trainee": 4},
            "step_map_preview": {"gold": [0, 0, 1], "trainee": [0, 0]},
            "gold_video_id": 1,
            "trainee_video_id": 2,
            "task_id": "task1",
            "embedding_model": "heuristic-v1",
        }
        r = ScoreResult(**raw)
        assert r.score == 90.0
        assert r.deviations[0].type == "step_missing"
        assert r.gold_video_id == 1


class TestScoreResultResponse:
    def test_with_typed_result(self) -> None:
        result = _make_score_result()
        resp = ScoreResultResponse(
            score_job_id="j1",
            status="completed",
            gold_video_id=1,
            trainee_video_id=2,
            score=85.0,
            result=result,
        )
        assert resp.result is not None
        assert resp.result.metrics.miss == 1

    def test_with_none_result(self) -> None:
        resp = ScoreResultResponse(
            score_job_id="j1",
            status="queued",
            gold_video_id=1,
            trainee_video_id=2,
            score=None,
            result=None,
        )
        assert resp.result is None

    def test_from_dict_result(self) -> None:
        """result field accepts a plain dict (backward compat with JSON files)."""
        raw_result = _make_score_result().model_dump()
        resp = ScoreResultResponse(
            score_job_id="j1",
            status="completed",
            gold_video_id=1,
            trainee_video_id=2,
            score=85.0,
            result=raw_result,
        )
        assert resp.result is not None
        assert resp.result.score == 85.0


# ---------------------------------------------------------------------------
# Training result nested models
# ---------------------------------------------------------------------------


class TestTrainingResult:
    def test_skipped(self) -> None:
        r = TrainingResult(
            status="skipped",
            trigger="nightly",
            reason="not enough new videos",
            new_videos=3,
            threshold=10,
            since="2026-01-01T00:00:00",
        )
        assert r.status == "skipped"
        assert r.mode is None

    def test_builtin(self) -> None:
        r = TrainingResult(
            status="completed",
            mode="builtin_feature_adapter",
            trigger="manual",
            adapter_path="/data/models/adapter.npz",
            videos_used=5,
            clips_used=100,
            embedding_dim=768,
            reindex=ReindexStats(
                old_index_version="v1",
                new_index_version="v2",
                videos_refreshed=5,
                clips_indexed=100,
                tasks_touched=["task1"],
            ),
        )
        assert r.mode == "builtin_feature_adapter"
        assert r.reindex is not None
        assert r.reindex.clips_indexed == 100

    def test_external(self) -> None:
        r = TrainingResult(
            status="completed",
            mode="external_command",
            command="python train.py",
            return_code=0,
            duration_sec=120.5,
            stdout_tail="done",
        )
        assert r.mode == "external_command"
        assert r.return_code == 0

    def test_extra_fields(self) -> None:
        r = TrainingResult(status="completed", unexpected="value")
        assert r.model_dump()["unexpected"] == "value"


class TestTrainingResultResponse:
    def test_with_typed_result(self) -> None:
        result = TrainingResult(status="skipped", reason="no data")
        resp = TrainingResultResponse(
            training_job_id="t1",
            trigger="manual",
            status="skipped",
            result=result,
        )
        assert resp.result is not None
        assert resp.result.reason == "no data"


# ---------------------------------------------------------------------------
# Other models: round-trip sanity
# ---------------------------------------------------------------------------


class TestVideoModels:
    def test_ingest_create(self) -> None:
        m = VideoIngestCreateResponse(ingest_job_id="j1", status="queued")
        assert m.ingest_job_id == "j1"

    def test_ingest_result(self) -> None:
        m = VideoIngestResultResponse(
            ingest_job_id="j1",
            status="completed",
            task_id="t1",
            role="trainee",
            video_id=1,
            num_clips=10,
        )
        assert m.video_id == 1

    def test_video_info(self) -> None:
        m = VideoInfoResponse(
            video_id=1,
            task_id="t1",
            role="gold",
            num_clips=5,
            embedding_model="heuristic-v1",
        )
        assert m.role == "gold"

    def test_video_list(self) -> None:
        m = VideoListResponse(
            items=[
                VideoInfoResponse(
                    video_id=1,
                    task_id="t1",
                    role="gold",
                    num_clips=5,
                    embedding_model="heuristic-v1",
                )
            ]
        )
        assert len(m.items) == 1

    def test_delete(self) -> None:
        m = VideoDeleteResponse(video_id=1, task_id="t1", removed_files=["/a"], reindexed_clips=3)
        assert m.reindexed_clips == 3


class TestScoreRequest:
    def test_valid(self) -> None:
        m = ScoreRequest(gold_video_id=1, trainee_video_id=2)
        assert m.gold_video_id == 1

    def test_zero_rejected(self) -> None:
        with pytest.raises(Exception):
            ScoreRequest(gold_video_id=0, trainee_video_id=1)


class TestSearchModels:
    def test_search_response(self) -> None:
        m = SearchResponse(
            task_id="t1",
            query_video_id=1,
            query_clip_idx=0,
            items=[
                SearchResultItem(
                    similarity=0.9,
                    video_id=2,
                    clip_idx=3,
                    start_sec=0.0,
                    end_sec=4.0,
                    role="gold",
                )
            ],
        )
        assert len(m.items) == 1
        assert m.items[0].similarity == 0.9


class TestAuditModels:
    def test_audit_trail(self) -> None:
        item = AuditTrailItem(job_id="j1", job_type="ingest", status="completed")
        resp = AuditTrailResponse(items=[item])
        assert len(resp.items) == 1

    def test_audit_export(self) -> None:
        sig = AuditSignature(
            algorithm="hmac-sha256",
            key_id="k1",
            payload_sha256="abc",
            signature_hex="def",
        )
        resp = AuditExportResponse(
            export_id="e1",
            generated_at="2026-01-01",
            item_count=5,
            file_path="/tmp/export.json",
            signature=sig,
        )
        assert resp.signature.algorithm == "hmac-sha256"


class TestQueueModels:
    def test_queue_metrics(self) -> None:
        m = QueueMetricsResponse(
            generated_at="2026-01-01",
            runtime_mode="api",
            queue=QueueBackendMetrics(
                backend="inline",
                queues=[
                    QueueStatsItem(
                        key="q1",
                        name="ingest",
                        queued=1,
                        started=0,
                        failed=0,
                        finished=5,
                        deferred=0,
                        scheduled=0,
                    )
                ],
            ),
            jobs={"ingest": {"queued": 1, "completed": 5}},
        )
        assert m.queue.backend == "inline"


class TestNightlyStatus:
    def test_basic(self) -> None:
        m = NightlyStatusResponse(enabled=True, hour_local=2, min_new_videos=10)
        assert m.next_run_local is None

    def test_hour_bounds(self) -> None:
        with pytest.raises(Exception):
            NightlyStatusResponse(enabled=True, hour_local=25, min_new_videos=0)
