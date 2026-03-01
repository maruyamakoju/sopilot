from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from sopilot.constants import DEFAULT_WEIGHTS

if TYPE_CHECKING:
    from sopilot.core.scoring import ScoreWeights as CoreScoreWeights


class VideoIngestRequest(BaseModel):
    task_id: str = Field(description="SOP task identifier for this video")
    site_id: str | None = Field(default=None, description="Site identifier where recording took place")
    camera_id: str | None = Field(default=None, description="Camera identifier used for recording")
    operator_id_hash: str | None = Field(default=None, description="Hashed operator identifier for privacy")
    recorded_at: str | None = Field(default=None, description="ISO-8601 timestamp when the video was recorded")


class VideoIngestResponse(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{"video_id": 1, "task_id": "hand-wash", "is_gold": True, "status": "ready", "clip_count": 12, "step_boundaries": [0, 3, 7, 12], "original_filename": "gold_001.mp4", "gold_version": 1}]}}

    video_id: int = Field(description="Unique video identifier")
    task_id: str = Field(description="SOP task identifier")
    is_gold: bool = Field(description="True if this is a gold (reference) video")
    status: str = Field(description="Processing status: processing, ready, failed")
    clip_count: int = Field(description="Number of clips extracted from the video")
    step_boundaries: list[int] = Field(description="Detected step boundary indices")
    original_filename: str | None = Field(default=None, description="Original upload filename")
    quality: dict[str, Any] | None = Field(default=None, description="Video quality report (informational)")
    gold_version: int | None = Field(default=None, description="Chronological version number of this gold video within its task (gold videos only)")


class VideoDetailResponse(BaseModel):
    video_id: int = Field(description="Unique video identifier")
    task_id: str = Field(description="SOP task identifier")
    is_gold: bool = Field(description="True if this is a gold (reference) video")
    status: str = Field(description="Processing status: processing, ready, failed")
    clip_count: int = Field(description="Number of clips extracted from the video")
    site_id: str | None = Field(default=None, description="Site identifier where recording took place")
    camera_id: str | None = Field(default=None, description="Camera identifier used for recording")
    operator_id_hash: str | None = Field(default=None, description="Hashed operator identifier for privacy")
    recorded_at: str | None = Field(default=None, description="ISO-8601 timestamp when the video was recorded")
    embedding_model: str = Field(description="Name of the embedding model used for clip embeddings")
    step_boundaries: list[int] = Field(description="Detected step boundary indices")
    error: str | None = Field(default=None, description="Error message if processing failed")
    original_filename: str | None = Field(default=None, description="Original upload filename")
    created_at: str | None = Field(default=None, description="ISO-8601 timestamp when the video was ingested")
    updated_at: str | None = Field(default=None, description="ISO-8601 timestamp when the video was last updated")
    gold_version: int | None = Field(default=None, description="Chronological version number of this gold video within its task (gold videos only)")


class VideoListItem(BaseModel):
    video_id: int = Field(description="Unique video identifier")
    task_id: str = Field(description="SOP task identifier")
    is_gold: bool = Field(description="True if this is a gold (reference) video")
    status: str = Field(description="Processing status: processing, ready, failed")
    site_id: str | None = Field(default=None, description="Site identifier where recording took place")
    camera_id: str | None = Field(default=None, description="Camera identifier used for recording")
    operator_id_hash: str | None = Field(default=None, description="Hashed operator identifier for privacy")
    recorded_at: str | None = Field(default=None, description="ISO-8601 timestamp when the video was recorded")
    created_at: str = Field(description="ISO-8601 timestamp when the video was ingested")
    clip_count: int = Field(description="Number of clips extracted from the video")
    original_filename: str | None = Field(default=None, description="Original upload filename")
    gold_version: int | None = Field(default=None, description="Chronological version number of this gold video within its task (gold videos only)")


class VideoListResponse(BaseModel):
    items: list[VideoListItem] = Field(description="List of video records")
    total: int = Field(default=0, description="Total number of videos matching the filter")
    has_more: bool = Field(default=False, description="True if there are more results beyond the current page")


class VideoUpdateRequest(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{"site_id": "factory-a", "camera_id": "cam-02"}]}}

    site_id: str | None = Field(default=None, description="Updated site identifier")
    camera_id: str | None = Field(default=None, description="Updated camera identifier")
    operator_id_hash: str | None = Field(default=None, description="Updated operator hash")
    recorded_at: str | None = Field(default=None, description="Updated recording timestamp (ISO-8601)")


class ScoreWeights(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{"w_miss": 1.0, "w_swap": 0.8, "w_dev": 0.6, "w_time": 0.3}]}}

    w_miss: float = Field(default=DEFAULT_WEIGHTS["w_miss"], ge=0.0, description="Weight for missed step penalty")
    w_swap: float = Field(default=DEFAULT_WEIGHTS["w_swap"], ge=0.0, description="Weight for step order swap penalty")
    w_dev: float = Field(default=DEFAULT_WEIGHTS["w_dev"], ge=0.0, description="Weight for quality deviation penalty")
    w_time: float = Field(default=DEFAULT_WEIGHTS["w_time"], ge=0.0, description="Weight for over-time penalty")

    def to_core_weights(self) -> CoreScoreWeights:
        from sopilot.core.scoring import ScoreWeights as CoreScoreWeights
        return CoreScoreWeights(w_miss=self.w_miss, w_swap=self.w_swap, w_dev=self.w_dev, w_time=self.w_time)


class ScoreRequest(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{"gold_video_id": 1, "trainee_video_id": 2}]}}

    gold_video_id: int = Field(description="ID of the gold (reference) video", ge=1)
    trainee_video_id: int = Field(description="ID of the trainee video to score", ge=1)
    weights: ScoreWeights | None = Field(default=None, description="Custom score weights (uses task profile defaults if omitted)")


class ScoreDecisionSummary(BaseModel):
    model_config = {"extra": "allow"}

    decision: str | None = Field(default=None, description="Decision label: pass/fail/needs_review/retrain")
    decision_reason: str | None = Field(default=None, description="Human-readable decision rationale")
    decision_basis: str | None = Field(
        default=None,
        description=(
            "Machine-readable decision driver: critical_deviation | score_above_threshold | "
            "score_below_retrain | score_between_thresholds"
        ),
    )
    score_band: str | None = Field(
        default=None,
        description="Performance band relative to task thresholds: excellent | passing | needs_review | poor",
    )
    severity_counts: dict[str, int] | None = Field(default=None, description="Deviation severity histogram")
    pass_score: float | None = Field(default=None, description="Pass threshold used for this result")
    retrain_score: float | None = Field(default=None, description="Retrain threshold used for this result")


class ScoreResultPayload(BaseModel):
    """Structured score result with permissive extras for forward compatibility."""

    model_config = {"extra": "allow"}

    score: float | None = Field(default=None, description="Overall numeric score")
    summary: ScoreDecisionSummary | None = Field(default=None, description="Decision summary")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Computed scoring metrics")
    deviations: list[dict[str, Any]] = Field(default_factory=list, description="Deviation events")
    boundaries: dict[str, list[int]] | None = Field(default=None, description="Detected clip boundaries")
    alignment: dict[str, Any] | None = Field(default=None, description="Alignment diagnostics")
    confidence: dict[str, Any] | None = Field(default=None, description="Heuristic confidence interval payload")


class ScoreJobResponse(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{"job_id": 1, "status": "completed", "result": {"score": 85.2}, "weights": None, "review": None, "error": None, "created_at": "2025-01-01T00:00:00+00:00", "started_at": "2025-01-01T00:00:01+00:00", "finished_at": "2025-01-01T00:00:05+00:00"}]}}

    job_id: int = Field(description="Unique score job identifier")
    status: str = Field(description="Job status: queued, running, completed, failed, cancelled")
    result: ScoreResultPayload | None = Field(default=None, description="Structured score result")
    weights: ScoreWeights | None = Field(default=None, description="Score weights used for this job")
    review: dict | None = Field(default=None, description="Human reviewer record if submitted")
    error: str | None = Field(default=None, description="Error message if job failed")
    created_at: str | None = Field(default=None, description="ISO-8601 timestamp when the job was created")
    started_at: str | None = Field(default=None, description="ISO-8601 timestamp when the job started running")
    finished_at: str | None = Field(default=None, description="ISO-8601 timestamp when the job finished")


class ScoreJobListItem(BaseModel):
    id: int = Field(description="Score job identifier")
    gold_video_id: int = Field(description="Gold video ID")
    trainee_video_id: int = Field(description="Trainee video ID")
    status: str = Field(description="Job status")
    score: float | None = Field(default=None, description="Overall score if completed")
    decision: str | None = Field(default=None, description="Pass/fail decision if completed")
    severity_counts: dict[str, int] | None = Field(default=None, description="Severity breakdown if completed")
    score_band: str | None = Field(default=None, description="Performance band: excellent | passing | needs_review | poor")
    task_id: str | None = Field(default=None, description="Task ID from gold video")
    created_at: str = Field(description="ISO-8601 creation timestamp")
    updated_at: str = Field(description="ISO-8601 last update timestamp")
    finished_at: str | None = Field(default=None, description="ISO-8601 finish timestamp")
    error: str | None = Field(default=None, description="Error message if failed")


class ScoreJobListResponse(BaseModel):
    items: list[ScoreJobListItem] = Field(description="List of score job summaries")
    total: int = Field(default=0, description="Total number of score jobs matching the filter")
    has_more: bool = Field(default=False, description="True if there are more results beyond the current page")


class SearchResult(BaseModel):
    video_id: int = Field(description="Video ID containing the matching clip")
    clip_index: int = Field(description="Index of the matching clip within the video")
    task_id: str = Field(description="SOP task identifier of the matching video")
    is_gold: bool = Field(description="True if the matching video is a gold (reference) video")
    similarity: float = Field(description="Cosine similarity score between query and result clip")
    start_sec: float = Field(description="Start time of the matching clip in seconds")
    end_sec: float = Field(description="End time of the matching clip in seconds")


class SearchResponse(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{"query_video_id": 1, "query_clip_index": 0, "results": [{"video_id": 2, "clip_index": 3, "task_id": "hand-wash", "is_gold": True, "similarity": 0.95, "start_sec": 6.0, "end_sec": 8.0}]}]}}

    query_video_id: int = Field(description="Video ID of the query clip")
    query_clip_index: int = Field(description="Clip index used as the query")
    results: list[SearchResult] = Field(description="Ranked list of similar clips")


class TaskProfileResponse(BaseModel):
    task_id: str = Field(description="Unique task identifier")
    task_name: str = Field(description="Human-readable task name")
    pass_score: float = Field(description="Minimum score threshold to pass")
    retrain_score: float = Field(description="Score threshold below which retraining is required")
    default_weights: ScoreWeights = Field(description="Default scoring weights for this task")
    deviation_policy: dict[str, str] = Field(description="Mapping of deviation types to their handling policies")


class TaskProfileUpdateRequest(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{"task_name": "Hand Wash v2", "pass_score": 80.0}]}}

    task_name: str | None = Field(default=None, description="Updated human-readable task name")
    pass_score: float | None = Field(default=None, ge=0.0, le=100.0, description="Updated minimum score threshold to pass")
    retrain_score: float | None = Field(default=None, ge=0.0, le=100.0, description="Updated retraining score threshold")
    default_weights: ScoreWeights | None = Field(default=None, description="Updated default scoring weights")
    deviation_policy: dict[str, str] | None = Field(default=None, description="Updated deviation handling policies")


class BatchScoreRequest(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{"pairs": [{"gold_video_id": 1, "trainee_video_id": 2}, {"gold_video_id": 1, "trainee_video_id": 3}]}]}}

    pairs: list[ScoreRequest] = Field(description="List of gold-trainee video pairs to score", min_length=1, max_length=50)


class ScoreReviewRequest(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{"verdict": "pass", "note": "Operator performed all steps correctly."}]}}

    verdict: str = Field(pattern="^(pass|retrain|fail|needs_review)$", description="Reviewer's verdict")
    note: str | None = Field(default=None, max_length=5000, description="Free-text reviewer comment")


class ScoreReviewResponse(BaseModel):
    job_id: int = Field(description="Score job identifier this review belongs to")
    verdict: str = Field(description="Reviewer's verdict: pass, retrain, fail, or needs_review")
    note: str | None = Field(default=None, description="Free-text reviewer comment")
    created_at: str = Field(description="ISO-8601 timestamp when the review was created")
    updated_at: str = Field(description="ISO-8601 timestamp when the review was last updated")


# --- Request models for endpoints that previously used plain dict ---


class EnsembleScoreRequest(BaseModel):
    """Request body for POST /score/ensemble."""
    model_config = {"json_schema_extra": {"examples": [{"gold_video_ids": [1, 2, 3], "trainee_video_id": 4}]}}

    gold_video_ids: list[int] = Field(description="IDs of gold reference videos (1-10)", min_length=1, max_length=10)
    trainee_video_id: int = Field(description="ID of the trainee video to score", ge=1)
    weights: ScoreWeights | None = Field(default=None, description="Custom score weights (uses task profile defaults if omitted)")


class SoftDTWRequest(BaseModel):
    """Request body for POST /research/soft-dtw."""
    model_config = {"json_schema_extra": {"examples": [{"gold_video_id": 1, "trainee_video_id": 2, "gamma": 1.0}]}}

    gold_video_id: int = Field(description="ID of the gold reference video", ge=1)
    trainee_video_id: int = Field(description="ID of the trainee video", ge=1)
    gamma: float = Field(default=1.0, gt=0.0, description="Smoothing temperature (> 0)")


class SOPStepsUpsertRequest(BaseModel):
    """Request body for PUT /tasks/steps."""
    model_config = {"json_schema_extra": {"examples": [{"steps": [{"step_index": 0, "name_ja": "手洗い", "name_en": "Handwashing", "expected_duration_sec": 30}]}]}}

    steps: list[dict[str, Any]] = Field(description="List of step definitions to upsert")
