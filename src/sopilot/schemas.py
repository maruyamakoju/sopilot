from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = Field(description="Service health: 'ok' or 'degraded'")
    db: bool = Field(description="Database connectivity status")


# ---------------------------------------------------------------------------
# Video ingest
# ---------------------------------------------------------------------------


class VideoIngestCreateResponse(BaseModel):
    ingest_job_id: str
    status: str


class VideoIngestResultResponse(BaseModel):
    ingest_job_id: str
    status: str
    task_id: str
    role: str
    requested_by: str | None = None
    video_id: int | None = None
    num_clips: int | None = None
    source_fps: float | None = None
    sampled_fps: float | None = None
    embedding_model: str | None = None
    error_message: str | None = None
    queued_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


# ---------------------------------------------------------------------------
# Score – request / create
# ---------------------------------------------------------------------------


class ScoreRequest(BaseModel):
    gold_video_id: int = Field(..., ge=1)
    trainee_video_id: int = Field(..., ge=1)


class ScoreCreateResponse(BaseModel):
    score_job_id: str
    status: str
    score: float | None


# ---------------------------------------------------------------------------
# Score – typed result sub-models
# ---------------------------------------------------------------------------


class TimeRange(BaseModel):
    """Start/end seconds for a clip range."""

    start_sec: float | None = None
    end_sec: float | None = None


class ScoreMetrics(BaseModel):
    """Detailed scoring metrics produced by the step engine."""

    miss: int = Field(description="Number of missed gold steps")
    swap: int = Field(description="Number of step-order swaps")
    deviation: float = Field(description="Fraction of low-similarity aligned pairs")
    over_time: float = Field(description="Relative excess duration vs gold")
    temporal_warp: float = Field(description="Mean temporal alignment distortion")
    path_stretch: float = Field(description="DTW path stretch beyond diagonal")
    duplicate_ratio: float
    order_violation_ratio: float
    temporal_drift: float
    confidence_loss: float
    local_similarity_gap: float
    adaptive_low_similarity_threshold: float
    effective_low_similarity_threshold: float
    hard_miss_ratio: float
    mean_alignment_cost: float


class StepBoundaries(BaseModel):
    gold: list[int]
    trainee: list[int]


class DeviationItem(BaseModel):
    type: str = Field(description="step_missing | order_swap | execution_deviation")
    gold_step: int
    gold_time: TimeRange
    trainee_time: TimeRange
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class AlignmentPreviewItem(BaseModel):
    gold_clip: int
    trainee_clip: int
    similarity: float


class ClipCount(BaseModel):
    gold: int
    trainee: int


class StepMapPreview(BaseModel):
    gold: list[int]
    trainee: list[int]


class NeuralScoreUncertainty(BaseModel):
    """MC Dropout uncertainty estimate for neural scoring."""

    score: float = Field(description="Mean score from MC samples")
    uncertainty: float = Field(description="Standard deviation across MC samples")
    ci_lower: float = Field(description="95% CI lower bound")
    ci_upper: float = Field(description="95% CI upper bound")
    calibrated_score: float | None = Field(default=None, description="Isotonic-calibrated score")
    n_samples: int = Field(default=30, description="Number of MC Dropout samples")


class ScoreResult(BaseModel):
    """Full evaluation output written to score_<job_id>.json."""

    model_config = ConfigDict(extra="allow")

    score: float = Field(ge=0.0, le=100.0, description="SOP compliance score 0-100")
    metrics: ScoreMetrics
    step_boundaries: StepBoundaries
    deviations: list[DeviationItem]
    alignment_preview: list[AlignmentPreviewItem]
    clip_count: ClipCount
    step_map_preview: StepMapPreview
    # Neural mode fields
    neural_score: NeuralScoreUncertainty | None = Field(
        default=None, description="Neural scoring with uncertainty (when neural mode enabled)"
    )
    neural_mode: bool = Field(default=False, description="Whether neural mode was used")
    # Fields added by scoring_service after evaluate_sop()
    gold_video_id: int | None = None
    trainee_video_id: int | None = None
    task_id: str | None = None
    embedding_model: str | None = None


# ---------------------------------------------------------------------------
# Score – result response
# ---------------------------------------------------------------------------


class ScoreResultResponse(BaseModel):
    score_job_id: str
    status: str
    gold_video_id: int
    trainee_video_id: int
    requested_by: str | None = None
    score: float | None
    error_message: str | None = None
    queued_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    result: ScoreResult | None = None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchResultItem(BaseModel):
    similarity: float
    video_id: int
    clip_idx: int
    start_sec: float
    end_sec: float
    role: str


class SearchResponse(BaseModel):
    task_id: str
    query_video_id: int
    query_clip_idx: int
    items: list[SearchResultItem]


# ---------------------------------------------------------------------------
# Video CRUD
# ---------------------------------------------------------------------------


class VideoInfoResponse(BaseModel):
    video_id: int
    task_id: str
    role: str
    site_id: str | None = None
    camera_id: str | None = None
    num_clips: int
    embedding_model: str
    created_at: str | None = None


class VideoListResponse(BaseModel):
    items: list[VideoInfoResponse]


class VideoDeleteResponse(BaseModel):
    video_id: int
    task_id: str
    removed_files: list[str]
    reindexed_clips: int


# ---------------------------------------------------------------------------
# Training – typed result sub-models
# ---------------------------------------------------------------------------


class ReindexStats(BaseModel):
    old_index_version: str
    new_index_version: str
    videos_refreshed: int
    clips_indexed: int
    tasks_touched: list[str]


class TrainingResult(BaseModel):
    """Training job output (polymorphic: skipped | builtin | external)."""

    model_config = ConfigDict(extra="allow")

    status: str = Field(description="completed | skipped")
    mode: str | None = Field(default=None, description="builtin_feature_adapter | external_command")
    trigger: str | None = None
    since: str | None = None
    new_videos: int | None = None
    reason: str | None = None
    # Builtin adapter fields
    adapter_path: str | None = None
    videos_used: int | None = None
    clips_used: int | None = None
    embedding_dim: int | None = None
    # External command fields
    command: str | None = None
    return_code: int | None = None
    duration_sec: float | None = None
    stdout_tail: str | None = None
    # Common
    reindex: ReindexStats | None = None
    threshold: int | None = None


# ---------------------------------------------------------------------------
# Training – responses
# ---------------------------------------------------------------------------


class TrainingCreateResponse(BaseModel):
    training_job_id: str
    status: str
    trigger: str


class TrainingResultResponse(BaseModel):
    training_job_id: str
    trigger: str
    status: str
    requested_by: str | None = None
    error_message: str | None = None
    queued_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    result: TrainingResult | None = None


# ---------------------------------------------------------------------------
# Nightly scheduler
# ---------------------------------------------------------------------------


class NightlyStatusResponse(BaseModel):
    enabled: bool
    next_run_local: str | None = None
    hour_local: int = Field(ge=0, le=23)
    min_new_videos: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


class AuditTrailItem(BaseModel):
    job_id: str
    job_type: str
    requested_by: str | None = None
    task_id: str | None = None
    subject: str | None = None
    status: str
    model_name: str | None = None
    score: float | None = None
    error_message: str | None = None
    queued_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str | None = None


class AuditTrailResponse(BaseModel):
    items: list[AuditTrailItem]


class AuditSignature(BaseModel):
    algorithm: str
    key_id: str
    payload_sha256: str
    signature_hex: str


class AuditExportResponse(BaseModel):
    export_id: str
    generated_at: str
    item_count: int
    file_path: str
    signature: AuditSignature


# ---------------------------------------------------------------------------
# Ops / queue metrics
# ---------------------------------------------------------------------------


class QueueStatsItem(BaseModel):
    key: str
    name: str
    queued: int
    started: int
    failed: int
    finished: int
    deferred: int
    scheduled: int


class QueueBackendMetrics(BaseModel):
    backend: str
    redis_ok: bool | None = None
    error: str | None = None
    queues: list[QueueStatsItem]


class QueueMetricsResponse(BaseModel):
    generated_at: str
    runtime_mode: str
    queue: QueueBackendMetrics
    jobs: dict[str, dict[str, int]]
