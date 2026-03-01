"""TypedDict definitions for database row shapes."""

from typing import TypedDict

from sopilot.core.score_result import ScoreResult


class VideoRow(TypedDict):
    id: int
    task_id: str
    site_id: str | None
    camera_id: str | None
    operator_id_hash: str | None
    recorded_at: str | None
    is_gold: bool
    file_path: str | None
    status: str
    clip_count: int
    step_boundaries_json: str
    embedding_model: str
    created_at: str
    updated_at: str
    error: str | None
    original_filename: str | None


class VideoListRow(TypedDict):
    id: int
    task_id: str
    is_gold: bool
    status: str
    site_id: str | None
    camera_id: str | None
    operator_id_hash: str | None
    recorded_at: str | None
    created_at: str
    clip_count: int
    original_filename: str | None


class ClipRow(TypedDict):
    clip_index: int
    start_sec: float
    end_sec: float
    embedding: list[float]
    quality_flag: str | None


class JoinedClipRow(TypedDict):
    video_id: int
    clip_index: int
    start_sec: float
    end_sec: float
    embedding: list[float]
    task_id: str
    is_gold: bool


class ScoreJobRow(TypedDict):
    id: int
    gold_video_id: int
    trainee_video_id: int
    status: str
    score: ScoreResult | None
    weights: dict[str, float] | None
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None
    error: str | None


class ScoreJobInputRow(TypedDict):
    id: int
    gold_video_id: int
    trainee_video_id: int
    status: str
    weights: dict[str, float] | None


class TaskProfileRow(TypedDict):
    task_id: str
    task_name: str
    pass_score: float
    retrain_score: float
    default_weights: dict[str, float]
    deviation_policy: dict[str, str]


class ScoreReviewRow(TypedDict):
    job_id: int
    verdict: str
    note: str | None
    created_at: str
    updated_at: str
