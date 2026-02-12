from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return default


def _env_str(key: str, default: str) -> str:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    raw_dir: Path
    embeddings_dir: Path
    reports_dir: Path
    index_dir: Path
    models_dir: Path
    db_path: Path
    target_fps: int
    clip_seconds: float
    max_side: int
    min_clip_coverage: float
    ingest_embed_batch_size: int
    upload_max_mb: int
    min_scoring_clips: int
    change_threshold_factor: float
    min_step_clips: int
    low_similarity_threshold: float
    w_miss: float
    w_swap: float
    w_dev: float
    w_time: float
    w_warp: float
    embedder_backend: str
    embedder_fallback_enabled: bool
    embedding_device: str
    vjepa2_repo: str
    vjepa2_variant: str
    vjepa2_pretrained: bool
    vjepa2_source: str
    vjepa2_local_repo: str
    vjepa2_local_checkpoint: str
    vjepa2_num_frames: int
    vjepa2_image_size: int
    vjepa2_batch_size: int
    queue_backend: str
    redis_url: str
    rq_queue_prefix: str
    rq_job_timeout_sec: int
    rq_result_ttl_sec: int
    rq_failure_ttl_sec: int
    rq_retry_max: int
    score_worker_threads: int
    training_worker_threads: int
    nightly_enabled: bool
    nightly_hour_local: int
    nightly_min_new_videos: int
    nightly_check_interval_sec: int
    adapt_command: str
    adapt_timeout_sec: int
    enable_feature_adapter: bool
    report_title: str
    auth_required: bool
    api_token: str
    api_token_role: str
    api_role_tokens: str
    basic_user: str
    basic_password: str
    basic_role: str
    auth_default_role: str
    audit_signing_key: str
    audit_signing_key_id: str
    privacy_mask_enabled: bool
    privacy_mask_mode: str
    privacy_mask_rects: str
    privacy_face_blur: bool
    watch_enabled: bool
    watch_dir: Path
    watch_poll_sec: int
    watch_task_id: str
    watch_role: str
    dtw_use_gpu: bool

    # Neural mode settings
    neural_mode: bool
    neural_model_dir: Path
    neural_projection_enabled: bool
    neural_segmenter_enabled: bool
    neural_scoring_enabled: bool
    neural_soft_dtw_enabled: bool
    neural_soft_dtw_gamma: float
    neural_uncertainty_samples: int
    neural_calibration_enabled: bool
    neural_device: str
    neural_asformer_enabled: bool
    neural_ot_alignment: bool
    neural_cuda_dtw: bool
    neural_conformal_alpha: float
    neural_dilate_alpha: float

    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.target_fps < 1:
            errors.append(f"target_fps must be >= 1, got {self.target_fps}")
        if self.clip_seconds <= 0:
            errors.append(f"clip_seconds must be > 0, got {self.clip_seconds}")
        if self.max_side < 1:
            errors.append(f"max_side must be >= 1, got {self.max_side}")
        if not (0.0 <= self.min_clip_coverage <= 1.0):
            errors.append(f"min_clip_coverage must be in [0, 1], got {self.min_clip_coverage}")
        if self.ingest_embed_batch_size < 1:
            errors.append(f"ingest_embed_batch_size must be >= 1, got {self.ingest_embed_batch_size}")
        if self.upload_max_mb < 1:
            errors.append(f"upload_max_mb must be >= 1, got {self.upload_max_mb}")
        if self.min_scoring_clips < 1:
            errors.append(f"min_scoring_clips must be >= 1, got {self.min_scoring_clips}")
        if self.min_step_clips < 1:
            errors.append(f"min_step_clips must be >= 1, got {self.min_step_clips}")
        if not (0.0 <= self.low_similarity_threshold <= 1.0):
            errors.append(f"low_similarity_threshold must be in [0, 1], got {self.low_similarity_threshold}")
        for name in ("w_miss", "w_swap", "w_dev", "w_time", "w_warp"):
            if getattr(self, name) < 0:
                errors.append(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.nightly_hour_local < 0 or self.nightly_hour_local > 23:
            errors.append(f"nightly_hour_local must be in [0, 23], got {self.nightly_hour_local}")
        if self.rq_job_timeout_sec < 1:
            errors.append(f"rq_job_timeout_sec must be >= 1, got {self.rq_job_timeout_sec}")
        if self.adapt_timeout_sec < 1:
            errors.append(f"adapt_timeout_sec must be >= 1, got {self.adapt_timeout_sec}")
        if self.queue_backend not in {"inline", "rq"}:
            errors.append(f"queue_backend must be 'inline' or 'rq', got {self.queue_backend!r}")
        if self.privacy_mask_mode and self.privacy_mask_mode not in {"black", "blur", "pixelate"}:
            errors.append(f"privacy_mask_mode must be black|blur|pixelate, got {self.privacy_mask_mode!r}")
        if self.watch_role and self.watch_role not in {"gold", "trainee", "audit"}:
            errors.append(f"watch_role must be gold|trainee|audit, got {self.watch_role!r}")
        if self.neural_soft_dtw_gamma <= 0:
            errors.append(f"neural_soft_dtw_gamma must be > 0, got {self.neural_soft_dtw_gamma}")
        if self.neural_uncertainty_samples < 1:
            errors.append(f"neural_uncertainty_samples must be >= 1, got {self.neural_uncertainty_samples}")
        if not (0.0 < self.neural_conformal_alpha < 1.0):
            errors.append(f"neural_conformal_alpha must be in (0, 1), got {self.neural_conformal_alpha}")
        if not (0.0 <= self.neural_dilate_alpha <= 1.0):
            errors.append(f"neural_dilate_alpha must be in [0, 1], got {self.neural_dilate_alpha}")
        if errors:
            raise ValueError("Settings validation failed:\n  - " + "\n  - ".join(errors))


def get_settings() -> Settings:
    data_dir = Path(os.getenv("SOPILOT_DATA_DIR", "data")).resolve()
    raw_dir = data_dir / "raw"
    embeddings_dir = data_dir / "embeddings"
    reports_dir = data_dir / "reports"
    index_dir = data_dir / "index"
    models_dir = data_dir / "models"

    return Settings(
        data_dir=data_dir,
        raw_dir=raw_dir,
        embeddings_dir=embeddings_dir,
        reports_dir=reports_dir,
        index_dir=index_dir,
        models_dir=models_dir,
        db_path=data_dir / "sopilot.db",
        target_fps=_env_int("SOPILOT_TARGET_FPS", 4),
        clip_seconds=_env_float("SOPILOT_CLIP_SECONDS", 4.0),
        max_side=_env_int("SOPILOT_MAX_SIDE", 320),
        min_clip_coverage=_env_float("SOPILOT_MIN_CLIP_COVERAGE", 0.6),
        ingest_embed_batch_size=_env_int("SOPILOT_INGEST_EMBED_BATCH_SIZE", 8),
        upload_max_mb=_env_int("SOPILOT_UPLOAD_MAX_MB", 1024),
        min_scoring_clips=_env_int("SOPILOT_MIN_SCORING_CLIPS", 4),
        change_threshold_factor=_env_float("SOPILOT_CHANGE_THRESHOLD_FACTOR", 1.0),
        min_step_clips=_env_int("SOPILOT_MIN_STEP_CLIPS", 2),
        low_similarity_threshold=_env_float("SOPILOT_LOW_SIM_THRESHOLD", 0.75),
        w_miss=_env_float("SOPILOT_WEIGHT_MISS", 12.0),
        w_swap=_env_float("SOPILOT_WEIGHT_SWAP", 8.0),
        w_dev=_env_float("SOPILOT_WEIGHT_DEV", 30.0),
        w_time=_env_float("SOPILOT_WEIGHT_TIME", 15.0),
        w_warp=_env_float("SOPILOT_WEIGHT_WARP", 12.0),
        embedder_backend=_env_str("SOPILOT_EMBEDDER_BACKEND", "auto"),
        embedder_fallback_enabled=_env_bool("SOPILOT_EMBEDDER_FALLBACK", True),
        embedding_device=_env_str("SOPILOT_EMBEDDING_DEVICE", "auto"),
        vjepa2_repo=_env_str("SOPILOT_VJEPA2_REPO", "facebookresearch/vjepa2"),
        vjepa2_variant=_env_str("SOPILOT_VJEPA2_VARIANT", "vjepa2_vit_large"),
        vjepa2_pretrained=_env_bool("SOPILOT_VJEPA2_PRETRAINED", True),
        vjepa2_source=_env_str("SOPILOT_VJEPA2_SOURCE", "hub"),
        vjepa2_local_repo=_env_str("SOPILOT_VJEPA2_LOCAL_REPO", "").strip(),
        vjepa2_local_checkpoint=_env_str("SOPILOT_VJEPA2_LOCAL_CHECKPOINT", "").strip(),
        vjepa2_num_frames=_env_int("SOPILOT_VJEPA2_NUM_FRAMES", 64),
        vjepa2_image_size=_env_int("SOPILOT_VJEPA2_IMAGE_SIZE", 256),
        vjepa2_batch_size=_env_int("SOPILOT_VJEPA2_BATCH_SIZE", 2),
        queue_backend=_env_str("SOPILOT_QUEUE_BACKEND", "rq").strip().lower(),
        redis_url=_env_str("SOPILOT_REDIS_URL", "redis://127.0.0.1:6379/0"),
        rq_queue_prefix=_env_str("SOPILOT_RQ_QUEUE_PREFIX", "sopilot").strip(),
        rq_job_timeout_sec=_env_int("SOPILOT_RQ_JOB_TIMEOUT_SEC", 21600),
        rq_result_ttl_sec=_env_int("SOPILOT_RQ_RESULT_TTL_SEC", 0),
        rq_failure_ttl_sec=_env_int("SOPILOT_RQ_FAILURE_TTL_SEC", 604800),
        rq_retry_max=_env_int("SOPILOT_RQ_RETRY_MAX", 2),
        score_worker_threads=_env_int("SOPILOT_SCORE_WORKERS", 2),
        training_worker_threads=_env_int("SOPILOT_TRAIN_WORKERS", 1),
        nightly_enabled=_env_bool("SOPILOT_NIGHTLY_ENABLED", False),
        nightly_hour_local=_env_int("SOPILOT_NIGHTLY_HOUR_LOCAL", 2),
        nightly_min_new_videos=_env_int("SOPILOT_NIGHTLY_MIN_NEW_VIDEOS", 10),
        nightly_check_interval_sec=_env_int("SOPILOT_NIGHTLY_CHECK_SEC", 30),
        adapt_command=_env_str("SOPILOT_ADAPT_COMMAND", "").strip(),
        adapt_timeout_sec=_env_int("SOPILOT_ADAPT_TIMEOUT_SEC", 14400),
        enable_feature_adapter=_env_bool("SOPILOT_ENABLE_FEATURE_ADAPTER", True),
        report_title=_env_str("SOPILOT_REPORT_TITLE", "SOPilot Audit Report"),
        auth_required=_env_bool("SOPILOT_AUTH_REQUIRED", True),
        api_token=_env_str("SOPILOT_API_TOKEN", "").strip(),
        api_token_role=_env_str("SOPILOT_API_TOKEN_ROLE", "admin").strip().lower(),
        api_role_tokens=_env_str("SOPILOT_API_ROLE_TOKENS", "").strip(),
        basic_user=_env_str("SOPILOT_BASIC_USER", "").strip(),
        basic_password=_env_str("SOPILOT_BASIC_PASSWORD", "").strip(),
        basic_role=_env_str("SOPILOT_BASIC_ROLE", "admin").strip().lower(),
        auth_default_role=_env_str("SOPILOT_AUTH_DEFAULT_ROLE", "admin").strip().lower(),
        audit_signing_key=_env_str("SOPILOT_AUDIT_SIGNING_KEY", "").strip(),
        audit_signing_key_id=_env_str("SOPILOT_AUDIT_SIGNING_KEY_ID", "local").strip(),
        privacy_mask_enabled=_env_bool("SOPILOT_PRIVACY_MASK_ENABLED", False),
        privacy_mask_mode=_env_str("SOPILOT_PRIVACY_MASK_MODE", "black").strip().lower(),
        privacy_mask_rects=_env_str("SOPILOT_PRIVACY_MASK_RECTS", "").strip(),
        privacy_face_blur=_env_bool("SOPILOT_PRIVACY_FACE_BLUR", False),
        watch_enabled=_env_bool("SOPILOT_WATCH_ENABLED", False),
        watch_dir=Path(_env_str("SOPILOT_WATCH_DIR", str(data_dir / "watch_inbox"))).resolve(),
        watch_poll_sec=_env_int("SOPILOT_WATCH_POLL_SEC", 5),
        watch_task_id=_env_str("SOPILOT_WATCH_TASK_ID", "").strip(),
        watch_role=_env_str("SOPILOT_WATCH_ROLE", "trainee").strip().lower(),
        dtw_use_gpu=_env_bool("SOPILOT_DTW_USE_GPU", True),
        # Neural mode
        neural_mode=_env_bool("SOPILOT_NEURAL_MODE", False),
        neural_model_dir=Path(_env_str("SOPILOT_NEURAL_MODEL_DIR", str(data_dir / "models" / "neural"))).resolve(),
        neural_projection_enabled=_env_bool("SOPILOT_NEURAL_PROJECTION", True),
        neural_segmenter_enabled=_env_bool("SOPILOT_NEURAL_SEGMENTER", True),
        neural_scoring_enabled=_env_bool("SOPILOT_NEURAL_SCORING", True),
        neural_soft_dtw_enabled=_env_bool("SOPILOT_NEURAL_SOFT_DTW", True),
        neural_soft_dtw_gamma=_env_float("SOPILOT_NEURAL_SOFT_DTW_GAMMA", 1.0),
        neural_uncertainty_samples=_env_int("SOPILOT_NEURAL_UNCERTAINTY_SAMPLES", 30),
        neural_calibration_enabled=_env_bool("SOPILOT_NEURAL_CALIBRATION", True),
        neural_device=_env_str("SOPILOT_NEURAL_DEVICE", "auto").strip().lower(),
        neural_asformer_enabled=_env_bool("SOPILOT_NEURAL_ASFORMER", True),
        neural_ot_alignment=_env_bool("SOPILOT_NEURAL_OT_ALIGNMENT", False),
        neural_cuda_dtw=_env_bool("SOPILOT_NEURAL_CUDA_DTW", True),
        neural_conformal_alpha=_env_float("SOPILOT_NEURAL_CONFORMAL_ALPHA", 0.1),
        neural_dilate_alpha=_env_float("SOPILOT_NEURAL_DILATE_ALPHA", 0.5),
    )
