import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    raw_video_dir: Path
    ui_dir: Path
    database_path: Path
    sample_fps: int = 4
    clip_seconds: int = 4
    frame_size: int = 256
    min_boundary_gap: int = 2
    boundary_z_threshold: float = 1.0
    deviation_threshold: float = 0.25
    embedder_backend: Literal["vjepa2", "color-motion"] = "vjepa2"
    vjepa2_variant: str = "vjepa2_vit_large"
    vjepa2_pretrained: bool = True
    vjepa2_device: str = "auto"
    vjepa2_crop_size: int = 256
    vjepa2_use_amp: bool = True
    vjepa2_pooling: Literal["mean_tokens", "first_token", "flatten"] = "mean_tokens"
    allow_embedder_fallback: bool = True
    score_worker_threads: int = 1
    primary_task_id: str = "pilot_task"
    primary_task_name: str = "PoC Primary Task"
    enforce_primary_task: bool = True
    default_pass_score: float = 60.0
    default_retrain_score: float = 50.0
    efficiency_over_time_threshold: float = 0.2
    log_level: str = "INFO"
    log_json: bool = True
    score_job_max_retries: int = 2
    api_key: str | None = None
    cors_origins: list[str] = ()  # type: ignore[assignment]
    max_upload_mb: int = 500
    rate_limit_rpm: int = 120
    rate_limit_burst: int = 20
    webhook_url: str | None = None

    def __post_init__(self) -> None:
        """Validate cross-field constraints at construction time."""
        if self.default_retrain_score > self.default_pass_score:
            raise ValueError(
                f"default_retrain_score ({self.default_retrain_score}) must be "
                f"<= default_pass_score ({self.default_pass_score})"
            )
        if self.deviation_threshold <= 0:
            raise ValueError(f"deviation_threshold must be > 0, got {self.deviation_threshold}")
        if self.boundary_z_threshold <= 0:
            raise ValueError(f"boundary_z_threshold must be > 0, got {self.boundary_z_threshold}")
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got {self.log_level!r}")

    @staticmethod
    def from_env() -> "Settings":
        base_dir = Path(os.getenv("SOPILOT_DATA_DIR", "data")).resolve()
        raw_video_dir = base_dir / "raw"
        ui_dir = Path(__file__).resolve().parent / "ui"
        database_path = base_dir / "sopilot.db"
        raw_video_dir.mkdir(parents=True, exist_ok=True)
        base_dir.mkdir(parents=True, exist_ok=True)
        backend_env = os.getenv("SOPILOT_EMBEDDER_BACKEND", "vjepa2").strip().lower()
        if backend_env == "color-motion":
            embedder_backend: Literal["vjepa2", "color-motion"] = "color-motion"
        else:
            embedder_backend = "vjepa2"

        _pooling_raw: str = os.getenv("SOPILOT_VJEPA2_POOLING", "mean_tokens").strip().lower()
        _pooling: Literal["mean_tokens", "first_token", "flatten"] = (
            _pooling_raw  # type: ignore[assignment]
            if _pooling_raw in {"mean_tokens", "first_token", "flatten"}
            else "mean_tokens"
        )
        return Settings(
            data_dir=base_dir,
            raw_video_dir=raw_video_dir,
            ui_dir=ui_dir,
            database_path=database_path,
            sample_fps=max(1, int(os.getenv("SOPILOT_SAMPLE_FPS", "4"))),
            clip_seconds=max(1, int(os.getenv("SOPILOT_CLIP_SECONDS", "4"))),
            frame_size=max(64, int(os.getenv("SOPILOT_FRAME_SIZE", "256"))),
            min_boundary_gap=max(1, int(os.getenv("SOPILOT_MIN_BOUNDARY_GAP", "2"))),
            boundary_z_threshold=float(os.getenv("SOPILOT_BOUNDARY_Z", "1.0")),
            deviation_threshold=float(os.getenv("SOPILOT_DEVIATION_THRESHOLD", "0.25")),
            embedder_backend=embedder_backend,
            vjepa2_variant=os.getenv("SOPILOT_VJEPA2_VARIANT", "vjepa2_vit_large"),
            vjepa2_pretrained=_env_bool("SOPILOT_VJEPA2_PRETRAINED", True),
            vjepa2_device=os.getenv("SOPILOT_VJEPA2_DEVICE", "auto"),
            vjepa2_crop_size=max(224, int(os.getenv("SOPILOT_VJEPA2_CROP_SIZE", "256"))),
            vjepa2_use_amp=_env_bool("SOPILOT_VJEPA2_USE_AMP", True),
            vjepa2_pooling=_pooling,
            allow_embedder_fallback=_env_bool("SOPILOT_ALLOW_EMBEDDER_FALLBACK", True),
            score_worker_threads=max(1, int(os.getenv("SOPILOT_SCORE_WORKERS", "1"))),
            primary_task_id=os.getenv("SOPILOT_PRIMARY_TASK_ID", "pilot_task").strip() or "pilot_task",
            primary_task_name=os.getenv("SOPILOT_PRIMARY_TASK_NAME", "PoC Primary Task").strip()
            or "PoC Primary Task",
            enforce_primary_task=_env_bool("SOPILOT_ENFORCE_PRIMARY_TASK", True),
            default_pass_score=float(os.getenv("SOPILOT_DEFAULT_PASS_SCORE", "60.0")),
            default_retrain_score=float(os.getenv("SOPILOT_DEFAULT_RETRAIN_SCORE", "50.0")),
            efficiency_over_time_threshold=float(os.getenv("SOPILOT_EFFICIENCY_OVER_TIME_THRESHOLD", "0.2")),
            log_level=os.getenv("SOPILOT_LOG_LEVEL", "INFO").strip().upper() or "INFO",
            log_json=_env_bool("SOPILOT_LOG_JSON", True),
            score_job_max_retries=max(0, int(os.getenv("SOPILOT_SCORE_JOB_MAX_RETRIES", "2"))),
            api_key=os.getenv("SOPILOT_API_KEY", "").strip() or None,
            cors_origins=[
                o.strip() for o in os.getenv("SOPILOT_CORS_ORIGINS", "").split(",") if o.strip()
            ] or ["http://localhost:8000", "http://127.0.0.1:8000"],
            max_upload_mb=max(1, int(os.getenv("SOPILOT_MAX_UPLOAD_MB", "500"))),
            rate_limit_rpm=max(0, int(os.getenv("SOPILOT_RATE_LIMIT_RPM", "120"))),
            rate_limit_burst=max(0, int(os.getenv("SOPILOT_RATE_LIMIT_BURST", "20"))),
            webhook_url=os.getenv("SOPILOT_WEBHOOK_URL", "").strip() or None,
        )
