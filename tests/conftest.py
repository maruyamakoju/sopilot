"""Shared test fixtures and helpers for SOPilot tests."""

from __future__ import annotations

from pathlib import Path

from sopilot.config import Settings


def make_test_settings(data_dir: Path | None = None, **overrides) -> Settings:
    """Build a valid Settings instance for testing with sensible defaults.

    Accepts the same keyword arguments as the Settings dataclass.
    Any field can be overridden via **overrides.
    """
    if data_dir is None:
        data_dir = overrides.pop("data_dir", Path("/tmp/sopilot_test"))
    else:
        overrides.pop("data_dir", None)

    defaults = dict(
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        embeddings_dir=data_dir / "embeddings",
        reports_dir=data_dir / "reports",
        index_dir=data_dir / "index",
        models_dir=data_dir / "models",
        db_path=data_dir / "sopilot.db",
        target_fps=4,
        clip_seconds=4.0,
        max_side=320,
        min_clip_coverage=0.6,
        ingest_embed_batch_size=8,
        upload_max_mb=512,
        min_scoring_clips=4,
        change_threshold_factor=1.0,
        min_step_clips=2,
        low_similarity_threshold=0.75,
        w_miss=12.0,
        w_swap=8.0,
        w_dev=30.0,
        w_time=15.0,
        w_warp=12.0,
        embedder_backend="heuristic",
        embedder_fallback_enabled=True,
        embedding_device="auto",
        vjepa2_repo="",
        vjepa2_variant="",
        vjepa2_pretrained=True,
        vjepa2_source="hub",
        vjepa2_local_repo="",
        vjepa2_local_checkpoint="",
        vjepa2_num_frames=64,
        vjepa2_image_size=256,
        vjepa2_batch_size=2,
        queue_backend="inline",
        redis_url="redis://127.0.0.1:6379/0",
        rq_queue_prefix="sopilot",
        rq_job_timeout_sec=21600,
        rq_result_ttl_sec=0,
        rq_failure_ttl_sec=604800,
        rq_retry_max=2,
        score_worker_threads=1,
        training_worker_threads=1,
        nightly_enabled=False,
        nightly_hour_local=2,
        nightly_min_new_videos=10,
        nightly_check_interval_sec=30,
        adapt_command="",
        adapt_timeout_sec=14400,
        enable_feature_adapter=True,
        report_title="SOPilot Test",
        auth_required=False,
        api_token="",
        api_token_role="admin",
        api_role_tokens="",
        basic_user="",
        basic_password="",
        basic_role="admin",
        auth_default_role="admin",
        audit_signing_key="",
        audit_signing_key_id="local",
        privacy_mask_enabled=False,
        privacy_mask_mode="black",
        privacy_mask_rects="",
        privacy_face_blur=False,
        watch_enabled=False,
        watch_dir=data_dir / "watch_inbox",
        watch_poll_sec=5,
        watch_task_id="",
        watch_role="trainee",
        dtw_use_gpu=True,
        neural_mode=False,
        neural_model_dir=data_dir / "models" / "neural",
        neural_projection_enabled=True,
        neural_segmenter_enabled=True,
        neural_scoring_enabled=True,
        neural_soft_dtw_enabled=True,
        neural_soft_dtw_gamma=1.0,
        neural_uncertainty_samples=30,
        neural_calibration_enabled=True,
        neural_device="cpu",
        neural_asformer_enabled=True,
        neural_ot_alignment=False,
        neural_cuda_dtw=True,
        neural_conformal_alpha=0.1,
        neural_dilate_alpha=0.5,
    )
    defaults.update(overrides)
    return Settings(**defaults)
