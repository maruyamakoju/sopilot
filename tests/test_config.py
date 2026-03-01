"""Tests for sopilot.config.Settings — defaults, env-var overrides, and validation."""
import os
from pathlib import Path

import pytest

from sopilot.config import Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_env(monkeypatch, keys=None):
    """Remove all SOPILOT_ env vars so from_env() uses pure defaults."""
    all_keys = keys or [k for k in os.environ if k.startswith("SOPILOT_")]
    for k in all_keys:
        monkeypatch.delenv(k, raising=False)


def _base_settings(tmp_path: Path) -> Settings:
    """Return a minimal valid Settings pointing at *tmp_path*."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        data_dir=data_dir,
        raw_video_dir=raw_dir,
        ui_dir=tmp_path / "ui",
        database_path=data_dir / "sopilot.db",
    )


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_sample_fps(self, tmp_path):
        s = _base_settings(tmp_path)
        assert s.sample_fps == 4

    def test_default_clip_seconds(self, tmp_path):
        s = _base_settings(tmp_path)
        assert s.clip_seconds == 4

    def test_default_frame_size(self, tmp_path):
        s = _base_settings(tmp_path)
        assert s.frame_size == 256

    def test_default_deviation_threshold(self, tmp_path):
        s = _base_settings(tmp_path)
        assert s.deviation_threshold == 0.25

    def test_default_embedder_backend(self, tmp_path):
        s = _base_settings(tmp_path)
        assert s.embedder_backend == "vjepa2"

    def test_default_log_level(self, tmp_path):
        s = _base_settings(tmp_path)
        assert s.log_level == "INFO"

    def test_default_pass_score(self, tmp_path):
        s = _base_settings(tmp_path)
        assert s.default_pass_score == 60.0

    def test_default_retrain_score(self, tmp_path):
        s = _base_settings(tmp_path)
        assert s.default_retrain_score == 50.0

    def test_default_api_key_is_none(self, tmp_path):
        s = _base_settings(tmp_path)
        assert s.api_key is None

    def test_default_cors_origins_is_empty_tuple(self, tmp_path):
        s = _base_settings(tmp_path)
        # Default is an empty tuple (which is falsy)
        assert s.cors_origins == ()

    def test_default_max_upload_mb(self, tmp_path):
        s = _base_settings(tmp_path)
        assert s.max_upload_mb == 500

    def test_default_score_job_max_retries(self, tmp_path):
        s = _base_settings(tmp_path)
        assert s.score_job_max_retries == 2


# ---------------------------------------------------------------------------
# Env-var overrides via from_env()
# ---------------------------------------------------------------------------

class TestEnvOverrides:
    def test_sample_fps_override(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SOPILOT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("SOPILOT_SAMPLE_FPS", "8")
        s = Settings.from_env()
        assert s.sample_fps == 8

    def test_log_level_override(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SOPILOT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("SOPILOT_LOG_LEVEL", "DEBUG")
        s = Settings.from_env()
        assert s.log_level == "DEBUG"

    def test_log_level_case_insensitive(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SOPILOT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("SOPILOT_LOG_LEVEL", "warning")
        s = Settings.from_env()
        assert s.log_level == "WARNING"

    def test_embedder_backend_color_motion(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SOPILOT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("SOPILOT_EMBEDDER_BACKEND", "color-motion")
        s = Settings.from_env()
        assert s.embedder_backend == "color-motion"

    def test_deviation_threshold_override(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SOPILOT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("SOPILOT_DEVIATION_THRESHOLD", "0.5")
        s = Settings.from_env()
        assert s.deviation_threshold == 0.5

    def test_api_key_set(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SOPILOT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("SOPILOT_API_KEY", "secret-key")
        s = Settings.from_env()
        assert s.api_key == "secret-key"

    def test_api_key_empty_string_becomes_none(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SOPILOT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("SOPILOT_API_KEY", "")
        s = Settings.from_env()
        assert s.api_key is None

    def test_api_key_whitespace_only_becomes_none(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SOPILOT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("SOPILOT_API_KEY", "   ")
        s = Settings.from_env()
        assert s.api_key is None

    def test_pass_score_override(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SOPILOT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("SOPILOT_DEFAULT_PASS_SCORE", "85.0")
        s = Settings.from_env()
        assert s.default_pass_score == 85.0

    def test_max_upload_mb_override(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SOPILOT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("SOPILOT_MAX_UPLOAD_MB", "1000")
        s = Settings.from_env()
        assert s.max_upload_mb == 1000


# ---------------------------------------------------------------------------
# CORS origins parsing
# ---------------------------------------------------------------------------

class TestCorsOrigins:
    def test_cors_comma_separated(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SOPILOT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("SOPILOT_CORS_ORIGINS", "http://a.com,http://b.com")
        s = Settings.from_env()
        assert s.cors_origins == ["http://a.com", "http://b.com"]

    def test_cors_wildcard(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SOPILOT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("SOPILOT_CORS_ORIGINS", "*")
        s = Settings.from_env()
        assert s.cors_origins == ["*"]

    def test_cors_defaults_when_empty(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SOPILOT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("SOPILOT_CORS_ORIGINS", "")
        s = Settings.from_env()
        assert "http://localhost:8000" in s.cors_origins
        assert "http://127.0.0.1:8000" in s.cors_origins

    def test_cors_strips_whitespace(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SOPILOT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("SOPILOT_CORS_ORIGINS", " http://x.com , http://y.com ")
        s = Settings.from_env()
        assert s.cors_origins == ["http://x.com", "http://y.com"]


# ---------------------------------------------------------------------------
# Cross-field validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_retrain_above_pass_raises(self, tmp_path):
        with pytest.raises(ValueError, match="default_retrain_score"):
            Settings(
                data_dir=tmp_path,
                raw_video_dir=tmp_path,
                ui_dir=tmp_path,
                database_path=tmp_path / "db",
                default_pass_score=80.0,
                default_retrain_score=90.0,
            )

    def test_deviation_threshold_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="deviation_threshold"):
            Settings(
                data_dir=tmp_path,
                raw_video_dir=tmp_path,
                ui_dir=tmp_path,
                database_path=tmp_path / "db",
                deviation_threshold=0.0,
            )

    def test_deviation_threshold_negative_raises(self, tmp_path):
        with pytest.raises(ValueError, match="deviation_threshold"):
            Settings(
                data_dir=tmp_path,
                raw_video_dir=tmp_path,
                ui_dir=tmp_path,
                database_path=tmp_path / "db",
                deviation_threshold=-0.1,
            )

    def test_boundary_z_threshold_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="boundary_z_threshold"):
            Settings(
                data_dir=tmp_path,
                raw_video_dir=tmp_path,
                ui_dir=tmp_path,
                database_path=tmp_path / "db",
                boundary_z_threshold=0.0,
            )

    def test_invalid_log_level_raises(self, tmp_path):
        with pytest.raises(ValueError, match="log_level"):
            Settings(
                data_dir=tmp_path,
                raw_video_dir=tmp_path,
                ui_dir=tmp_path,
                database_path=tmp_path / "db",
                log_level="VERBOSE",
            )

    def test_valid_log_levels_accepted(self, tmp_path):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            s = Settings(
                data_dir=tmp_path,
                raw_video_dir=tmp_path,
                ui_dir=tmp_path,
                database_path=tmp_path / "db",
                log_level=level,
            )
            assert s.log_level == level

    def test_retrain_equal_to_pass_is_ok(self, tmp_path):
        s = Settings(
            data_dir=tmp_path,
            raw_video_dir=tmp_path,
            ui_dir=tmp_path,
            database_path=tmp_path / "db",
            default_pass_score=85.0,
            default_retrain_score=85.0,
        )
        assert s.default_pass_score == 85.0
        assert s.default_retrain_score == 85.0
