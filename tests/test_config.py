from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from conftest import make_test_settings
from sopilot.config import Settings, _env_bool, _env_float, _env_int, _env_str


class TestEnvInt:
    def test_returns_default_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _env_int("MISSING_KEY", 42) == 42

    def test_returns_parsed_value(self) -> None:
        with patch.dict(os.environ, {"MY_INT": "99"}):
            assert _env_int("MY_INT", 0) == 99

    def test_returns_default_on_invalid(self) -> None:
        with patch.dict(os.environ, {"MY_INT": "not_a_number"}):
            assert _env_int("MY_INT", 7) == 7

    def test_handles_negative(self) -> None:
        with patch.dict(os.environ, {"MY_INT": "-5"}):
            assert _env_int("MY_INT", 0) == -5

    def test_handles_zero(self) -> None:
        with patch.dict(os.environ, {"MY_INT": "0"}):
            assert _env_int("MY_INT", 99) == 0


class TestEnvFloat:
    def test_returns_default_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _env_float("MISSING_KEY", 3.14) == 3.14

    def test_returns_parsed_value(self) -> None:
        with patch.dict(os.environ, {"MY_FLOAT": "2.718"}):
            assert abs(_env_float("MY_FLOAT", 0.0) - 2.718) < 1e-6

    def test_returns_default_on_invalid(self) -> None:
        with patch.dict(os.environ, {"MY_FLOAT": "abc"}):
            assert _env_float("MY_FLOAT", 1.0) == 1.0

    def test_handles_scientific_notation(self) -> None:
        with patch.dict(os.environ, {"MY_FLOAT": "1e-6"}):
            assert abs(_env_float("MY_FLOAT", 0.0) - 1e-6) < 1e-12


class TestEnvBool:
    def test_returns_default_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _env_bool("MISSING_KEY", True) is True
            assert _env_bool("MISSING_KEY", False) is False

    def test_true_variants(self) -> None:
        for val in ["1", "true", "True", "TRUE", "yes", "YES", "on", "ON"]:
            with patch.dict(os.environ, {"MY_BOOL": val}):
                assert _env_bool("MY_BOOL", False) is True, f"failed for {val!r}"

    def test_false_variants(self) -> None:
        for val in ["0", "false", "False", "FALSE", "no", "NO", "off", "OFF"]:
            with patch.dict(os.environ, {"MY_BOOL": val}):
                assert _env_bool("MY_BOOL", True) is False, f"failed for {val!r}"

    def test_returns_default_on_unrecognized(self) -> None:
        with patch.dict(os.environ, {"MY_BOOL": "maybe"}):
            assert _env_bool("MY_BOOL", True) is True
            assert _env_bool("MY_BOOL", False) is False

    def test_strips_whitespace(self) -> None:
        with patch.dict(os.environ, {"MY_BOOL": "  true  "}):
            assert _env_bool("MY_BOOL", False) is True


class TestEnvStr:
    def test_returns_default_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _env_str("MISSING_KEY", "default") == "default"

    def test_returns_value_as_is(self) -> None:
        with patch.dict(os.environ, {"MY_STR": "  hello  "}):
            assert _env_str("MY_STR", "") == "  hello  "

    def test_returns_empty_string(self) -> None:
        with patch.dict(os.environ, {"MY_STR": ""}):
            assert _env_str("MY_STR", "fallback") == ""


# ---------------------------------------------------------------------------
# Settings __post_init__ validation
# ---------------------------------------------------------------------------


def _valid_settings(**overrides) -> Settings:
    return make_test_settings(**overrides)


class TestSettingsValidation:
    def test_valid_defaults(self) -> None:
        s = _valid_settings()
        assert s.target_fps == 4

    def test_target_fps_zero(self) -> None:
        with pytest.raises(ValueError, match="target_fps"):
            _valid_settings(target_fps=0)

    def test_clip_seconds_negative(self) -> None:
        with pytest.raises(ValueError, match="clip_seconds"):
            _valid_settings(clip_seconds=-1.0)

    def test_max_side_zero(self) -> None:
        with pytest.raises(ValueError, match="max_side"):
            _valid_settings(max_side=0)

    def test_min_clip_coverage_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="min_clip_coverage"):
            _valid_settings(min_clip_coverage=1.5)

    def test_upload_max_mb_zero(self) -> None:
        with pytest.raises(ValueError, match="upload_max_mb"):
            _valid_settings(upload_max_mb=0)

    def test_low_similarity_threshold_too_high(self) -> None:
        with pytest.raises(ValueError, match="low_similarity_threshold"):
            _valid_settings(low_similarity_threshold=2.0)

    def test_negative_weight(self) -> None:
        with pytest.raises(ValueError, match="w_miss"):
            _valid_settings(w_miss=-1.0)

    def test_nightly_hour_too_high(self) -> None:
        with pytest.raises(ValueError, match="nightly_hour_local"):
            _valid_settings(nightly_hour_local=25)

    def test_invalid_queue_backend(self) -> None:
        with pytest.raises(ValueError, match="queue_backend"):
            _valid_settings(queue_backend="celery")

    def test_invalid_privacy_mask_mode(self) -> None:
        with pytest.raises(ValueError, match="privacy_mask_mode"):
            _valid_settings(privacy_mask_mode="invert")

    def test_invalid_watch_role(self) -> None:
        with pytest.raises(ValueError, match="watch_role"):
            _valid_settings(watch_role="unknown_role")

    def test_empty_watch_role_allowed(self) -> None:
        s = _valid_settings(watch_role="")
        assert s.watch_role == ""

    def test_empty_privacy_mask_mode_allowed(self) -> None:
        s = _valid_settings(privacy_mask_mode="")
        assert s.privacy_mask_mode == ""

    def test_multiple_errors(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _valid_settings(target_fps=0, clip_seconds=-1.0)
        msg = str(exc_info.value)
        assert "target_fps" in msg
        assert "clip_seconds" in msg

    def test_rq_job_timeout_zero(self) -> None:
        with pytest.raises(ValueError, match="rq_job_timeout_sec"):
            _valid_settings(rq_job_timeout_sec=0)

    def test_adapt_timeout_zero(self) -> None:
        with pytest.raises(ValueError, match="adapt_timeout_sec"):
            _valid_settings(adapt_timeout_sec=0)

    def test_zero_weight_allowed(self) -> None:
        s = _valid_settings(w_miss=0.0, w_swap=0.0)
        assert s.w_miss == 0.0

    def test_edge_valid_values(self) -> None:
        s = _valid_settings(
            min_clip_coverage=0.0,
            low_similarity_threshold=0.0,
            nightly_hour_local=0,
        )
        assert s.min_clip_coverage == 0.0
        s2 = _valid_settings(
            min_clip_coverage=1.0,
            low_similarity_threshold=1.0,
            nightly_hour_local=23,
        )
        assert s2.nightly_hour_local == 23
