"""Extended tests for insurance_mvp.cosmos.client — VideoLLMClient.

Focuses on coverage gaps in the mock backend path:
- VLMConfig extended attributes
- health_check (mock vs real)
- _validate_and_construct edge cases (missing fields, validation failures)
- _extract_fields_from_text edge cases
- _run_inference dispatching
- _run_inference_with_retry logic
- create_client factory
- _load_cosmos raises
- assess_claim clip-duration limiting
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from insurance_mvp.cosmos.client import (
    VideoLLMClient,
    VLMConfig,
    create_client,
)
from insurance_mvp.insurance.schema import ClaimAssessment, create_default_claim_assessment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    """Create a VideoLLMClient in mock mode."""
    return create_client(model_name="mock", device="cpu")


@pytest.fixture
def test_video(tmp_path):
    """Create a short test video: 3 seconds @ 30 FPS."""
    video_path = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
    for i in range(90):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :] = (i % 256, 0, 0)
        writer.write(frame)
    writer.release()
    return video_path


# ---------------------------------------------------------------------------
# VLMConfig extended attributes
# ---------------------------------------------------------------------------


class TestVLMConfigExtended:

    def test_gpu_cleanup_default(self):
        config = VLMConfig()
        assert config.gpu_cleanup is True

    def test_jpeg_quality(self):
        config = VLMConfig()
        assert config.jpeg_quality == 75

    def test_max_clip_duration(self):
        config = VLMConfig()
        assert config.max_clip_duration_sec == 60.0

    def test_enable_cpu_fallback(self):
        config = VLMConfig()
        assert config.enable_cpu_fallback is True

    def test_quantize_default_none(self):
        config = VLMConfig()
        assert config.quantize is None

    def test_custom_quantize(self):
        config = VLMConfig(quantize="int4")
        assert config.quantize == "int4"


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:

    def test_mock_mode_ok(self, mock_client):
        result = mock_client.health_check()
        assert result["status"] == "ok"
        assert result["is_mock"] is True
        assert result["model_loaded"] is False

    def test_mock_mode_fields(self, mock_client):
        result = mock_client.health_check()
        assert "model_name" in result
        assert "device" in result
        assert "backend" in result


# ---------------------------------------------------------------------------
# _validate_and_construct
# ---------------------------------------------------------------------------


class TestValidateAndConstruct:

    def test_missing_fault_assessment_gets_default(self, mock_client):
        """When fault_assessment is None, defaults are inserted."""
        data = {
            "severity": "LOW",
            "confidence": 0.8,
            "prediction_set": ["LOW"],
            "review_priority": "LOW_PRIORITY",
            "fraud_risk": {"risk_score": 0.0, "indicators": [], "reasoning": "OK"},
            "hazards": [],
            "evidence": [],
            "causal_reasoning": "Test",
            "recommended_action": "APPROVE",
        }
        assessment = mock_client._validate_and_construct(data, "v1", 1.0)
        assert isinstance(assessment, ClaimAssessment)
        assert assessment.fault_assessment.fault_ratio == 50.0

    def test_missing_fraud_risk_gets_default(self, mock_client):
        """When fraud_risk is None, defaults are inserted."""
        data = {
            "severity": "MEDIUM",
            "confidence": 0.7,
            "prediction_set": ["MEDIUM"],
            "review_priority": "STANDARD",
            "fault_assessment": {
                "fault_ratio": 30.0,
                "reasoning": "X",
                "scenario_type": "test",
            },
            "hazards": [],
            "evidence": [],
            "causal_reasoning": "Test",
            "recommended_action": "REVIEW",
        }
        assessment = mock_client._validate_and_construct(data, "v1", 1.0)
        assert isinstance(assessment, ClaimAssessment)
        assert assessment.fraud_risk.risk_score == 0.0

    def test_partial_fault_assessment_fills_defaults(self, mock_client):
        """Existing fault_assessment dict gets missing keys filled."""
        data = {
            "severity": "HIGH",
            "confidence": 0.9,
            "prediction_set": ["HIGH"],
            "review_priority": "URGENT",
            "fault_assessment": {},  # empty dict — all defaults
            "fraud_risk": {"risk_score": 0.0, "indicators": [], "reasoning": "OK"},
            "hazards": [],
            "evidence": [],
            "causal_reasoning": "Test",
            "recommended_action": "REVIEW",
        }
        assessment = mock_client._validate_and_construct(data, "v1", 1.0)
        assert assessment.fault_assessment.fault_ratio == 50.0
        assert assessment.fault_assessment.scenario_type == "unknown"

    def test_validation_failure_preserves_severity(self, mock_client):
        """On ValidationError, severity/confidence from parsed data are preserved."""
        data = {
            "severity": "HIGH",
            "confidence": 0.95,
            # Missing required fields -> will cause ValidationError in ClaimAssessment
            # But _validate_and_construct fills defaults, so we need to trigger a real error
            # We'll provide an invalid fault_ratio to trigger it
            "fault_assessment": {"fault_ratio": 999.0, "reasoning": "X", "scenario_type": "t"},
            "fraud_risk": {"risk_score": 0.0, "indicators": [], "reasoning": "OK"},
            "prediction_set": ["HIGH"],
            "review_priority": "URGENT",
            "hazards": [],
            "evidence": [],
            "causal_reasoning": "Test",
            "recommended_action": "REVIEW",
        }
        assessment = mock_client._validate_and_construct(data, "v2", 1.0)
        # fault_ratio=999 exceeds le=100 constraint -> fallback to default
        # but severity should be preserved from parsed data
        assert isinstance(assessment, ClaimAssessment)
        assert assessment.severity == "HIGH"
        assert assessment.confidence == 0.95


# ---------------------------------------------------------------------------
# _extract_fields_from_text
# ---------------------------------------------------------------------------


class TestExtractFieldsFromText:

    def test_extract_severity_and_confidence(self, mock_client):
        text = '"severity": "MEDIUM" and "confidence": 0.88'
        data = mock_client._extract_fields_from_text(text)
        assert data["severity"] == "MEDIUM"
        assert data["confidence"] == pytest.approx(0.88)

    def test_extract_fault_ratio(self, mock_client):
        text = '"fault_ratio": 75.5 and "reasoning": "Driver at fault"'
        data = mock_client._extract_fields_from_text(text)
        assert "fault_assessment" in data
        assert data["fault_assessment"]["fault_ratio"] == pytest.approx(75.5)
        assert data["causal_reasoning"] == "Driver at fault"

    def test_no_fields_found(self, mock_client):
        text = "This text has no recognizable fields whatsoever."
        data = mock_client._extract_fields_from_text(text)
        assert data == {}


# ---------------------------------------------------------------------------
# _run_inference dispatch
# ---------------------------------------------------------------------------


class TestRunInference:

    def test_mock_dispatch(self, mock_client):
        """_run_inference delegates to _mock_inference for mock mode."""
        result = mock_client._run_inference([], "test prompt")
        # Should return valid JSON string
        parsed = json.loads(result)
        assert "severity" in parsed

    def test_unknown_model_raises(self):
        """_run_inference raises for unsupported model name."""
        config = VLMConfig(model_name="mock")
        client = VideoLLMClient(config)
        # Temporarily override model_name to something unsupported
        client.config.model_name = "nvidia-cosmos-reason-2"
        with pytest.raises(RuntimeError, match="Inference not implemented"):
            client._run_inference([], "prompt")


# ---------------------------------------------------------------------------
# _run_inference_with_retry
# ---------------------------------------------------------------------------


class TestRunInferenceWithRetry:

    def test_retry_exhaustion(self, mock_client, monkeypatch):
        """All retries fail -> RuntimeError."""
        call_count = [0]

        def always_fail(*args, **kwargs):
            call_count[0] += 1
            raise RuntimeError("fail")

        monkeypatch.setattr(mock_client, "_run_inference", always_fail)
        with pytest.raises(RuntimeError, match="All 2 inference attempts failed"):
            mock_client._run_inference_with_retry([], "prompt", max_retries=2)
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# _load_cosmos
# ---------------------------------------------------------------------------


class TestLoadCosmos:

    def test_cosmos_not_implemented(self):
        config = VLMConfig(model_name="mock")
        client = VideoLLMClient(config)
        with pytest.raises(RuntimeError, match="not yet implemented"):
            client._load_cosmos()


# ---------------------------------------------------------------------------
# create_client factory
# ---------------------------------------------------------------------------


class TestCreateClient:

    def test_mock_factory(self):
        client = create_client(model_name="mock", device="cpu")
        assert client.config.model_name == "mock"
        assert client.config.device == "cpu"
        assert client.config.dtype == "float16"

    def test_qwen_factory_dtype(self):
        """create_client('qwen2.5-vl-7b') sets dtype=bfloat16."""
        # We can't actually load the model, but we can check the config
        # by patching _ensure_model_loaded
        with patch.object(VideoLLMClient, "_ensure_model_loaded"):
            client = create_client(model_name="qwen2.5-vl-7b", device="cpu")
            assert client.config.dtype == "bfloat16"


# ---------------------------------------------------------------------------
# assess_claim clip-duration limiting
# ---------------------------------------------------------------------------


class TestAssessClaimClipDuration:

    def test_clip_duration_limit_applied(self, mock_client, test_video):
        """When video exceeds max_clip_duration_sec, end_sec is clamped."""
        mock_client.config.max_clip_duration_sec = 1.0  # 1 second limit
        assessment = mock_client.assess_claim(test_video, "clip_test")
        assert isinstance(assessment, ClaimAssessment)
        assert assessment.video_id == "clip_test"

    def test_full_pipeline_mock(self, mock_client, test_video):
        """End-to-end assess_claim in mock mode produces valid output."""
        assessment = mock_client.assess_claim(test_video, "e2e_mock")
        assert isinstance(assessment, ClaimAssessment)
        assert assessment.severity in {"NONE", "LOW", "MEDIUM", "HIGH"}
        assert 0.0 <= assessment.confidence <= 1.0
        assert assessment.processing_time_sec > 0.0
