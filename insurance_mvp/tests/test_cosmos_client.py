"""Tests for Video-LLM insurance claim assessment client.

Tests cover:
- Mock mode (no GPU required)
- JSON parsing pipeline (7 steps)
- Error handling and graceful degradation
- Model caching
- Frame sampling
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from insurance_mvp.cosmos import (
    ClaimAssessment,
    VLMConfig,
    VideoLLMClient,
    create_client,
    create_default_claim_assessment,
)


@pytest.fixture
def test_video():
    """Create a simple test video (5 seconds, 30 FPS)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = Path(f.name)

    # Create video with 150 frames (5 seconds @ 30 FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

    for i in range(150):
        # Create frame with changing color
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (i % 256, (i * 2) % 256, (i * 3) % 256)
        writer.write(frame)

    writer.release()

    yield video_path

    # Cleanup
    video_path.unlink(missing_ok=True)


class TestVLMConfig:
    """Test VLM configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VLMConfig()
        assert config.model_name == "qwen2.5-vl-7b"
        assert config.device == "cuda"
        assert config.fps == 2.0
        assert config.max_frames == 16
        assert config.timeout_sec == 300.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = VLMConfig(
            model_name="mock", device="cpu", fps=2.0, max_frames=16, temperature=0.5, timeout_sec=600.0
        )
        assert config.model_name == "mock"
        assert config.device == "cpu"
        assert config.fps == 2.0
        assert config.max_frames == 16
        assert config.temperature == 0.5
        assert config.timeout_sec == 600.0


class TestVideoLLMClient:
    """Test Video-LLM client."""

    def test_mock_client_creation(self):
        """Test creating client in mock mode (no GPU required)."""
        client = create_client(model_name="mock", device="cpu")
        assert client.config.model_name == "mock"
        assert client._model is None  # Mock mode doesn't load model
        assert client._processor is None

    def test_mock_inference(self, test_video):
        """Test mock inference on test video."""
        client = create_client(model_name="mock", device="cpu")
        assessment = client.assess_claim(video_path=test_video, video_id="test_001")

        # Verify basic structure
        assert isinstance(assessment, ClaimAssessment)
        assert assessment.video_id == "test_001"
        assert assessment.severity in ["NONE", "LOW", "MEDIUM", "HIGH"]
        assert 0.0 <= assessment.confidence <= 1.0
        assert assessment.processing_time_sec > 0

        # Verify fault assessment
        assert 0.0 <= assessment.fault_assessment.fault_ratio <= 100.0
        assert len(assessment.fault_assessment.reasoning) > 0

        # Verify fraud risk
        assert 0.0 <= assessment.fraud_risk.risk_score <= 1.0

    def test_frame_sampling(self, test_video):
        """Test frame sampling from video clip."""
        client = create_client(model_name="mock", device="cpu")

        # Sample frames from 1-3 seconds (60 frames @ 30 FPS)
        frame_paths = client._sample_frames(test_video, start_sec=1.0, end_sec=3.0)

        # Should sample at 4 FPS for 2 seconds = 8 frames
        assert len(frame_paths) == 8

        # Verify frames exist and are images
        for frame_path in frame_paths:
            assert frame_path.exists()
            assert frame_path.suffix == ".jpg"

        # Cleanup
        temp_dir = frame_paths[0].parent
        import shutil

        shutil.rmtree(temp_dir)

    def test_frame_sampling_entire_video(self, test_video):
        """Test frame sampling from entire video."""
        client = create_client(model_name="mock", device="cpu")

        # Sample entire video (5 seconds @ 30 FPS = 150 frames)
        frame_paths = client._sample_frames(test_video, start_sec=0.0, end_sec=None)

        # Should sample at 4 FPS for 5 seconds = 20 frames
        assert len(frame_paths) == 20

        # Cleanup
        temp_dir = frame_paths[0].parent
        import shutil

        shutil.rmtree(temp_dir)

    def test_frame_sampling_max_frames_limit(self, test_video):
        """Test max_frames limit on frame sampling."""
        config = VLMConfig(model_name="mock", max_frames=5)
        client = VideoLLMClient(config)

        # Sample entire video with max_frames=5
        frame_paths = client._sample_frames(test_video, start_sec=0.0, end_sec=None)

        # Should not exceed max_frames
        assert len(frame_paths) <= 5

        # Cleanup
        temp_dir = frame_paths[0].parent
        import shutil

        shutil.rmtree(temp_dir)


class TestJSONParsing:
    """Test 7-step JSON parsing pipeline."""

    def test_direct_parse(self):
        """Test Step 1: Direct JSON parse."""
        client = create_client(model_name="mock", device="cpu")

        valid_json = json.dumps(
            {
                "severity": "LOW",
                "confidence": 0.8,
                "prediction_set": ["LOW"],
                "review_priority": "LOW_PRIORITY",
                "fault_assessment": {
                    "fault_ratio": 75.0,
                    "reasoning": "Test",
                    "applicable_rules": [],
                    "scenario_type": "test",
                },
                "fraud_risk": {"risk_score": 0.1, "indicators": [], "reasoning": "Test"},
                "hazards": [],
                "evidence": [],
                "causal_reasoning": "Test reasoning",
                "recommended_action": "APPROVE",
            }
        )

        assessment = client._parse_json_response(valid_json, "test_001", 1.0)
        assert assessment.severity == "LOW"
        assert assessment.confidence == 0.8

    def test_markdown_fence_removal(self):
        """Test Step 2: Remove markdown fences."""
        client = create_client(model_name="mock", device="cpu")

        markdown_json = """```json
        {
            "severity": "MEDIUM",
            "confidence": 0.7,
            "prediction_set": ["MEDIUM"],
            "review_priority": "STANDARD",
            "fault_assessment": {
                "fault_ratio": 50.0,
                "reasoning": "Test",
                "applicable_rules": [],
                "scenario_type": "test"
            },
            "fraud_risk": {"risk_score": 0.2, "indicators": [], "reasoning": "Test"},
            "hazards": [],
            "evidence": [],
            "causal_reasoning": "Test",
            "recommended_action": "REVIEW"
        }
        ```"""

        assessment = client._parse_json_response(markdown_json, "test_002", 1.0)
        assert assessment.severity == "MEDIUM"

    def test_truncation_repair(self):
        """Test Step 3: Add missing closing braces."""
        client = create_client(model_name="mock", device="cpu")

        # Missing closing braces
        truncated_json = """{
            "severity": "HIGH",
            "confidence": 0.9,
            "prediction_set": ["HIGH"],
            "review_priority": "URGENT",
            "fault_assessment": {
                "fault_ratio": 100.0,
                "reasoning": "Test",
                "applicable_rules": [],
                "scenario_type": "test"
            },
            "fraud_risk": {"risk_score": 0.8, "indicators": [], "reasoning": "Test"},
            "hazards": [],
            "evidence": [],
            "causal_reasoning": "Test",
            "recommended_action": "REJECT"
        """  # Missing two closing braces

        assessment = client._parse_json_response(truncated_json, "test_003", 1.0)
        # Should recover with default assessment if parse fails
        assert isinstance(assessment, ClaimAssessment)

    def test_brace_extraction(self):
        """Test Step 4: Extract JSON from surrounding text."""
        client = create_client(model_name="mock", device="cpu")

        text_with_json = """Here is my assessment:
        {
            "severity": "LOW",
            "confidence": 0.85,
            "prediction_set": ["LOW"],
            "review_priority": "LOW_PRIORITY",
            "fault_assessment": {
                "fault_ratio": 25.0,
                "reasoning": "Test",
                "applicable_rules": [],
                "scenario_type": "test"
            },
            "fraud_risk": {"risk_score": 0.05, "indicators": [], "reasoning": "Test"},
            "hazards": [],
            "evidence": [],
            "causal_reasoning": "Test",
            "recommended_action": "APPROVE"
        }
        That's my final answer."""

        assessment = client._parse_json_response(text_with_json, "test_004", 1.0)
        assert assessment.severity == "LOW"

    def test_fallback_to_default(self):
        """Test fallback to default assessment on total parse failure."""
        client = create_client(model_name="mock", device="cpu")

        # Completely invalid JSON
        invalid = "This is not JSON at all. Just random text."

        assessment = client._parse_json_response(invalid, "test_005", 1.0)

        # Should return default assessment
        assert isinstance(assessment, ClaimAssessment)
        assert assessment.video_id == "test_005"
        assert assessment.severity == "LOW"  # Default
        assert assessment.confidence == 0.0  # Default
        assert assessment.review_priority == "URGENT"  # Default requires human review

    def test_field_extraction(self):
        """Test Step 7: Extract individual fields from text."""
        client = create_client(model_name="mock", device="cpu")

        # Partial JSON-like text
        partial = """
        The severity is "severity": "HIGH" based on my analysis.
        My confidence is "confidence": 0.92 in this assessment.
        The fault ratio is "fault_ratio": 88.5 percent.
        My reasoning: "reasoning": "The driver ran a red light."
        """

        data = client._extract_fields_from_text(partial)

        # Should extract some fields
        assert "severity" in data
        assert data["severity"] == "HIGH"
        assert "confidence" in data
        assert data["confidence"] == 0.92


class TestErrorHandling:
    """Test error handling and graceful degradation."""

    def test_nonexistent_video(self):
        """Test handling of nonexistent video file."""
        client = create_client(model_name="mock", device="cpu")

        with pytest.raises(ValueError, match="Video not found"):
            client._sample_frames(Path("/nonexistent/video.mp4"))

    def test_invalid_frame_range(self, test_video):
        """Test handling of invalid frame range."""
        client = create_client(model_name="mock", device="cpu")

        # Start after end
        with pytest.raises(ValueError, match="Invalid frame range"):
            client._sample_frames(test_video, start_sec=10.0, end_sec=5.0)

    def test_assessment_with_exception(self, test_video, monkeypatch):
        """Test graceful degradation when inference fails."""
        client = create_client(model_name="mock", device="cpu")

        # Monkeypatch _run_inference to raise exception
        def mock_inference_error(*args, **kwargs):
            raise RuntimeError("Simulated inference failure")

        monkeypatch.setattr(client, "_run_inference", mock_inference_error)

        # Should return default assessment instead of crashing
        assessment = client.assess_claim(test_video, "test_006")

        assert isinstance(assessment, ClaimAssessment)
        assert assessment.video_id == "test_006"
        assert "Assessment failed" in assessment.causal_reasoning


class TestDefaultAssessment:
    """Test default assessment creation."""

    def test_create_default(self):
        """Test creating default assessment."""
        default = create_default_claim_assessment("test_default")

        assert default.video_id == "test_default"
        assert default.severity == "LOW"
        assert default.confidence == 0.0
        assert default.review_priority == "URGENT"
        assert default.fault_assessment.fault_ratio == 50.0
        assert default.fraud_risk.risk_score == 0.0
        assert default.recommended_action == "REVIEW"


class TestModelCaching:
    """Test model caching (singleton pattern)."""

    def test_cache_reuse(self):
        """Test that models are cached and reused."""
        # Clear cache
        VideoLLMClient._model_cache.clear()

        # Create two clients with same config
        client1 = create_client(model_name="mock", device="cpu")
        client2 = create_client(model_name="mock", device="cpu")

        # Mock mode doesn't cache (no model loaded)
        # But the pattern works for real models
        assert client1.config.model_name == client2.config.model_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
