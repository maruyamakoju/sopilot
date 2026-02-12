"""Tests for Video-LLM service.

Note: These tests use mock mode since Video-LLM models are not available
in test environment. Real model tests should be added to E2E suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from sopilot.video_llm_service import (
    VideoLLMConfig,
    VideoLLMService,
    VideoQAResult,
    get_default_config,
)


class TestVideoLLMConfig:
    """Tests for VideoLLMConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = VideoLLMConfig()
        assert config.model_name == "qwen2.5-vl-7b"
        assert config.device == "cuda"
        assert config.dtype == "float16"
        assert config.load_in_8bit is False
        assert config.load_in_4bit is False
        assert config.max_frames == 32
        assert config.frame_sample_strategy == "uniform"
        assert config.max_new_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9

    def test_custom_config(self):
        """Test custom configuration."""
        config = VideoLLMConfig(
            model_name="llava-video-7b",
            device="cpu",
            max_frames=16,
            temperature=0.5,
        )
        assert config.model_name == "llava-video-7b"
        assert config.device == "cpu"
        assert config.max_frames == 16
        assert config.temperature == 0.5


class TestGetDefaultConfig:
    """Tests for get_default_config."""

    def test_internvideo_config(self):
        """Test InternVideo2.5 default config."""
        config = get_default_config("internvideo2.5-chat-8b")
        assert config.model_name == "internvideo2.5-chat-8b"
        assert config.max_frames == 32
        assert config.device == "cuda"

    def test_llava_config(self):
        """Test LLaVA-Video default config."""
        config = get_default_config("llava-video-7b")
        assert config.model_name == "llava-video-7b"
        assert config.max_frames == 16  # Fewer frames than InternVideo

    def test_mock_config(self):
        """Test mock mode config."""
        config = get_default_config("mock")
        assert config.model_name == "mock"
        assert config.device == "cpu"
        assert config.max_frames == 8

    def test_unknown_model(self):
        """Test unknown model raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_default_config("unknown-model")


class TestVideoLLMServiceMock:
    """Tests for VideoLLMService in mock mode."""

    def test_init_mock_mode(self):
        """Test initialization in mock mode."""
        config = VideoLLMConfig(model_name="mock")
        service = VideoLLMService(config)
        assert service.config == config
        assert service._model is None
        assert service._processor is None

    def test_init_unimplemented_model_raises(self):
        """Test initialization with unimplemented model raises error."""
        config = VideoLLMConfig(model_name="internvideo2.5-chat-8b")
        with pytest.raises(RuntimeError, match="not yet implemented"):
            VideoLLMService(config)

    def test_extract_embedding_mock(self):
        """Test embedding extraction in mock mode."""
        config = VideoLLMConfig(model_name="mock")
        service = VideoLLMService(config)

        # Mock embedding (random)
        embedding = service.extract_embedding("dummy.mp4", start_sec=0.0, end_sec=5.0)

        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        # Mock InternVideo2.5 returns 768-dim
        assert embedding.shape == (768,)

    def test_answer_question_mock(self):
        """Test question answering in mock mode."""
        config = VideoLLMConfig(model_name="mock")
        service = VideoLLMService(config)

        result = service.answer_question(
            "dummy.mp4",
            "What is happening in the video?",
            start_sec=0.0,
            end_sec=10.0,
        )

        assert isinstance(result, VideoQAResult)
        assert result.question == "What is happening in the video?"
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert result.confidence == 0.5  # Mock confidence

    def test_answer_question_with_cot_mock(self):
        """Test question answering with chain-of-thought."""
        config = VideoLLMConfig(model_name="mock")
        service = VideoLLMService(config)

        result = service.answer_question(
            "dummy.mp4",
            "Why did this happen?",
            enable_cot=True,
        )

        assert isinstance(result, VideoQAResult)
        assert result.reasoning == "Mock reasoning"

    def test_batch_extract_embeddings_mock(self):
        """Test batch embedding extraction in mock mode."""
        config = VideoLLMConfig(model_name="mock")
        service = VideoLLMService(config)

        clips = [
            ("video1.mp4", 0.0, 5.0),
            ("video2.mp4", 10.0, 15.0),
            ("video3.mp4", 20.0, 25.0),
        ]

        embeddings = service.batch_extract_embeddings(clips)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 768)  # 3 clips, 768-dim each
        assert embeddings.dtype == np.float32

    def test_batch_extract_embeddings_empty(self):
        """Test batch extraction with empty list."""
        config = VideoLLMConfig(model_name="mock")
        service = VideoLLMService(config)

        embeddings = service.batch_extract_embeddings([])
        assert embeddings.shape == (0, 768)


class TestVideoQAResult:
    """Tests for VideoQAResult dataclass."""

    def test_basic_result(self):
        """Test basic QA result."""
        result = VideoQAResult(
            question="What color is the car?",
            answer="The car is red.",
        )
        assert result.question == "What color is the car?"
        assert result.answer == "The car is red."
        assert result.confidence is None
        assert result.reasoning is None

    def test_result_with_confidence(self):
        """Test QA result with confidence."""
        result = VideoQAResult(
            question="Is this a factory?",
            answer="Yes, this appears to be a factory.",
            confidence=0.92,
        )
        assert result.confidence == 0.92

    def test_result_with_reasoning(self):
        """Test QA result with reasoning."""
        result = VideoQAResult(
            question="Why did the machine stop?",
            answer="The machine stopped due to a safety sensor trigger.",
            reasoning="First, I noticed the red light. Then, I saw the worker step back.",
        )
        assert result.reasoning is not None
        assert "red light" in result.reasoning


# Note: Real frame sampling tests require a test video file
# These should be added to integration tests with real video assets
