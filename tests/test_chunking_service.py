"""Tests for multi-scale video chunking service."""

from __future__ import annotations

import pytest

from sopilot.chunking_service import Chunk, ChunkConfig, ChunkingService

# Skip tests if scenedetect not available
pytest.importorskip("scenedetect")


class TestChunkConfig:
    """Tests for ChunkConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ChunkConfig()
        assert config.shot_detector == "adaptive"
        assert config.shot_threshold == 3.0
        assert config.micro_min == 2.0
        assert config.micro_max == 4.0
        assert config.meso_min == 8.0
        assert config.meso_max == 16.0
        assert config.macro_min == 32.0
        assert config.macro_max == 64.0

    def test_factory_domain_config(self):
        """Test factory domain configuration."""
        config = ChunkConfig.for_domain("factory")
        assert config.shot_threshold == 2.5  # More sensitive
        assert config.micro_max == 3.0  # Shorter chunks
        assert config.keyframes_per_micro == 4  # More keyframes

    def test_surveillance_domain_config(self):
        """Test surveillance domain configuration."""
        config = ChunkConfig.for_domain("surveillance")
        assert config.shot_threshold == 4.0  # Less sensitive
        assert config.micro_min == 3.0  # Longer chunks
        assert config.keyframes_per_micro == 2  # Fewer keyframes

    def test_sports_domain_config(self):
        """Test sports domain configuration."""
        config = ChunkConfig.for_domain("sports")
        assert config.shot_threshold == 2.0  # Very sensitive
        assert config.keyframes_per_micro == 5  # Many keyframes

    def test_generic_domain_config(self):
        """Test generic domain returns default config."""
        config = ChunkConfig.for_domain("generic")
        default = ChunkConfig()
        assert config.shot_threshold == default.shot_threshold
        assert config.micro_min == default.micro_min


class TestChunkingService:
    """Tests for ChunkingService."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        service = ChunkingService()
        assert service.config is not None
        assert isinstance(service.config, ChunkConfig)

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ChunkConfig(shot_threshold=5.0)
        service = ChunkingService(config)
        assert service.config.shot_threshold == 5.0

    def test_distribute_keyframes_single(self):
        """Test keyframe distribution with single keyframe."""
        service = ChunkingService()
        keyframes = service._distribute_keyframes(0, 100, 1)
        assert len(keyframes) == 1
        assert keyframes[0] == 50  # Middle frame

    def test_distribute_keyframes_multiple(self):
        """Test keyframe distribution with multiple keyframes."""
        service = ChunkingService()
        keyframes = service._distribute_keyframes(0, 100, 3)
        assert len(keyframes) == 3
        # Should be evenly distributed
        assert keyframes[0] == 25
        assert keyframes[1] == 50
        assert keyframes[2] == 75

    def test_distribute_keyframes_zero(self):
        """Test keyframe distribution with zero keyframes."""
        service = ChunkingService()
        keyframes = service._distribute_keyframes(0, 100, 0)
        assert len(keyframes) == 0

    def test_distribute_keyframes_invalid_range(self):
        """Test keyframe distribution with invalid frame range."""
        service = ChunkingService()
        keyframes = service._distribute_keyframes(100, 100, 3)
        assert len(keyframes) == 0

    def test_generate_fixed_chunks_basic(self):
        """Test fixed chunk generation."""
        service = ChunkingService()
        base_chunks: list[Chunk] = []  # Empty base chunks
        fps = 30.0
        duration = 60.0  # 60 second video

        chunks = service._generate_fixed_chunks(
            base_chunks,
            fps,
            duration,
            level="micro",
            min_dur=2.0,
            max_dur=4.0,
            keyframes=3,
        )

        assert len(chunks) > 0
        # Each chunk should have target duration of ~3 seconds
        for chunk in chunks:
            assert chunk.level == "micro"
            assert chunk.end_sec > chunk.start_sec
            assert len(chunk.keyframe_indices) == 3

        # All chunks should cover the full duration
        assert chunks[0].start_sec == 0.0
        assert chunks[-1].end_sec == duration

    def test_generate_fixed_chunks_short_video(self):
        """Test fixed chunk generation for short video."""
        service = ChunkingService()
        base_chunks: list[Chunk] = []
        fps = 30.0
        duration = 2.0  # Very short video

        chunks = service._generate_fixed_chunks(
            base_chunks,
            fps,
            duration,
            level="micro",
            min_dur=2.0,
            max_dur=4.0,
            keyframes=3,
        )

        # Should have at least 1 chunk
        assert len(chunks) == 1
        assert chunks[0].start_sec == 0.0
        assert chunks[0].end_sec == duration


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a chunk."""
        chunk = Chunk(
            level="shot",
            start_sec=0.0,
            end_sec=2.5,
            start_frame=0,
            end_frame=75,
            keyframe_indices=[37],
        )
        assert chunk.level == "shot"
        assert chunk.start_sec == 0.0
        assert chunk.end_sec == 2.5
        assert chunk.start_frame == 0
        assert chunk.end_frame == 75
        assert chunk.keyframe_indices == [37]
        assert chunk.parent_chunks is None

    def test_chunk_with_parents(self):
        """Test chunk with parent chunks."""
        parent = Chunk(
            level="micro",
            start_sec=0.0,
            end_sec=3.0,
            start_frame=0,
            end_frame=90,
            keyframe_indices=[30, 60],
        )
        chunk = Chunk(
            level="meso",
            start_sec=0.0,
            end_sec=10.0,
            start_frame=0,
            end_frame=300,
            keyframe_indices=[50, 150, 250],
            parent_chunks=[parent],
        )
        assert len(chunk.parent_chunks) == 1
        assert chunk.parent_chunks[0] == parent


# Note: Full integration tests that call chunk_video() require:
# 1. A real video file
# 2. SceneDetect properly installed
# These should be added to integration tests or E2E tests
