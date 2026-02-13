"""Tests for retrieval embeddings (OpenCLIP)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from sopilot.retrieval_embeddings import (
    RetrievalConfig,
    RetrievalEmbedder,
    create_embedder,
)


class TestRetrievalConfig:
    """Tests for RetrievalConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RetrievalConfig()
        assert config.model_name == "ViT-B-32"
        assert config.pretrained == "laion2b_s34b_b79k"
        assert config.device == "cuda"
        assert config.batch_size == 32
        assert config.embedding_dim == 512

    def test_for_model_vit_b_32(self):
        """Test ViT-B-32 config."""
        config = RetrievalConfig.for_model("ViT-B-32")
        assert config.model_name == "ViT-B-32"
        assert config.embedding_dim == 512

    def test_for_model_vit_l_14(self):
        """Test ViT-L-14 config."""
        config = RetrievalConfig.for_model("ViT-L-14")
        assert config.model_name == "ViT-L-14"
        assert config.embedding_dim == 768

    def test_for_model_vit_h_14(self):
        """Test ViT-H-14 config."""
        config = RetrievalConfig.for_model("ViT-H-14")
        assert config.model_name == "ViT-H-14"
        assert config.embedding_dim == 1024

    def test_for_model_unknown(self):
        """Test unknown model falls back to ViT-B-32."""
        config = RetrievalConfig.for_model("unknown-model")
        assert config.model_name == "ViT-B-32"
        assert config.embedding_dim == 512


class TestRetrievalEmbedderMock:
    """Tests for RetrievalEmbedder in mock mode (no OpenCLIP)."""

    def test_init_mock_mode(self):
        """Test initialization (mock mode if OpenCLIP unavailable, else real model)."""
        embedder = RetrievalEmbedder()
        assert embedder.config is not None
        assert isinstance(embedder.config, RetrievalConfig)
        # Model may be None (mock) or loaded (if OpenCLIP installed)
        # Both are valid - we just need config to be set

    def test_encode_text_mock(self):
        """Test text encoding in mock mode."""
        embedder = RetrievalEmbedder()

        texts = ["a photo of a cat", "a photo of a dog"]
        embeddings = embedder.encode_text(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 512)  # Default ViT-B-32 dim
        assert embeddings.dtype == np.float32

    def test_encode_text_single(self):
        """Test encoding single text."""
        embedder = RetrievalEmbedder()

        texts = ["hello world"]
        embeddings = embedder.encode_text(texts)

        assert embeddings.shape == (1, 512)

    def test_encode_images_mock(self):
        """Test image encoding in mock mode."""
        embedder = RetrievalEmbedder()

        # Create dummy PIL images
        images = [
            Image.new("RGB", (224, 224), color="red"),
            Image.new("RGB", (224, 224), color="blue"),
        ]
        embeddings = embedder.encode_images(images)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 512)
        assert embeddings.dtype == np.float32

    def test_encode_images_numpy(self):
        """Test encoding numpy array images."""
        embedder = RetrievalEmbedder()

        # Create dummy numpy arrays
        images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        ]
        embeddings = embedder.encode_images(images)

        assert embeddings.shape == (2, 512)

    def test_encode_images_mixed(self):
        """Test encoding mixed PIL and numpy images."""
        embedder = RetrievalEmbedder()

        images = [
            Image.new("RGB", (224, 224), color="red"),
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        ]
        embeddings = embedder.encode_images(images)

        assert embeddings.shape == (2, 512)

    def test_different_embedding_dims(self):
        """Test different embedding dimensions."""
        # ViT-L-14 has 768-dim
        config = RetrievalConfig.for_model("ViT-L-14")
        embedder = RetrievalEmbedder(config)

        texts = ["test"]
        embeddings = embedder.encode_text(texts)

        assert embeddings.shape == (1, 768)


class TestCreateEmbedder:
    """Tests for create_embedder factory function."""

    def test_create_default(self):
        """Test creating embedder with defaults."""
        embedder = create_embedder()

        assert isinstance(embedder, RetrievalEmbedder)
        assert embedder.config.model_name == "ViT-B-32"
        assert embedder.config.device == "cuda"

    def test_create_custom_model(self):
        """Test creating embedder with custom model."""
        embedder = create_embedder(model_name="ViT-L-14")

        assert embedder.config.model_name == "ViT-L-14"
        assert embedder.config.embedding_dim == 768

    def test_create_cpu_device(self):
        """Test creating embedder with CPU device."""
        embedder = create_embedder(device="cpu")

        assert embedder.config.device == "cpu"


# Note: Real OpenCLIP tests require the library to be installed
# and would download model weights. These should be added to
# integration tests with appropriate fixtures.
