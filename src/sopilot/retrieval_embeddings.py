"""Retrieval embeddings using OpenCLIP for VIGIL-RAG.

This module provides text-image embedding alignment for retrieval.
Separate from video_llm_service.py which handles QA inference.

Key difference:
- Retrieval: Text ↔ Image same embedding space (CLIP)
- QA: Video → Answer (InternVideo2.5 Chat / LLaVA)

Design:
- OpenCLIP for production (best text-image alignment)
- Mock mode for testing (random embeddings)
- Simple API: encode_text(), encode_images()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import open_clip
    import torch
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False


@dataclass
class RetrievalConfig:
    """Configuration for retrieval embeddings."""

    model_name: str = "ViT-B-32"  # OpenCLIP model
    pretrained: str = "laion2b_s34b_b79k"  # Pretrained dataset
    device: str = "cuda"  # cuda / cpu
    batch_size: int = 32
    embedding_dim: int = 512  # ViT-B-32 outputs 512-dim

    @classmethod
    def for_model(cls, model_name: str) -> RetrievalConfig:
        """Get config for a specific OpenCLIP model.

        Args:
            model_name: OpenCLIP model name

        Returns:
            RetrievalConfig with appropriate settings
        """
        # Common model configs
        configs = {
            "ViT-B-32": cls(
                model_name="ViT-B-32",
                pretrained="laion2b_s34b_b79k",
                embedding_dim=512,
            ),
            "ViT-B-16": cls(
                model_name="ViT-B-16",
                pretrained="laion2b_s34b_b82k",
                embedding_dim=512,
            ),
            "ViT-L-14": cls(
                model_name="ViT-L-14",
                pretrained="laion2b_s32b_b82k",
                embedding_dim=768,
            ),
            "ViT-H-14": cls(
                model_name="ViT-H-14",
                pretrained="laion2b_s32b_b79k",
                embedding_dim=1024,
            ),
        }

        if model_name in configs:
            return configs[model_name]
        else:
            # Default to ViT-B-32
            logger.warning(f"Unknown model {model_name}, using ViT-B-32")
            return configs["ViT-B-32"]


class RetrievalEmbedder:
    """OpenCLIP-based retrieval embedder."""

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        """Initialize retrieval embedder.

        Args:
            config: Retrieval configuration (defaults to ViT-B-32)

        Raises:
            RuntimeError: If OpenCLIP not available in non-mock mode
        """
        self.config = config or RetrievalConfig()
        self._model = None
        self._preprocess = None
        self._tokenizer = None

        if not OPENCLIP_AVAILABLE:
            logger.warning(
                "open_clip_torch not installed, using mock mode. "
                "Install with: pip install -e '.[vigil]'"
            )
        else:
            self._load_model()

    def _load_model(self) -> None:
        """Load OpenCLIP model."""
        logger.info(
            "Loading OpenCLIP model: %s (pretrained=%s)",
            self.config.model_name,
            self.config.pretrained,
        )

        try:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.config.model_name,
                pretrained=self.config.pretrained,
                device=self.config.device,
            )
            self._tokenizer = open_clip.get_tokenizer(self.config.model_name)

            self._model.eval()  # Set to evaluation mode
            logger.info("OpenCLIP model loaded successfully")

        except Exception as e:
            logger.error("Failed to load OpenCLIP model: %s", e)
            logger.warning("Falling back to mock mode")
            self._model = None

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of text strings

        Returns:
            Embeddings (N, D) where D is embedding_dim
        """
        if self._model is None:
            # Mock mode
            return self._encode_text_mock(texts)

        with torch.no_grad():
            tokens = self._tokenizer(texts).to(self.config.device)
            embeddings = self._model.encode_text(tokens)

            # Normalize embeddings (for cosine similarity)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            return embeddings.cpu().numpy().astype(np.float32)

    def encode_images(self, images: list[Image.Image | np.ndarray]) -> np.ndarray:
        """Encode images to embeddings.

        Args:
            images: List of PIL Images or numpy arrays (H, W, 3)

        Returns:
            Embeddings (N, D) where D is embedding_dim
        """
        if self._model is None:
            # Mock mode
            return self._encode_images_mock(images)

        # Convert numpy arrays to PIL Images
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)

        with torch.no_grad():
            # Preprocess and stack
            image_tensors = torch.stack(
                [self._preprocess(img) for img in pil_images]
            ).to(self.config.device)

            embeddings = self._model.encode_image(image_tensors)

            # Normalize embeddings
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            return embeddings.cpu().numpy().astype(np.float32)

    def encode_image_batch(
        self,
        image_paths: list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode images from file paths in batches.

        Args:
            image_paths: List of image file paths
            batch_size: Batch size (defaults to config.batch_size)

        Returns:
            Embeddings (N, D)
        """
        batch_size = batch_size or self.config.batch_size
        all_embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]

            # Load images
            images = []
            for path in batch_paths:
                try:
                    images.append(Image.open(path).convert("RGB"))
                except Exception as e:
                    logger.warning("Failed to load image %s: %s", path, e)
                    continue

            if images:
                embeddings = self.encode_images(images)
                all_embeddings.append(embeddings)

        if not all_embeddings:
            return np.zeros((0, self.config.embedding_dim), dtype=np.float32)

        return np.vstack(all_embeddings)

    def _encode_text_mock(self, texts: list[str]) -> np.ndarray:
        """Mock text encoding (for testing)."""
        logger.debug("Mock mode: encoding %d texts", len(texts))
        return np.random.randn(len(texts), self.config.embedding_dim).astype(np.float32)

    def _encode_images_mock(self, images: list) -> np.ndarray:
        """Mock image encoding (for testing)."""
        logger.debug("Mock mode: encoding %d images", len(images))
        return np.random.randn(len(images), self.config.embedding_dim).astype(np.float32)


# Convenience function for quick usage
def create_embedder(
    model_name: str = "ViT-B-32",
    device: str = "cuda",
) -> RetrievalEmbedder:
    """Create a retrieval embedder with sensible defaults.

    Args:
        model_name: OpenCLIP model name (ViT-B-32, ViT-B-16, ViT-L-14, ViT-H-14)
        device: Device (cuda or cpu)

    Returns:
        RetrievalEmbedder instance
    """
    config = RetrievalConfig.for_model(model_name)
    config.device = device
    return RetrievalEmbedder(config)
