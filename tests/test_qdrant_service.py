"""Tests for Qdrant vector database service.

Note: These tests focus on FAISS fallback mode since a running Qdrant
server is not available in test environment. Integration tests with
real Qdrant should be added to E2E test suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from sopilot.qdrant_service import QdrantConfig, QdrantService, SearchResult


class TestQdrantConfig:
    """Tests for QdrantConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = QdrantConfig()
        assert config.host == "localhost"
        assert config.port == 6333
        assert config.grpc_port == 6334
        assert config.prefer_grpc is False
        assert config.api_key is None
        assert config.https is False
        assert config.embedding_dim == 768

    def test_connection_url_http(self):
        """Test HTTP connection URL."""
        config = QdrantConfig(host="example.com", port=6333, https=False)
        assert config.connection_url == "http://example.com:6333"

    def test_connection_url_https(self):
        """Test HTTPS connection URL."""
        config = QdrantConfig(host="example.com", port=6333, https=True)
        assert config.connection_url == "https://example.com:6333"


def _faiss_config() -> QdrantConfig:
    """Create config that always falls back to FAISS (unreachable port)."""
    return QdrantConfig(host="localhost", port=19999)


class TestQdrantServiceFAISSFallback:
    """Tests for QdrantService using FAISS fallback."""

    def test_init_with_fallback(self):
        """Test initialization with FAISS fallback enabled."""
        config = _faiss_config()
        service = QdrantService(config, use_faiss_fallback=True)
        assert service.config == config
        assert service.use_faiss_fallback is True
        # Should fall back to FAISS if Qdrant not available
        assert service._faiss_indexes == {}

    def test_collection_name(self):
        """Test collection name generation."""
        config = _faiss_config()
        service = QdrantService(config, use_faiss_fallback=True)
        assert service._get_collection_name("shot") == "vigil_shot"
        assert service._get_collection_name("micro") == "vigil_micro"
        assert service._get_collection_name("meso") == "vigil_meso"
        assert service._get_collection_name("macro") == "vigil_macro"

    def test_add_embeddings_faiss(self):
        """Test adding embeddings with FAISS fallback."""
        config = _faiss_config()
        service = QdrantService(config, use_faiss_fallback=True)

        embeddings = np.random.randn(5, 768).astype(np.float32)
        metadata = [
            {
                "clip_id": f"clip-{i}",
                "video_id": "video-123",
                "start_sec": float(i * 2),
                "end_sec": float((i + 1) * 2),
            }
            for i in range(5)
        ]

        count = service.add_embeddings("micro", embeddings, metadata)
        assert count == 5

        # Verify FAISS index was created
        assert "micro" in service._faiss_indexes
        assert len(service._faiss_indexes["micro"]["vectors"]) == 1
        assert len(service._faiss_indexes["micro"]["metadata"]) == 5

    def test_add_embeddings_validation(self):
        """Test embedding validation."""
        config = _faiss_config()
        service = QdrantService(config, use_faiss_fallback=True)

        embeddings = np.random.randn(5, 768).astype(np.float32)

        # Mismatch count
        metadata = [{"clip_id": "clip-0", "video_id": "video-123", "start_sec": 0.0, "end_sec": 2.0}]
        with pytest.raises(ValueError, match="mismatch"):
            service.add_embeddings("micro", embeddings, metadata)

        # Missing required key
        metadata = [
            {
                "clip_id": f"clip-{i}",
                "video_id": "video-123",
                # Missing start_sec
                "end_sec": float((i + 1) * 2),
            }
            for i in range(5)
        ]
        with pytest.raises(ValueError, match="missing keys"):
            service.add_embeddings("micro", embeddings, metadata)

    def test_add_embeddings_empty(self):
        """Test adding empty embeddings."""
        config = _faiss_config()
        service = QdrantService(config, use_faiss_fallback=True)

        embeddings = np.zeros((0, 768), dtype=np.float32)
        metadata = []

        count = service.add_embeddings("micro", embeddings, metadata)
        assert count == 0

    def test_search_faiss_basic(self):
        """Test basic search with FAISS fallback."""
        config = _faiss_config()
        service = QdrantService(config, use_faiss_fallback=True)

        # Add some embeddings
        embeddings = np.random.randn(10, 768).astype(np.float32)
        metadata = [
            {
                "clip_id": f"clip-{i}",
                "video_id": "video-123",
                "start_sec": float(i * 2),
                "end_sec": float((i + 1) * 2),
            }
            for i in range(10)
        ]
        service.add_embeddings("micro", embeddings, metadata)

        # Search with first embedding as query (should return itself as top result)
        query = embeddings[0]
        results = service.search("micro", query, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        # Top result should be the query itself (highest similarity)
        assert results[0].clip_id == "clip-0"
        assert results[0].video_id == "video-123"
        assert results[0].level == "micro"
        assert results[0].score > 0.99  # Near perfect match

    def test_search_faiss_with_video_filter(self):
        """Test search with video_id filter."""
        config = _faiss_config()
        service = QdrantService(config, use_faiss_fallback=True)

        # Add embeddings from two videos
        embeddings_v1 = np.random.randn(5, 768).astype(np.float32)
        metadata_v1 = [
            {
                "clip_id": f"clip-v1-{i}",
                "video_id": "video-1",
                "start_sec": float(i * 2),
                "end_sec": float((i + 1) * 2),
            }
            for i in range(5)
        ]
        service.add_embeddings("micro", embeddings_v1, metadata_v1)

        embeddings_v2 = np.random.randn(5, 768).astype(np.float32)
        metadata_v2 = [
            {
                "clip_id": f"clip-v2-{i}",
                "video_id": "video-2",
                "start_sec": float(i * 2),
                "end_sec": float((i + 1) * 2),
            }
            for i in range(5)
        ]
        service.add_embeddings("micro", embeddings_v2, metadata_v2)

        # Search with video_id filter
        query = embeddings_v1[0]
        results = service.search("micro", query, top_k=10, video_id="video-1")

        # Should only return results from video-1
        assert len(results) == 5
        assert all(r.video_id == "video-1" for r in results)

    def test_search_faiss_with_min_score(self):
        """Test search with minimum score threshold."""
        config = _faiss_config()
        service = QdrantService(config, use_faiss_fallback=True)

        embeddings = np.random.randn(10, 768).astype(np.float32)
        metadata = [
            {
                "clip_id": f"clip-{i}",
                "video_id": "video-123",
                "start_sec": float(i * 2),
                "end_sec": float((i + 1) * 2),
            }
            for i in range(10)
        ]
        service.add_embeddings("micro", embeddings, metadata)

        # Search with high min_score (should return fewer results)
        query = embeddings[0]
        results = service.search("micro", query, top_k=10, min_score=0.9)

        # Should return at least 1 (the query itself)
        assert len(results) >= 1
        assert all(r.score >= 0.9 for r in results)

    def test_search_faiss_empty_index(self):
        """Test search on empty index."""
        config = _faiss_config()
        service = QdrantService(config, use_faiss_fallback=True)

        query = np.random.randn(768).astype(np.float32)
        results = service.search("micro", query, top_k=10)

        assert results == []

    def test_coarse_to_fine_search(self):
        """Test coarse-to-fine search across all levels."""
        config = _faiss_config()
        service = QdrantService(config, use_faiss_fallback=True)

        # Add embeddings at each level
        for level in ["shot", "micro", "meso", "macro"]:
            embeddings = np.random.randn(5, 768).astype(np.float32)
            metadata = [
                {
                    "clip_id": f"{level}-{i}",
                    "video_id": "video-123",
                    "start_sec": float(i * 2),
                    "end_sec": float((i + 1) * 2),
                }
                for i in range(5)
            ]
            service.add_embeddings(level, embeddings, metadata)

        # Perform coarse-to-fine search
        query = np.random.randn(768).astype(np.float32)
        results = service.coarse_to_fine_search(query, macro_k=3, meso_k=3, micro_k=3, shot_k=3)

        # Should have results for all levels
        assert "macro" in results
        assert "meso" in results
        assert "micro" in results
        assert "shot" in results

        # Each level should have results
        assert len(results["macro"]) == 3
        assert len(results["meso"]) == 3
        assert len(results["micro"]) == 3
        assert len(results["shot"]) == 3

    def test_coarse_to_fine_search_empty(self):
        """Test coarse-to-fine search with no data."""
        config = _faiss_config()
        service = QdrantService(config, use_faiss_fallback=True)

        query = np.random.randn(768).astype(np.float32)
        results = service.coarse_to_fine_search(query)

        # Should return empty results for all levels
        assert results["macro"] == []
        assert results["meso"] == []
        assert results["micro"] == []
        assert results["shot"] == []


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            clip_id="clip-123",
            video_id="video-456",
            level="micro",
            start_sec=5.0,
            end_sec=8.0,
            score=0.95,
        )
        assert result.clip_id == "clip-123"
        assert result.video_id == "video-456"
        assert result.level == "micro"
        assert result.start_sec == 5.0
        assert result.end_sec == 8.0
        assert result.score == 0.95
        assert result.embedding is None

    def test_search_result_with_embedding(self):
        """Test SearchResult with embedding."""
        emb = np.random.randn(768).astype(np.float32)
        result = SearchResult(
            clip_id="clip-123",
            video_id="video-456",
            level="shot",
            start_sec=0.0,
            end_sec=2.0,
            score=0.87,
            embedding=emb,
        )
        assert result.embedding is not None
        assert np.array_equal(result.embedding, emb)
