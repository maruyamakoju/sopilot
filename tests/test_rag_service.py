"""Tests for RAG service."""

from __future__ import annotations

import numpy as np
import pytest
from sopilot.rag_service import (
    RAGService,
    RetrievalConfig,
    Evidence,
    RAGResult,
    create_rag_service,
)
from sopilot.qdrant_service import QdrantConfig, QdrantService, SearchResult
from sopilot.video_llm_service import VideoLLMConfig, VideoLLMService


class TestRetrievalConfig:
    """Tests for RetrievalConfig."""

    def test_default_config(self):
        """Test default retrieval configuration."""
        config = RetrievalConfig()
        assert config.macro_k == 5
        assert config.meso_k == 10
        assert config.micro_k == 20
        assert config.shot_k == 30
        assert config.enable_rerank is False
        assert config.rerank_top_k == 10
        assert config.min_score is None
        assert config.video_id_filter is None

    def test_custom_config(self):
        """Test custom retrieval configuration."""
        config = RetrievalConfig(
            macro_k=3,
            meso_k=5,
            enable_rerank=True,
            min_score=0.8,
        )
        assert config.macro_k == 3
        assert config.meso_k == 5
        assert config.enable_rerank is True
        assert config.min_score == 0.8


class TestEvidence:
    """Tests for Evidence dataclass."""

    def test_evidence_creation(self):
        """Test creating Evidence."""
        evidence = Evidence(
            clip_id="clip-123",
            video_id="video-456",
            level="micro",
            start_sec=5.0,
            end_sec=8.0,
            score=0.92,
        )
        assert evidence.clip_id == "clip-123"
        assert evidence.video_id == "video-456"
        assert evidence.level == "micro"
        assert evidence.start_sec == 5.0
        assert evidence.end_sec == 8.0
        assert evidence.score == 0.92
        assert evidence.rerank_score is None

    def test_evidence_with_rerank(self):
        """Test Evidence with re-ranking score."""
        evidence = Evidence(
            clip_id="clip-789",
            video_id="video-101",
            level="shot",
            start_sec=0.0,
            end_sec=2.0,
            score=0.85,
            rerank_score=0.95,
        )
        assert evidence.rerank_score == 0.95


class TestRAGResult:
    """Tests for RAGResult dataclass."""

    def test_basic_result(self):
        """Test basic RAG result."""
        evidence = [
            Evidence(
                clip_id="clip-1",
                video_id="video-1",
                level="micro",
                start_sec=0.0,
                end_sec=3.0,
                score=0.9,
            )
        ]
        result = RAGResult(
            question="What happened?",
            answer="An event occurred.",
            evidence=evidence,
        )
        assert result.question == "What happened?"
        assert result.answer == "An event occurred."
        assert len(result.evidence) == 1
        assert result.confidence is None
        assert result.reasoning is None

    def test_result_with_confidence_and_reasoning(self):
        """Test RAG result with confidence and reasoning."""
        result = RAGResult(
            question="Why did it happen?",
            answer="Because of reason X.",
            evidence=[],
            confidence=0.87,
            reasoning="Step 1: ... Step 2: ...",
        )
        assert result.confidence == 0.87
        assert result.reasoning is not None


class TestRAGService:
    """Tests for RAGService."""

    def _create_mock_services(self):
        """Create mock Qdrant and Video-LLM services."""
        qdrant_config = QdrantConfig()
        qdrant_service = QdrantService(qdrant_config, use_faiss_fallback=True)

        llm_config = VideoLLMConfig(model_name="mock")
        llm_service = VideoLLMService(llm_config)

        return qdrant_service, llm_service

    def test_init(self):
        """Test RAG service initialization."""
        qdrant_service, llm_service = self._create_mock_services()
        rag_service = RAGService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )
        assert rag_service.vector_service == qdrant_service
        assert rag_service.llm_service == llm_service
        assert isinstance(rag_service.retrieval_config, RetrievalConfig)

    def test_init_with_custom_config(self):
        """Test RAG service with custom retrieval config."""
        qdrant_service, llm_service = self._create_mock_services()
        config = RetrievalConfig(macro_k=3, enable_rerank=True)
        rag_service = RAGService(
            vector_service=qdrant_service,
            llm_service=llm_service,
            retrieval_config=config,
        )
        assert rag_service.retrieval_config.macro_k == 3
        assert rag_service.retrieval_config.enable_rerank is True

    def test_encode_query_mock(self):
        """Test query encoding (mock mode)."""
        qdrant_service, llm_service = self._create_mock_services()
        rag_service = RAGService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        query_emb = rag_service._encode_query("What is happening?")
        assert isinstance(query_emb, np.ndarray)
        assert query_emb.ndim == 1  # 1-D embedding vector (dim depends on model)

    def test_flatten_search_results(self):
        """Test flattening multi-level search results."""
        qdrant_service, llm_service = self._create_mock_services()
        rag_service = RAGService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        search_results = {
            "shot": [
                SearchResult("clip-1", "video-1", "shot", 0.0, 2.0, 0.9),
                SearchResult("clip-2", "video-1", "shot", 2.0, 4.0, 0.85),
            ],
            "micro": [
                SearchResult("clip-3", "video-1", "micro", 0.0, 3.0, 0.95),
            ],
            "meso": [],
            "macro": [
                SearchResult("clip-4", "video-1", "macro", 0.0, 32.0, 0.8),
            ],
        }

        flattened = rag_service._flatten_search_results(search_results)

        assert len(flattened) == 4
        # Should be sorted by score descending
        assert flattened[0].score == 0.95  # micro clip
        assert flattened[1].score == 0.9  # shot clip-1
        assert flattened[2].score == 0.85  # shot clip-2
        assert flattened[3].score == 0.8  # macro clip

    def test_answer_question_with_data(self):
        """Test answering question with pre-populated data."""
        qdrant_service, llm_service = self._create_mock_services()

        # Add some mock embeddings to all levels
        for level in ["shot", "micro", "meso", "macro"]:
            embeddings = np.random.randn(3, 512).astype(np.float32)
            metadata = [
                {
                    "clip_id": f"{level}-{i}",
                    "video_id": "video-123",
                    "start_sec": float(i * 2),
                    "end_sec": float((i + 1) * 2),
                }
                for i in range(3)
            ]
            qdrant_service.add_embeddings(level, embeddings, metadata)

        rag_service = RAGService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        result = rag_service.answer_question("What is in the video?")

        assert isinstance(result, RAGResult)
        assert result.question == "What is in the video?"
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        # Should have evidence clips
        assert len(result.evidence) > 0
        assert all(isinstance(e, Evidence) for e in result.evidence)

    def test_answer_question_no_data(self):
        """Test answering question with no data."""
        qdrant_service, llm_service = self._create_mock_services()
        rag_service = RAGService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        result = rag_service.answer_question("What is in the video?")

        assert isinstance(result, RAGResult)
        assert result.question == "What is in the video?"
        # No data, so answer should indicate no clips found
        assert "No relevant clips found" in result.answer
        assert result.confidence == 0.0
        assert len(result.evidence) == 0

    def test_answer_question_with_video_filter(self):
        """Test answering question with video_id filter."""
        qdrant_service, llm_service = self._create_mock_services()

        # Add embeddings from two videos to all levels
        for level in ["shot", "micro", "meso", "macro"]:
            embeddings_v1 = np.random.randn(2, 512).astype(np.float32)
            metadata_v1 = [
                {
                    "clip_id": f"{level}-v1-{i}",
                    "video_id": "video-1",
                    "start_sec": float(i * 2),
                    "end_sec": float((i + 1) * 2),
                }
                for i in range(2)
            ]
            qdrant_service.add_embeddings(level, embeddings_v1, metadata_v1)

            embeddings_v2 = np.random.randn(2, 512).astype(np.float32)
            metadata_v2 = [
                {
                    "clip_id": f"{level}-v2-{i}",
                    "video_id": "video-2",
                    "start_sec": float(i * 2),
                    "end_sec": float((i + 1) * 2),
                }
                for i in range(2)
            ]
            qdrant_service.add_embeddings(level, embeddings_v2, metadata_v2)

        rag_service = RAGService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        # Filter by video-1
        result = rag_service.answer_question("What happened?", video_id="video-1")

        assert len(result.evidence) > 0
        # All evidence should be from video-1
        assert all(e.video_id == "video-1" for e in result.evidence)

    def test_answer_question_from_clip(self):
        """Test direct clip QA (no retrieval)."""
        qdrant_service, llm_service = self._create_mock_services()
        rag_service = RAGService(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        result = rag_service.answer_question_from_clip(
            "dummy.mp4",
            "What is happening?",
            start_sec=5.0,
            end_sec=10.0,
        )

        assert isinstance(result, RAGResult)
        assert result.question == "What is happening?"
        assert isinstance(result.answer, str)
        # Should have exactly 1 evidence (the direct clip)
        assert len(result.evidence) == 1
        assert result.evidence[0].clip_id == "direct-clip"
        assert result.evidence[0].start_sec == 5.0
        assert result.evidence[0].end_sec == 10.0


class TestCreateRAGService:
    """Tests for create_rag_service factory function."""

    def test_create_with_defaults(self):
        """Test factory with default parameters."""
        qdrant_config = QdrantConfig()
        qdrant_service = QdrantService(qdrant_config, use_faiss_fallback=True)

        llm_config = VideoLLMConfig(model_name="mock")
        llm_service = VideoLLMService(llm_config)

        rag_service = create_rag_service(
            vector_service=qdrant_service,
            llm_service=llm_service,
        )

        assert isinstance(rag_service, RAGService)
        assert rag_service.retrieval_config.macro_k == 5
        assert rag_service.retrieval_config.meso_k == 10
        assert rag_service.retrieval_config.enable_rerank is False

    def test_create_with_custom_params(self):
        """Test factory with custom parameters."""
        qdrant_config = QdrantConfig()
        qdrant_service = QdrantService(qdrant_config, use_faiss_fallback=True)

        llm_config = VideoLLMConfig(model_name="mock")
        llm_service = VideoLLMService(llm_config)

        rag_service = create_rag_service(
            vector_service=qdrant_service,
            llm_service=llm_service,
            macro_k=3,
            meso_k=6,
            enable_rerank=True,
        )

        assert rag_service.retrieval_config.macro_k == 3
        assert rag_service.retrieval_config.meso_k == 6
        assert rag_service.retrieval_config.enable_rerank is True
