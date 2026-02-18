"""Tests for VIGIL-RAG FastAPI router."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sopilot.vigil_router import router


@pytest.fixture()
def app():
    """Create a minimal FastAPI app with the VIGIL router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture()
def client(app):
    """Test client for the VIGIL router."""
    return TestClient(app)


@pytest.fixture()
def _init_services(app):
    """Pre-initialize VIGIL services with mock/FAISS backend."""
    from sopilot.qdrant_service import QdrantConfig, QdrantService
    from sopilot.rag_service import RAGService, RetrievalConfig
    from sopilot.video_llm_service import VideoLLMConfig, VideoLLMService

    qdrant = QdrantService(QdrantConfig(host="localhost", port=19999), use_faiss_fallback=True)
    llm = VideoLLMService(VideoLLMConfig(model_name="mock"))
    rag = RAGService(
        vector_service=qdrant,
        llm_service=llm,
        retrieval_config=RetrievalConfig(),
    )

    app.state._vigil_qdrant = qdrant
    app.state._vigil_llm = llm
    app.state._vigil_rag = rag
    app.state._vigil_embedder = None  # No real embedder for unit tests
    app.state._vigil_indexed_videos = set()


class TestVigilSearchEndpoint:
    """Tests for POST /vigil/search."""

    @pytest.mark.usefixtures("_init_services")
    def test_search_no_embedder_returns_503(self, client):
        """Search without embedder should return 503."""
        resp = client.post("/vigil/search", json={"query": "test"})
        assert resp.status_code == 503
        assert "embedder" in resp.json()["detail"].lower()


class TestVigilIndexEndpoint:
    """Tests for POST /vigil/index."""

    @pytest.mark.usefixtures("_init_services")
    def test_index_no_embedder_returns_503(self, client):
        """Index without embedder should return 503."""
        resp = client.post("/vigil/index", json={"video_path": "nonexistent.mp4"})
        assert resp.status_code == 503

    @pytest.mark.usefixtures("_init_services")
    def test_index_missing_video_returns_404(self, client, app):
        """Index with non-existent video should return 404."""
        from unittest.mock import MagicMock

        # Provide a mock embedder so we get past the 503 check
        mock_embedder = MagicMock()
        mock_embedder.config.embedding_dim = 512
        app.state._vigil_embedder = mock_embedder

        resp = client.post("/vigil/index", json={"video_path": "/nonexistent/path/video.mp4"})
        assert resp.status_code == 404


class TestVigilStatusEndpoint:
    """Tests for GET /vigil/status/{video_id}."""

    @pytest.mark.usefixtures("_init_services")
    def test_status_unknown_video(self, client):
        """Status for unknown video should show 0 counts."""
        resp = client.get("/vigil/status/unknown-vid")
        assert resp.status_code == 200
        data = resp.json()
        assert data["video_id"] == "unknown-vid"
        assert data["indexed"] is False
        assert data["num_micro"] == 0

    @pytest.mark.usefixtures("_init_services")
    def test_status_after_manual_insert(self, client, app):
        """Status should reflect embeddings added to FAISS."""
        qdrant = app.state._vigil_qdrant
        embeddings = np.random.randn(3, 8).astype(np.float32)
        metadata = [
            {"clip_id": f"c{i}", "video_id": "test-vid", "start_sec": i * 5.0, "end_sec": (i + 1) * 5.0}
            for i in range(3)
        ]
        qdrant.add_embeddings("micro", embeddings, metadata)

        resp = client.get("/vigil/status/test-vid")
        assert resp.status_code == 200
        data = resp.json()
        assert data["video_id"] == "test-vid"
        assert data["indexed"] is True
        assert data["num_micro"] == 3


class TestVigilAskEndpoint:
    """Tests for POST /vigil/ask."""

    @pytest.mark.usefixtures("_init_services")
    def test_ask_missing_video_returns_404(self, client):
        """Ask with non-existent video path should return 404."""
        resp = client.post(
            "/vigil/ask",
            json={
                "question": "What happens?",
                "video_path": "/nonexistent/video.mp4",
            },
        )
        assert resp.status_code == 404

    @pytest.mark.usefixtures("_init_services")
    def test_ask_with_mock_video(self, client, app):
        """Ask with a valid video should return a response (mock LLM)."""
        # Create a tiny test video
        import cv2

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            pass
        video_path = Path(tmp.name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30, (64, 64))
        for i in range(30):
            frame = np.full((64, 64, 3), fill_value=(i * 8 % 256), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        try:
            resp = client.post(
                "/vigil/ask",
                json={
                    "question": "What happens in the video?",
                    "video_path": str(video_path),
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["question"] == "What happens in the video?"
            assert "answer" in data
            assert isinstance(data["evidence"], list)
        finally:
            video_path.unlink(missing_ok=True)
