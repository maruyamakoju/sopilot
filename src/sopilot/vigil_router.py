"""VIGIL-RAG FastAPI router.

Provides endpoints for:
- ``POST /vigil/index`` — Index a video (chunk → embed → store)
- ``POST /vigil/search`` — Semantic search across indexed videos
- ``POST /vigil/ask`` — RAG question-answering with evidence
- ``GET  /vigil/status/{video_id}`` — Check indexing status

All endpoints use FAISS fallback when Qdrant is not available.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vigil", tags=["vigil-rag"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    video_path: str = Field(description="Absolute path to video file on server")
    video_id: str | None = Field(default=None, description="Video ID (auto-computed SHA-256 if omitted)")
    domain: str = Field(default="generic", description="Video domain (factory, surveillance, sports, generic)")
    hierarchical: bool = Field(default=False, description="Also index meso+macro levels for coarse-to-fine retrieval")
    transcribe: bool = Field(default=False, description="Run Whisper transcription for hybrid audio search")


class IndexResponse(BaseModel):
    video_id: str
    num_micro: int = Field(description="Number of micro-level embeddings stored")
    num_meso: int = Field(default=0, description="Number of meso-level embeddings stored")
    num_macro: int = Field(default=0, description="Number of macro-level embeddings stored")
    num_text: int = Field(default=0, description="Number of micro_text (transcript) embeddings stored")
    num_transcript_segments: int = Field(default=0, description="Number of Whisper transcript segments")


class SearchRequest(BaseModel):
    query: str = Field(description="Natural language search query")
    video_id: str | None = Field(default=None, description="Filter to specific video")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    use_hybrid: bool = Field(default=True, description="Enable hybrid visual+audio search")
    alpha: float = Field(default=0.3, ge=0.0, le=1.0, description="Audio fusion weight")


class SearchResultItem(BaseModel):
    clip_id: str
    video_id: str
    start_sec: float
    end_sec: float
    score: float
    transcript_text: str | None = None


class SearchResponse(BaseModel):
    query: str
    num_results: int
    results: list[SearchResultItem]


class AskRequest(BaseModel):
    question: str = Field(description="Question to answer about the video(s)")
    video_path: str = Field(description="Path to the video file")
    video_id: str | None = Field(default=None, description="Filter retrieval to this video")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of evidence clips")
    enable_rerank: bool = Field(default=False, description="Enable MMR diversity re-ranking")


class EvidenceItem(BaseModel):
    clip_id: str
    video_id: str
    level: str
    start_sec: float
    end_sec: float
    score: float
    transcript_text: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float | None = None
    evidence: list[EvidenceItem]
    num_clips_observed: int = 0


class VideoStatusResponse(BaseModel):
    video_id: str
    num_micro: int
    num_meso: int
    num_macro: int
    num_text: int
    indexed: bool


# ---------------------------------------------------------------------------
# Service accessor (lazy init)
# ---------------------------------------------------------------------------


def _get_vigil_services(request: Request):
    """Get or lazily create VIGIL-RAG services from app state.

    Services are cached on ``app.state`` after first creation.
    """
    app = request.app

    if not hasattr(app.state, "_vigil_qdrant"):
        from .qdrant_service import QdrantConfig, QdrantService
        from .rag_service import RAGService, RetrievalConfig
        from .video_llm_service import VideoLLMConfig, VideoLLMService

        qdrant_config = QdrantConfig()
        qdrant = QdrantService(qdrant_config, use_faiss_fallback=True)

        llm_config = VideoLLMConfig(model_name="mock")
        llm = VideoLLMService(llm_config)

        retrieval_config = RetrievalConfig()

        # Try loading OpenCLIP embedder (optional)
        retrieval_embedder = None
        try:
            from .retrieval_embeddings import RetrievalConfig as EmbConfig
            from .retrieval_embeddings import RetrievalEmbedder

            emb_config = EmbConfig(model_name="ViT-B-32", device="cpu")
            retrieval_embedder = RetrievalEmbedder(emb_config)
        except Exception as exc:
            logger.warning("OpenCLIP not available, using mock embeddings: %s", exc)

        rag = RAGService(
            vector_service=qdrant,
            llm_service=llm,
            retrieval_config=retrieval_config,
            retrieval_embedder=retrieval_embedder,
        )

        app.state._vigil_qdrant = qdrant
        app.state._vigil_llm = llm
        app.state._vigil_rag = rag
        app.state._vigil_embedder = retrieval_embedder
        app.state._vigil_indexed_videos: set[str] = set()
        logger.info("VIGIL-RAG services initialized")

    return {
        "qdrant": app.state._vigil_qdrant,
        "llm": app.state._vigil_llm,
        "rag": app.state._vigil_rag,
        "embedder": app.state._vigil_embedder,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/index", response_model=IndexResponse)
def index_video(payload: IndexRequest, request: Request) -> IndexResponse:
    """Index a video: chunk, encode keyframes, store embeddings."""
    services = _get_vigil_services(request)
    qdrant = services["qdrant"]
    embedder = services["embedder"]

    if embedder is None:
        raise HTTPException(status_code=503, detail="OpenCLIP embedder not available")

    video_path = Path(payload.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {payload.video_path}")

    # Compute video_id if not provided
    video_id = payload.video_id
    if video_id is None:
        from .rag_service import compute_video_id

        video_id = compute_video_id(video_path)

    try:
        from .chunking_service import ChunkingService

        chunker = ChunkingService()

        tx_service = None
        if payload.transcribe:
            from .transcription_service import TranscriptionConfig, TranscriptionService

            tx_config = TranscriptionConfig(backend="openai-whisper")
            tx_service = TranscriptionService(tx_config)

        if payload.hierarchical:
            from .vigil_helpers import index_video_all_levels

            result = index_video_all_levels(
                video_path,
                video_id,
                chunker,
                embedder,
                qdrant,
                domain=payload.domain,
                transcription_service=tx_service,
            )
        else:
            from .vigil_helpers import index_video_micro

            result = index_video_micro(
                video_path,
                video_id,
                chunker,
                embedder,
                qdrant,
                domain=payload.domain,
                transcription_service=tx_service,
            )

        request.app.state._vigil_indexed_videos.add(video_id)

        return IndexResponse(
            video_id=video_id,
            num_micro=result.get("num_added", 0),
            num_meso=result.get("num_meso_added", 0),
            num_macro=result.get("num_macro_added", 0),
            num_text=result.get("num_text_added", 0),
            num_transcript_segments=len(result.get("transcript_segments", [])),
        )
    except Exception as exc:
        logger.error("Index failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc


@router.post("/search", response_model=SearchResponse)
def search_clips(payload: SearchRequest, request: Request) -> SearchResponse:
    """Search indexed videos by natural language query."""
    services = _get_vigil_services(request)
    qdrant = services["qdrant"]
    embedder = services["embedder"]

    if embedder is None:
        raise HTTPException(status_code=503, detail="OpenCLIP embedder not available")

    try:
        import contextlib

        query_embedding = embedder.encode_text([payload.query])[0]

        # Visual search
        visual_results = qdrant.search(
            level="micro",
            query_vector=query_embedding,
            top_k=payload.top_k,
            video_id=payload.video_id,
        )

        # Hybrid fusion
        if payload.use_hybrid and payload.alpha > 0:
            audio_results = []
            with contextlib.suppress(Exception):
                audio_results = qdrant.search(
                    level="micro_text",
                    query_vector=query_embedding,
                    top_k=payload.top_k,
                    video_id=payload.video_id,
                )

            if audio_results:
                visual_map = {r.clip_id: r for r in visual_results}
                audio_map = {r.clip_id: r for r in audio_results}
                all_ids = set(visual_map) | set(audio_map)

                fused = []
                for cid in all_ids:
                    v = visual_map.get(cid)
                    a = audio_map.get(cid)
                    v_score = v.score if v else 0.0
                    a_score = (a.score * payload.alpha) if a else 0.0
                    base = v or a
                    fused.append(
                        SearchResultItem(
                            clip_id=base.clip_id,
                            video_id=base.video_id,
                            start_sec=base.start_sec,
                            end_sec=base.end_sec,
                            score=max(v_score, a_score),
                            transcript_text=base.transcript_text or (a.transcript_text if a else None),
                        )
                    )

                fused.sort(key=lambda x: x.score, reverse=True)
                items = fused[: payload.top_k]
                return SearchResponse(query=payload.query, num_results=len(items), results=items)

        items = [
            SearchResultItem(
                clip_id=r.clip_id,
                video_id=r.video_id,
                start_sec=r.start_sec,
                end_sec=r.end_sec,
                score=r.score,
                transcript_text=r.transcript_text,
            )
            for r in visual_results
        ]
        return SearchResponse(query=payload.query, num_results=len(items), results=items)

    except Exception as exc:
        logger.error("Search failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc


@router.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest, request: Request) -> AskResponse:
    """Answer a question about video(s) using RAG pipeline."""
    services = _get_vigil_services(request)
    rag = services["rag"]

    video_path = Path(payload.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {payload.video_path}")

    try:
        if payload.enable_rerank:
            rag.retrieval_config.enable_rerank = True

        result = rag.answer_question_topk(
            video_path,
            payload.question,
            video_id=payload.video_id,
            top_k=payload.top_k,
        )

        evidence = [
            EvidenceItem(
                clip_id=e.clip_id,
                video_id=e.video_id,
                level=e.level,
                start_sec=e.start_sec,
                end_sec=e.end_sec,
                score=e.score,
                transcript_text=e.transcript_text,
            )
            for e in result.evidence
        ]

        return AskResponse(
            question=result.question,
            answer=result.answer,
            confidence=result.confidence,
            evidence=evidence,
            num_clips_observed=len(result.clip_observations or []),
        )
    except Exception as exc:
        logger.error("Ask failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Question answering failed: {exc}") from exc


@router.get("/status/{video_id}", response_model=VideoStatusResponse)
def get_video_status(video_id: str, request: Request) -> VideoStatusResponse:
    """Check indexing status for a video."""
    services = _get_vigil_services(request)
    qdrant = services["qdrant"]

    micro = qdrant.count_by_video("micro", video_id)
    meso = qdrant.count_by_video("meso", video_id)
    macro = qdrant.count_by_video("macro", video_id)
    text = qdrant.count_by_video("micro_text", video_id)

    return VideoStatusResponse(
        video_id=video_id,
        num_micro=micro,
        num_meso=meso,
        num_macro=macro,
        num_text=text,
        indexed=micro > 0,
    )
