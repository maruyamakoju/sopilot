"""Qdrant vector database integration for VIGIL-RAG.

This module provides:
- Qdrant client wrapper with connection pooling
- Multi-level embedding storage (shot/micro/meso/macro)
- Similarity search with filtering by level/video_id
- FAISS fallback for development (when Qdrant not available)

Design principles:
- Separate collections per level for optimized retrieval
- Metadata stored alongside vectors (video_id, start_sec, end_sec)
- Coarse-to-fine retrieval pattern (macro → meso → micro → shot)
- Graceful degradation to FAISS when Qdrant unavailable
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

logger = logging.getLogger(__name__)

ChunkLevel = Literal["shot", "micro", "meso", "macro"]


@dataclass
class SearchResult:
    """Vector search result."""

    clip_id: str  # UUID as string
    video_id: str  # UUID as string
    level: ChunkLevel
    start_sec: float
    end_sec: float
    score: float  # Similarity score (higher is better)
    embedding: np.ndarray | None = None  # Optional: return embedding


@dataclass
class QdrantConfig:
    """Qdrant connection configuration."""

    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False
    api_key: str | None = None
    https: bool = False
    embedding_dim: int = 768  # Default for InternVideo2
    distance_metric: Distance = Distance.COSINE

    @property
    def connection_url(self) -> str:
        """Get connection URL for Qdrant."""
        protocol = "https" if self.https else "http"
        return f"{protocol}://{self.host}:{self.port}"


class QdrantService:
    """Qdrant vector database service."""

    # Collection name format: vigil_{level} (e.g., vigil_shot, vigil_micro)
    COLLECTION_PREFIX = "vigil"

    def __init__(
        self,
        config: QdrantConfig,
        *,
        use_faiss_fallback: bool = True,
    ) -> None:
        """Initialize Qdrant service.

        Args:
            config: Qdrant connection configuration
            use_faiss_fallback: If True, use FAISS when Qdrant unavailable

        Raises:
            RuntimeError: If Qdrant not available and fallback disabled
        """
        self.config = config
        self.use_faiss_fallback = use_faiss_fallback
        self._client: QdrantClient | None = None
        self._faiss_indexes: dict[ChunkLevel, dict] = {}  # Fallback FAISS indexes

        if not QDRANT_AVAILABLE and not use_faiss_fallback:
            raise RuntimeError(
                "qdrant-client not installed and fallback disabled. "
                "Install with: pip install qdrant-client"
            )

        if QDRANT_AVAILABLE:
            try:
                self._client = self._connect()
                logger.info("Vector backend: qdrant (%s)", self.config.connection_url)
            except Exception as exc:
                if use_faiss_fallback:
                    logger.warning("Qdrant connection failed, using FAISS fallback: %s", exc)
                    self._client = None
                    logger.info("Vector backend: faiss (in-memory)")
                else:
                    raise
        else:
            logger.info("Vector backend: faiss (in-memory, qdrant-client not installed)")

    def _connect(self) -> QdrantClient:
        """Establish connection to Qdrant server.

        Returns:
            QdrantClient instance

        Raises:
            Exception: If connection fails
        """
        logger.info("Connecting to Qdrant at %s", self.config.connection_url)

        if self.config.prefer_grpc:
            client = QdrantClient(
                host=self.config.host,
                grpc_port=self.config.grpc_port,
                api_key=self.config.api_key,
                https=self.config.https,
                prefer_grpc=True,
            )
        else:
            client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key,
                https=self.config.https,
            )

        # Test connection
        client.get_collections()
        logger.info("Successfully connected to Qdrant")
        return client

    def _get_collection_name(self, level: ChunkLevel) -> str:
        """Get collection name for a chunk level.

        Args:
            level: Chunk level

        Returns:
            Collection name (e.g., "vigil_micro")
        """
        return f"{self.COLLECTION_PREFIX}_{level}"

    def ensure_collections(
        self,
        levels: list[ChunkLevel] | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        """Ensure collections exist for specified levels.

        Args:
            levels: List of levels to create (defaults to all 4 levels)
            embedding_dim: Embedding dimension (defaults to config value)

        Raises:
            RuntimeError: If using FAISS fallback (collections not applicable)
        """
        if self._client is None:
            logger.warning("Using FAISS fallback, skipping collection creation")
            return

        if levels is None:
            levels = ["shot", "micro", "meso", "macro"]

        dim = embedding_dim or self.config.embedding_dim

        for level in levels:
            collection_name = self._get_collection_name(level)

            # Check if collection exists
            collections = self._client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)

            if not exists:
                logger.info("Creating collection: %s (dim=%d)", collection_name, dim)
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=dim,
                        distance=self.config.distance_metric,
                    ),
                )
            else:
                logger.debug("Collection already exists: %s", collection_name)

    def add_embeddings(
        self,
        level: ChunkLevel,
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> int:
        """Add embeddings to a collection.

        Args:
            level: Chunk level
            embeddings: Embedding vectors (N, D)
            metadata: List of metadata dicts (one per embedding)
                Required keys: clip_id, video_id, start_sec, end_sec

        Returns:
            Number of embeddings added

        Raises:
            ValueError: If embeddings/metadata mismatch or missing keys
        """
        if len(embeddings) != len(metadata):
            raise ValueError(f"Embeddings/metadata count mismatch: {len(embeddings)} vs {len(metadata)}")

        if len(embeddings) == 0:
            return 0

        # Validate metadata
        required_keys = {"clip_id", "video_id", "start_sec", "end_sec"}
        for i, meta in enumerate(metadata):
            missing = required_keys - set(meta.keys())
            if missing:
                raise ValueError(f"Metadata[{i}] missing keys: {missing}")

        if self._client is None:
            # FAISS fallback
            return self._add_embeddings_faiss(level, embeddings, metadata)

        # Qdrant storage
        collection_name = self._get_collection_name(level)
        points = [
            PointStruct(
                id=str(meta["clip_id"]),
                vector=embeddings[i].tolist(),
                payload={
                    "video_id": str(meta["video_id"]),
                    "level": level,
                    "start_sec": float(meta["start_sec"]),
                    "end_sec": float(meta["end_sec"]),
                    **{k: v for k, v in meta.items() if k not in required_keys},
                },
            )
            for i, meta in enumerate(metadata)
        ]

        self._client.upsert(collection_name=collection_name, points=points)
        logger.info("Added %d embeddings to %s", len(points), collection_name)
        return len(points)

    def search(
        self,
        level: ChunkLevel,
        query_vector: np.ndarray,
        *,
        top_k: int = 10,
        video_id: str | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Search for similar embeddings.

        Args:
            level: Chunk level to search
            query_vector: Query embedding vector (D,)
            top_k: Number of results to return
            video_id: Optional filter by video ID
            min_score: Optional minimum similarity score

        Returns:
            List of SearchResult objects, sorted by score (descending)
        """
        if self._client is None:
            # FAISS fallback
            return self._search_faiss(level, query_vector, top_k, video_id, min_score)

        # Qdrant search
        collection_name = self._get_collection_name(level)

        # Build filter if video_id specified
        search_filter = None
        if video_id is not None:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="video_id",
                        match=MatchValue(value=str(video_id)),
                    )
                ]
            )

        response = self._client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=search_filter,
            score_threshold=min_score,
        )

        results: list[SearchResult] = []
        for hit in response.points:
            results.append(
                SearchResult(
                    clip_id=str(hit.id),
                    video_id=str(hit.payload["video_id"]),
                    level=level,
                    start_sec=float(hit.payload["start_sec"]),
                    end_sec=float(hit.payload["end_sec"]),
                    score=float(hit.score),
                )
            )

        logger.debug("Found %d results for level=%s, top_k=%d", len(results), level, top_k)
        return results

    def coarse_to_fine_search(
        self,
        query_vector: np.ndarray,
        *,
        video_id: str | None = None,
        macro_k: int = 5,
        meso_k: int = 10,
        micro_k: int = 20,
        shot_k: int = 30,
    ) -> dict[ChunkLevel, list[SearchResult]]:
        """Perform coarse-to-fine retrieval across all levels.

        Search order: macro → meso → micro → shot
        Each level filters to relevant time ranges from coarser level.

        Args:
            query_vector: Query embedding vector
            video_id: Optional filter by video ID
            macro_k: Top-k for macro level
            meso_k: Top-k for meso level
            micro_k: Top-k for micro level
            shot_k: Top-k for shot level

        Returns:
            Dictionary mapping level to search results
        """
        results: dict[ChunkLevel, list[SearchResult]] = {}

        # Step 1: Macro search (coarsest)
        macro_results = self.search(
            level="macro",
            query_vector=query_vector,
            top_k=macro_k,
            video_id=video_id,
        )
        results["macro"] = macro_results

        if not macro_results:
            # No macro results, return empty for all levels
            results["meso"] = []
            results["micro"] = []
            results["shot"] = []
            return results

        # Step 2: Meso search (within macro time ranges)
        # For simplicity, search all meso chunks (filtering by time range is TODO)
        meso_results = self.search(
            level="meso",
            query_vector=query_vector,
            top_k=meso_k,
            video_id=video_id,
        )
        results["meso"] = meso_results

        # Step 3: Micro search
        micro_results = self.search(
            level="micro",
            query_vector=query_vector,
            top_k=micro_k,
            video_id=video_id,
        )
        results["micro"] = micro_results

        # Step 4: Shot search (finest)
        shot_results = self.search(
            level="shot",
            query_vector=query_vector,
            top_k=shot_k,
            video_id=video_id,
        )
        results["shot"] = shot_results

        logger.info(
            "Coarse-to-fine search: macro=%d, meso=%d, micro=%d, shot=%d",
            len(macro_results),
            len(meso_results),
            len(micro_results),
            len(shot_results),
        )

        return results

    def count_by_video(self, level: ChunkLevel, video_id: str) -> int:
        """Count embeddings stored for a specific video.

        Args:
            level: Chunk level to check.
            video_id: Video ID to filter.

        Returns:
            Number of stored embeddings for this video.
        """
        if self._client is None:
            return self._count_by_video_faiss(level, video_id)

        collection_name = self._get_collection_name(level)
        try:
            result = self._client.count(
                collection_name=collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="video_id",
                            match=MatchValue(value=str(video_id)),
                        )
                    ]
                ),
                exact=True,
            )
            return result.count
        except Exception:
            # Collection may not exist yet
            return 0

    # ===== FAISS Fallback Implementation =====

    def _add_embeddings_faiss(
        self,
        level: ChunkLevel,
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> int:
        """FAISS fallback for add_embeddings."""
        if level not in self._faiss_indexes:
            self._faiss_indexes[level] = {
                "vectors": [],
                "metadata": [],
            }

        index = self._faiss_indexes[level]
        index["vectors"].append(embeddings)
        index["metadata"].extend(metadata)

        logger.debug("Added %d embeddings to FAISS %s index", len(embeddings), level)
        return len(embeddings)

    def _count_by_video_faiss(self, level: ChunkLevel, video_id: str) -> int:
        """FAISS fallback for count_by_video."""
        if level not in self._faiss_indexes:
            return 0
        metadata = self._faiss_indexes[level].get("metadata", [])
        return sum(1 for m in metadata if str(m.get("video_id")) == str(video_id))

    def _search_faiss(
        self,
        level: ChunkLevel,
        query_vector: np.ndarray,
        top_k: int,
        video_id: str | None,
        min_score: float | None,
    ) -> list[SearchResult]:
        """FAISS fallback for search."""
        if level not in self._faiss_indexes or not self._faiss_indexes[level]["vectors"]:
            return []

        index = self._faiss_indexes[level]
        all_vectors = np.vstack(index["vectors"])
        all_metadata = index["metadata"]

        # Filter by video_id if specified
        if video_id is not None:
            mask = np.array([str(m["video_id"]) == str(video_id) for m in all_metadata])
            if not np.any(mask):
                return []
            vectors = all_vectors[mask]
            metadata = [m for m, keep in zip(all_metadata, mask) if keep]
        else:
            vectors = all_vectors
            metadata = all_metadata

        # Compute cosine similarity
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        vector_norms = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
        scores = vector_norms @ query_norm

        # Filter by min_score
        if min_score is not None:
            mask = scores >= min_score
            scores = scores[mask]
            metadata = [m for m, keep in zip(metadata, mask) if keep]

        # Sort by score descending and take top-k
        sorted_indices = np.argsort(-scores)[:top_k]

        results: list[SearchResult] = []
        for idx in sorted_indices:
            meta = metadata[idx]
            results.append(
                SearchResult(
                    clip_id=str(meta["clip_id"]),
                    video_id=str(meta["video_id"]),
                    level=level,
                    start_sec=float(meta["start_sec"]),
                    end_sec=float(meta["end_sec"]),
                    score=float(scores[idx]),
                )
            )

        return results
