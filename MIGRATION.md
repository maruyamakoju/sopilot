# Migration Guide: SOPilot → VIGIL-RAG

**Version**: 1.0
**Status**: Active Development

---

## Overview

This document describes the migration strategy from **SOPilot** (short-form SOP scoring) to **VIGIL-RAG** (long-form video understanding + event detection).

**Key Principle**: **Additive, not destructive**
- SOPilot functionality remains intact
- VIGIL-RAG features added incrementally
- Shared infrastructure where beneficial

---

## Migration Phases

### Phase 0: Foundation (Week 1) ✓ Partially Complete

#### Completed
- [x] Git repository with CI (lint/test/smoke)
- [x] 443 tests passing
- [x] Baseline refactoring (3 waves)
- [x] Planning docs (VIGIL.md, ARCHITECTURE.md)

#### Remaining
- [ ] PostgreSQL schema design
- [ ] Qdrant integration
- [ ] Multi-scale chunking implementation
- [ ] Evaluation framework

---

### Phase 1: Database Migration (Current)

#### 1.1 PostgreSQL Schema Setup

**New Tables** (additive, no breaking changes):
```sql
-- Enhanced videos table (superset of SQLite schema)
CREATE TABLE videos_v2 (
  video_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- SOPilot compatibility fields
  file_path TEXT NOT NULL,
  task_id VARCHAR(200),
  role VARCHAR(20),  -- gold/trainee/audit
  site_id VARCHAR(100),
  camera_id VARCHAR(100),
  num_clips INT,
  embedding_model VARCHAR(100),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

  -- VIGIL-RAG new fields
  uri TEXT,  -- S3/HTTP URI (optional)
  storage_path TEXT,  -- Object storage path
  duration_sec FLOAT,
  fps FLOAT,
  width INT,
  height INT,
  domain VARCHAR(50),  -- factory/surveillance/sports
  checksum VARCHAR(64),
  ingest_time TIMESTAMP,

  -- Indexes
  CONSTRAINT unique_file_path UNIQUE(file_path)
);

CREATE INDEX idx_videos_task_id ON videos_v2(task_id);
CREATE INDEX idx_videos_domain ON videos_v2(domain);
CREATE INDEX idx_videos_camera_id ON videos_v2(camera_id);
```

**Migration Strategy**:
1. Run SQLite + PostgreSQL in parallel
2. Dual-write to both databases
3. Validate data consistency
4. Switch reads to PostgreSQL
5. Deprecate SQLite (Phase 2)

**Implementation**:
```python
# src/sopilot/db_postgres.py (new)
class PostgresDatabase(Database):
    """PostgreSQL implementation with SQLite compatibility."""

    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)

    def create_video(self, input: VideoCreateInput) -> int:
        # Map SOPilot fields to PostgreSQL schema
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                INSERT INTO videos_v2 (file_path, task_id, role, ...)
                VALUES (:file_path, :task_id, :role, ...)
                RETURNING video_id
                """),
                input.dict()
            )
            return result.fetchone()[0]
```

#### 1.2 Alembic Migrations

```bash
# Initialize Alembic
alembic init migrations

# Create migration
alembic revision --autogenerate -m "Add VIGIL-RAG schema"

# Apply migration
alembic upgrade head
```

**Migration Files**:
- `migrations/versions/001_initial_vigil_schema.py`
- `migrations/versions/002_add_clips_table.py`
- `migrations/versions/003_add_events_table.py`

---

### Phase 2: Vector Store Migration

#### 2.1 Qdrant Setup

**Docker Compose**:
```yaml
# docker-compose.yml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6333:6333"  # HTTP API
      - "6334:6334"  # gRPC API
    volumes:
      - ./data/qdrant:/qdrant/storage
    environment:
      - QDRANT_LOG_LEVEL=INFO
```

**Collection Schema**:
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="vigil_clips",
    vectors_config=VectorParams(
        size=768,  # InternVideo2 dimension
        distance=Distance.COSINE
    )
)

# Create payload index for filtering
client.create_payload_index(
    collection_name="vigil_clips",
    field_name="video_id",
    field_schema=PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name="vigil_clips",
    field_name="level",
    field_schema=PayloadSchemaType.KEYWORD
)
```

#### 2.2 FAISS → Qdrant Migration

**Migration Script**:
```python
# scripts/migrate_faiss_to_qdrant.py

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from sopilot.vector_index import NpyVectorIndex

def migrate_task(task_id: str, faiss_index: NpyVectorIndex, qdrant: QdrantClient):
    """Migrate a single task from FAISS to Qdrant."""

    # Load FAISS vectors
    vectors, metadata = faiss_index.load(task_id)

    # Convert to Qdrant points
    points = []
    for i, (vector, meta) in enumerate(zip(vectors, metadata)):
        point = PointStruct(
            id=f"{task_id}_{i}",
            vector=vector.tolist(),
            payload={
                "video_id": meta["video_id"],
                "clip_idx": meta["clip_idx"],
                "start_sec": meta.get("start_sec", 0.0),
                "end_sec": meta.get("end_sec", 0.0),
                "level": "legacy",  # Mark as pre-VIGIL
                "task_id": task_id
            }
        )
        points.append(point)

    # Batch upsert
    qdrant.upsert(
        collection_name="vigil_clips",
        points=points,
        wait=True
    )

    print(f"Migrated {len(points)} vectors for task {task_id}")
```

**Backward Compatibility**:
```python
# src/sopilot/vector_index.py (updated)

class VectorIndexAdapter:
    """Adapter pattern for FAISS/Qdrant abstraction."""

    def __init__(self, backend: str = "faiss"):
        if backend == "qdrant":
            self.impl = QdrantVectorIndex()
        else:
            self.impl = NpyVectorIndex()  # Original FAISS

    def add(self, task_id, vectors, metadata):
        return self.impl.add(task_id, vectors, metadata)

    def search(self, task_id, query_vector, k):
        return self.impl.search(task_id, query_vector, k)
```

---

### Phase 3: Multi-Scale Chunking

#### 3.1 Shot Detection

**New Module**: `src/sopilot/shot_detector.py`
```python
from scenedetect import detect, ContentDetector

class ShotDetector:
    def __init__(self, threshold: float = 27.0):
        self.threshold = threshold

    def detect_shots(self, video_path: str) -> list[tuple[float, float]]:
        """Detect shot boundaries using content-based detection.

        Returns:
            List of (start_sec, end_sec) tuples
        """
        scene_list = detect(video_path, ContentDetector(threshold=self.threshold))

        shots = []
        for i, scene in enumerate(scene_list):
            start_sec = scene[0].get_seconds()
            end_sec = scene[1].get_seconds()
            shots.append((start_sec, end_sec))

        return shots
```

#### 3.2 Multi-Level Chunker

**Enhanced**: `src/sopilot/video.py`
```python
from dataclasses import dataclass
from enum import Enum

class ChunkLevel(str, Enum):
    SHOT = "shot"
    MICRO = "micro"  # 2-4s
    MESO = "meso"    # 8-16s
    MACRO = "macro"  # 32-64s

@dataclass
class VideoChunk:
    level: ChunkLevel
    start_sec: float
    end_sec: float
    start_frame: int
    end_frame: int
    keyframes: list[int]  # Frame indices

class MultiScaleChunker:
    def __init__(self, config: dict):
        self.shot_threshold = config.get("shot_threshold", 27.0)
        self.micro_window = config.get("micro_window_sec", 3.0)
        self.micro_stride = config.get("micro_stride_sec", 1.5)
        self.meso_window = config.get("meso_window_sec", 12.0)
        self.meso_stride = config.get("meso_stride_sec", 6.0)
        self.macro_window = config.get("macro_window_sec", 48.0)
        self.macro_stride = config.get("macro_stride_sec", 24.0)

    def chunk_video(self, video_path: str, fps: float, duration_sec: float) -> list[VideoChunk]:
        """Generate multi-scale chunks for a video."""

        chunks = []

        # Level 0: Shots (adaptive)
        shot_detector = ShotDetector(self.shot_threshold)
        shots = shot_detector.detect_shots(video_path)
        for start_sec, end_sec in shots:
            chunks.append(VideoChunk(
                level=ChunkLevel.SHOT,
                start_sec=start_sec,
                end_sec=end_sec,
                start_frame=int(start_sec * fps),
                end_frame=int(end_sec * fps),
                keyframes=self._select_keyframes(start_sec, end_sec, fps, n=4)
            ))

        # Level 1: Micro (sliding windows)
        chunks.extend(self._sliding_window_chunks(
            ChunkLevel.MICRO,
            duration_sec,
            fps,
            window_sec=self.micro_window,
            stride_sec=self.micro_stride,
            n_keyframes=8
        ))

        # Level 2: Meso
        chunks.extend(self._sliding_window_chunks(
            ChunkLevel.MESO,
            duration_sec,
            fps,
            window_sec=self.meso_window,
            stride_sec=self.meso_stride,
            n_keyframes=12
        ))

        # Level 3: Macro
        chunks.extend(self._sliding_window_chunks(
            ChunkLevel.MACRO,
            duration_sec,
            fps,
            window_sec=self.macro_window,
            stride_sec=self.macro_stride,
            n_keyframes=16
        ))

        return chunks

    def _sliding_window_chunks(self, level, duration_sec, fps, window_sec, stride_sec, n_keyframes):
        chunks = []
        current_sec = 0.0
        while current_sec + window_sec <= duration_sec:
            end_sec = current_sec + window_sec
            chunks.append(VideoChunk(
                level=level,
                start_sec=current_sec,
                end_sec=end_sec,
                start_frame=int(current_sec * fps),
                end_frame=int(end_sec * fps),
                keyframes=self._select_keyframes(current_sec, end_sec, fps, n_keyframes)
            ))
            current_sec += stride_sec
        return chunks

    def _select_keyframes(self, start_sec, end_sec, fps, n):
        # Uniform sampling for now (TODO: motion saliency)
        duration = end_sec - start_sec
        interval = duration / (n + 1)
        return [int((start_sec + interval * (i + 1)) * fps) for i in range(n)]
```

#### 3.3 Integration with Ingest Service

**Updated**: `src/sopilot/ingest_service.py`
```python
class IngestService:
    def __init__(self, settings, db, index, embedding_mgr, queue):
        self.settings = settings
        self.chunker = MultiScaleChunker(settings.chunk_config)  # NEW
        # ... existing fields

    def _process_video_vigil(self, video_path: Path, video_id: int, fps: float, duration_sec: float):
        """VIGIL-RAG processing path (multi-scale)."""

        # Generate all chunks
        chunks = self.chunker.chunk_video(str(video_path), fps, duration_sec)

        # Group by level for batch embedding
        by_level = {
            "shot": [c for c in chunks if c.level == ChunkLevel.SHOT],
            "micro": [c for c in chunks if c.level == ChunkLevel.MICRO],
            "meso": [c for c in chunks if c.level == ChunkLevel.MESO],
            "macro": [c for c in chunks if c.level == ChunkLevel.MACRO],
        }

        # Process each level
        for level, level_chunks in by_level.items():
            embeddings = []
            for chunk in level_chunks:
                # Extract keyframes
                frames = self._extract_frames(video_path, chunk.keyframes)

                # Embed
                emb = self.embedding_mgr.embed_frames(frames)
                embeddings.append(emb)

            # Store to Qdrant
            self._store_embeddings_qdrant(video_id, level, level_chunks, embeddings)
```

---

### Phase 4: API Evolution

#### 4.1 Endpoint Namespacing

**SOPilot (Legacy)**: `/sop/*`
```http
POST /sop/gold
POST /sop/videos
POST /sop/score
GET  /sop/score/{job_id}
GET  /sop/score/{job_id}/report.pdf
```

**VIGIL-RAG (New)**: `/vigil/*`
```http
POST /vigil/videos          # Upload with domain/camera_id
POST /vigil/videos/{id}/index  # Trigger multi-scale indexing
POST /vigil/query           # Long-form QA
GET  /vigil/videos/{id}/events  # Event timeline
POST /vigil/events/definitions  # Define new event types
```

**Shared**: `/videos/*`, `/health`, `/metrics`

#### 4.2 Config-Based Routing

```python
# src/sopilot/config.py (updated)

@dataclass(frozen=True)
class Settings:
    # ... existing fields

    # VIGIL-RAG mode toggle
    vigil_enabled: bool
    vigil_postgres_url: str
    vigil_qdrant_url: str
    vigil_llm_model: str  # internvideo_chat / llava_video
    vigil_chunk_config: dict
```

```python
# src/sopilot/api.py (updated)

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="SOPilot + VIGIL-RAG", version="2.0.0")

    # Always include SOPilot endpoints
    app.include_router(sop_router, prefix="/sop", tags=["SOPilot"])

    # Conditionally include VIGIL-RAG
    if settings.vigil_enabled:
        app.include_router(vigil_router, prefix="/vigil", tags=["VIGIL-RAG"])

    return app
```

---

### Phase 5: Testing Strategy

#### 5.1 Backward Compatibility Tests

**Goal**: Ensure SOPilot functionality unaffected

```python
# tests/test_backward_compat.py

def test_sop_scoring_still_works():
    """Original SOPilot scoring unchanged."""
    # Use existing test from test_service_integration.py
    with tempfile.TemporaryDirectory() as td:
        settings = make_test_settings(Path(td) / "data")
        service = SopilotService(settings, db, runtime_mode="api")

        # Seed gold + trainee (original method)
        gold_id = _seed_video(service, "test_task", "gold")
        trainee_id = _seed_video(service, "test_task", "trainee")

        # Score
        result = service.enqueue_score(gold_id, trainee_id)
        assert result["status"] == "queued"
```

#### 5.2 Migration Tests

```python
# tests/test_migration.py

def test_sqlite_to_postgres_migration():
    """Verify data migrates correctly."""
    # 1. Create records in SQLite
    sqlite_db = Database(":memory:")
    video_id = sqlite_db.create_video(VideoCreateInput(...))

    # 2. Run migration script
    postgres_db = PostgresDatabase("postgresql://test")
    migrate_database(sqlite_db, postgres_db)

    # 3. Verify data
    migrated = postgres_db.get_video(video_id)
    assert migrated is not None
    assert migrated["task_id"] == original["task_id"]

def test_faiss_to_qdrant_migration():
    """Verify FAISS vectors migrate to Qdrant."""
    # ... similar pattern
```

#### 5.3 VIGIL-RAG Feature Tests

```python
# tests/test_vigil_rag.py

def test_multi_scale_chunking():
    """Verify chunking generates all 4 levels."""
    chunker = MultiScaleChunker(config)
    chunks = chunker.chunk_video(video_path, fps=30, duration_sec=600)

    levels = {c.level for c in chunks}
    assert ChunkLevel.SHOT in levels
    assert ChunkLevel.MICRO in levels
    assert ChunkLevel.MESO in levels
    assert ChunkLevel.MACRO in levels

def test_qdrant_payload_filtering():
    """Verify metadata filtering works."""
    client = QdrantClient(":memory:")
    # ... create collection, insert points

    results = client.search(
        collection_name="vigil_clips",
        query_vector=[...],
        query_filter=Filter(
            must=[
                FieldCondition(key="domain", match=MatchValue(value="factory"))
            ]
        ),
        limit=10
    )
    assert all(r.payload["domain"] == "factory" for r in results)
```

---

## Migration Checklist

### Week 1: Foundation
- [ ] Add `VIGIL.md`, `ARCHITECTURE.md`, `MIGRATION.md` to repo
- [ ] Design PostgreSQL schema (backward compatible)
- [ ] Set up Alembic migrations
- [ ] Implement `PostgresDatabase` class
- [ ] Add Qdrant to `docker-compose.yml`
- [ ] Implement `QdrantVectorIndex` class
- [ ] Create `VectorIndexAdapter` for abstraction

### Week 2: Chunking & Embedding
- [ ] Implement `ShotDetector` (PySceneDetect)
- [ ] Implement `MultiScaleChunker`
- [ ] Update `IngestService` with VIGIL path
- [ ] Add config toggle: `vigil_enabled`
- [ ] Test multi-scale chunking on 10-min video
- [ ] Benchmark: Indexing speed vs SOPilot

### Week 3: RAG & LLM
- [ ] Create `llm/` module structure
- [ ] Implement `InternVideoChatLLM` wrapper
- [ ] Implement `LLaVAVideoLLM` wrapper (alternative)
- [ ] Create `RAGService` with retrieval pipeline
- [ ] Add `POST /vigil/query` endpoint
- [ ] Evaluation: Recall@10 on 10-video QA set

### Week 4: Event Detection
- [ ] Implement zero-shot event detection
- [ ] Integrate MMAction2 (optional for Week 4)
- [ ] Add `GET /vigil/videos/{id}/events` endpoint
- [ ] Add `POST /vigil/events/definitions` endpoint
- [ ] Test: FAH (False Alarms per Hour) tracking

### Week 5: UI & Integration
- [ ] Timeline UI component (React/Vue)
- [ ] QA chat interface
- [ ] Update smoke test for VIGIL endpoints
- [ ] Load testing: Concurrent queries
- [ ] Documentation: API reference, deployment guide

---

## Rollback Plan

If migration causes critical issues:

1. **Disable VIGIL-RAG**: Set `VIGIL_ENABLED=false`
2. **Revert to SQLite**: Change connection string
3. **Revert to FAISS**: Change `vector_backend=faiss`
4. **Git revert**: Roll back to `baseline-refactor` tag

**Data Safety**:
- PostgreSQL runs alongside SQLite (dual-write)
- Qdrant is additive (FAISS index preserved)
- No destructive schema changes until Phase 5

---

## Performance Comparison

| Metric | SOPilot (Baseline) | VIGIL-RAG (Target) |
|--------|-------------------|-------------------|
| Video length | 2-10 min | 10 min - 2 hrs |
| Indexing time (10 min) | ~1 min | ~5 min |
| Query latency | N/A | <10s (indexed) |
| Concurrent videos | 3-5 | 2-3 (GPU-bound) |
| Storage overhead | 1x | 2-3x (multi-scale) |
| VRAM usage | ~12GB | ~26GB (peak) |

---

**Last Updated**: 2026-02-12
**Next Update**: After Phase 1 completion
