# VIGIL-RAG System Architecture

**Version**: 1.0
**Date**: 2026-02-12

---

## 1. Architecture Overview

VIGIL-RAG extends SOPilot's video understanding to long-form videos through a **dual-path RAG architecture**:

```
                    ┌──────────────────────────────────┐
                    │       Video Input (Hours)         │
                    └───────────┬──────────────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │   Ingestion Pipeline      │
                    │  (Decode + Segment)       │
                    └───────────┬──────────────┘
                                │
                ┌───────────────┴────────────────┐
                │                                 │
        ┌───────▼────────┐              ┌───────▼────────┐
        │  Multi-Scale    │              │   Multi-Modal  │
        │   Chunking      │              │   Features     │
        │ (4 granularities)│             │ (V+A+T)        │
        └───────┬─────────┘              └────────┬───────┘
                │                                  │
                └──────────────┬───────────────────┘
                               │
                   ┌───────────▼────────────┐
                   │  Embedding Extraction   │
                   │  (InternVideo/V-JEPA2)  │
                   └───────────┬────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                  │
       ┌──────▼────────┐              ┌────────▼────────┐
       │  Vector Store  │              │  Metadata Store  │
       │   (Qdrant)     │              │  (PostgreSQL)    │
       └──────┬─────────┘              └─────────┬────────┘
              │                                   │
              └──────────────┬────────────────────┘
                             │
          ┌──────────────────┴─────────────────┐
          │                                     │
   ┌──────▼──────┐                     ┌───────▼───────┐
   │  RAG Engine  │                     │ Event Detector │
   │  (Retrieve   │                     │  (3 Methods)   │
   │   + Reason)  │                     │                │
   └──────┬───────┘                     └───────┬────────┘
          │                                     │
          └──────────────┬──────────────────────┘
                         │
                ┌────────▼─────────┐
                │   API Gateway     │
                │  (FastAPI + gRPC) │
                └────────┬──────────┘
                         │
          ┌──────────────┴───────────────┐
          │                               │
   ┌──────▼──────┐              ┌────────▼────────┐
   │   QA Chat    │              │ Event Timeline  │
   │     UI       │              │      UI         │
   └──────────────┘              └─────────────────┘
```

---

## 2. Component Details

### 2.1 Ingestion Pipeline

**Responsibilities:**
- Video decoding (PyAV/Decord)
- Shot detection (scene boundary detection)
- Multi-scale chunking (4 levels)
- Keyframe extraction
- Audio/transcript extraction (optional)

**Technology Stack:**
- **Decord**: Fast GPU-accelerated video loader
- **PySceneDetect**: Shot boundary detection
- **Whisper** (optional): Audio transcription for multi-modal

**Configuration:**
```python
CHUNK_CONFIG = {
    "shot": {"method": "adaptive", "min_sec": 1.0, "max_sec": 10.0},
    "micro": {"window_sec": 3.0, "stride_sec": 1.5},
    "meso": {"window_sec": 12.0, "stride_sec": 6.0},
    "macro": {"window_sec": 48.0, "stride_sec": 24.0}
}
```

### 2.2 Feature Extraction

**Visual Embeddings:**
- **Primary**: InternVideo2 (6B params, 768-dim)
- **Fallback**: V-JEPA2 (existing SOPilot)
- **Sampling**: 2 fps (micro/meso), 1 fps (macro)

**Audio Embeddings** (optional):
- **CLAP** or **ImageBind** for audio-text alignment

**Text Embeddings:**
- **Shared with visual**: InternVideo text encoder
- **Fallback**: SentenceTransformers (all-MiniLM-L6-v2)

### 2.3 Vector Store (Qdrant)

**Why Qdrant:**
- Payload filtering (camera_id, domain, time_range)
- HNSW indexing (fast approximate search)
- gRPC API (low latency)
- Persistence + replication

**Index Structure:**
```python
Collection: "vigil_clips"
  Vector Config:
    - size: 768
    - distance: Cosine
  Payload Schema:
    - video_id: str (indexed)
    - clip_id: str (indexed)
    - level: str (shot/micro/meso/macro)
    - start_sec: float
    - end_sec: float
    - domain: str (factory/surveillance/sports)
    - camera_id: str (indexed)
    - timestamp: datetime (indexed)
```

**Backward Compatibility:**
- FAISS wrapper for local/dev deployments
- Migration script: FAISS → Qdrant export/import

### 2.4 Metadata Store (PostgreSQL)

**Why PostgreSQL:**
- ACID compliance (audit trails)
- JSONB support (flexible event metadata)
- PostGIS (optional spatial queries for camera coverage)
- Time-series extensions (TimescaleDB for event analytics)

**Schema Migration:**
- Alembic for version control
- Backward compatible with SQLite (dev mode)
- Migration path: SQLite → PostgreSQL export

**Key Tables:**
1. `videos` - Video metadata + domain/camera tagging
2. `clips` - Multi-level clip records
3. `embeddings` - Model version tracking
4. `events` - Detected events with evidence links
5. `queries` - QA history for analytics

### 2.5 RAG Engine

**Components:**

#### A. Retrieval (Multi-Stage)
```python
# Stage 1: Coarse retrieval (macro level)
macro_candidates = vector_search(
    query_embedding,
    collection="vigil_clips",
    filter={"level": "macro", "video_id": target_video},
    limit=50
)

# Stage 2: Focused retrieval (meso/micro in candidate regions)
focused_candidates = []
for macro_clip in macro_candidates:
    time_range = (macro_clip.start_sec, macro_clip.end_sec)
    meso_clips = vector_search(
        query_embedding,
        filter={"level": "meso", "start_sec": {"$gte": time_range[0]}},
        limit=10
    )
    focused_candidates.extend(meso_clips)

# Stage 3: Re-ranking
top_k = rerank(focused_candidates, query_text, k=10)
```

#### B. Evidence Packing
```python
EvidencePack = {
    "clips": [
        {
            "clip_id": uuid,
            "start_sec": 123.4,
            "end_sec": 135.2,
            "keyframes": [url1, url2, ...],  # 8-16 frames
            "transcript": "...",  # optional
            "context_before": {...},  # ±10s context
            "context_after": {...}
        }
    ],
    "query": "What caused the machine stop?",
    "video_metadata": {...}
}
```

#### C. Video-LLM Inference
```python
# Prompt Template
PROMPT = """
You are analyzing a factory surveillance video.

Question: {question}

Evidence clips (timestamp + keyframes):
{evidence_pack}

Instructions:
1. Answer based ONLY on provided evidence
2. Cite specific timestamps
3. If evidence is insufficient, say "Unknown - insufficient evidence"

Output JSON:
{
  "answer": "...",
  "confidence": 0.0-1.0,
  "evidence_used": [{"start_sec": ..., "why": "..."}],
  "unknowns": [...]
}
"""

# Inference
response = video_llm.generate(
    prompt=PROMPT,
    images=[keyframes_batch],
    max_tokens=512,
    temperature=0.3  # Lower = more deterministic
)
```

**Model Selection:**
- **Primary**: InternVideo2.5 Chat 8B (long-context optimized)
- **Alternative**: LLaVA-Video 7B (strong visual grounding)
- **Quantization**: 4-bit (fits in 8GB VRAM)

### 2.6 Event Detection Engine

**Method 1: Zero-Shot (Fast Deployment)**
```python
# Define event via text
event_def = {
    "type": "forklift_intrusion",
    "description": "Yellow forklift enters red-marked restricted zone",
    "visual_cues": ["forklift", "red floor marking", "crossing boundary"]
}

# Embed definition
event_embedding = text_encoder(event_def["description"])

# Search for similar clips
candidates = vector_search(
    event_embedding,
    filter={"domain": "factory"},
    score_threshold=0.75
)

# Calibrate threshold with 10-50 labeled examples
calibrated_threshold = calibrate(candidates, ground_truth_examples)
```

**Method 2: Supervised (High Precision)**
```python
# MMAction2 Temporal Action Localization
from mmaction.apis import init_recognizer, inference_recognizer

model = init_recognizer(
    config='configs/localization/bmn/bmn_400x100_9e_activitynet_video.py',
    checkpoint='checkpoints/bmn_400x100_9e.pth',
    device='cuda:0'
)

# Inference
results = inference_recognizer(model, video_path)
# Returns: List[(event_type, start_sec, end_sec, confidence)]
```

**Method 3: Anomaly Detection**
```python
# Train on "normal" embeddings
from sklearn.ensemble import IsolationForest

normal_embeddings = get_normal_clips(video_history)
anomaly_detector = IsolationForest(contamination=0.05)
anomaly_detector.fit(normal_embeddings)

# Score new clips
new_clips = get_latest_clips()
anomaly_scores = anomaly_detector.decision_function(new_clips)
anomalies = new_clips[anomaly_scores < threshold]
```

---

## 3. Data Flow Diagrams

### 3.1 Indexing Flow (Offline/Batch)

```
┌──────────────┐
│ Upload Video │
└──────┬───────┘
       │
┌──────▼──────────────────────────────────────────┐
│ 1. Decode + Validate                             │
│    - Check codec, duration, fps                  │
│    - Extract metadata (resolution, bitrate)      │
└──────┬──────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────┐
│ 2. Shot Detection                                │
│    - PySceneDetect (adaptive threshold)          │
│    - Output: List[(start_frame, end_frame)]      │
└──────┬──────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────┐
│ 3. Multi-Scale Chunking                          │
│    - Micro: 3s sliding windows (stride 1.5s)     │
│    - Meso: 12s sliding windows (stride 6s)       │
│    - Macro: 48s sliding windows (stride 24s)     │
│    - Shot: Variable (adaptive to scene changes)  │
└──────┬──────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────┐
│ 4. Keyframe Extraction (per clip)                │
│    - Select N frames (N=8 for micro, 16 for macro)│
│    - Criteria: Motion saliency + diversity       │
└──────┬──────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────┐
│ 5. Embedding Extraction                          │
│    - Video encoder: InternVideo2 (768-dim)       │
│    - Batch size: 16 clips/batch                  │
│    - GPU: RTX 5090 (batch through all clips)     │
└──────┬──────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────┐
│ 6. Optional: Transcript Extraction               │
│    - Whisper large-v3 (if audio present)         │
│    - Aligned to clip timestamps                  │
└──────┬──────────────────────────────────────────┘
       │
       ├───────────────────┬──────────────────────┐
       │                   │                      │
┌──────▼──────┐  ┌────────▼────────┐  ┌──────────▼────────┐
│ Qdrant       │  │ PostgreSQL       │  │ Object Storage    │
│ (vectors)    │  │ (metadata)       │  │ (keyframes)       │
└──────────────┘  └─────────────────┘  └───────────────────┘
```

### 3.2 Query Flow (Online)

```
┌───────────────────────────────────────┐
│ User Question: "What caused the stop?" │
└───────────────┬───────────────────────┘
                │
┌───────────────▼────────────────────────────────┐
│ 1. Query Understanding                          │
│    - Text embedding (InternVideo text encoder)  │
│    - Extract filters (time_range, domain)       │
└───────────────┬────────────────────────────────┘
                │
┌───────────────▼────────────────────────────────┐
│ 2. Coarse Retrieval (Macro level)              │
│    - Vector search: Top 50 macro clips          │
│    - Filter: video_id, time_range               │
└───────────────┬────────────────────────────────┘
                │
┌───────────────▼────────────────────────────────┐
│ 3. Focused Retrieval (Meso/Micro)              │
│    - Expand macro → search meso/micro in range  │
│    - Top 20 focused clips                       │
└───────────────┬────────────────────────────────┘
                │
┌───────────────▼────────────────────────────────┐
│ 4. Re-Ranking                                   │
│    - Cross-encoder or LMM-based scoring         │
│    - Reduce to Top 10                           │
└───────────────┬────────────────────────────────┘
                │
┌───────────────▼────────────────────────────────┐
│ 5. Evidence Pack Construction                   │
│    - Fetch keyframes (8-16 per clip)            │
│    - Fetch transcript (if available)            │
│    - Add context (±10s before/after)            │
└───────────────┬────────────────────────────────┘
                │
┌───────────────▼────────────────────────────────┐
│ 6. Video-LLM Inference                          │
│    - Model: InternVideo2.5 Chat 8B (4-bit)      │
│    - Input: Prompt + Evidence Pack              │
│    - Output: Structured JSON (answer + citations)│
└───────────────┬────────────────────────────────┘
                │
┌───────────────▼────────────────────────────────┐
│ 7. Response Formatting                          │
│    {                                            │
│      "answer": "...",                           │
│      "confidence": 0.87,                        │
│      "evidence": [                              │
│        {"start_sec": 245, "end_sec": 258, ...}  │
│      ]                                          │
│    }                                            │
└─────────────────────────────────────────────────┘
```

---

## 4. Deployment Architecture

### 4.1 Single-GPU Development (RTX 5090)

```
┌─────────────────────────────────────────────────┐
│              RTX 5090 (32GB GDDR7)               │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────┐         ┌──────────────┐     │
│  │ Embedding    │         │ Video-LLM    │     │
│  │ Extraction   │         │ Inference    │     │
│  │ (Batch)      │         │ (4-bit)      │     │
│  │ 16GB VRAM    │         │ 8GB VRAM     │     │
│  └──────────────┘         └──────────────┘     │
│                                                  │
│  Shared: 8GB buffer for decoding/pre-processing │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│                CPU Resources                     │
├─────────────────────────────────────────────────┤
│  - Video Decoding (PyAV/Decord)                 │
│  - Database (PostgreSQL + Qdrant)               │
│  - API Server (FastAPI)                         │
│  - Worker Queue (RQ + Redis)                    │
└─────────────────────────────────────────────────┘
```

**VRAM Budget:**
- Embedding model (InternVideo2 FP16): ~12GB
- Video-LLM (8B @ 4-bit): ~6GB
- Batch buffers: ~8GB
- **Total**: ~26GB (6GB margin)

### 4.2 Production Deployment (Future)

```
┌──────────────────────────────────────────────────┐
│              Load Balancer (Nginx)                │
└───────┬──────────────────────────────────────────┘
        │
        ├─────────────┬────────────────┬────────────┐
        │             │                │            │
┌───────▼──────┐ ┌───▼────────┐ ┌────▼───────┐ ┌──▼───────┐
│ API Gateway  │ │ API Gateway │ │   ...      │ │   ...    │
│  (FastAPI)   │ │  (FastAPI)  │ │            │ │          │
└───────┬──────┘ └───┬─────────┘ └────────────┘ └──────────┘
        │            │
        └────────┬───┘
                 │
        ┌────────▼──────────┐
        │   Message Queue    │
        │   (Redis/RabbitMQ) │
        └────────┬───────────┘
                 │
     ┌───────────┼───────────┬─────────────┐
     │           │           │             │
┌────▼─────┐ ┌──▼──────┐ ┌──▼──────┐ ┌───▼──────┐
│ Indexer  │ │  RAG    │ │  Event  │ │  ...     │
│ Worker   │ │ Worker  │ │ Worker  │ │          │
│ (GPU)    │ │ (GPU)   │ │ (GPU)   │ │          │
└────┬─────┘ └──┬──────┘ └──┬──────┘ └──────────┘
     │          │           │
     └──────────┼───────────┘
                │
     ┌──────────▼──────────┐
     │   PostgreSQL (RDS)   │
     └──────────┬───────────┘
                │
     ┌──────────▼──────────┐
     │   Qdrant Cluster    │
     └──────────┬───────────┘
                │
     ┌──────────▼──────────┐
     │  Object Storage      │
     │  (S3/Minio)          │
     └──────────────────────┘
```

---

## 5. Configuration Management

### 5.1 Environment Variables

```bash
# Core
VIGIL_DATA_DIR=/data/vigil
VIGIL_LOG_LEVEL=INFO

# Database
VIGIL_POSTGRES_URL=postgresql://user:pass@localhost:5432/vigil
VIGIL_QDRANT_URL=http://localhost:6333

# Models
VIGIL_EMBEDDING_MODEL=internvideo2  # or vjepa2
VIGIL_LLM_MODEL=internvideo_chat    # or llava_video
VIGIL_LLM_QUANTIZATION=4bit

# Chunking
VIGIL_CHUNK_MICRO_SEC=3.0
VIGIL_CHUNK_MESO_SEC=12.0
VIGIL_CHUNK_MACRO_SEC=48.0

# RAG
VIGIL_RETRIEVAL_TOP_K=50
VIGIL_RERANK_TOP_K=10
VIGIL_CONTEXT_WINDOW_SEC=10.0

# Event Detection
VIGIL_EVENT_METHODS=zero_shot,supervised,anomaly
VIGIL_EVENT_CONFIDENCE_THRESHOLD=0.7
VIGIL_FAH_TARGET=2.0  # False Alarms per Hour

# GPU
VIGIL_GPU_DEVICE=cuda:0
VIGIL_BATCH_SIZE_EMBED=16
VIGIL_BATCH_SIZE_LLM=1
```

### 5.2 Model Registry

```python
# config/models.yaml
embedding_models:
  internvideo2:
    repo: OpenGVLab/InternVideo2-1B
    variant: InternVideo2-1B
    dim: 768
    device: cuda
    quantization: fp16

  vjepa2:
    repo: facebookresearch/vjepa2
    variant: vjepa2_vit_large
    dim: 1024
    device: cuda
    quantization: fp16

llm_models:
  internvideo_chat:
    repo: OpenGVLab/InternVideo2-Chat-8B
    max_tokens: 2048
    quantization: 4bit
    context_frames: 32

  llava_video:
    repo: llava-hf/LLaVA-Video-7B-hf
    max_tokens: 2048
    quantization: 4bit
    context_frames: 24
```

---

## 6. Performance Targets

### 6.1 Latency (Indexed Videos)

| Operation | Target | Measurement |
|-----------|--------|-------------|
| 10-min video indexing | <5 min | End-to-end |
| 1-hr video indexing | <30 min | End-to-end |
| Query (10-min video) | <10s | P95 latency |
| Query (1-hr video) | <30s | P95 latency |
| Event detection (real-time) | <15s | From frame to alert |

### 6.2 Throughput

| Workload | Target | Hardware |
|----------|--------|----------|
| Concurrent indexing jobs | 2-3 | RTX 5090 |
| Concurrent queries | 5-10 | RTX 5090 (with queue) |
| Videos indexed per day | 100+ | 24/7 operation |

### 6.3 Accuracy

| Metric | Target | Evaluation Set |
|--------|--------|----------------|
| Evidence Recall@10 | >90% | 50-video QA benchmark |
| Answer Faithfulness | >85% | Human evaluation (n=100) |
| Event F1 (per type) | >75% | Domain-specific test set |
| FAH (Surveillance) | <2 | 24-hour continuous test |

---

## 7. Migration Path from SOPilot

### Phase 1: Parallel Operation
- SOPilot endpoints: `/sop/*` (unchanged)
- VIGIL-RAG endpoints: `/vigil/*` (new)
- Shared: Video storage, worker infrastructure

### Phase 2: Unified Backend
- Merge embedding extraction (single pipeline)
- Unified vector DB (Qdrant with SOPilot compatibility)
- Shared evaluation framework

### Phase 3: Feature Convergence
- SOPilot uses VIGIL chunking for better granularity
- VIGIL-RAG supports SOP scoring as event type

---

**Last Updated**: 2026-02-12
**Next Review**: After Phase 1 MVP completion
