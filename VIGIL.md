# VIGIL-RAG: Long-Form Video Understanding & Event Intelligence Platform

**Version:** 1.0
**Status:** Active Development
**Target Hardware:** RTX 5090 (32GB GDDR7) single GPU

---

## Executive Summary

VIGIL-RAG extends SOPilot's video understanding capabilities to **long-form videos (10min - hours)** with:
- **Video RAG**: Question answering with evidence-based retrieval
- **Event Detection**: Automated timeline generation for surveillance/factory/sports
- **Multi-modal Understanding**: Visual + audio + transcript integration

**Key Differentiation from SOPilot:**
- SOPilot: Short-form (minutes) SOP compliance scoring (2 videos compared)
- VIGIL-RAG: Long-form (hours) video understanding + event detection (1 video analyzed)

---

## 1. Problem Statement

### Current Pain Points
- **Manual review cost**: 10min+ videos require full human viewing
- **Existing Video-LLM limitations**: Token/frame constraints prevent holistic understanding
- **Event detection fragility**: Class-based detectors fail on novel events

### Target Use Cases

#### B: Long-Form Understanding (10min - hours)
- **Factory**: "What caused yesterday's machine stop? Show 30s before stop."
- **Surveillance**: "Summarize suspicious activity 2-3pm, list evidence clips."
- **Sports**: "Compare defensive positioning in 1st vs 2nd half goals."

#### C: Event Detection
- **Factory**: Intrusion, PPE violations, unsafe actions, machine failures
- **Surveillance**: Falls, violence, loitering, vehicle incidents
- **Sports**: Goals, cards, substitutions, highlight moments

---

## 2. Technical Strategy

### 2.1 Dual-Model Architecture

**Retrieval Path** (Fast, Scalable)
- InternVideo family for video/text embeddings
- Multi-granularity indexing (shot/micro/meso/macro)
- Vector DB: Qdrant (production) / FAISS (development)

**Reasoning Path** (Accurate, Explainable)
- Video-LLM: InternVideo2.5 Chat 8B or LLaVA-Video 7B
- Structured output (JSON with answer + evidence timestamps)
- Quantization: 4-bit/8-bit for VRAM efficiency

**Why Separation?**
- RAG bottleneck is *Recall* (retrieval), not reasoning
- Video-LLM only sees retrieved clips (not full video)
- Models are independently swappable

### 2.2 Multi-Scale Chunking (Critical Design Choice)

```
Level 0 (Shot):     Scene boundaries (variable length)
Level 1 (Micro):    2-4s  - Capture moments/changes
Level 2 (Meso):     8-16s - Action units
Level 3 (Macro):    32-64s - Context/causality
```

**Rationale**: Same timestamp indexed at multiple scales enables:
- Moment detection (micro)
- Action recognition (meso)
- Contextual reasoning (macro)

**Sampling FPS**:
- Micro/Meso: 2 fps (motion-sensitive)
- Macro: 1 fps (compute-efficient)

### 2.3 Event Detection: 3-Method Ensemble

1. **Zero-Shot** (Fast deployment)
   - Text definition → embedding → similarity search
   - Calibration with 10-50 examples per event
   - Best for: Novel events, rapid iteration

2. **Supervised** (High precision)
   - MMAction2 temporal localization
   - Requires: 100+ labeled clips per event
   - Best for: High-frequency events (goals, stops)

3. **Anomaly Detection** (Unknown unknowns)
   - Normal distribution learning + outlier detection
   - UCF-Crime methodology
   - Best for: Safety-critical monitoring

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      VIGIL-RAG Platform                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │   Ingestion │───▶│  Preprocessor│───▶│Feature Extract│  │
│  │   Service   │    │  (Chunking)  │    │  (Embeddings) │  │
│  └─────────────┘    └──────────────┘    └───────┬───────┘  │
│                                                   │           │
│                     ┌─────────────────────────────┘           │
│                     ▼                                          │
│         ┌─────────────────────────┐                           │
│         │   Index Store (Qdrant)  │                           │
│         │  + Metadata DB (Postgres)│                          │
│         └────────────┬─────────────┘                          │
│                      │                                         │
│        ┌─────────────┴────────────────┐                       │
│        ▼                               ▼                       │
│  ┌────────────┐               ┌──────────────┐               │
│  │ RAG Engine │               │Event Detector│               │
│  │ (QA/Search)│               │ (3 Methods)  │               │
│  └─────┬──────┘               └──────┬───────┘               │
│        │                              │                       │
│        └──────────────┬───────────────┘                       │
│                       ▼                                        │
│              ┌─────────────────┐                              │
│              │  API / UI Layer │                              │
│              └─────────────────┘                              │
└───────────────────────────────────────────────────────────────┘
```

### Data Flow

**Indexing (Batch)**
```
Video → Decode → Shot Detection → Multi-Scale Chunking
  → Keyframe Selection → Embedding Extraction
  → Vector DB Upsert + Metadata DB Insert
```

**Query (Online)**
```
User Question → Query Embedding → Vector Search (TopK)
  → Re-Rank → Evidence Pack (frames + timestamps + context)
  → Video-LLM Inference → Structured Answer (JSON)
```

**Event Detection (Online/Batch)**
```
Video Clips → Event Proposals (3 methods)
  → Threshold Calibration → Temporal Localization
  → Event DB Insert → Alert/Timeline Update
```

---

## 4. Database Schema

### PostgreSQL (Metadata)

```sql
-- Videos
CREATE TABLE videos (
  video_id UUID PRIMARY KEY,
  uri TEXT NOT NULL,
  storage_path TEXT,
  duration_sec FLOAT,
  fps FLOAT,
  width INT,
  height INT,
  domain VARCHAR(50), -- factory/surveillance/sports
  camera_id VARCHAR(100),
  ingest_time TIMESTAMP,
  checksum VARCHAR(64)
);

-- Clips (Multi-level)
CREATE TABLE clips (
  clip_id UUID PRIMARY KEY,
  video_id UUID REFERENCES videos(video_id),
  level VARCHAR(20), -- shot/micro/meso/macro
  start_sec FLOAT,
  end_sec FLOAT,
  keyframe_paths JSONB,
  transcript_text TEXT,
  embedding_id UUID
);

-- Embeddings
CREATE TABLE embeddings (
  embedding_id UUID PRIMARY KEY,
  model_name VARCHAR(100),
  model_version VARCHAR(50),
  dimension INT,
  vector_ref TEXT, -- Qdrant point ID
  created_at TIMESTAMP
);

-- Events
CREATE TABLE events (
  event_id UUID PRIMARY KEY,
  video_id UUID REFERENCES videos(video_id),
  event_type VARCHAR(100),
  start_sec FLOAT,
  end_sec FLOAT,
  confidence FLOAT,
  method VARCHAR(50), -- zero_shot/supervised/anomaly
  evidence_clip_ids UUID[],
  status VARCHAR(20), -- new/confirmed/false_alarm
  created_at TIMESTAMP
);
```

### Qdrant (Vectors)

```python
{
  "point_id": "embedding_id",
  "vector": [0.1, 0.2, ...],  # dim=512 or 768
  "payload": {
    "video_id": "uuid",
    "clip_id": "uuid",
    "start_sec": 123.4,
    "end_sec": 127.8,
    "level": "meso",
    "domain": "factory",
    "camera_id": "cam_01"
  }
}
```

---

## 5. API Specification

### 5.1 Ingestion
```http
POST /videos
Content-Type: application/json
{
  "uri": "s3://bucket/video.mp4",
  "domain": "factory",
  "camera_id": "line_a_cam_1"
}
Response: {"video_id": "uuid", "status": "queued"}

POST /videos/{video_id}/index
Response: {"job_id": "uuid", "status": "processing"}
```

### 5.2 Query (RAG)
```http
POST /query
Content-Type: application/json
{
  "video_id": "uuid",
  "question": "What happened before the machine stopped?",
  "time_range": {"start_sec": 0, "end_sec": 600},
  "top_k": 10
}
Response: {
  "answer": "Operator entered restricted zone without PPE...",
  "confidence": 0.87,
  "evidence": [
    {
      "video_id": "uuid",
      "start_sec": 245.2,
      "end_sec": 258.6,
      "why_relevant": "Shows operator entry",
      "keyframes": ["url1", "url2"]
    }
  ],
  "unknowns": ["Exact reason for stop unclear"]
}
```

### 5.3 Events
```http
GET /videos/{video_id}/events?type=intrusion&min_conf=0.8
Response: {
  "events": [
    {
      "event_id": "uuid",
      "event_type": "intrusion",
      "start_sec": 123.4,
      "end_sec": 130.2,
      "confidence": 0.92,
      "method": "zero_shot",
      "status": "new"
    }
  ]
}

POST /events/definitions
{
  "event_type": "forklift_intrusion",
  "description": "Forklift enters restricted assembly area",
  "visual_cues": ["yellow forklift", "red floor marking"],
  "method": "zero_shot"
}
```

---

## 6. Evaluation Framework

### 6.1 Long-Form RAG Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Evidence Recall@5 | >85% | Correct clip in Top-5 |
| Evidence Recall@10 | >95% | Correct clip in Top-10 |
| Answer Faithfulness | >90% | Answer matches evidence |
| Time-to-Answer (10min video) | <10s | Latency (indexed) |
| Time-to-Answer (1hr video) | <30s | Latency (indexed) |

### 6.2 Event Detection Metrics

| Metric | Target | Domain |
|--------|--------|--------|
| mAP (per event) | >0.7 | Factory/Surveillance |
| F1 (macro avg) | >0.75 | All events |
| False Alarms/Hour | <2 | Surveillance |
| Detection Latency | <15s | Real-time |

### 6.3 Benchmark Datasets

**Public Benchmarks:**
- LongVideoBench (video-language QA, up to 1hr)
- Video-MME (video MLLM evaluation)
- EgoSchema / Ego4D (long-form QA)
- SoccerNet Action Spotting (sparse events)
- UCF-Crime (surveillance anomaly)

**Internal Benchmark (Required):**
- 50 videos per domain (factory/surveillance/sports)
- 200+ QA pairs (Who/What/When/Where/Why/Compare)
- 30+ event types with ground truth timestamps

---

## 7. Implementation Roadmap

### Phase 0: Foundation (Week 1)
- [x] CI passing (lint/test/smoke)
- [ ] Add VIGIL.md, ARCHITECTURE.md, EVALUATION.md
- [ ] PostgreSQL schema + Alembic migrations
- [ ] Qdrant integration + FAISS compatibility layer
- [ ] Multi-scale chunking implementation

### Phase 1: RAG MVP (Week 2-3)
- [ ] Video-LLM integration (InternVideo2.5 Chat 8B)
- [ ] RAG service implementation
- [ ] POST /query API endpoint
- [ ] Evaluation harness (Recall@K, Faithfulness)
- [ ] 10-video benchmark

### Phase 2: Event Detection (Week 4-5)
- [ ] Zero-shot event detection (text similarity)
- [ ] MMAction2 integration (temporal localization)
- [ ] Anomaly detection baseline
- [ ] GET /events API endpoint
- [ ] FAH (False Alarms/Hour) tracking

### Phase 3: Production Hardening (Week 6-8)
- [ ] Timeline UI (events + evidence clips)
- [ ] QA chat interface
- [ ] Alert system (Slack/email/webhook)
- [ ] Active learning loop (feedback → re-training)
- [ ] Docker deployment + k8s manifests

---

## 8. Success Criteria

### MVP Success (End of Phase 1)
- 10-minute factory video → QA working with >80% Evidence Recall@10
- Indexed video responds in <15 seconds
- At least 3 event types detected with >70% F1

### Production Ready (End of Phase 3)
- 1-hour video → QA working with >85% Evidence Recall@5
- <5s query latency for indexed videos
- FAH <2 for surveillance domain
- UI deployed and usable by non-technical operators

---

## 9. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hallucination (wrong answers) | High | Evidence-first design; return "unknown" if no evidence |
| Recall failure (missing events) | High | Multi-granularity + multi-modal indexing; re-ranking |
| False alarms | High | Calibration with domain data; confidence thresholds |
| VRAM limits (RTX 5090 32GB) | Medium | 4-bit quantization; batch size tuning |
| Model licensing | Medium | Verify Apache-2.0 for all weights before deployment |

---

## 10. Open Questions (To Be Resolved)

1. **Domain Priority**: Factory (safety/stops) vs Surveillance (intrusion/anomaly) vs Sports (highlights)?
   - **Decision**: Start with **Factory** (extends SOPilot use case)

2. **Video-LLM Choice**: InternVideo2.5 Chat vs LLaVA-Video?
   - **Approach**: Test both, benchmark on 10-video QA set

3. **SOPilot Coexistence**: Keep short-form SOP scoring?
   - **Decision**: Yes, as separate endpoint `/sop/score` (existing functionality preserved)

4. **Deployment Target**: Cloud (AWS/Azure) or On-prem?
   - **Approach**: On-prem first (RTX 5090 assumption), cloud-ready architecture

---

## References

**Models:**
- InternVideo2.5 Chat: https://github.com/OpenGVLab/InternVideo
- LLaVA-Video: https://github.com/LLaVA-VL/LLaVA-NeXT
- MMAction2: https://github.com/open-mmlab/mmaction2

**Datasets:**
- LongVideoBench: https://github.com/longvideobench/LongVideoBench
- SoccerNet: https://www.soccer-net.org/
- UCF-Crime: https://www.crcv.ucf.edu/projects/real-world/

**Papers:**
- InternVideo (ICLR 2023)
- LLaVA (NeurIPS 2023)
- Video-LLaVA (EMNLP 2023)
- Soft-DTW (ICML 2017)
- Anomaly Detection Survey (CVPR 2022)

---

**Last Updated**: 2026-02-12
**Maintainer**: SOPilot Team
**License**: Apache 2.0
