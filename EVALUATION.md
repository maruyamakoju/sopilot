# VIGIL-RAG Evaluation Framework

**Version**: 1.0
**Purpose**: Define metrics, benchmarks, and evaluation procedures for VIGIL-RAG

---

## 1. Overview

VIGIL-RAG evaluation consists of **3 pillars**:

1. **Long-Form RAG** (B): Video QA with evidence retrieval
2. **Event Detection** (C): Temporal event localization
3. **System Performance**: Latency, throughput, resource usage

All metrics must be **reproducible** and **continuously tracked** (CI integration).

---

## 2. Long-Form RAG Evaluation

### 2.1 Core Metrics

| Metric | Definition | Formula | Target |
|--------|-----------|---------|--------|
| **Evidence Recall@K** | % of queries where correct clip is in Top-K | (Queries with correct clip in Top-K) / Total queries | R@5: >85%<br>R@10: >95% |
| **Answer Faithfulness** | % of answers that match provided evidence | Human evaluation: Answer ⊆ Evidence | >90% |
| **Time-to-Answer** | Query latency (indexed video) | End-to-end from POST /query to response | 10min video: <10s<br>1hr video: <30s |
| **Retrieval Precision@K** | % of Top-K clips that are relevant | Relevant clips in Top-K / K | P@10: >60% |

### 2.2 Evaluation Procedure

#### Step 1: Question Set Construction

**Coverage Matrix** (20 question types):

| Category | Examples | Count |
|----------|----------|-------|
| **Who** | "Who entered the restricted zone?" | 3 |
| **What** | "What caused the machine stop?" | 5 |
| **When** | "When did the operator remove PPE?" | 3 |
| **Where** | "Where did the incident occur?" | 2 |
| **Why** | "Why did the alarm trigger?" | 2 |
| **How** | "How did the worker bypass safety?" | 2 |
| **Compare** | "Compare safety violations in shift A vs B" | 2 |
| **Count** | "How many times did forklift enter zone?" | 1 |

**Annotation Format**:
```json
{
  "question_id": "factory_001",
  "video_id": "abc123",
  "question": "What caused the emergency stop at 14:32?",
  "ground_truth_answer": "Operator entered light curtain zone without clearance",
  "evidence_clips": [
    {
      "start_sec": 872.0,
      "end_sec": 885.5,
      "relevance": "primary"  // or "context"
    }
  ],
  "difficulty": "medium",
  "domain": "factory"
}
```

#### Step 2: Automated Evaluation

```python
# evaluation/rag_metrics.py

from dataclasses import dataclass
from typing import List

@dataclass
class RAGPrediction:
    question_id: str
    answer: str
    confidence: float
    retrieved_clips: List[dict]  # [{"start_sec": ..., "end_sec": ...}]

@dataclass
class RAGGroundTruth:
    question_id: str
    evidence_clips: List[dict]
    answer: str

def compute_recall_at_k(
    predictions: List[RAGPrediction],
    ground_truths: List[RAGGroundTruth],
    k: int = 10,
    iou_threshold: float = 0.3
) -> float:
    """Compute Evidence Recall@K with temporal IoU matching."""

    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        # Check if any retrieved clip matches ground truth
        for retrieved in pred.retrieved_clips[:k]:
            for gt_clip in gt.evidence_clips:
                iou = temporal_iou(retrieved, gt_clip)
                if iou >= iou_threshold:
                    correct += 1
                    break
            else:
                continue
            break

    return correct / len(predictions)

def temporal_iou(clip1: dict, clip2: dict) -> float:
    """Compute Intersection over Union for temporal intervals."""
    start1, end1 = clip1["start_sec"], clip1["end_sec"]
    start2, end2 = clip2["start_sec"], clip2["end_sec"]

    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = (end1 - start1) + (end2 - start2) - intersection

    return intersection / union if union > 0 else 0.0
```

#### Step 3: Human Evaluation (Faithfulness)

**Protocol**:
1. Annotator views evidence clips provided by system
2. Reads system answer
3. Rates: `{Faithful, Partially Faithful, Unfaithful, Unknown}`

**Inter-Annotator Agreement**: Require Cohen's Kappa > 0.7

**Sampling**: Evaluate 100 random queries per benchmark release

### 2.3 Benchmark Datasets

#### 2.3.1 Public Benchmarks

| Dataset | Domain | Length | QA Pairs | Usage |
|---------|--------|--------|----------|-------|
| **LongVideoBench** | Mixed | Up to 1hr | 3,763 | Baseline comparison |
| **Video-MME** | General | 1-60min | 2,700+ | MLLM capability |
| **EgoSchema** | Egocentric | 3min avg | 5,031 | Long-context reasoning |
| **Ego4D NLQ** | Egocentric | Hours | 13,000+ | Moment retrieval |
| **ActivityNet Captions** | Action | 2min avg | 100K | Dense video captioning |

#### 2.3.2 Internal Benchmark (Required)

**Minimum Viable Benchmark**:
- **50 videos** (10min - 1hr each)
- **200 QA pairs** (4 per video)
- **3 domains**: Factory (20), Surveillance (20), Sports (10)

**Annotation Tool**: Custom web UI for:
- Video playback with timestamp marking
- Question input
- Evidence clip annotation (drag-to-select)
- Answer typing + difficulty rating

**Storage Format**:
```
data/benchmarks/
  factory/
    videos/
      video_001.mp4
      video_002.mp4
    annotations/
      video_001_qa.json
      video_002_qa.json
  surveillance/
    ...
  sports/
    ...
```

### 2.4 Continuous Evaluation (CI)

**Regression Test**:
```python
# tests/test_rag_regression.py

def test_rag_recall_regression():
    """Ensure RAG Recall@10 does not degrade."""

    # Load 10-video benchmark (fast subset)
    benchmark = load_benchmark("data/benchmarks/quick_10.json")

    # Run RAG pipeline
    predictions = []
    for qa in benchmark:
        pred = rag_service.query(
            video_id=qa["video_id"],
            question=qa["question"],
            top_k=10
        )
        predictions.append(pred)

    # Compute metrics
    recall_10 = compute_recall_at_k(predictions, benchmark, k=10)

    # Assert threshold (allow 5% variance)
    assert recall_10 >= 0.80, f"Recall@10 = {recall_10:.2%} below threshold"
```

**CI Integration**:
```yaml
# .github/workflows/ci.yml

  rag_benchmark:
    runs-on: ubuntu-latest
    needs: [test]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Download benchmark data
        run: |
          aws s3 cp s3://vigil-benchmarks/quick_10.tar.gz .
          tar -xzf quick_10.tar.gz -C data/benchmarks/

      - name: Install project
        run: pip install -e ".[dev,ml]"

      - name: Run RAG regression test
        run: python -m pytest tests/test_rag_regression.py -v
```

---

## 3. Event Detection Evaluation

### 3.1 Core Metrics

| Metric | Definition | Formula | Target |
|--------|-----------|---------|--------|
| **mAP** (mean Average Precision) | Average precision across all event types | Mean of AP per event type | >0.70 |
| **F1 Score** (per event) | Harmonic mean of precision & recall | 2 * (P * R) / (P + R) | >0.75 (macro avg) |
| **FAH** (False Alarms per Hour) | False positives per hour of video | FP count / (Video duration in hours) | <2 (surveillance) |
| **Detection Latency** (real-time) | Time from event occurrence to alert | Alert timestamp - Event timestamp | <15s (P95) |
| **Temporal IoU** | Overlap between predicted and ground truth intervals | Intersection / Union | >0.5 (avg) |

### 3.2 Evaluation Procedure

#### Step 1: Event Annotation

**Format** (ActivityNet-style):
```json
{
  "video_id": "factory_cam1_20260212",
  "duration_sec": 3600.0,
  "events": [
    {
      "event_id": "evt_001",
      "event_type": "intrusion",
      "start_sec": 245.2,
      "end_sec": 258.6,
      "confidence": 1.0,  // ground truth
      "description": "Forklift enters restricted assembly area"
    },
    {
      "event_id": "evt_002",
      "event_type": "ppe_violation",
      "start_sec": 512.0,
      "end_sec": 520.5,
      "confidence": 1.0,
      "description": "Worker without helmet in hard hat zone"
    }
  ]
}
```

**Annotation Guidelines**:
- **Start timestamp**: First frame where event condition is met
- **End timestamp**: Last frame where event condition persists
- **Ambiguous cases**: Mark with `confidence: 0.5` and review flag

#### Step 2: Automated Metrics

```python
# evaluation/event_metrics.py

from typing import List, Tuple
import numpy as np

def compute_map(
    predictions: List[dict],
    ground_truths: List[dict],
    iou_threshold: float = 0.5
) -> dict:
    """Compute mean Average Precision (ActivityNet protocol)."""

    event_types = set(gt["event_type"] for gt in ground_truths)

    ap_per_type = {}
    for event_type in event_types:
        # Filter predictions and ground truths for this event type
        preds_type = [p for p in predictions if p["event_type"] == event_type]
        gts_type = [gt for gt in ground_truths if gt["event_type"] == event_type]

        # Sort predictions by confidence (descending)
        preds_type = sorted(preds_type, key=lambda x: x["confidence"], reverse=True)

        # Compute precision-recall curve
        tp = np.zeros(len(preds_type))
        fp = np.zeros(len(preds_type))

        matched_gts = set()
        for i, pred in enumerate(preds_type):
            # Find best matching ground truth
            best_iou = 0.0
            best_gt_idx = -1
            for j, gt in enumerate(gts_type):
                if j in matched_gts:
                    continue
                iou = temporal_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            # Check if match is valid
            if best_iou >= iou_threshold:
                tp[i] = 1
                matched_gts.add(best_gt_idx)
            else:
                fp[i] = 1

        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recall = tp_cumsum / max(len(gts_type), 1)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Compute AP (area under PR curve)
        ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
        ap_per_type[event_type] = ap

    return {
        "mAP": np.mean(list(ap_per_type.values())),
        "AP_per_type": ap_per_type
    }

def compute_fah(
    predictions: List[dict],
    ground_truths: List[dict],
    video_duration_hours: float,
    iou_threshold: float = 0.5
) -> float:
    """Compute False Alarms per Hour."""

    # Match predictions to ground truths
    matched_preds = set()
    for pred in predictions:
        for gt in ground_truths:
            if temporal_iou(pred, gt) >= iou_threshold:
                matched_preds.add(pred["event_id"])
                break

    # Count false positives
    fp_count = len(predictions) - len(matched_preds)

    return fp_count / video_duration_hours
```

### 3.3 Benchmark Datasets

| Dataset | Domain | Events | Annotations | Usage |
|---------|--------|--------|-------------|-------|
| **SoccerNet Action Spotting** | Sports | 17 types | 550 games | Sparse events baseline |
| **UCF-Crime** | Surveillance | 13 anomalies | 1,900 videos | Anomaly detection |
| **THUMOS14** | Action | 20 classes | 413 videos | Temporal localization |
| **MultiTHUMOS** | Action | 65 classes | 400 videos | Dense multi-label events |

**Internal Benchmark** (Week 4 target):
- **30 event types** across 3 domains
- **100+ labeled videos** (10-60min each)
- **500+ event instances**

### 3.4 Domain-Specific Targets

| Domain | Priority Events | F1 Target | FAH Target |
|--------|----------------|-----------|------------|
| **Factory** | Intrusion, PPE violation, unsafe action | >0.80 | <1.0 |
| **Surveillance** | Fall, violence, loitering, vehicle incident | >0.75 | <2.0 |
| **Sports** | Goal, card, substitution, foul | >0.85 | N/A |

---

## 4. System Performance Evaluation

### 4.1 Latency Benchmarks

| Operation | Target (P50) | Target (P95) | Measurement |
|-----------|-------------|-------------|-------------|
| 10-min video indexing | <3min | <5min | End-to-end batch |
| 1-hr video indexing | <20min | <30min | End-to-end batch |
| Query (10-min video, indexed) | <5s | <10s | POST /vigil/query → response |
| Query (1-hr video, indexed) | <15s | <30s | POST /vigil/query → response |
| Event detection (real-time) | <10s | <15s | Frame to alert notification |

**Profiling Tool**:
```python
# evaluation/profiler.py

import time
from contextlib import contextmanager

@contextmanager
def profile_operation(operation_name: str):
    """Context manager for latency profiling."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[PROFILE] {operation_name}: {elapsed:.2f}s")
    # Log to metrics store (Prometheus/InfluxDB)
    METRICS.timing(operation_name, elapsed)
```

### 4.2 Throughput Benchmarks

| Workload | Target | Hardware | Notes |
|----------|--------|----------|-------|
| Concurrent indexing jobs | 2-3 | RTX 5090 | GPU-bound |
| Concurrent queries | 5-10 | RTX 5090 | With request queue |
| Videos indexed per day | 100+ | 24/7 operation | Assumes 30min avg |

### 4.3 Resource Usage

| Resource | Baseline (SOPilot) | VIGIL-RAG Target | Headroom |
|----------|-------------------|-----------------|----------|
| **VRAM (peak)** | ~12GB | <26GB | 6GB margin |
| **VRAM (idle)** | ~2GB | ~8GB | LLM loaded |
| **Storage (per hour video)** | ~500MB | ~1.5GB | Multi-scale overhead |
| **Postgres DB size (1K videos)** | N/A | ~2GB | Metadata only |
| **Qdrant index (1K videos)** | ~200MB | ~600MB | Vector storage |

---

## 5. Evaluation Harness

### 5.1 Command-Line Interface

```bash
# Run full evaluation suite
python -m evaluation.run_benchmark \
  --dataset data/benchmarks/factory_50 \
  --output results/factory_eval_$(date +%Y%m%d).json \
  --metrics recall,faithfulness,map,fah

# Run quick regression (10 videos)
python -m evaluation.run_benchmark \
  --dataset data/benchmarks/quick_10 \
  --output results/quick_regression.json \
  --fast

# Compare two model versions
python -m evaluation.compare_models \
  --baseline results/v1.0_eval.json \
  --current results/v1.1_eval.json \
  --output results/comparison.md
```

### 5.2 Output Format

```json
{
  "benchmark_id": "factory_50_20260212",
  "timestamp": "2026-02-12T15:30:00Z",
  "config": {
    "embedding_model": "internvideo2",
    "llm_model": "internvideo_chat_8b_4bit",
    "retrieval_top_k": 50,
    "rerank_top_k": 10
  },
  "rag_metrics": {
    "recall_at_5": 0.87,
    "recall_at_10": 0.94,
    "precision_at_10": 0.68,
    "mean_time_to_answer_sec": 8.3,
    "faithfulness": 0.91
  },
  "event_metrics": {
    "map": 0.74,
    "f1_macro": 0.78,
    "fah": 1.2,
    "ap_per_type": {
      "intrusion": 0.82,
      "ppe_violation": 0.71,
      "unsafe_action": 0.69
    }
  },
  "performance": {
    "p50_latency_sec": 7.2,
    "p95_latency_sec": 12.8,
    "avg_indexing_time_min": 4.5,
    "peak_vram_gb": 24.3
  }
}
```

### 5.3 Visualization Dashboard

**Metrics to Track**:
- Recall@K over time (trend chart)
- mAP per event type (bar chart)
- Latency distribution (histogram)
- FAH by domain (line chart)

**Tool**: Grafana + Prometheus (or MLflow)

---

## 6. Ablation Studies

### 6.1 Retrieval Ablations

| Configuration | Recall@10 | Precision@10 | Notes |
|--------------|-----------|-------------|-------|
| Single-scale (meso only) | Baseline | Baseline | Current SOPilot |
| Multi-scale (no rerank) | +8% | -5% | More recall, less precision |
| Multi-scale + rerank | +12% | +3% | **Best overall** |
| Multi-scale + transcript | +15% | +5% | If audio available |

### 6.2 Event Detection Ablations

| Method | mAP | F1 | FAH | Speed |
|--------|-----|-----|-----|-------|
| Zero-shot (no calibration) | 0.45 | 0.50 | 8.5 | Fast |
| Zero-shot (calibrated) | 0.62 | 0.68 | 3.2 | Fast |
| Supervised (MMAction2) | 0.78 | 0.82 | 1.5 | Medium |
| Ensemble (zero-shot + supervised) | 0.81 | 0.84 | 1.2 | Slow |

### 6.3 LLM Ablations

| Model | Answer Quality | Latency | VRAM | License |
|-------|---------------|---------|------|---------|
| InternVideo Chat 8B (4-bit) | Good | 3.5s | 6GB | Apache-2.0 |
| LLaVA-Video 7B (4-bit) | Good | 3.2s | 5GB | Apache-2.0 (check weights) |
| LLaVA-Video 72B (4-bit) | Best | 12s | 28GB | Apache-2.0 (check weights) |

---

## 7. Error Analysis

### 7.1 Failure Modes (RAG)

| Error Type | Frequency | Root Cause | Mitigation |
|-----------|-----------|------------|------------|
| **No evidence retrieved** | 15% | Query embedding mismatch | Improve text encoder, add query expansion |
| **Wrong clip retrieved** | 10% | Ambiguous visual similarity | Add re-ranking, multi-modal fusion |
| **Hallucinated answer** | 8% | LLM ignores evidence | Strengthen prompt, add faithfulness check |
| **Incomplete answer** | 12% | Missing context clips | Expand context window (±10s → ±30s) |

### 7.2 Failure Modes (Event Detection)

| Error Type | Frequency | Root Cause | Mitigation |
|-----------|-----------|------------|------------|
| **False alarm** | 30% | Over-sensitive threshold | Calibration with domain data |
| **Missed event** | 20% | Rare event type | Active learning, few-shot tuning |
| **Wrong event type** | 15% | Visually similar events | Fine-grained classifier, temporal context |
| **Imprecise boundaries** | 25% | Gradual event onset | Temporal localization model refinement |

---

## 8. Continuous Improvement Loop

```
1. Collect production feedback (user corrections, false alarms)
   ↓
2. Add to labeled dataset (active learning)
   ↓
3. Re-train/calibrate models (weekly)
   ↓
4. Run regression tests (CI)
   ↓
5. A/B test in production (shadow mode)
   ↓
6. Deploy if metrics improve
```

**Tracking**:
- **Feedback rate**: Target >5% of queries annotated
- **Improvement cadence**: Weekly model updates
- **Regression checks**: Automated via CI

---

## 9. Evaluation Schedule

| Milestone | Evaluation | Acceptance Criteria |
|-----------|-----------|---------------------|
| **Phase 1 MVP** | 10-video QA benchmark | Recall@10 >80% |
| **Phase 2 Events** | 30-event factory benchmark | mAP >0.65, FAH <3 |
| **Phase 3 Production** | 50-video full benchmark | Recall@5 >85%, mAP >0.70, FAH <2 |
| **Quarterly Review** | Public benchmarks (LongVideoBench, SoccerNet) | Track relative to SOTA |

---

## 10. Appendix: Benchmark Creation Checklist

### A. Long-Form RAG Benchmark

- [ ] Select 50 videos (diverse, 10min - 1hr)
- [ ] Annotate 4 QA pairs per video (200 total)
- [ ] Cover all 8 question types (Who/What/When/Where/Why/How/Compare/Count)
- [ ] Mark evidence clips with start/end timestamps
- [ ] Validate with 2nd annotator (Cohen's Kappa >0.7)
- [ ] Store in standardized JSON format
- [ ] Upload to secure storage (S3/GCS)

### B. Event Detection Benchmark

- [ ] Define 30 event types (10 per domain)
- [ ] Collect 100 videos with events (10-60min each)
- [ ] Annotate event timestamps + types
- [ ] Balance dataset (avoid class imbalance >10:1)
- [ ] Add negative samples (videos with no events)
- [ ] Validate annotations (spot-check 20%)
- [ ] Compute baseline metrics (random/heuristic)

---

**Last Updated**: 2026-02-12
**Maintainers**: SOPilot + VIGIL-RAG Team
**License**: Apache 2.0
