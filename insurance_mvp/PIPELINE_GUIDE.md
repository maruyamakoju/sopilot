# Insurance MVP Pipeline - User Guide

## Overview

The Insurance MVP pipeline is a production-grade end-to-end system for automated dashcam video analysis. It orchestrates multiple AI components to assess accident severity, fault ratio, and fraud risk.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INSURANCE PIPELINE                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ B1: Multimodal Signal Mining                                        │
│ ├─ Audio Analysis (brake sounds, horns, crashes)                    │
│ ├─ Motion Analysis (optical flow, sudden movements)                 │
│ ├─ Proximity Analysis (YOLO object detection)                       │
│ └─ Signal Fusion → Top-20 danger clips                              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ B2: Video-LLM Inference (Qwen2.5-VL-7B)                             │
│ ├─ Clip → Severity Assessment (NONE/LOW/MEDIUM/HIGH)                │
│ ├─ Hazard Detection (collision, near-miss, violation)               │
│ └─ Causal Reasoning (why did this happen?)                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ B3: Insurance Domain Logic                                          │
│ ├─ Fault Assessment (0-100% fault ratio)                            │
│ └─ Fraud Detection (0.0-1.0 risk score)                             │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ B4: Conformal Prediction (Uncertainty Quantification)               │
│ ├─ Prediction Set (e.g., {MEDIUM, HIGH})                            │
│ └─ 90% Confidence Guarantee                                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ B5: Review Priority Assignment                                      │
│ ├─ URGENT: High severity × high uncertainty                         │
│ ├─ STANDARD: Medium severity or low uncertainty                     │
│ └─ LOW_PRIORITY: Low severity × high certainty                      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Output                                                               │
│ ├─ JSON Report (results.json)                                       │
│ └─ HTML Visualization (report.html)                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
cd insurance_mvp
pip install -e ".[all]"
```

### 2. Configuration

Copy the example config and customize:

```bash
cp config.example.yaml config.yaml
nano config.yaml
```

**Key settings:**

```yaml
# Output directory
output_dir: results

# Cosmos backend (qwen2.5-vl-7b or mock)
cosmos:
  backend: qwen2.5-vl-7b  # Use "mock" for testing without GPU
  device: auto  # auto, cuda, cpu

# Mining parameters
mining:
  top_k_clips: 20  # Number of danger clips to extract

# Feature flags
enable_conformal: true  # Uncertainty quantification
enable_transcription: true  # Audio transcription (requires Whisper)
enable_fraud_detection: true
enable_fault_assessment: true
```

### 3. Process a Single Video

```bash
python -m insurance_mvp.pipeline \
    --video-path data/dashcam001.mp4 \
    --output-dir results/
```

**Output:**

```
results/
└── dashcam001/
    ├── results.json       # JSON report
    ├── report.html        # HTML visualization
    └── checkpoint.json    # Intermediate checkpoint
```

### 4. Batch Processing

Process multiple videos in parallel:

```bash
python -m insurance_mvp.pipeline \
    --video-dir data/dashcam/ \
    --output-dir results/ \
    --parallel 4
```

### 5. View Results

Open the HTML report in your browser:

```bash
# Windows
start results/dashcam001/report.html

# Linux/Mac
open results/dashcam001/report.html
```

---

## CLI Reference

### Basic Usage

```bash
python -m insurance_mvp.pipeline [OPTIONS]
```

### Required Arguments

One of the following:

- `--video-path PATH` - Process a single video file
- `--video-dir PATH` - Process all videos in a directory

### Optional Arguments

#### General

- `--output-dir PATH` - Output directory (default: `results`)
- `--config PATH` - YAML config file
- `--parallel N` - Number of parallel workers (default: 1)
- `--log-level LEVEL` - Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)

#### Backend Selection

- `--cosmos-backend BACKEND` - Cosmos backend: `qwen2.5-vl-7b` or `mock`

#### Feature Flags

- `--no-conformal` - Disable conformal prediction
- `--no-transcription` - Disable audio transcription

### Examples

#### Mock Backend (No GPU Required)

```bash
python -m insurance_mvp.pipeline \
    --video-path data/test.mp4 \
    --cosmos-backend mock \
    --output-dir results/
```

#### Debug Mode

```bash
python -m insurance_mvp.pipeline \
    --video-path data/test.mp4 \
    --log-level DEBUG \
    --output-dir results/
```

#### Batch with Custom Config

```bash
python -m insurance_mvp.pipeline \
    --config my_config.yaml \
    --video-dir data/dashcam/ \
    --parallel 8
```

---

## Configuration Guide

### Configuration Priority

Configuration is loaded with the following priority (highest to lowest):

1. **CLI arguments** (e.g., `--parallel 4`)
2. **Environment variables** (e.g., `INSURANCE_PARALLEL_WORKERS=4`)
3. **YAML config file** (e.g., `config.yaml`)
4. **Default values**

### Environment Variables

All config options can be set via environment variables with the `INSURANCE_` prefix:

```bash
# General
export INSURANCE_OUTPUT_DIR=results
export INSURANCE_LOG_LEVEL=DEBUG
export INSURANCE_PARALLEL_WORKERS=4

# Cosmos
export INSURANCE_COSMOS_BACKEND=mock
export INSURANCE_COSMOS_DEVICE=cuda

# Mining
export INSURANCE_MINING_TOP_K=10

# Feature flags
export INSURANCE_ENABLE_CONFORMAL=true
export INSURANCE_ENABLE_TRANSCRIPTION=false
```

### YAML Structure

See `config.example.yaml` for full reference. Key sections:

```yaml
# Pipeline settings
parallel_workers: 1
resume_from_checkpoint: true
continue_on_error: true

# Video processing
video:
  max_resolution: [1280, 720]
  fps_sampling: 2

# Mining
mining:
  top_k_clips: 20
  audio_weight: 0.3
  motion_weight: 0.3
  proximity_weight: 0.4

# Cosmos
cosmos:
  backend: qwen2.5-vl-7b
  device: auto
  max_concurrent_inferences: 2

# Conformal
conformal:
  alpha: 0.1  # 90% confidence

# Whisper
whisper:
  backend: openai-whisper
  model_size: base
  language: ja
```

---

## Output Format

### JSON Report (`results.json`)

```json
{
  "video_id": "dashcam001",
  "timestamp": "2026-02-17T12:34:56.789Z",
  "config": { ... },
  "danger_clips": [
    {
      "clip_id": "dashcam001_clip_0",
      "start_sec": 10.5,
      "end_sec": 15.5,
      "danger_score": 0.95
    }
  ],
  "assessments": [
    {
      "severity": "HIGH",
      "confidence": 0.85,
      "prediction_set": ["MEDIUM", "HIGH"],
      "review_priority": "URGENT",
      "fault_assessment": {
        "fault_ratio": 80.0,
        "reasoning": "Driver failed to yield right of way",
        "scenario_type": "intersection_collision"
      },
      "fraud_risk": {
        "risk_score": 0.15,
        "indicators": [],
        "reasoning": "No fraud indicators detected"
      },
      "causal_reasoning": "Vehicle turned left without checking for oncoming traffic",
      "recommended_action": "REVIEW"
    }
  ],
  "summary": {
    "total_clips": 1,
    "severity_distribution": { "HIGH": 1 },
    "priority_distribution": { "URGENT": 1 },
    "avg_confidence": 0.85,
    "avg_fault_ratio": 80.0,
    "avg_fraud_score": 0.15
  }
}
```

### HTML Report (`report.html`)

Interactive table with:

- Severity color-coding (RED = HIGH, YELLOW = MEDIUM, GREEN = LOW)
- Prediction sets (uncertainty visualization)
- Review priority highlighting
- Fault ratio and fraud scores
- Recommended actions

---

## Advanced Features

### Resume from Checkpoint

If processing is interrupted, the pipeline can resume from the last checkpoint:

```bash
# Enable in config
resume_from_checkpoint: true
```

The pipeline saves a `checkpoint.json` after each clip. On restart, it skips already processed clips.

### Error Handling

```bash
# Continue processing on error (default: true)
continue_on_error: true

# Max retries for transient errors (default: 3)
max_retries: 3
```

### GPU Resource Management

Limit concurrent GPU inferences to avoid OOM:

```yaml
cosmos:
  max_concurrent_inferences: 2  # Max 2 clips processed simultaneously
```

### Performance Profiling

```yaml
enable_profiling: true  # Enable performance profiling
enable_metrics: true    # Track detailed metrics
```

Results saved to `batch_summary.json`:

```json
{
  "metrics": {
    "total_videos": 10,
    "successful_videos": 9,
    "failed_videos": 1,
    "total_clips_analyzed": 45,
    "total_processing_time_sec": 1234.56,
    "avg_processing_time_per_video": 123.46
  }
}
```

---

## Troubleshooting

### Out of Memory (CUDA OOM)

**Solution:** Reduce concurrent inferences

```yaml
cosmos:
  max_concurrent_inferences: 1  # Process one clip at a time
```

### Slow Processing

**Solutions:**

1. **Enable parallel processing:**
   ```bash
   --parallel 4
   ```

2. **Use mock backend for testing:**
   ```bash
   --cosmos-backend mock
   ```

3. **Reduce top-K clips:**
   ```yaml
   mining:
     top_k_clips: 10  # Reduce from 20 to 10
   ```

### Missing Dependencies

**Whisper not installed:**

```bash
pip install -e ".[whisper]"
```

**CUDA not available:**

```yaml
cosmos:
  device: cpu  # Force CPU mode
```

---

## Production Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -e ".[all]"

CMD ["python", "-m", "insurance_mvp.pipeline", \
     "--config", "config.yaml", \
     "--video-dir", "/data/videos", \
     "--output-dir", "/data/results"]
```

### Kubernetes

See `deployment/kubernetes.yaml` for example deployment.

### Monitoring

The pipeline exposes metrics via:

- `batch_summary.json` - Per-batch statistics
- Log files - Detailed processing logs
- (Optional) Prometheus metrics endpoint

---

## Performance Benchmarks

**Hardware:** NVIDIA RTX 5090 (24GB VRAM)

| Stage | Time per Clip | Notes |
|-------|--------------|-------|
| Mining | ~2s | Audio + Motion + YOLO |
| VLM Inference | ~8s | Qwen2.5-VL-7B |
| Fault Assessment | ~0.1s | Rule-based |
| Fraud Detection | ~0.2s | Statistical analysis |
| Conformal | ~0.01s | Calibrated predictor |
| **Total** | **~10s** | Single clip end-to-end |

**Throughput:**

- Single video (20 clips): ~200s
- Batch (10 videos, 4 workers): ~500s

---

## API Integration

For API-based access, see:

- `insurance_mvp/api/main.py` - FastAPI server
- `insurance_mvp/api/README.md` - API documentation

---

## Support

**Documentation:**
- [README.md](README.md) - Project overview
- [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - This guide
- [API Reference](api/README.md) - REST API docs

**Issues:**
- File bug reports on internal issue tracker
- For urgent issues, contact development team

---

## License

Proprietary - Internal use only
