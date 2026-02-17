# Insurance MVP Pipeline

**Production-Grade End-to-End Orchestration for Automated Dashcam Video Analysis**

---

## Features

- **Multimodal Signal Mining** - Audio, motion, and proximity analysis
- **Video-LLM Inference** - Qwen2.5-VL-7B for severity assessment
- **Insurance Domain Logic** - Fault ratio and fraud detection
- **Uncertainty Quantification** - Conformal prediction with 90% confidence
- **Review Priority Assignment** - Automatic triage (URGENT/STANDARD/LOW)
- **Production-Grade**:
  - Batch processing with parallel workers
  - GPU resource management (semaphore-based)
  - Incremental saving and resume capability
  - Comprehensive error handling and retry logic
  - Progress tracking with tqdm
  - Performance profiling and metrics
  - Configuration via YAML + env vars + CLI

---

## Quick Start

### Installation

```bash
cd insurance_mvp
pip install -e ".[all]"
```

### Run Demo

```bash
# Demo with mock data (no GPU required)
python demo_pipeline.py --output-dir demo_results/

# View results
start demo_results\dashcam_collision\report.html  # Windows
open demo_results/dashcam_collision/report.html   # Linux/Mac
```

### Process Real Video

```bash
# Single video
python -m insurance_mvp.pipeline \
    --video-path data/dashcam001.mp4 \
    --output-dir results/

# Batch processing (4 parallel workers)
python -m insurance_mvp.pipeline \
    --video-dir data/dashcam/ \
    --output-dir results/ \
    --parallel 4
```

---

## Architecture

### Pipeline Stages

```
Input Video
    │
    ├─> B1: Signal Mining (Audio + Motion + Proximity)
    │       └─> Top-20 danger clips
    │
    ├─> B2: Video-LLM Inference (Qwen2.5-VL)
    │       └─> Severity + Hazards + Reasoning
    │
    ├─> B3: Insurance Domain Logic
    │       ├─> Fault Assessment (0-100%)
    │       └─> Fraud Detection (0.0-1.0)
    │
    ├─> B4: Conformal Prediction
    │       └─> Prediction Sets + Uncertainty
    │
    └─> B5: Review Priority
            └─> URGENT / STANDARD / LOW_PRIORITY
                │
                └─> Output (JSON + HTML)
```

### Component Integration

```python
# config.py - Configuration management
PipelineConfig
├─ VideoConfig          # Video processing settings
├─ MiningConfig         # Signal mining thresholds
├─ CosmosConfig         # Video-LLM settings
├─ ConformalConfig      # Uncertainty quantification
└─ WhisperConfig        # Transcription settings

# pipeline.py - Main orchestrator
InsurancePipeline
├─ AudioAnalyzer        # mining/audio.py
├─ MotionAnalyzer       # mining/motion.py
├─ ProximityAnalyzer    # mining/proximity.py
├─ SignalFuser          # mining/fuse.py
├─ CosmosClient         # cosmos/client.py
├─ FaultAssessor        # insurance/fault_assessment.py
├─ FraudDetector        # insurance/fraud_detection.py
└─ SplitConformal       # conformal/split_conformal.py
```

---

## Configuration

### YAML Configuration

Copy `config.example.yaml` to `config.yaml`:

```yaml
# Output
output_dir: results
log_level: INFO

# Processing
parallel_workers: 1
resume_from_checkpoint: true
continue_on_error: true

# Mining
mining:
  top_k_clips: 20
  audio_weight: 0.3
  motion_weight: 0.3
  proximity_weight: 0.4

# Cosmos (Video-LLM)
cosmos:
  backend: qwen2.5-vl-7b  # or "mock"
  device: auto  # auto, cuda, cpu
  max_concurrent_inferences: 2

# Conformal Prediction
conformal:
  alpha: 0.1  # 90% confidence

# Feature Flags
enable_conformal: true
enable_transcription: true
enable_fraud_detection: true
enable_fault_assessment: true
```

### Environment Variables

```bash
export INSURANCE_OUTPUT_DIR=results
export INSURANCE_COSMOS_BACKEND=mock
export INSURANCE_PARALLEL_WORKERS=4
export INSURANCE_ENABLE_CONFORMAL=true
```

### CLI Overrides

```bash
python -m insurance_mvp.pipeline \
    --video-path data/test.mp4 \
    --cosmos-backend mock \
    --parallel 4 \
    --log-level DEBUG \
    --no-conformal
```

**Priority:** CLI > Environment Variables > YAML > Defaults

---

## Usage

### Single Video

```bash
python -m insurance_mvp.pipeline \
    --video-path data/dashcam001.mp4 \
    --output-dir results/
```

**Output:**

```
results/
└── dashcam001/
    ├── results.json       # Full assessment data
    ├── report.html        # HTML visualization
    └── checkpoint.json    # Resume checkpoint
```

### Batch Processing

```bash
python -m insurance_mvp.pipeline \
    --video-dir data/dashcam/ \
    --output-dir results/ \
    --parallel 4
```

**Output:**

```
results/
├── batch_summary.json     # Batch statistics
├── video001/
│   ├── results.json
│   └── report.html
├── video002/
│   ├── results.json
│   └── report.html
└── ...
```

### With Configuration File

```bash
python -m insurance_mvp.pipeline \
    --config my_config.yaml \
    --video-path data/test.mp4
```

---

## Output Format

### JSON Report (`results.json`)

```json
{
  "video_id": "dashcam001",
  "timestamp": "2026-02-17T12:34:56.789Z",
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
      "causal_reasoning": "Vehicle turned left without checking...",
      "recommended_action": "REVIEW"
    }
  ],
  "summary": {
    "total_clips": 5,
    "severity_distribution": {"HIGH": 2, "MEDIUM": 2, "LOW": 1},
    "priority_distribution": {"URGENT": 2, "STANDARD": 3},
    "avg_confidence": 0.75,
    "avg_fault_ratio": 60.0,
    "avg_fraud_score": 0.12
  }
}
```

### HTML Report (`report.html`)

Interactive table with:
- **Color-coded severity** (RED = HIGH, YELLOW = MEDIUM, GREEN = LOW)
- **Prediction sets** (uncertainty visualization)
- **Review priority** (highlighted)
- **Fault ratio and fraud scores**
- **Recommended actions**

---

## Advanced Features

### Resume from Checkpoint

Process interrupted? No problem:

```yaml
resume_from_checkpoint: true
```

The pipeline saves `checkpoint.json` after each clip. On restart, it skips already processed clips.

### GPU Resource Management

Prevent CUDA OOM with semaphore-based concurrency control:

```yaml
cosmos:
  max_concurrent_inferences: 2  # Max 2 clips in GPU memory
```

### Error Handling

```yaml
continue_on_error: true  # Skip failed clips, continue processing
max_retries: 3           # Retry transient errors
retry_delay_sec: 5.0     # Wait 5s between retries
```

Failed clips get default assessment with `review_priority: URGENT`.

### Performance Profiling

```yaml
enable_profiling: true
enable_metrics: true
```

Metrics saved to `batch_summary.json`:

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

## Testing

### Unit Tests

```bash
pytest insurance_mvp/tests/test_pipeline.py -v
```

### Demo (Mock Backend)

```bash
python demo_pipeline.py --verbose
```

### Integration Test (Real Backend)

```bash
# Requires GPU + Qwen2.5-VL
python -m insurance_mvp.pipeline \
    --video-path data/test.mp4 \
    --cosmos-backend qwen2.5-vl-7b \
    --output-dir test_results/
```

---

## Performance

**Hardware:** NVIDIA RTX 5090 (24GB VRAM)

| Stage | Time per Clip |
|-------|--------------|
| Mining | ~2s |
| VLM Inference | ~8s |
| Fault + Fraud | ~0.3s |
| Conformal | ~0.01s |
| **Total** | **~10s** |

**Throughput:**
- Single video (20 clips): ~200s
- Batch (10 videos, 4 workers): ~500s

---

## Troubleshooting

### CUDA Out of Memory

**Solution:** Reduce concurrent inferences

```yaml
cosmos:
  max_concurrent_inferences: 1
```

### Slow Processing

**Solutions:**

1. **Parallel processing:**
   ```bash
   --parallel 4
   ```

2. **Mock backend (testing):**
   ```bash
   --cosmos-backend mock
   ```

3. **Reduce top-K:**
   ```yaml
   mining:
     top_k_clips: 10
   ```

### Missing Dependencies

```bash
# Whisper
pip install -e ".[whisper]"

# CUDA not available
cosmos:
  device: cpu
```

---

## API Integration

For REST API access, see:

```bash
# Start API server
python -m insurance_mvp.api.main

# Upload video via API
curl -X POST http://localhost:8000/api/analyze \
     -F "video=@data/dashcam001.mp4"
```

See `api/README.md` for full API documentation.

---

## File Structure

```
insurance_mvp/
├── config.py                 # Configuration management
├── config.example.yaml       # Example config
├── pipeline.py               # Main orchestrator
├── demo_pipeline.py          # Demo script
├── PIPELINE_GUIDE.md         # User guide
├── PIPELINE_README.md        # This file
│
├── mining/                   # B1: Signal mining
│   ├── audio.py
│   ├── motion.py
│   ├── proximity.py
│   └── fuse.py
│
├── cosmos/                   # B2: Video-LLM
│   ├── client.py
│   └── prompt.py
│
├── insurance/                # B3: Domain logic
│   ├── fault_assessment.py
│   ├── fraud_detection.py
│   └── schema.py
│
├── conformal/                # B4: Uncertainty
│   └── split_conformal.py
│
└── tests/
    ├── test_pipeline.py      # Pipeline tests
    └── test_*.py
```

---

## Next Steps

1. **Run demo:**
   ```bash
   python demo_pipeline.py
   ```

2. **Process real video:**
   ```bash
   python -m insurance_mvp.pipeline --video-path data/test.mp4
   ```

3. **Customize config:**
   ```bash
   cp config.example.yaml config.yaml
   nano config.yaml
   ```

4. **Batch processing:**
   ```bash
   python -m insurance_mvp.pipeline --video-dir data/ --parallel 4
   ```

5. **Deploy API:**
   ```bash
   python -m insurance_mvp.api.main
   ```

---

## Documentation

- **[PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)** - Comprehensive user guide
- **[config.example.yaml](config.example.yaml)** - Configuration reference
- **[README.md](README.md)** - Project overview
- **[api/README.md](api/README.md)** - API documentation

---

## Support

**Issues:** Internal issue tracker
**Team:** SOPilot Development Team
**License:** Proprietary - Internal use only

---

## Changelog

### 2026-02-17 - Initial Release

- ✅ Configuration management (YAML + env + CLI)
- ✅ End-to-end pipeline orchestration
- ✅ Batch processing with parallel workers
- ✅ GPU resource management
- ✅ Incremental saving and resume
- ✅ Error handling and retry logic
- ✅ Performance profiling and metrics
- ✅ Comprehensive tests (pytest)
- ✅ Demo script with mock data
- ✅ HTML and JSON output formats
- ✅ Documentation (README + Guide)
