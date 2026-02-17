# Cosmos Module - Video-LLM Insurance Claim Assessment

Production-ready Video-LLM inference for automated insurance claim evaluation from dashcam footage.

## Overview

This module provides robust Video-LLM integration with:
- **NVIDIA Cosmos Reason 2** (when available)
- **Qwen2.5-VL-7B-Instruct** (current default, 7B parameters)
- **Mock mode** (for testing without GPU)

## Key Features

### 1. Production-Ready Error Handling
- **7-step JSON repair pipeline**:
  1. Direct JSON parse
  2. Markdown fence removal (```json ... ```)
  3. Truncation repair (add closing braces)
  4. Brace extraction (find JSON boundaries)
  5. Missing comma insertion
  6. Orphaned key removal
  7. Field extraction from text (last resort)
- Graceful degradation to default assessment on total failure
- Never crashes - always returns valid `ClaimAssessment` object

### 2. Model Caching
- Singleton pattern: models loaded once, cached for all future inference
- Significant speedup for batch processing
- Memory efficient

### 3. GPU Memory Management
- Automatic device placement (cuda/cpu)
- Mixed precision support (float16/bfloat16/float32)
- Configurable max_pixels to control VRAM usage

### 4. Timeout Handling
- 1200 second (20 minute) default timeout
- Prevents indefinite hangs on edge cases
- Configurable per use case

### 5. Calibrated Prompts
- Prevents over-prediction of HIGH severity
- Targets realistic distribution:
  - NONE: ~20% (near-miss, no damage)
  - LOW: ~40% (minor damage) - MOST COMMON
  - MEDIUM: ~25% (moderate damage)
  - HIGH: ~15% (severe damage, injury risk)
- Structured JSON output with fault ratio and fraud indicators

## Installation

### Dependencies
```bash
# Core dependencies (required)
pip install torch transformers opencv-python pillow pydantic numpy

# Qwen2.5-VL specific (recommended)
pip install qwen-vl-utils

# For testing
pip install pytest
```

### Hardware Requirements
- **Minimum**: CPU-only mode (slow, use mock mode for testing)
- **Recommended**: NVIDIA GPU with 16GB+ VRAM (e.g., RTX 4090, A6000)
- **Optimal**: NVIDIA GPU with 24GB+ VRAM (e.g., RTX 5090, A100)

## Quick Start

### Basic Usage
```python
from insurance_mvp.cosmos import create_client

# Create client (loads model once)
client = create_client(model_name="qwen2.5-vl-7b", device="cuda")

# Assess claim from dashcam video
assessment = client.assess_claim(
    video_path="dashcam_incident.mp4",
    video_id="claim_12345",
    start_sec=10.0,  # Start at 10 seconds
    end_sec=30.0     # End at 30 seconds
)

# Access structured results
print(f"Severity: {assessment.severity}")
print(f"Confidence: {assessment.confidence:.2%}")
print(f"Fault Ratio: {assessment.fault_assessment.fault_ratio}%")
print(f"Fraud Risk: {assessment.fraud_risk.risk_score:.2%}")
print(f"Recommended Action: {assessment.recommended_action}")
```

### Demo Script
```bash
# Mock mode (no GPU required, instant results)
python -m insurance_mvp.cosmos.demo --video dashcam.mp4 --mock

# Real inference with Qwen2.5-VL
python -m insurance_mvp.cosmos.demo --video dashcam.mp4 --output results.json

# Analyze specific time range
python -m insurance_mvp.cosmos.demo \
    --video dashcam.mp4 \
    --start-sec 15.0 \
    --end-sec 25.0 \
    --output analysis.json
```

### Testing
```bash
# Run all tests (uses mock mode, no GPU required)
pytest insurance_mvp/tests/test_cosmos_client.py -v

# Run specific test
pytest insurance_mvp/tests/test_cosmos_client.py::TestJSONParsing::test_markdown_fence_removal -v
```

## Architecture

### Files
```
cosmos/
├── __init__.py        # Module exports
├── schema.py          # Pydantic data models (ClaimAssessment, etc.)
├── prompt.py          # Insurance-specific prompts (calibrated)
├── client.py          # VLM client with 7-step JSON parsing
├── demo.py            # Command-line demo script
└── README.md          # This file
```

### Data Flow
```
Video File → Frame Sampling (4 FPS) → VLM Inference → JSON Parsing → ClaimAssessment
                                          ↓
                                    Timeout Guard
                                    (1200 seconds)
                                          ↓
                                    7-Step Repair
                                    Pipeline
                                          ↓
                                    Pydantic
                                    Validation
                                          ↓
                                    Default Fallback
                                    (on failure)
```

## Configuration

### VLMConfig Parameters
```python
from insurance_mvp.cosmos import VLMConfig, VideoLLMClient

config = VLMConfig(
    model_name="qwen2.5-vl-7b",    # Model to use
    device="cuda",                  # Device (cuda/cpu)
    dtype="bfloat16",               # Precision (float16/bfloat16/float32)
    fps=4.0,                        # Frame sampling rate
    max_frames=32,                  # Max frames per clip
    max_new_tokens=1024,            # Max output tokens
    temperature=0.3,                # Sampling temperature (lower = more conservative)
    timeout_sec=1200.0,             # Inference timeout
    min_pixels=256 * 28 * 28,       # Min resolution (Qwen only)
    max_pixels=768 * 28 * 28,       # Max resolution (Qwen only)
)

client = VideoLLMClient(config)
```

### Model Selection
- **qwen2.5-vl-7b**: Best balance of quality and speed (default)
- **nvidia-cosmos-reason-2**: Coming soon (higher quality, slower)
- **mock**: Instant fake results for testing

## Output Schema

### ClaimAssessment
```python
{
    "video_id": "claim_12345",
    "processing_time_sec": 45.2,

    # Severity assessment
    "severity": "MEDIUM",                    # NONE/LOW/MEDIUM/HIGH
    "confidence": 0.85,                      # 0.0-1.0
    "prediction_set": ["MEDIUM", "HIGH"],    # Conformal prediction set
    "review_priority": "STANDARD",           # URGENT/STANDARD/LOW_PRIORITY

    # Fault analysis
    "fault_assessment": {
        "fault_ratio": 75.0,                 # 0-100% (insured's fault)
        "reasoning": "Following too closely, failed to brake in time",
        "applicable_rules": ["Following Distance Rule"],
        "scenario_type": "rear_end",
        "traffic_signal": null,
        "right_of_way": "Lead vehicle"
    },

    # Fraud detection
    "fraud_risk": {
        "risk_score": 0.15,                  # 0.0-1.0
        "indicators": [],                     # Empty = no fraud detected
        "reasoning": "Accident appears genuine"
    },

    # Evidence
    "hazards": [
        {
            "type": "collision",
            "actors": ["insured_vehicle", "lead_vehicle"],
            "spatial_relation": "front",
            "timestamp_sec": 12.5
        }
    ],

    "evidence": [
        {
            "timestamp_sec": 12.5,
            "description": "Impact occurs - insured strikes rear of lead vehicle"
        }
    ],

    # Reasoning and action
    "causal_reasoning": "Driver failed to maintain safe following distance...",
    "recommended_action": "APPROVE"          # APPROVE/REVIEW/REJECT/REQUEST_MORE_INFO
}
```

## Prompts

### Main Assessment Prompt
- **Calibrated severity guidance**: Prevents over-prediction of HIGH severity
- **Structured JSON output**: All fields documented with examples
- **Fault assessment logic**: Common scenarios (rear-end, T-bone, etc.)
- **Fraud indicators**: Staged accidents, inconsistent damage, suspicious behavior
- **Prediction set rules**: Based on confidence level

### Specialized Prompts
```python
from insurance_mvp.cosmos.prompt import (
    get_claim_assessment_prompt,    # Full assessment (default)
    get_quick_severity_prompt,      # Just severity classification
    get_fault_assessment_prompt,    # Just fault ratio
    get_fraud_detection_prompt,     # Just fraud detection
)
```

## Performance

### Benchmarks (RTX 5090, Qwen2.5-VL-7B)
- **Model loading**: ~15 seconds (once, then cached)
- **Frame sampling**: ~0.5 seconds (30-second clip at 4 FPS)
- **Inference**: ~30-45 seconds (depends on clip length)
- **Total**: ~45-60 seconds per claim (first run), ~35-50 seconds (cached)

### Optimization Tips
1. **Batch processing**: Reuse client for multiple videos (model stays loaded)
2. **Reduce FPS**: Lower fps (e.g., 2.0) for faster processing
3. **Reduce max_pixels**: Lower max_pixels (e.g., 512*28*28) for less VRAM
4. **Use bfloat16**: Best quality/speed tradeoff on modern GPUs

## Troubleshooting

### Out of Memory (CUDA OOM)
```python
# Reduce max_pixels
config = VLMConfig(max_pixels=512 * 28 * 28)

# Or use float16 instead of bfloat16
config = VLMConfig(dtype="float16")

# Or reduce max_frames
config = VLMConfig(max_frames=16)
```

### Slow Inference
```python
# Reduce FPS (fewer frames)
config = VLMConfig(fps=2.0)

# Or reduce max_new_tokens
config = VLMConfig(max_new_tokens=512)

# Or use CPU (very slow, for testing only)
config = VLMConfig(device="cpu")
```

### JSON Parsing Errors
- The 7-step pipeline handles most cases automatically
- Check logs for warnings: `logger.warning("Failed to parse LLM JSON...")`
- If persistent, report issue with raw model output

### Timeout Errors
```python
# Increase timeout for long videos
config = VLMConfig(timeout_sec=1800.0)  # 30 minutes
```

## Roadmap

### Planned Features
- [x] Qwen2.5-VL-7B support
- [x] 7-step JSON repair pipeline
- [x] Model caching
- [x] Timeout handling
- [ ] NVIDIA Cosmos Reason 2 support (when released)
- [ ] Batch inference optimization
- [ ] Async/await API
- [ ] Streaming partial results
- [ ] Multi-GPU support

## License

Proprietary - Sompo Japan Insurance PoC MVP

## Contact

For issues or questions, contact the SOPilot team.
