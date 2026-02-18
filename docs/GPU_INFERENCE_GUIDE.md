# GPU Inference Guide - Insurance MVP

## Prerequisites

### Hardware
- NVIDIA GPU with **14GB+ VRAM** (RTX 3090, RTX 4090, A100, etc.)
- CUDA 11.8 or 12.x compatible

### Software
- Python 3.10+
- CUDA Toolkit 11.8+ or 12.x
- cuDNN 8.x+

## Quick Setup

### 1. Check GPU readiness

```bash
python scripts/insurance_gpu_check.py
```

This checks CUDA availability, VRAM, dependencies, and model download status.

### 2. Install dependencies

```bash
# Base install
pip install -e ".[vigil]"

# GPU-specific PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Qwen2.5-VL dependencies
pip install transformers>=4.51.3 qwen-vl-utils>=0.0.10
```

### 3. Download Qwen2.5-VL-7B model

The model downloads automatically on first use (~14GB). To pre-download:

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Download model and processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
print("Model downloaded successfully")
```

## Mock vs Real Backend

### Environment variable

```bash
# Mock mode (default, no GPU needed)
export INSURANCE_VLM_BACKEND=mock

# Real inference
export INSURANCE_VLM_BACKEND=qwen2.5-vl-7b
```

### Programmatic

```python
from insurance_mvp.cosmos.client import VideoLLMClient, VLMConfig

# Mock mode
client = VideoLLMClient(VLMConfig(model_name="mock"))

# Real mode
client = VideoLLMClient(VLMConfig(
    model_name="qwen2.5-vl-7b",
    device="cuda",
    dtype="bfloat16",
))
```

### Health check

```python
client = VideoLLMClient(VLMConfig(model_name="mock"))
status = client.health_check()
print(status)
# {'status': 'ok', 'model_name': 'mock', 'model_loaded': False, ...}
```

## Running Benchmarks

### Mock benchmark (no GPU)

```bash
python scripts/insurance_e2e_benchmark.py --output results/benchmark_mock.json
```

### Real benchmark (requires GPU)

```bash
export INSURANCE_VLM_BACKEND=qwen2.5-vl-7b
python scripts/insurance_e2e_benchmark.py --output results/benchmark_real.json
```

### Accuracy report

```bash
python scripts/insurance_accuracy_report.py --output reports/
```

## Troubleshooting

### "CUDA out of memory"
- Close other GPU-using applications
- Reduce `max_pixels` in VLMConfig (default: 602112)
- Use `float16` instead of `bfloat16` dtype
- Reduce `max_frames` (default: 32) to 16

### "No module named 'qwen_vl_utils'"
```bash
pip install qwen-vl-utils>=0.0.10
```

### "transformers version too old"
```bash
pip install transformers>=4.51.3
```

### Windows: "Timeout not supported"
Inference timeout uses Unix SIGALRM, unavailable on Windows. The system will proceed without timeout protection. Monitor GPU memory usage manually.

### Model download fails
Set HuggingFace cache directory:
```bash
export HF_HOME=/path/to/large/drive/.cache/huggingface
```

### qwen_vl_utils video_kwargs issue
The `process_vision_info()` function wraps `fps` in a list `[2.0]`. The client automatically unwraps this. If you encounter shape errors, ensure you're using the latest `qwen-vl-utils`.

## Architecture Notes

- Model is cached at class level (singleton pattern) - loaded once, reused across calls
- GPU semaphore limits concurrent inferences (default: 2)
- Inference timeout: 1200 seconds (Unix only)
- max_pixels hard limit: 602112 (768 * 28 * 28)
