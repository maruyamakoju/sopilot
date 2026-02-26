# Reproducibility Checklist

This document records the exact steps, commands, seeds, and hardware used to
reproduce all reported results for the Insurance MVP dashcam evaluation pipeline.

**Last Updated**: 2026-02-24

---

## Hardware Requirements

| Backend          | Minimum Hardware                                    |
|------------------|-----------------------------------------------------|
| `mock`           | Any machine with Python 3.10+, no GPU needed        |
| `qwen2.5-vl-7b`  | GPU with >= 16 GB VRAM (tested: NVIDIA RTX 5090)    |

> All CI/CD pipeline checks use `--backend mock`.
> The 90% real-VLM accuracy result was measured on an RTX 5090 with 24 GB VRAM.

---

## Software Dependencies

Install the insurance extras (includes scipy, ultralytics, jinja2, pyyaml, tqdm):

```bash
pip install -e ".[insurance]"
```

Full version pins (from `pyproject.toml`):

| Package                   | Version constraint   |
|---------------------------|----------------------|
| Python                    | >= 3.10              |
| fastapi                   | >= 0.115.0           |
| uvicorn[standard]         | >= 0.30.0            |
| numpy                     | >= 1.26.0            |
| opencv-python-headless    | >= 4.10.0            |
| pydantic                  | >= 2.8.0             |
| reportlab                 | >= 4.2.0             |
| sqlalchemy                | >= 2.0.0             |
| jinja2                    | >= 3.1.0             |
| pyyaml                    | >= 6.0.0             |
| scipy                     | >= 1.11.0            |
| ultralytics               | >= 8.1.0             |
| tqdm                      | >= 4.66.0            |

For real VLM evaluation add:

```bash
pip install "transformers>=4.51.3" "qwen-vl-utils[decord]>=0.0.10" "accelerate>=0.34.0"
```

---

## Seed Management

All experiments use **seed = 42** (set via `PipelineConfig.seed`).

The seed is propagated at pipeline initialization:

```python
# insurance_mvp/pipeline/orchestrator.py
import random, numpy as np, torch
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
```

To override in scripts:

```bash
python scripts/research_benchmark.py --backend mock --seed 42
python -m insurance_mvp.pipeline benchmark --backend mock --seed 42
```

---

## Reported Results (2026-02-25)

| Experiment                               | Result                                      | Script                              |
|------------------------------------------|---------------------------------------------|-------------------------------------|
| Mock VLM accuracy (10 demo videos)       | 10/10 (100%)                                | `vlm_accuracy_benchmark.py`         |
| Real VLM accuracy (RTX 5090, 10 videos)  | 9/10 (90%)                                  | `real_vlm_eval_10.py`               |
| E2E pipeline benchmark (mock)            | 9/9 checks                                  | `insurance_e2e_benchmark.py`        |
| Real mining batch (5 JP dashcam videos)  | 5/5 success                                 | `batch_process.py`                  |
| Full system benchmark, mock (10 videos)  | 1.00 [1.00, 1.00] (95% CI, BCa, n=10)      | `research_benchmark.py`             |
| Ablation: remove motion signal           | 0.20 (catastrophic, motion is critical)     | `research_benchmark.py --ablation`  |
| Sensitivity: motion_weight threshold     | ≥ 0.30 required for 100% accuracy           | `research_benchmark.py --sensitivity-analysis` |
| McNemar vs MajorityClass baseline        | p=0.041 (significant)                       | `research_benchmark.py --ablation`  |
| **Nexar mock (50 real videos)**          | **86.0% [72.0%, 92.0%] (95% CI, BCa, n=50)** | `real_data_benchmark.py`          |
| **Nexar collision recall**               | **100% (25/25 HIGH correctly identified)**  | `real_data_benchmark.py`            |
| **Nexar real VLM (50 videos, RTX 5090)** | **20.0% [8.0%, 32.0%] (95% CI, BCa, n=50)** | `real_data_benchmark.py`           |
| **Nexar real VLM — collision recall**    | **0% (0/25 HIGH detected — 2fps misses <1s events)** | `real_data_benchmark.py`    |
| **Nexar real VLM — normal recall**       | **40% (10/25 NONE; 12→MEDIUM, 3→LOW)**    | `real_data_benchmark.py`            |
| **Nexar fps=4 max_frames=80 (RTX 5090)** | **INVALID — 52/53 OOM (39.8 GiB req > 31.84 GiB avail)** | `real_data_benchmark.py --fps 4 --max-frames 80` |
| **Nexar fps=4 max_frames=48 (RTX 5090)** | **18.0% [8.0%, 30.0%] (95% CI, BCa, n=50) — collision recall 0%, normal recall 36%** | `real_data_benchmark.py --fps 4 --max-frames 48` |

---

## Quick Start (Mock Backend — no GPU needed)

```bash
# 1. Install dependencies
pip install -e ".[insurance]"

# 2. Run E2E benchmark (mock VLM, deterministic, seed=42)
python scripts/insurance_e2e_benchmark.py --backend mock

# 3. Run unified research benchmark with ablation study
python scripts/research_benchmark.py --backend mock --ablation

# 4. Reproduce VLM accuracy (mock): expected 10/10
python scripts/vlm_accuracy_benchmark.py --backend mock

# 5. Multi-source evaluation (demo + optional jp/nexar)
python scripts/expanded_video_eval.py --backend mock --sources dashcam_demo

# 6. Batch processing on demo videos
python scripts/batch_process.py --input-dir data/dashcam_demo --backend mock

# 7. CLI benchmark subcommand (equivalent to insurance_e2e_benchmark.py)
python -m insurance_mvp.pipeline benchmark --backend mock

# 8. CLI benchmark + ablation via the pipeline CLI
python -m insurance_mvp.pipeline benchmark --backend mock --ablation
```

---

## Real VLM Evaluation (GPU required)

```bash
# Requires: GPU >= 16 GB VRAM, transformers >= 4.51.3, qwen-vl-utils >= 0.0.10

# 1. Install real VLM deps
pip install -e ".[insurance,vigil]"

# 2. Run real VLM eval on 10 demo videos: expected 9/10 (90%)
python scripts/real_vlm_eval_10.py --backend qwen2.5-vl-7b

# 3. Batch on JP dashcam data (20 videos, ~2.3 min/video)
python scripts/batch_process.py --input-dir data/jp_dashcam --backend qwen2.5-vl-7b

# 4. Benchmark using real VLM
python -m insurance_mvp.pipeline benchmark --backend qwen2.5-vl-7b
```

---

## Data Sources

### Demo videos (synthetic, included in repo)

```
data/dashcam_demo/
    collision.mp4
    near_miss.mp4
    normal.mp4
    metadata.json       <- ground truth labels
```

### JP Dashcam (yt-dlp)

```bash
# Download ~20 Japanese dashcam compilation videos
# On Windows use: python -m yt_dlp (not the yt-dlp command directly)
python -m yt_dlp -o "data/jp_dashcam/%(title)s.%(ext)s" \
    --match-filter "duration > 60" \
    "ytsearch20:ドライブレコーダー 事故 まとめ 2025"
```

### Nexar Collision Dataset (Hugging Face)

```bash
# Download via hf_hub_download (avoids torchcodec dependency)
# --n-per-class 25 gives 25 collision + 25 normal = 50 videos (seed=42)
python scripts/download_nexar.py --n-per-class 25 --output data/real_dashcam/nexar

# Run benchmark with mock VLM (deterministic, no GPU)
# Expected: 86.0% [72.0%, 92.0%] (95% CI, BCa, n=50)
python scripts/real_data_benchmark.py \
    --input data/real_dashcam/nexar \
    --backend mock \
    --output reports/nexar_mock_benchmark.json

# Run benchmark with real VLM (requires RTX >= 16 GB VRAM)
python scripts/real_data_benchmark.py \
    --input data/real_dashcam/nexar \
    --backend real \
    --output reports/nexar_real_vlm_benchmark.json
```

**Naming convention**: `pos_XXXXX.mp4` → gt_severity=HIGH, `neg_XXXXX.mp4` → gt_severity=NONE.
`ground_truth.json` is auto-generated by `download_nexar.py`.

---

## Step-by-Step: Data Download → Evaluation → Report Generation

### Step 1 — Environment

```bash
git clone https://github.com/maruyamakoju/sopilot.git
cd sopilot
pip install -e ".[insurance,dev]"
```

### Step 2 — Verify install

```bash
python scripts/smoke_e2e.py --verbose
```

Expected: all checks pass (no GPU needed).

### Step 3 — Generate / download data

```bash
# Demo videos are already in data/dashcam_demo/
# For JP dashcam or Nexar, see "Data Sources" section above
```

### Step 4 — Training benchmark (optional, for SOP neural pipeline)

```bash
python scripts/train_benchmark.py --device cpu --epochs-multiplier 0.1
# GPU: --device cuda --epochs-multiplier 1.0
```

### Step 5 — Run evaluations

```bash
# Mock (deterministic, no GPU)
python scripts/insurance_e2e_benchmark.py --backend mock
python scripts/vlm_accuracy_benchmark.py --backend mock
python scripts/research_benchmark.py --backend mock --ablation

# Real VLM (GPU >= 16 GB VRAM)
python scripts/real_vlm_eval_10.py --backend qwen2.5-vl-7b
```

### Step 6 — Generate HTML report

```bash
# Single video report
python -m insurance_mvp.pipeline run \
    --video-path data/dashcam_demo/collision.mp4 \
    --output-dir results/ \
    --cosmos-backend mock

# Multi-clip HTML report is written to results/<video_id>_report.html
```

### Step 7 — Run full test suite

```bash
# Insurance MVP tests (~24 s)
python -m pytest insurance_mvp/tests/ -q

# SOPilot tests
python -m pytest tests/ -q

# Full suite
python -m pytest -q
```

---

## Conformal Prediction Calibration

The conformal predictor uses pre-trained calibration scores stored inside the
`insurance_mvp/` package. No external calibration data download is needed for
the mock backend.

Configuration:

```python
# insurance_mvp/config.py
ConformalConfig(
    alpha=0.1,               # 90% coverage guarantee
    severity_levels=["NONE", "LOW", "MEDIUM", "HIGH"],
    use_pretrained_calibration=True,
)
```

---

## Signal Mining Configuration

The three mining analyzers are fused with these default weights:

| Analyzer  | Weight |
|-----------|--------|
| Audio     | 0.30   |
| Motion    | 0.40   |
| Proximity | 0.30   |

Speed optimizations active by default:

- `downscale_factor = 0.5`
- `frame_skip = 10`
- `max_analysis_duration = 600 s`

Real-world timing: ~66 s for a 20-minute 720p video (9x speedup vs naive).

---

## Known Limitations / Edge Cases

- **GPU device configuration (fixed 2026-02-25)**: `DeviceType.AUTO` (default) now
  correctly sets `device_map="auto"` (GPU) and moves inputs to the model device.
  Before the fix, `AUTO` resolved to `device_map="cpu"` — model loaded on CPU even
  on GPU machines. If you observe ~170s/video inference (vs. expected ~28s), check
  that `cosmos.device` is `"auto"` or `"cuda"`, not `"cpu"`.
- **CUDA retry loop corruption (fixed 2026-02-25)**: `_run_inference_with_retry()`
  had two bugs that corrupted long-running benchmarks (50+ videos):
  (1) GPU cleanup checked `device == "cuda"` but default is `"auto"` → `empty_cache()`
  never ran between retries; (2) CPU fallback permanently set `self.config.device = "cpu"`,
  causing all subsequent videos to use CPU. Fixed: CUDA errors now call
  `cuda.synchronize() + empty_cache()` and retry on GPU without touching the device config.
  The `finally` block in `_run_inference()` also now clears cache for `device="auto"`.
- **Nexar 2fps frame sampling vs sub-second collisions**: Real VLM achieves only 20%
  accuracy on Nexar (vs 90% on demo videos). The Nexar collision events are typically
  <1 second; at 2fps (one frame per 500ms), the impact frame is often not sampled.
  Collision recall = 0% (0/25 HIGH). The VLM sees surrounding context (traffic → MEDIUM,
  calm roads → NONE) but misses the actual impact.
- **fps=4 / max_frames=80 OOM (RTX 5090, 31.84 GiB)**: Attempting 80 frames at 4fps
  requires ~39.8 GiB for the KV cache (80 frames × ~512 visual tokens/frame → 40,960 tokens;
  KV: 2 × 28 layers × 40960 × 3584 hidden × 2 bytes ≈ 16.5 GB, plus 14 GB model weights).
  Result: 52/53 inference OOM, all error assessments (conf=0), recalibration bumps
  LOW→MEDIUM → 0% accuracy — **this result is invalid/not comparable**.
  SDPA (`attn_implementation="sdpa"`) reduces peak activation memory but NOT KV-cache size.
  Maximum safe frame count on this GPU: **max_frames=48** (covers full clip at ~2.5fps effective).
- **fps=4 / max_frames=48 does NOT improve over fps=2 / max_frames=48 for Nexar**: Both achieve
  0% collision recall. For 40-second Nexar clips, both configurations produce exactly 48 frames
  uniformly sampled → identical effective rate of ~1.2fps. The theoretical 4fps benefit only
  applies to clips shorter than 24s. Collision recall remained 0% (0/25 HIGH). The root cause
  is the **VLM's systematic MEDIUM bias on dashcam footage** — it predicts MEDIUM for 68% of
  collision clips and 52% of normal clips regardless of temporal sampling rate.
- **swerve_avoidance** scenario: real VLM predicts LOW (correct label: MEDIUM).
  No nearby objects trigger recalibration, so the VLM score stands. This is the
  single miss in the 9/10 real-VLM result.
- **Windows paths**: use raw strings or forward slashes when passing video paths
  to `qwen_vl_utils`. Do NOT use the `file:///` prefix.
- **Batch size = 1**: `BatchNorm1d` in the neural ScoringHead fails with a single
  sample; `_forward_single()` is used automatically by the MC Dropout path.
- **Qwen2.5-VL**: requires `transformers >= 4.51.3` and `qwen-vl-utils >= 0.0.10`.
- **yt-dlp on Windows**: use `python -m yt_dlp`, not the `yt-dlp` command directly.
- **Nexar dataset**: use `hf_hub_download()` directly; the `datasets` library
  requires `torchcodec` which is not available on all platforms.

---

## Platform Verification

### Fresh Linux / Ubuntu 22.04

```bash
git clone https://github.com/maruyamakoju/sopilot.git
cd sopilot
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[insurance,dev]"
python scripts/smoke_e2e.py --verbose
python -m pytest insurance_mvp/tests/ -q
```

### Windows 10/11

```powershell
git clone https://github.com/maruyamakoju/sopilot.git
cd sopilot
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[insurance,dev]"
python scripts/smoke_e2e.py --verbose
python -m pytest insurance_mvp/tests/ -q
```

### Docker (CPU-only)

```bash
docker build -t sopilot:test -f Dockerfile .
docker run sopilot:test python scripts/smoke_e2e.py --verbose
docker run sopilot:test python -m pytest insurance_mvp/tests/ -q
```

---

## File Checksums (demo data)

Run to verify demo data integrity after checkout:

```bash
python -c "
import hashlib, pathlib
for p in sorted(pathlib.Path('data/dashcam_demo').glob('*.mp4')):
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    print(f'{h}  {p.name}')
"
```

*(Actual checksums depend on your local demo video generation run.
Re-generate with `python scripts/generate_test_video.py` if needed.)*
