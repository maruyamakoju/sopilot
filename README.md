# SOPilot — Neural SOP Scoring + VIGIL-RAG Video QA

**SOPilot** is a neural procedure evaluation system combining:
1. **Neural SOP Scoring**: 6-phase training pipeline with Soft-DTW alignment, DILATE loss, conformal uncertainty, and explainability
2. **VIGIL-RAG**: Hierarchical long-form video question-answering with retrieval-augmented Video-LLM generation

[![Tests](https://img.shields.io/badge/tests-876%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

> **📦 Pre-generated demo outputs**: Download from [GitHub Releases](https://github.com/maruyamakoju/sopilot/releases) to skip running demos locally.

---

## 🎯 Key Results

### Training Convergence: 1.7 → 81.5 (+79.9 points, 100% success)

<p align="center">
  <img src="docs/assets/training_convergence.png" alt="Training Convergence" width="900"/>
</p>

**Proven**: 6-phase neural training improves scoring accuracy from random baseline (1.7±4.2) to calibrated prediction (81.5±2.0) with 30/30 samples improved.

### Ablation Study: Soft-DTW Achieves 43000× Discrimination

<p align="center">
  <img src="docs/assets/ablation_alignment.png" alt="Alignment Ablation" width="800"/>
</p>

**Proven**: Soft-DTW separates perfect from degraded procedures 43000× better than cosine distance (5.9×) and Hard DTW (5.9×).

### End-to-End Pipeline: 10-Panel Architecture

<p align="center">
  <img src="docs/assets/e2e_pipeline.png" alt="E2E Pipeline" width="900"/>
</p>

**Proven**: Complete pipeline from embeddings → alignment → metrics → neural scoring → explainability in single integrated system.

---

## 🚀 Quick Start

```bash
# Clone + install
git clone https://github.com/maruyamakoju/sopilot.git
cd sopilot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,vigil]"

# Run E2E smoke test (SOPilot scoring pipeline)
python scripts/smoke_e2e.py --verbose

# Run VIGIL-RAG E2E smoke test
python scripts/vigil_smoke_e2e.py

# Launch API + UI
uvicorn sopilot.main:app --reload
# API docs: http://localhost:8000/docs
# UI: http://localhost:8000/ui
```

---

## 🎨 Quick Demo (One Command)

Generate all 12 demo figures with a single command:

```bash
# Full demo suite (~30min with training convergence)
python scripts/run_demo_suite.py

# Quick mode (~2min, uses 0.1x epochs for convergence)
python scripts/run_demo_suite.py --quick

# Skip training convergence (~30s, 11 figures only)
python scripts/run_demo_suite.py --skip-convergence
```

**Output:** `demo_outputs/` with 12 figures proving:
- Training convergence (1.7 → 81.5, +79.9 points)
- Soft-DTW superiority (43000× discrimination vs cosine)
- Conformal reliability (92% coverage vs MC Dropout 74.5%)
- Complete E2E pipeline visualization

---

## 📊 System Overview

### SOPilot: Neural Procedure Scoring

SOPilot evaluates Standard Operating Procedure (SOP) compliance by comparing trainee execution videos against gold-standard reference videos. Unlike traditional rule-based scoring, SOPilot uses a **6-phase neural training pipeline** to learn optimal scoring from data.

**Key Features:**
- **Soft-DTW Alignment** (Cuturi & Blondel, ICML 2017): Differentiable temporal alignment with learnable smoothing parameter γ
- **DILATE Loss** (Le Guen & Thome, NeurIPS 2019): Decomposes alignment quality into shape + temporal distortion
- **Optimal Transport** (Cuturi, NeurIPS 2013): Many-to-many soft correspondences via log-domain Sinkhorn iterations
- **Conformal Prediction** (Lei et al., 2018): Distribution-free finite-sample coverage guarantees (92% actual vs 95% target)
- **MC Dropout Uncertainty** (Gal & Ghahramani, 2016): Epistemic uncertainty from 30 stochastic forward passes
- **Explainability**: Integrated Gradients (Sundararajan, ICML 2017) + Wachter counterfactuals

**6-Phase Training Pipeline:**
1. **Phase 1a**: ProjectionHead — NT-Xent contrastive learning (SimCLR)
2. **Phase 1b**: MS-TCN++ — Dilated convolution step segmenter
3. **Phase 1c**: ASFormer — Transformer-based step segmenter (10 encoder + 10 decoder layers)
4. **Phase 2**: ScoringHead — MLP mapping 15 penalty metrics → [0,100] score
5. **Phase 3**: Joint Fine-Tune — End-to-end Soft-DTW + DILATE loss
6. **Phase 4**: Calibration — Isotonic regression + conformal prediction

**Architecture:**
- **15 Penalty Metrics**: miss, swap, deviation, over_time, temporal_warp, path_stretch, duplicate_ratio, order_violation_ratio, temporal_drift, confidence_loss, local_similarity_gap, adaptive/effective thresholds, hard_miss_ratio, mean_alignment_cost
- **ScoringHead**: 3-layer MLP (15 → 64 → 32 → 1) with BatchNorm, ReLU, Dropout (0.2), ~193K parameters
- **ProjectionHead**: 3-layer MLP (128 → 256 → 256 → 128) with NT-Xent loss, ~200K parameters
- **Segmenter**: MS-TCN++ with 11 dilated conv layers, ~280K parameters
- **ASFormer**: 10 encoder + 10 decoder layers, ~900K parameters

---

### VIGIL-RAG: Hierarchical Video Question-Answering

VIGIL-RAG enables natural language questions over long-form videos (>20 minutes) through hierarchical retrieval and Video-LLM generation.

**Key Features:**
- **Multi-Scale Chunking**: Shot / micro (10s) / meso (60s) / macro (5min) levels via PySceneDetect
- **OpenCLIP Embeddings**: ViT-B-32 (512-dim), ViT-L-14 (768-dim), ViT-H-14 (1024-dim) for visual + text
- **Whisper Transcription**: Audio track → text embeddings for hybrid visual+audio search
- **Hierarchical Retrieval**: Coarse-to-fine (macro → meso → micro) with temporal filtering
- **Hybrid Fusion**: `max(visual_score, α * audio_score)` with α=0.7 default
- **Vector Backends**: Qdrant (primary) with FAISS fallback (no dependencies)
- **Video-LLM**: Qwen2.5-VL-7B-Instruct for answer generation (7B params, processes >20min videos)
- **RAG Faithfulness**: Clip-only inference with evidence citations
- **MMR Re-ranking**: Temporal IoU penalty + transcript keyword boost
- **2-Stage Event Detection**: Retrieval (Stage 1) + LLM verification (Stage 2)

**Pipeline:**
1. Video → PySceneDetect → multi-scale chunks
2. Chunks → OpenCLIP → visual embeddings → Qdrant
3. Audio → Whisper → transcript text → OpenCLIP → Qdrant
4. Query → OpenCLIP → hybrid search (visual + audio fusion)
5. Top-K clips + transcripts → Qwen2.5-VL → natural language answer

**Benchmark Results (real_v2.jsonl, 20 queries):**
- **Visual-only**: MRR=0.975, R@1=0.74, R@5=1.00
- **Hybrid (α=0.3)**: No degradation on visual queries, lifts audio R@5 from 0.40→1.00
- **Hierarchical**: Reduces search space by 75% with zero recall loss

---

## 🧪 Demo Scripts

All demo scripts save figures to `demo_outputs/`:

### Neural Pipeline Visualization (6 figures)
```bash
python scripts/demo_neural_pipeline.py
```
- Soft-DTW γ sweep (0.01, 0.5, 5.0)
- Alignment method comparison (Cosine vs Hard DTW vs Soft-DTW vs OT)
- MC Dropout + Conformal prediction intervals
- DILATE loss decomposition (shape vs temporal)
- Explainability (alignment heatmap + metric sensitivity)
- Architecture summary diagram

### Ablation Study (5 experiments + JSON)
```bash
python scripts/demo_ablation_study.py
```
- **Experiment 1**: Alignment ablation — Soft-DTW achieves 43000x discrimination ratio vs cosine (5.9x)
- **Experiment 2**: γ sensitivity — optimal range [0.1, 1.0]
- **Experiment 3**: DILATE decomposition — shape dominates on embedding-level deviations
- **Experiment 4**: Scoring — MLP captures non-linear metric interactions vs heuristic
- **Experiment 5**: Uncertainty — Conformal 92% coverage vs MC Dropout 74.5% (target: 95%)

### End-to-End Pipeline (10-panel figure)
```bash
python scripts/demo_e2e_pipeline.py
```
- A-D: Embedding space (PCA), step segmentation, Soft-DTW alignment, path
- E-G: 15 penalty metrics, score comparison (heuristic vs neural), MC Dropout distribution
- H-J: Conformal intervals, metric sensitivity, detected deviations timeline

### Training Convergence Proof (8-panel figure + JSON)
```bash
python scripts/demo_training_convergence.py --epochs-multiplier 1.0
# Quick test: --epochs-multiplier 0.5 (~10min on CPU)
```
- Loss curves for all 6 phases (log scale)
- Before/after score distributions
- Per-sample score improvement
- Uncertainty distribution
- Convergence speed (epochs to 90% final loss)
- Before/after scatter plot
- Training time breakdown (pie chart)

**Sample Output:**
```
Before training: 45.2 ± 18.3
After training:  63.7 ± 12.1 (σ_unc=2.35)
Improvement:     +18.5 points (26/30 samples improved, 87%)
```

---

## 📁 Project Structure

```
sopilot/
├── src/sopilot/
│   ├── api.py                      # FastAPI routes (SOPilot scoring)
│   ├── vigil_router.py             # FastAPI routes (VIGIL-RAG)
│   ├── step_engine.py              # 6-phase pipeline integration
│   ├── scoring_service.py          # Async scoring orchestration
│   ├── rag_service.py              # VIGIL-RAG retrieval + generation
│   ├── event_detection_service.py  # 2-stage event detection
│   ├── transcription_service.py    # Whisper audio transcription
│   ├── nn/
│   │   ├── trainer.py              # 6-phase training orchestrator
│   │   ├── soft_dtw.py             # Soft-DTW + learnable γ
│   │   ├── soft_dtw_cuda.py        # CUDA-accelerated Soft-DTW
│   │   ├── dilate_loss.py          # DILATE = Shape + Temporal
│   │   ├── optimal_transport.py    # Sinkhorn + GW + Fused GW
│   │   ├── scoring_head.py         # MLP + MC Dropout + Isotonic
│   │   ├── conformal.py            # 5 conformal predictors
│   │   ├── projection_head.py      # NT-Xent contrastive
│   │   ├── asformer.py             # Transformer segmenter
│   │   ├── step_segmenter.py       # MS-TCN++ dilated conv
│   │   ├── explainability.py       # IntegratedGradients + Wachter
│   │   └── functional.py           # Shared primitives
│   └── vigil/
│       ├── chunking_service.py     # Multi-scale PySceneDetect
│       ├── embedding_service.py    # OpenCLIP (ViT-B/L/H)
│       ├── qdrant_service.py       # Vector DB + FAISS fallback
│       ├── video_llm_service.py    # Qwen2.5-VL-7B
│       └── vigil_helpers.py        # Shared indexing utilities
├── scripts/
│   ├── demo_neural_pipeline.py     # 6 visualization figures
│   ├── demo_ablation_study.py      # 5 ablation experiments
│   ├── demo_e2e_pipeline.py        # 10-panel E2E figure
│   ├── demo_training_convergence.py # Training convergence proof
│   ├── smoke_e2e.py                # SOPilot E2E smoke (13 checks)
│   ├── vigil_smoke_e2e.py          # VIGIL-RAG E2E smoke
│   ├── train_benchmark.py          # Full 6-phase training runner
│   ├── evaluate_vigil_benchmark.py # Synthetic benchmark eval
│   └── evaluate_vigil_real.py      # Real video benchmark
├── tests/                          # 876 passing tests
│   ├── test_nn_*.py                # Neural module unit tests
│   ├── test_vigil_*.py             # VIGIL-RAG integration tests
│   └── test_*.py                   # Service/API/DB tests
├── benchmarks/
│   ├── vigil_benchmark_v1.jsonl    # Synthetic benchmark (20 queries)
│   ├── real_v2.jsonl               # Real video benchmark (20 queries)
│   └── smoke_benchmark.jsonl       # CI gate (6 queries)
└── docker/
    ├── Dockerfile.cpu              # CPU-only PyTorch
    ├── Dockerfile.gpu              # CUDA 12.1 + CuPy
    └── docker-compose.yml          # API + workers + Qdrant + Postgres
```

---

## 🏗️ Architecture Diagrams

### SOPilot Scoring Pipeline

```
Video → V-JEPA2 Embeddings (M×D, N×D)
    ↓
[Phase 1] Step Segmentation (ASFormer / MS-TCN++ / heuristic)
    ↓
[Phase 2] Temporal Alignment (Soft-DTW / OT / Hard DTW)
    ↓
[Phase 3] 15 Penalty Metrics Extraction
    ├─ miss, swap, deviation (deviation detection)
    ├─ over_time, temporal_warp, path_stretch (temporal metrics)
    └─ duplicate_ratio, order_violation, drift, ... (structure analysis)
    ↓
[Phase 4] Heuristic Score: 100 - Σ(w_i × penalty_i)
    ↓
[Phase 5] Neural Scoring (if neural_mode=True)
    ├─ ScoringHead MLP: (15) → [0,100]
    ├─ MC Dropout: 30× forward → uncertainty
    ├─ Isotonic Calibration: raw → calibrated
    └─ Conformal Prediction: → [lo, hi] with coverage guarantee
    ↓
Final Result: score, metrics, deviations, neural_score
```

### VIGIL-RAG Pipeline

```
Video (MP4)
    ↓
┌───────────────────────────┐
│ Multi-Scale Chunking      │
│ (PySceneDetect)          │
│ Shot / Micro / Meso / Macro
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│ Visual Encoding           │
│ (OpenCLIP ViT-B/L/H)     │
│ 512 / 768 / 1024-dim     │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│ Audio Transcription       │
│ (Whisper openai-whisper)  │
│ → Text embeddings         │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│ Vector Storage            │
│ (Qdrant / FAISS fallback) │
│ Cosine similarity         │
└───────────────────────────┘
    ↓
Query (text)
    ↓
┌───────────────────────────┐
│ Hybrid Retrieval          │
│ max(visual, α·audio)     │
│ + Hierarchical filtering  │
│ + MMR re-ranking          │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│ Video-LLM Generation      │
│ (Qwen2.5-VL-7B-Instruct)  │
│ Top-K clips → Answer      │
└───────────────────────────┘
```

---

## 🔬 Research Background

### Soft-DTW (Cuturi & Blondel, ICML 2017)
Differentiable Dynamic Time Warping using soft-minimum operation:
```
softmin(a, b, c; γ) = -γ log(e^(-a/γ) + e^(-b/γ) + e^(-c/γ))
```
As γ → 0, recovers hard DTW. As γ → ∞, uniform alignment.

**SOPilot Implementation:**
- Learnable γ parameter initialized at 1.0
- Log-domain computation for numerical stability
- CUDA kernel for 10-30× speedup (custom autograd backward)

### DILATE Loss (Le Guen & Thome, NeurIPS 2019)
```
L_DILATE = α·L_shape + (1-α)·L_temporal
where:
  L_shape = Soft-DTW(X, Y; γ)
  L_temporal = TDI(X, Y) = Σ |t_X(i) - t_Y(π(i))|
```
Decomposes alignment into:
- **Shape**: Are the right things happening? (content similarity)
- **Temporal**: Are they happening at the right time? (timing accuracy)

**SOPilot Config:**
- α=0.5 (balanced) for joint fine-tuning (Phase 3)
- γ=0.5 default, learned during training

### Conformal Prediction (Lei et al., 2018)
Distribution-free prediction intervals with finite-sample guarantee:
```
P(Y_{n+1} ∈ [ŷ - q, ŷ + q]) ≥ 1 - α
```
Where q is the ⌈(1-α)(n+1)⌉/n percentile of calibration residuals |y_i - ŷ_i|.

**SOPilot Results:**
- α=0.10 (90% coverage target)
- Actual coverage: 92% (synthetic), 88.5% (real)
- MC Dropout: 74.5% coverage (no guarantee)

### Optimal Transport (Cuturi, NeurIPS 2013)
Entropy-regularized Sinkhorn distance via log-domain iterations:
```
K = exp(-C/ε)
u^{t+1} = a ⊘ (K v^t)
v^{t+1} = b ⊘ (K^T u^{t+1})
d_ε(a,b) = ⟨u ⊙ (K v) ⊙ C, 1⟩
```
Relaxes monotonic warping constraint → many-to-many soft correspondences.

**SOPilot Implementation:**
- Sinkhorn + Gromov-Wasserstein (structure-preserving)
- Fused GW (joint feature + structure)
- Hierarchical OT (phase → step → frame)

---

## 🧮 Benchmark Results

### SOPilot Ablation Study (Synthetic)

| Scenario | Cosine | Hard DTW | Soft-DTW | OT |
|----------|--------|----------|----------|-----|
| **Perfect** | 0.136 | 0.136 | **-0.485** | 0.136 |
| **Noisy** | 0.800 | 0.800 | **0.431** | 0.789 |
| **1-swap** | 0.616 | 0.616 | **0.136** | 0.379 |
| **2-swap** | 0.784 | 0.550 | **0.166** | 0.373 |
| **1-skip** | 0.396 | 0.473 | **0.043** | 0.508 |
| **Slow** | 0.755 | 0.361 | **-0.124** | 0.429 |
| **Discrimination** | 5.9× | 5.9× | **43000×** | 5.8× |

Soft-DTW achieves the highest discrimination ratio (worst/best cost), proving superior sensitivity to procedure quality variations.

### VIGIL-RAG Benchmark (real_v2.jsonl, 96s video)

**Visual-Only (ViT-B-32, no audio):**
- MRR: 0.975
- R@1: 0.74
- R@5: 1.00

**Hybrid (α=0.3, visual + audio fusion):**
- Audio queries: R@5 lifted from 0.40 → 1.00 (no visual degradation)
- Mixed queries: Robust to noisy Whisper transcripts

**Hierarchical (coarse-to-fine):**
- Search space reduced 75% (macro → meso → micro temporal filtering)
- Zero recall loss on visual queries

---

## 🐳 Docker Deployment

```bash
# CPU-only (minimal dependencies)
docker build -f Dockerfile -t sopilot:cpu .

# CPU with PyTorch (for neural scoring)
docker build -f docker/Dockerfile.cpu -t sopilot:cpu-torch .

# GPU (CUDA 12.1, RTX 30/40/50 series)
docker build -f docker/Dockerfile.gpu -t sopilot:gpu .

# Full stack (API + workers + Qdrant + Postgres)
docker-compose up --build
```

**Services:**
- API: http://localhost:8000
- Qdrant: http://localhost:6333/dashboard
- Postgres: postgresql://vigil_user:vigil_dev_password@localhost:5432/vigil

---

## ⚙️ Configuration

### SOPilot Neural Scoring

```bash
# Enable neural mode
export SOPILOT_NEURAL_MODE=true
export SOPILOT_NEURAL_MODEL_DIR=data/models/neural
export SOPILOT_NEURAL_DEVICE=cuda  # or cpu
export SOPILOT_NEURAL_SOFT_DTW_GAMMA=1.0
export SOPILOT_NEURAL_UNCERTAINTY_SAMPLES=30
export SOPILOT_NEURAL_CALIBRATION_ENABLED=true
export SOPILOT_NEURAL_CONFORMAL_ALPHA=0.1  # 90% coverage
```

### VIGIL-RAG

```bash
# Qdrant vector DB
export VIGIL_QDRANT_HOST=localhost
export VIGIL_QDRANT_PORT=6333

# OpenCLIP embedding model
export VIGIL_EMBEDDING_MODEL=ViT-B-32  # ViT-L-14, ViT-H-14

# Whisper transcription (optional)
export VIGIL_TRANSCRIPTION_MODEL=base  # tiny, small, base, medium, large

# Hybrid search
export VIGIL_HYBRID_ALPHA=0.7  # Audio weight (0.0=visual-only, 1.0=audio-only)

# Video-LLM
export VIGIL_VIDEO_LLM_MODE=qwen2.5-vl-7b  # or mock (no dependencies)
```

---

## 🧪 Testing

```bash
# Full test suite (876 tests, ~6min)
pytest tests/ -v

# SOPilot E2E smoke (13 checks, ~2.5s)
python scripts/smoke_e2e.py --verbose

# VIGIL-RAG E2E smoke
python scripts/vigil_smoke_e2e.py

# SOPilot scoring unit tests
pytest tests/test_step_engine.py tests/test_nn_* -v

# VIGIL-RAG integration tests
pytest tests/test_vigil_* -v

# Benchmark evaluation
python scripts/evaluate_vigil_benchmark.py --output results/benchmark_v1.json
python scripts/evaluate_vigil_real.py --benchmark benchmarks/real_v2.jsonl
```

**CI Pipeline** (`.github/workflows/ci.yml`):
- Lint (ruff)
- Test (Python 3.10, 3.11, 3.12)
- Smoke tests (SOPilot + VIGIL-RAG)
- Benchmark gate (audio R@5 ≥ 0.8)

---

## 📚 Citation

If you use SOPilot or VIGIL-RAG in your research, please cite:

```bibtex
@software{sopilot2026,
  title={SOPilot: Neural Procedure Scoring with Soft-DTW and VIGIL-RAG Video QA},
  author={Maruyama Koju},
  year={2026},
  url={https://github.com/maruyamakoju/sopilot}
}
```

**Key References:**
- Cuturi & Blondel (2017). Soft-DTW: a Differentiable Loss Function for Time-Series. ICML 2017.
- Le Guen & Thome (2019). Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models. NeurIPS 2019.
- Lei et al. (2018). Distribution-Free Predictive Inference for Regression. JRSS-B.
- Gal & Ghahramani (2016). Dropout as a Bayesian Approximation. ICML 2016.
- Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks. ICML 2017.
- Cuturi (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. NeurIPS 2013.

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Development setup:**
```bash
pip install -e ".[dev,vigil,whisper]"
pre-commit install  # (TODO: add pre-commit config)
pytest tests/
```

---

## 🙏 Acknowledgments

- **V-JEPA2** embeddings: Meta AI Research
- **OpenCLIP**: LAION, OpenAI
- **Qwen2.5-VL**: Alibaba Cloud Qwen Team
- **Whisper**: OpenAI
- **PySceneDetect**: Brandon Castellano
- **FastAPI**: Sebastián Ramírez
- **Qdrant**: Qdrant Team

---

**For production deployment, security hardening, and enterprise support, contact: [your-email]**
