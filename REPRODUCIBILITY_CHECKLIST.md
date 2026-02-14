# Reproducibility Verification Checklist

This checklist ensures SOPilot can be reproduced in clean environments.

**Last Verified**: 2026-02-15
**Verified By**: [Pending manual verification]

---

## ✅ Automated Checks (Already Done)

- [x] pyproject.toml dependencies complete
- [x] quick_demo.py created (30-second minimal demo)
- [x] README Quick Start updated with tiered approach
- [x] Troubleshooting section added to README
- [x] CI coverage gate (75% minimum)
- [x] CI type checking (core modules)

---

## 📋 Manual Verification Checklist

### Environment 1: Fresh Linux/Ubuntu

**Setup**:
```bash
git clone https://github.com/maruyamakoju/sopilot.git
cd sopilot
python3 -m venv .venv
source .venv/bin/activate
```

**Tests**:
- [ ] `pip install -e "."` succeeds (no errors)
- [ ] `python scripts/quick_demo.py` runs and shows "Success!"
- [ ] `pip install -e ".[dev]"` succeeds
- [ ] `python scripts/smoke_e2e.py --verbose` passes all 13 checks
- [ ] `python -m pytest tests/ -k "not vigil" -v` runs without vigil deps

**Time**: ~10 minutes
**Blockers**: Document any errors

---

### Environment 2: Fresh Windows 10/11

**Setup**:
```powershell
git clone https://github.com/maruyamakoju/sopilot.git
cd sopilot
python -m venv .venv
.venv\Scripts\activate
```

**Tests**:
- [ ] `pip install -e "."` succeeds
- [ ] `python scripts/quick_demo.py` runs and shows "Success!"
- [ ] Path handling works (no Unix-specific assumptions)
- [ ] OpenCV works (video codec available)

**Time**: ~10 minutes
**Blockers**: Document any Windows-specific issues

---

### Environment 3: Docker (CPU-only)

**Setup**:
```bash
docker build -t sopilot:test -f Dockerfile .
```

**Tests**:
- [ ] Build completes without errors
- [ ] Image size < 3GB
- [ ] `docker run sopilot:test python scripts/quick_demo.py` works
- [ ] `docker run sopilot:test python scripts/smoke_e2e.py --verbose` passes

**Time**: ~15 minutes (build + test)
**Blockers**: Document Docker-specific issues

---

### Environment 4: macOS (M1/M2 Silicon)

**Setup**:
```bash
git clone https://github.com/maruyamakoju/sopilot.git
cd sopilot
python3 -m venv .venv
source .venv/bin/activate
```

**Tests**:
- [ ] `pip install -e "."` succeeds (numpy/opencv work on ARM)
- [ ] `python scripts/quick_demo.py` runs
- [ ] No x86 vs ARM64 conflicts

**Time**: ~10 minutes
**Blockers**: Document ARM-specific issues

---

## 🔬 Advanced Checks (Optional)

### VIGIL-RAG Full Stack

**Prerequisites**:
- Docker Compose installed
- 8GB+ RAM

**Setup**:
```bash
docker-compose up -d qdrant postgres
pip install -e ".[vigil]"
```

**Tests**:
- [ ] Qdrant starts and is accessible
- [ ] `python scripts/generate_test_video.py` creates video
- [ ] VIGIL smoke test runs (with real video)

**Time**: ~20 minutes
**Blockers**: Document infrastructure issues

---

### Neural Training Pipeline

**Prerequisites**:
- PyTorch installed
- 4GB+ RAM

**Setup**:
```bash
pip install -e ".[ml]"
```

**Tests**:
- [ ] `python scripts/demo_training_convergence.py --epochs-multiplier 0.01` runs
- [ ] Training completes without OOM
- [ ] Models save/load correctly

**Time**: ~30 minutes
**Blockers**: Document ML-specific issues

---

## 🐛 Known Issues

### Issue 1: PySceneDetect slow on large videos
- **Impact**: Indexing >1hour videos can take 10+ minutes
- **Workaround**: Pre-chunk videos or use lower frame sample rate
- **Fix**: Future optimization

### Issue 2: Qwen2.5-VL requires 14GB VRAM
- **Impact**: Video-LLM won't work on small GPUs
- **Workaround**: Use mock mode or CPU (very slow)
- **Fix**: Add quantization support

### Issue 3: Windows path handling in some scripts
- **Impact**: Some demo scripts may fail on Windows
- **Workaround**: Use WSL or Linux
- **Fix**: Audit all Path() usage for Windows compatibility

---

## 📊 Reproducibility Score

**Target**: 90% (9/10 environments reproduce successfully)

**Current**: [To be measured after manual verification]

**Environments Tested**:
- [ ] Linux/Ubuntu 22.04 (Python 3.10)
- [ ] Linux/Ubuntu 22.04 (Python 3.11)
- [ ] Linux/Ubuntu 22.04 (Python 3.12)
- [ ] Windows 11 (Python 3.10)
- [ ] macOS Intel (Python 3.10)
- [ ] macOS ARM (Python 3.11)
- [ ] Docker CPU-only
- [ ] Docker GPU (CUDA 12.1)
- [ ] Google Colab (Python 3.10)
- [ ] GitHub Codespaces

---

## 🚀 Next Steps

1. **Manual verification**: Test on 3+ environments (Linux, Windows, Docker minimum)
2. **Document blockers**: Add any issues to GitHub Issues
3. **Update README**: Add platform-specific notes if needed
4. **Create quick start video**: 3-minute screen recording of installation → quick_demo.py

---

**Goal**: Partner should be able to run `quick_demo.py` in under 5 minutes on any platform.
