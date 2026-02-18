# SOPilot Issue Tracker

Remaining refactoring and improvement items, prioritized by impact.

---

## P0 - High Priority (Security / Stability)

### ~~[P0-1] Add tests for `audit_service.py`~~ ✅ DONE (2026-02-12)
Added `tests/test_audit_service.py` (19 tests): audit trail, HMAC-SHA256 signed export, signature roundtrip, path sanitization.

### ~~[P0-2] Add tests for `db.py`~~ ✅ DONE (2026-02-12)
Added `tests/test_db.py` (46 tests): schema init, _ensure_column guards, video CRUD, clips, job lifecycles, audit queries.

### ~~[P0-3] Add tests for `dtw_gpu.py`~~ ✅ DONE (2026-02-12)
Added `tests/test_dtw_gpu.py` (13 tests): DtwAlignment, CPU fallback, auto-routing, GPU availability, diagnostics.

---

## P1 - Medium Priority (Core Functionality)

### ~~[P1-1] Add tests for `embeddings.py`~~ ✅ DONE (2026-02-13)
Added `tests/test_embeddings.py` (47 tests): HeuristicClipEmbedder feature extraction, VJepa2Embedder temporal sampling/pooling/error paths, AutoEmbedder fallback/recovery, build_embedder factory.

### ~~[P1-2] Add tests for `worker_tasks.py`~~ ✅ DONE (2026-02-13)
Added `tests/test_worker_tasks.py` (10 tests): singleton init, job delegation, shutdown idempotency, error propagation.

### ~~[P1-3] Comprehensive API endpoint tests~~ ✅ DONE (2026-02-13)
Added `tests/test_api_edge_cases.py` (45 tests): role/auth helpers, search endpoint validation, query param boundaries, 404 paths, metrics endpoint.

### ~~[P1-4] Add tests for `storage.py`~~ ✅ DONE (2026-02-13)
Added `tests/test_storage.py` (5 tests): ensure_directories creation, idempotency, nested paths.

### ~~[P1-5] Add tests for `nn/functional.py`~~ ✅ DONE (2026-02-12)
Added `tests/test_nn_functional_unit.py` with 19 tests covering softmin3, pairwise_euclidean_sq, pairwise_cosine_dist.

---

## P2 - Low Priority (Hardening)

### ~~[P2-1] Add tests for `adapt/train_domain_adapter.py`~~ ✅ DONE (2026-02-18)
Added `tests/test_train_domain_adapter.py` (44 tests): DistRuntime, _init_dist_runtime, _load_video_rows, _select_source_path, _infer_dim, _write_skip_report, _parse_args, main() integration (adapter creation, skip reports, filters).

### ~~[P2-2] Add tests for `nn/constants.py`~~ ✅ DONE (2026-02-12)
Added `tests/test_nn_constants_unit.py` with 3 tests.

### ~~[P2-3] Add tests for `logging_config.py`~~ ✅ DONE (2026-02-18)
Added `tests/test_logging_config.py` (22 tests): configure_logging, get_logger, LogContext, log_execution_time.

### ~~[P2-4] Add tests for `metrics.py`~~ ✅ DONE (2026-02-18)
Added `tests/test_metrics.py` (20 tests): job counters, duration trackers, gauge updates, build info, GPU metrics.

### ~~[P2-5] Add tests for `main.py`~~ ✅ DONE (2026-02-18)
Added `tests/test_main.py` (3 tests): app creation, routes, run() function.

---

## P3 - Tech Debt / Nice-to-Have

### ~~[P3-1] Pin exact CI dependency versions~~ ✅ DONE (2026-02-18)
`requirements-ci-base.txt` + `requirements-ci-insurance.txt` with pinned versions. CI jobs install lock files before editable install.

### ~~[P3-2] Add pre-commit hooks~~ ✅ DONE (2026-02-18)
`.pre-commit-config.yaml` with ruff check --fix + ruff format. pre-commit added to dev deps.

### ~~[P3-3] Add type checking~~ ✅ DONE (2026-02-18)
`mypy.ini` config + mypy in CI lint job (non-blocking). Checks core modules: scoring_head, soft_dtw, rag_service, step_engine.

### ~~[P3-4] Coverage reporting~~ ✅ DONE (2026-02-18)
pytest-cov added to dev deps. Insurance CI: `--cov-fail-under=60` (current: 74%). SOPilot CI: `--cov-fail-under=75`.

### ~~[P3-5] Docker / container support~~ ✅ DONE (2026-02-13)
Multi-stage Dockerfile (root=CPU-only, docker/Dockerfile.cpu=with PyTorch, docker/Dockerfile.gpu=CUDA 12.1). docker-compose.yml with API+worker+redis+qdrant+postgres. .env.example with all vars.

### ~~[P3-6] API documentation~~ ✅ DONE (2026-02-18)
Added `json_schema_extra` examples to key Pydantic models (HealthResponse, VideoIngest, ScoreRequest, ScoreCreate, VideoInfo, TrainingCreate). OpenAPI /docs now shows example payloads.

---

## Insurance MVP (2026-02-18)

### ~~[INS-1] Test coverage~~ ✅ DONE
312 tests across 12 test files: conformal (25), fault assessment (30), fraud detection (25), schema validation (20), report generator (31), cosmos client (26), E2E pipeline (10), API basic (20+), insurance domain (41), mining pipeline (22), pipeline (21).

### ~~[INS-2] CI integration~~ ✅ DONE
Insurance test job in `.github/workflows/ci.yml` (py3.10+3.12), E2E benchmark as CI gate, lint+format checks for `insurance_mvp/`.

### ~~[INS-3] Lint pass~~ ✅ DONE
418 ruff errors auto-fixed + 10 manual fixes. All checks passing.

### ~~[INS-4] Pydantic V2 migration~~ ✅ DONE
All `@validator` → `@field_validator`, all `class Config` → `model_config = ConfigDict(...)`.

### ~~[INS-5] FastAPI modernization~~ ✅ DONE
`@app.on_event` → `lifespan` context manager.

### ~~[INS-6] VLM prompt redesign~~ ✅ DONE
Removed calibration bias, added chain-of-thought, visual evidence criteria, system message. 8 regression tests.

### [INS-7] Real VLM benchmark validation
Run `python scripts/insurance_real_benchmark.py` on GPU. Target: severity accuracy >= 67%. Requires GPU with >= 14GB VRAM.

### ~~[INS-8] Japanese localization~~ ✅ DONE
Full JA/EN bilingual reports with executive summary.

---

## Completed (2026-02-11 Refactoring)

- [x] Wave 1: Dead code removal, double `_now()` bug fix, `NeuralModelCache`, `evaluate_sop()` decomposition, `make_test_settings()` fixture, generalized DB job helpers
- [x] Wave 2: `nn/functional.py` extraction, DDL injection prevention, soft_dtw input validation, NaN threshold fix (50% -> 20%), cost matrix deduplication
- [x] Wave 3: `nn/constants.py` (GAMMA_MIN, INF), registry-based neural model loader, V-JEPA2 retry on transient errors, AutoEmbedder periodic re-probe
- [x] Git init + initial commit + `baseline-refactor` tag
- [x] E2E smoke test (`scripts/smoke_e2e.py`) - 13 checks
- [x] CI setup (lint + test + smoke jobs)
- [x] Ruff lint/format pass (134 auto-fixed, 48 reformatted)
- [x] Dependency audit (added ruff to dev deps)

## Completed (2026-02-12 Priority 5 + Refactoring)

- [x] Priority 5: 2-Stage Event Detection (`event_detection_service.py`, `event_detection_demo.py`, 22 tests)
- [x] Extracted `temporal.py`: shared `temporal_iou()` (eliminated 3x duplication across rag_service, event_detection, vigil_metrics)
- [x] Extracted `llm_utils.py`: shared `parse_llm_json()` (eliminated 2x duplication across rag_service, event_detection)
- [x] Unified `compute_video_checksum` → delegates to `rag_service.compute_video_id` (eliminated SHA-256 reimplementation)
- [x] Centralized mock VIGIL services: `make_mock_vigil_services()` in conftest.py (deduped test_rag + test_event_detection)
- [x] Fixed `evaluation/__init__.py`: exports vigil_metrics (EvidenceRecallResult, EventDetectionResult, etc.)
- [x] Simplified `chunking_result_to_clip_records`: 4 loops → `itertools.chain` one-liner
- [x] Extracted `index_video_micro()` into vigil_helpers.py (deduped smoke_e2e + event_detection_demo)
- [x] Added `tests/test_nn_functional_unit.py` (19 tests) — closes P1-5
- [x] Added `tests/test_nn_constants_unit.py` (3 tests) — closes P2-2
- [x] Full lint pass: 46 ruff errors → 0 (import sorting, unused imports, B905 strict=, F821, B007)
- [x] Full test suite: 600 tests passing
