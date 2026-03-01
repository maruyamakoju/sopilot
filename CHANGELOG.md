# Changelog

All notable changes to SOPilot are documented in this file.

---

## [1.0.0] — 2026-03-02

### Added
- **Gold Builder**: `POST /gold` accepts `enforce_quality=true` to reject low-quality gold
  videos at upload time (HTTP 422 with per-axis quality breakdown). Prevents degraded reference
  videos from affecting downstream scores. UI: "品質ゲート" checkbox + detailed quality wizard
  card showing per-axis pass/fail results.
- **SOP Versioning**: Each gold video receives a sequential `gold_version` number per task
  (v1, v2, …). Returned in `POST /gold`, `GET /videos`, `GET /videos/{id}` responses.
  UI: version badge in gold list + version prefix in video detail pane.
- **Evidence Clips**: Every deviation includes `gold_timecode` and `trainee_timecode`
  (`[start_sec, end_sec]`) pinpointing the exact moment in each video. UI: "ジャンプ" button
  seeks both video players to the deviation timestamp.
- `VideoIngestResponse` schema now includes `gold_version` and `quality` fields.
- 18 new automated tests for v1.0 features (`tests/test_v1_features.py`).
- `DEMO.md` — 5-minute demo script for technical reviewers.

### Changed
- Version bump: 0.9.0 → 1.0.0 (`pyproject.toml`, `sopilot/__init__.py`, UI topbar).
- Test count: 877 → 895.
- `QUICKSTART.md` updated with Gold Builder (§3.1.1), Evidence Clips, gold_version examples.
- `TECHNICAL_SUMMARY.md` updated with v1.0 features section, API surface, algorithm table.
- `README.md` updated with v1.0 New Features section, test count, DEMO.md link.

### Fixed
- Nothing — zero regressions from v0.9.0.

---

## [0.9.0] — 2026-02-28

### Added
- Production-grade deployment: Docker Compose single-container, non-root user (uid 1000).
- LOSO evaluation: 99.40% accuracy, FP=0, on 3,507 production score jobs.
- Decision thresholds evidence-based at 60.0/50.0 (LOSO-validated).
- `POST /admin/rescore` — re-apply thresholds to all stored jobs.
- `score_band` (excellent/passing/needs_review/poor) and `decision_basis` fields.
- Ensemble scoring with ICC(2,1), Grubbs outlier removal, bootstrap CI.
- Learning curve analysis with GP + CUSUM changepoint detection.
- Operator trend analytics (`GET /analytics/operators/{id}/trend`).
- SOP step definition CRUD (`PUT /tasks/steps`).
- PDF report generation (`GET /score/{id}/report/pdf`).
- Japanese template comments per deviation type and severity.
- Video quality gate: 5-axis check (brightness, sharpness, stability, resolution, duration).
- Structured JSON error handling with error codes.
- Rate limiting (sliding-window deque, 120 req/min + burst 20).
- SQLite connection pooling with atomic job claims.
- Webhook notification with exponential backoff retry.
- Audit logging: structured JSON events.
- 877 automated tests (unit + integration + property-based + concurrency).
- `QUICKSTART.md` production deployment guide.

### Infrastructure
- FastAPI + SQLite (WAL mode) + Pydantic v2, Python 3.11.
- ColorMotion-v1 CPU embedder (no GPU required).
- Vectorized DTW with Sakoe-Chiba band.
- Middleware stack: CorrelationID → CORS → APIKey → RateLimit.
