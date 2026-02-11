# SOPilot Configuration Reference

Complete reference for all environment variables and configuration options.

---

## Table of Contents

1. [Core Infrastructure](#core-infrastructure)
2. [Video Processing](#video-processing)
3. [Embedder Configuration](#embedder-configuration)
4. [Queue & Workers](#queue--workers)
5. [Scoring Algorithm](#scoring-algorithm)
6. [Training & Adaptation](#training--adaptation)
7. [Authentication & Security](#authentication--security)
8. [Privacy Controls](#privacy-controls)
9. [Watch Folder](#watch-folder)
10. [Logging & Monitoring](#logging--monitoring)

---

## Core Infrastructure

### SOPILOT_DATA_DIR
- **Type:** Path
- **Default:** `data`
- **Description:** Root directory for all persistent data (videos, embeddings, database, reports)
- **Example:** `/var/lib/sopilot/data`

### SOPILOT_QUEUE_BACKEND
- **Type:** Enum (`inline`, `rq`)
- **Default:** `rq`
- **Description:** Queue backend for async job processing
  - `inline`: Synchronous execution (testing/dev only)
  - `rq`: Redis Queue (production)
- **Example:** `rq`

### SOPILOT_REDIS_URL
- **Type:** URL
- **Default:** `redis://127.0.0.1:6379/0`
- **Description:** Redis connection URL for RQ backend
- **Example:** `redis://:password@redis-host:6379/1`

### SOPILOT_RQ_QUEUE_PREFIX
- **Type:** String
- **Default:** `sopilot`
- **Description:** Prefix for RQ queue names
- **Example:** `sopilot` → queues: `sopilot_ingest`, `sopilot_score`, `sopilot_training`

### SOPILOT_RQ_JOB_TIMEOUT_SEC
- **Type:** Integer
- **Default:** `21600` (6 hours)
- **Description:** Maximum job execution time before timeout
- **Example:** `3600` (1 hour)

### SOPILOT_RQ_RESULT_TTL_SEC
- **Type:** Integer
- **Default:** `0` (disabled)
- **Description:** How long to keep job results in Redis (0 = forever)
- **Example:** `86400` (1 day)

### SOPILOT_RQ_FAILURE_TTL_SEC
- **Type:** Integer
- **Default:** `604800` (7 days)
- **Description:** How long to keep failed job info
- **Example:** `604800`

### SOPILOT_RQ_RETRY_MAX
- **Type:** Integer
- **Default:** `2`
- **Description:** Maximum number of automatic retries for failed jobs
- **Example:** `3`

---

## Video Processing

### SOPILOT_TARGET_FPS
- **Type:** Integer (≥1)
- **Default:** `4`
- **Description:** Target FPS for video sampling during ingest
- **Example:** `2` (slower, fewer clips), `8` (faster, more clips)
- **Impact:** Higher FPS = more clips = longer processing time

### SOPILOT_CLIP_SECONDS
- **Type:** Float (>0)
- **Default:** `4.0`
- **Description:** Duration of each video clip in seconds
- **Example:** `2.0` (shorter clips, more granular), `8.0` (longer clips, coarser)

### SOPILOT_MAX_SIDE
- **Type:** Integer (≥1)
- **Default:** `320`
- **Description:** Maximum resolution (longest side) for video frames
- **Example:** `256` (faster), `640` (higher quality, slower)

### SOPILOT_MIN_CLIP_COVERAGE
- **Type:** Float (0-1)
- **Default:** `0.6`
- **Description:** Minimum fraction of clip duration that must have valid frames
- **Example:** `0.5` (more lenient), `0.8` (stricter)

### SOPILOT_INGEST_EMBED_BATCH_SIZE
- **Type:** Integer (≥1)
- **Default:** `8`
- **Description:** Number of clips to embed in one batch during ingest
- **Example:** `16` (GPU: faster), `4` (CPU: lower memory)

### SOPILOT_UPLOAD_MAX_MB
- **Type:** Integer (≥1)
- **Default:** `1024` (1GB)
- **Description:** Maximum upload file size in megabytes
- **Example:** `2048` (2GB), `512` (512MB)

---

## Embedder Configuration

### SOPILOT_EMBEDDER_BACKEND
- **Type:** Enum (`auto`, `vjepa2`, `heuristic`)
- **Default:** `auto`
- **Description:** Video embedding model selection
  - `auto`: Try V-JEPA2, fallback to heuristic if unavailable
  - `vjepa2`: V-JEPA2 only (requires PyTorch + GPU)
  - `heuristic`: Fast CPU-based heuristic embedder
- **Example:** `vjepa2`

### SOPILOT_EMBEDDER_FALLBACK
- **Type:** Boolean
- **Default:** `true`
- **Description:** Enable automatic fallback to heuristic if primary embedder fails
- **Example:** `false` (fail hard if V-JEPA2 unavailable)

### SOPILOT_EMBEDDING_DEVICE
- **Type:** Enum (`auto`, `cuda`, `cpu`)
- **Default:** `auto`
- **Description:** Device for embedding inference
  - `auto`: Use CUDA if available, else CPU
  - `cuda`: Force CUDA (will fail if unavailable)
  - `cpu`: Force CPU
- **Example:** `cuda`

### V-JEPA2 Configuration

#### SOPILOT_VJEPA2_REPO
- **Type:** String
- **Default:** `facebookresearch/vjepa2`
- **Description:** torch.hub repository for V-JEPA2
- **Example:** `facebookresearch/vjepa2`

#### SOPILOT_VJEPA2_VARIANT
- **Type:** Enum
- **Default:** `vjepa2_vit_large`
- **Options:** `vjepa2_vit_large`, `vjepa2_vit_huge`, `vjepa2_vit_giant`
- **Description:** Model size variant
- **Example:** `vjepa2_vit_large` (1.2B params, fastest), `vjepa2_vit_giant` (slowest, best quality)

#### SOPILOT_VJEPA2_PRETRAINED
- **Type:** Boolean
- **Default:** `true`
- **Description:** Load pretrained weights from torch.hub
- **Example:** `false` (use local checkpoint only)

#### SOPILOT_VJEPA2_SOURCE
- **Type:** Enum (`hub`, `local`)
- **Default:** `hub`
- **Description:** Model loading source
  - `hub`: Download from torch.hub
  - `local`: Load from local repository
- **Example:** `local`

#### SOPILOT_VJEPA2_LOCAL_REPO
- **Type:** Path
- **Default:** `` (empty)
- **Description:** Path to local V-JEPA2 repository (required if source=local)
- **Example:** `/models/vjepa2`

#### SOPILOT_VJEPA2_LOCAL_CHECKPOINT
- **Type:** Path
- **Default:** `` (empty)
- **Description:** Path to local checkpoint file (.pt)
- **Example:** `/models/checkpoints/vjepa2_vitl.pt`

#### SOPILOT_VJEPA2_NUM_FRAMES
- **Type:** Integer
- **Default:** `64`
- **Description:** Number of frames to sample per clip for V-JEPA2
- **Example:** `32` (faster), `128` (slower, more temporal info)

#### SOPILOT_VJEPA2_IMAGE_SIZE
- **Type:** Integer
- **Default:** `256`
- **Description:** Input image resolution for V-JEPA2
- **Example:** `224`, `384`

#### SOPILOT_VJEPA2_BATCH_SIZE
- **Type:** Integer
- **Default:** `2`
- **Description:** Batch size for V-JEPA2 inference
- **Example:** `8` (RTX 3090), `16` (RTX 4090), `24` (RTX 5090)
- **Note:** Auto-detected based on GPU memory if using optimized embedder

---

## Queue & Workers

### SOPILOT_SCORE_WORKERS
- **Type:** Integer
- **Default:** `2`
- **Description:** Number of parallel score worker threads
- **Example:** `4` (more CPU cores), `1` (single GPU)

### SOPILOT_TRAIN_WORKERS
- **Type:** Integer
- **Default:** `1`
- **Description:** Number of parallel training worker threads
- **Example:** `1` (training is typically sequential)

---

## Scoring Algorithm

### SOPILOT_MIN_SCORING_CLIPS
- **Type:** Integer (≥1)
- **Default:** `4`
- **Description:** Minimum clips required in each video for scoring
- **Example:** `2` (lenient), `10` (strict)

### SOPILOT_CHANGE_THRESHOLD_FACTOR
- **Type:** Float
- **Default:** `1.0`
- **Description:** Multiplier for step boundary detection threshold (mean + factor*std)
- **Example:** `0.5` (more boundaries), `2.0` (fewer boundaries)

### SOPILOT_MIN_STEP_CLIPS
- **Type:** Integer (≥1)
- **Default:** `2`
- **Description:** Minimum clips per detected step
- **Example:** `1` (fine-grained), `5` (coarse-grained)

### SOPILOT_LOW_SIM_THRESHOLD
- **Type:** Float (0-1)
- **Default:** `0.75`
- **Description:** Similarity threshold below which clips are considered misaligned
- **Example:** `0.6` (stricter), `0.9` (more lenient)

### Scoring Weights

#### SOPILOT_WEIGHT_MISS
- **Type:** Float (≥0)
- **Default:** `12.0`
- **Description:** Penalty weight for missing steps
- **Example:** `20.0` (emphasize completeness)

#### SOPILOT_WEIGHT_SWAP
- **Type:** Float (≥0)
- **Default:** `8.0`
- **Description:** Penalty weight for step order violations
- **Example:** `15.0` (emphasize correct order)

#### SOPILOT_WEIGHT_DEV
- **Type:** Float (≥0)
- **Default:** `30.0`
- **Description:** Penalty weight for execution deviations
- **Example:** `40.0` (emphasize precision)

#### SOPILOT_WEIGHT_TIME
- **Type:** Float (≥0)
- **Default:** `15.0`
- **Description:** Penalty weight for timing issues
- **Example:** `10.0` (less emphasis on speed)

#### SOPILOT_WEIGHT_WARP
- **Type:** Float (≥0)
- **Default:** `12.0`
- **Description:** Penalty weight for temporal warping in DTW alignment
- **Example:** `5.0` (allow more temporal variation)

### SOPILOT_DTW_USE_GPU
- **Type:** Boolean
- **Default:** `true`
- **Description:** Enable GPU acceleration for DTW computation (requires CuPy)
- **Example:** `false` (force CPU)

---

## Training & Adaptation

### SOPILOT_NIGHTLY_ENABLED
- **Type:** Boolean
- **Default:** `false`
- **Description:** Enable automatic nightly training scheduler
- **Example:** `true` (production with continuous learning)

### SOPILOT_NIGHTLY_HOUR_LOCAL
- **Type:** Integer (0-23)
- **Default:** `2`
- **Description:** Local hour (24h format) to run nightly training
- **Example:** `3` (3 AM)

### SOPILOT_NIGHTLY_MIN_NEW_VIDEOS
- **Type:** Integer (≥0)
- **Default:** `10`
- **Description:** Minimum new videos since last training to trigger nightly run
- **Example:** `5` (more frequent), `20` (less frequent)

### SOPILOT_NIGHTLY_CHECK_SEC
- **Type:** Integer
- **Default:** `30`
- **Description:** Interval in seconds to check if nightly training should run
- **Example:** `60` (check every minute)

### SOPILOT_ADAPT_COMMAND
- **Type:** String
- **Default:** `` (empty, use builtin)
- **Description:** External command for custom feature adaptation
- **Example:** `python /scripts/train_adapter.py --job-id {job_id}`
- **Placeholders:** `{job_id}`, `{data_dir}`, `{models_dir}`, `{reports_dir}`, `{since}`

### SOPILOT_ADAPT_TIMEOUT_SEC
- **Type:** Integer (≥1)
- **Default:** `14400` (4 hours)
- **Description:** Maximum time for external adaptation command
- **Example:** `7200` (2 hours)

### SOPILOT_ENABLE_FEATURE_ADAPTER
- **Type:** Boolean
- **Default:** `true`
- **Description:** Enable feature adaptation (z-score normalization)
- **Example:** `false` (disable adaptation, use raw embeddings)

---

## Authentication & Security

### SOPILOT_AUTH_REQUIRED
- **Type:** Boolean
- **Default:** `true`
- **Description:** Require authentication for all non-public endpoints
- **Example:** `false` (dev/testing only)

### SOPILOT_API_TOKEN
- **Type:** String (secret)
- **Default:** `` (empty)
- **Description:** Single bearer token with configurable role
- **Example:** `changeme_production_secret_token`
- **Security:** Generate with `openssl rand -hex 32`

### SOPILOT_API_TOKEN_ROLE
- **Type:** Enum (`admin`, `operator`, `viewer`)
- **Default:** `admin`
- **Description:** Role assigned to SOPILOT_API_TOKEN
- **Example:** `operator`

### SOPILOT_API_ROLE_TOKENS
- **Type:** String (comma-separated)
- **Default:** `` (empty)
- **Description:** Multiple role-specific tokens: `role1:token1,role2:token2`
- **Example:** `admin:secret123,operator:secret456,viewer:secret789`

### SOPILOT_BASIC_USER
- **Type:** String
- **Default:** `` (empty)
- **Description:** Username for HTTP Basic auth
- **Example:** `admin`

### SOPILOT_BASIC_PASSWORD
- **Type:** String (secret)
- **Default:** `` (empty)
- **Description:** Password for HTTP Basic auth
- **Example:** `changeme_password`

### SOPILOT_BASIC_ROLE
- **Type:** Enum (`admin`, `operator`, `viewer`)
- **Default:** `admin`
- **Description:** Role assigned to Basic auth user
- **Example:** `operator`

### SOPILOT_AUTH_DEFAULT_ROLE
- **Type:** Enum (`admin`, `operator`, `viewer`)
- **Default:** `admin`
- **Description:** Default role when auth is disabled
- **Example:** `viewer` (most restrictive)

### SOPILOT_AUDIT_SIGNING_KEY
- **Type:** String (secret)
- **Default:** `` (empty)
- **Description:** HMAC secret key for audit trail signing
- **Example:** `changeme_audit_signing_key`
- **Security:** Generate with `openssl rand -hex 32`

### SOPILOT_AUDIT_SIGNING_KEY_ID
- **Type:** String
- **Default:** `local`
- **Description:** Identifier for audit signing key (for key rotation)
- **Example:** `prod_2026_q1`

---

## Privacy Controls

### SOPILOT_PRIVACY_MASK_ENABLED
- **Type:** Boolean
- **Default:** `false`
- **Description:** Enable privacy masking of video regions
- **Example:** `true`

### SOPILOT_PRIVACY_MASK_MODE
- **Type:** Enum (`black`, `blur`, `pixelate`)
- **Default:** `black`
- **Description:** Method for masking privacy regions
- **Example:** `blur`

### SOPILOT_PRIVACY_MASK_RECTS
- **Type:** String (semicolon-separated)
- **Default:** `` (empty)
- **Description:** Normalized rectangles to mask: `x1:y1:x2:y2;x1:y1:x2:y2`
- **Example:** `0:0:1:0.15;0.7:0.7:1:1` (top strip + bottom-right corner)
- **Coordinates:** (0,0) = top-left, (1,1) = bottom-right

### SOPILOT_PRIVACY_FACE_BLUR
- **Type:** Boolean
- **Default:** `false`
- **Description:** Automatically detect and blur faces
- **Example:** `true`

---

## Watch Folder

### SOPILOT_WATCH_ENABLED
- **Type:** Boolean
- **Default:** `false`
- **Description:** Enable watch folder daemon for automatic ingestion
- **Example:** `true`

### SOPILOT_WATCH_DIR
- **Type:** Path
- **Default:** `{data_dir}/watch_inbox`
- **Description:** Directory to watch for new video files
- **Example:** `/mnt/nfs/sopilot_inbox`

### SOPILOT_WATCH_POLL_SEC
- **Type:** Integer
- **Default:** `5`
- **Description:** Polling interval in seconds
- **Example:** `10` (less frequent), `1` (more responsive)

### SOPILOT_WATCH_TASK_ID
- **Type:** String
- **Default:** `` (empty, derive from path)
- **Description:** Default task_id for videos in watch folder
- **Example:** `assembly_line_1`
- **Note:** If empty, expects structure: `watch_dir/{task_id}/{role}/video.mp4`

### SOPILOT_WATCH_ROLE
- **Type:** Enum (`gold`, `trainee`, `audit`)
- **Default:** `trainee`
- **Description:** Default role for videos in watch folder
- **Example:** `gold`

---

## Logging & Monitoring

### SOPILOT_LOG_FORMAT
- **Type:** Enum (`json`, `console`)
- **Default:** `console`
- **Description:** Log output format
  - `json`: Structured JSON (for log aggregators)
  - `console`: Human-readable colored output
- **Example:** `json` (production)

### SOPILOT_LOG_LEVEL
- **Type:** Enum (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- **Default:** `INFO`
- **Description:** Minimum log level
- **Example:** `WARNING` (production, less verbose), `DEBUG` (development)

### SOPILOT_REPORT_TITLE
- **Type:** String
- **Default:** `SOPilot Audit Report`
- **Description:** Title for generated PDF reports
- **Example:** `Factory A - SOP Compliance Report`

---

## Configuration Examples

### Minimal Development Setup
```bash
export SOPILOT_DATA_DIR=./data
export SOPILOT_QUEUE_BACKEND=inline
export SOPILOT_EMBEDDER_BACKEND=heuristic
export SOPILOT_AUTH_REQUIRED=false
```

### Production with GPU
```bash
export SOPILOT_DATA_DIR=/var/lib/sopilot/data
export SOPILOT_QUEUE_BACKEND=rq
export SOPILOT_REDIS_URL=redis://redis:6379/0
export SOPILOT_EMBEDDER_BACKEND=vjepa2
export SOPILOT_EMBEDDING_DEVICE=cuda
export SOPILOT_DTW_USE_GPU=true
export SOPILOT_VJEPA2_BATCH_SIZE=16
export SOPILOT_AUTH_REQUIRED=true
export SOPILOT_API_TOKEN=$(openssl rand -hex 32)
export SOPILOT_AUDIT_SIGNING_KEY=$(openssl rand -hex 32)
export SOPILOT_LOG_FORMAT=json
export SOPILOT_LOG_LEVEL=INFO
```

### High-Throughput Factory Floor
```bash
export SOPILOT_TARGET_FPS=2
export SOPILOT_CLIP_SECONDS=8.0
export SOPILOT_MAX_SIDE=256
export SOPILOT_VJEPA2_BATCH_SIZE=24
export SOPILOT_INGEST_EMBED_BATCH_SIZE=24
export SOPILOT_SCORE_WORKERS=8
export SOPILOT_NIGHTLY_ENABLED=true
export SOPILOT_NIGHTLY_MIN_NEW_VIDEOS=5
```

### Privacy-Enhanced Setup
```bash
export SOPILOT_PRIVACY_MASK_ENABLED=true
export SOPILOT_PRIVACY_MASK_MODE=blur
export SOPILOT_PRIVACY_MASK_RECTS="0:0:1:0.2"  # Mask top 20%
export SOPILOT_PRIVACY_FACE_BLUR=true
```

---

## Validation

On startup, SOPilot validates all configuration:
- Numeric ranges (e.g., fps ≥ 1)
- Valid enums (e.g., queue_backend in {inline, rq})
- Path existence (for local model loading)
- Weight non-negativity

Invalid configuration will fail fast with detailed error messages.

---

## Configuration Priority

1. **Environment variables** (highest priority)
2. **`.env` file** in working directory
3. **Default values** (as documented)

---

## Next Steps

- **Deployment:** See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **API Usage:** See [API_REFERENCE.md](API_REFERENCE.md)
- **Troubleshooting:** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
