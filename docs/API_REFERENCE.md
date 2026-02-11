# SOPilot API Reference

Complete reference for all REST API endpoints.

**Base URL:** `http://localhost:8000`

**Authentication:** Bearer token or Basic auth (configurable)

---

## Table of Contents

1. [Health & Monitoring](#health--monitoring)
2. [Video Ingest](#video-ingest)
3. [Scoring](#scoring)
4. [Search](#search)
5. [Video Management](#video-management)
6. [Training](#training)
7. [Audit Trail](#audit-trail)
8. [Operations](#operations)

---

## Health & Monitoring

### GET `/health`

Health check endpoint.

**Authentication:** None required

**Response:**
```json
{
  "status": "ok",
  "db": true
}
```

**Status Codes:**
- `200 OK`: Service healthy
- `503 Service Unavailable`: Database unavailable

---

### GET `/metrics`

Prometheus metrics endpoint for monitoring.

**Authentication:** None required

**Response:** Prometheus text format

**Metrics Exposed:**
- `sopilot_ingest_jobs_total`: Counter by status
- `sopilot_score_jobs_total`: Counter by status
- `sopilot_training_jobs_total`: Counter by status/trigger
- `sopilot_job_duration_seconds`: Histogram (p50, p95, p99)
- `sopilot_dtw_execution_seconds`: Histogram by GPU usage
- `sopilot_embedding_generation_seconds`: Histogram by embedder
- `sopilot_queue_depth`: Gauge by queue name
- `sopilot_gpu_memory_bytes`: Gauge (allocated, reserved, total)
- `sopilot_active_workers`: Gauge by queue

---

## Video Ingest

### POST `/videos`

Upload a trainee video for processing.

**Authentication:** Operator role required

**Request:**
```http
POST /videos
Content-Type: multipart/form-data

file: (binary video file)
task_id: "assembly_line_1"
role: "trainee"  (optional, default: trainee)
site_id: "factory_a"  (optional)
camera_id: "cam_01"  (optional)
operator_id_hash: "abc123"  (optional)
```

**Response:**
```json
{
  "ingest_job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

**Status Codes:**
- `200 OK`: Video queued successfully
- `400 Bad Request`: Invalid file or parameters
- `403 Forbidden`: Insufficient permissions
- `413 Payload Too Large`: Exceeds SOPILOT_UPLOAD_MAX_MB

**Example (curl):**
```bash
curl -X POST http://localhost:8000/videos \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@trainee_video.mp4" \
  -F "task_id=assembly_line_1" \
  -F "role=trainee"
```

---

### POST `/gold`

Upload a gold standard video.

**Authentication:** Operator role required

**Request:** Same as `/videos` but role is forced to "gold"

**Response:**
```json
{
  "ingest_job_id": "550e8400-e29b-41d4-a716-446655440001",
  "status": "queued"
}
```

---

### GET `/videos/jobs/{ingest_job_id}`

Poll ingest job status.

**Authentication:** Viewer role required

**Response:**
```json
{
  "ingest_job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "task_id": "assembly_line_1",
  "role": "trainee",
  "requested_by": "token:operator",
  "video_id": 42,
  "num_clips": 75,
  "source_fps": 30.0,
  "sampled_fps": 4.0,
  "embedding_model": "vjepa2:vit_large:pt",
  "error_message": null,
  "queued_at": "2026-02-08T12:00:00Z",
  "started_at": "2026-02-08T12:00:01Z",
  "finished_at": "2026-02-08T12:00:45Z"
}
```

**Status Values:**
- `queued`: Waiting for worker
- `running`: Processing in progress
- `completed`: Successfully processed
- `failed`: Processing failed (see error_message)

**Status Codes:**
- `200 OK`: Job found
- `404 Not Found`: Job ID does not exist

---

## Scoring

### POST `/score`

Request a score comparison between gold and trainee videos.

**Authentication:** Operator role required

**Request:**
```json
{
  "gold_video_id": 1,
  "trainee_video_id": 42
}
```

**Response:**
```json
{
  "score_job_id": "score_20260208_120000_abc123",
  "status": "queued",
  "score": null
}
```

**Status Codes:**
- `200 OK`: Score job queued
- `400 Bad Request`: Invalid video IDs or mismatched task_id
- `403 Forbidden`: Insufficient permissions

---

### GET `/score/{score_job_id}`

Poll score job status and retrieve results.

**Authentication:** Viewer role required

**Response:**
```json
{
  "score_job_id": "score_20260208_120000_abc123",
  "status": "completed",
  "gold_video_id": 1,
  "trainee_video_id": 42,
  "requested_by": "token:operator",
  "score": 87.5,
  "error_message": null,
  "queued_at": "2026-02-08T12:00:00Z",
  "started_at": "2026-02-08T12:00:01Z",
  "finished_at": "2026-02-08T12:00:15Z",
  "result": {
    "score": 87.5,
    "metrics": {
      "miss": 1,
      "swap": 0,
      "deviation": 0.05,
      "over_time": 0.1,
      "temporal_warp": 0.02,
      "path_stretch": 0.01,
      "duplicate_ratio": 0.0,
      "order_violation_ratio": 0.0,
      "temporal_drift": 0.0,
      "confidence_loss": 0.0,
      "local_similarity_gap": 0.0,
      "adaptive_low_similarity_threshold": 0.75,
      "effective_low_similarity_threshold": 0.75,
      "hard_miss_ratio": 0.0,
      "mean_alignment_cost": 0.12
    },
    "step_boundaries": {
      "gold": [0, 15, 30, 45, 60],
      "trainee": [0, 14, 29, 45, 61]
    },
    "deviations": [
      {
        "type": "step_missing",
        "gold_step": 2,
        "gold_time": {"start_sec": 30.0, "end_sec": 45.0},
        "trainee_time": {"start_sec": null, "end_sec": null},
        "confidence": 1.0,
        "reason": "no aligned trainee clips"
      }
    ],
    "alignment_preview": [
      {"gold_clip": 0, "trainee_clip": 0, "similarity": 0.95}
    ],
    "clip_count": {"gold": 60, "trainee": 61},
    "step_map_preview": {
      "gold": [0, 0, 0, 1, 1, 1],
      "trainee": [0, 0, 0, 1, 1, 1]
    }
  }
}
```

---

### GET `/score/{score_job_id}/report.pdf`

Download PDF report for a completed score job.

**Authentication:** Viewer role required

**Response:** PDF file (application/pdf)

**Status Codes:**
- `200 OK`: PDF generated successfully
- `404 Not Found`: Job not found or not completed

---

## Search

### GET `/search`

Search for similar clips across videos in a task.

**Authentication:** Viewer role required

**Parameters:**
- `task_id` (required): Task identifier
- `video_id` (required): Query video ID
- `clip_idx` (required): Query clip index
- `k` (optional): Number of results (default: 5, max: 50)

**Example:**
```http
GET /search?task_id=assembly_line_1&video_id=42&clip_idx=10&k=5
```

**Response:**
```json
{
  "task_id": "assembly_line_1",
  "query_video_id": 42,
  "query_clip_idx": 10,
  "items": [
    {
      "similarity": 0.95,
      "video_id": 1,
      "clip_idx": 12,
      "start_sec": 48.0,
      "end_sec": 52.0,
      "role": "gold"
    },
    {
      "similarity": 0.89,
      "video_id": 38,
      "clip_idx": 9,
      "start_sec": 36.0,
      "end_sec": 40.0,
      "role": "trainee"
    }
  ]
}
```

---

## Video Management

### GET `/videos`

List videos with optional filtering.

**Authentication:** Viewer role required

**Parameters:**
- `task_id` (optional): Filter by task
- `limit` (optional): Max results (default: 50, max: 500)

**Response:**
```json
{
  "items": [
    {
      "video_id": 42,
      "task_id": "assembly_line_1",
      "role": "trainee",
      "site_id": "factory_a",
      "camera_id": "cam_01",
      "num_clips": 75,
      "embedding_model": "vjepa2:vit_large:pt",
      "created_at": "2026-02-08T12:00:00Z"
    }
  ]
}
```

---

### GET `/videos/{video_id}`

Get video metadata.

**Authentication:** Viewer role required

**Response:**
```json
{
  "video_id": 42,
  "task_id": "assembly_line_1",
  "role": "trainee",
  "site_id": "factory_a",
  "camera_id": "cam_01",
  "num_clips": 75,
  "embedding_model": "vjepa2:vit_large:pt",
  "created_at": "2026-02-08T12:00:00Z"
}
```

---

### GET `/videos/{video_id}/file`

Download original video file.

**Authentication:** Viewer role required

**Response:** Video file (video/mp4)

---

### DELETE `/videos/{video_id}`

Delete a video and all associated artifacts.

**Authentication:** Admin role required

**Response:**
```json
{
  "video_id": 42,
  "task_id": "assembly_line_1",
  "removed_files": [
    "/data/raw/video_42.mp4",
    "/data/embeddings/video_42.npy",
    "/data/embeddings/video_42.raw.npy",
    "/data/embeddings/video_42.json"
  ],
  "reindexed_clips": 180
}
```

**Status Codes:**
- `200 OK`: Video deleted successfully
- `403 Forbidden`: Admin role required
- `404 Not Found`: Video not found

---

## Training

### POST `/train/nightly`

Manually trigger a training job.

**Authentication:** Admin role required

**Response:**
```json
{
  "training_job_id": "train_20260208_120000_xyz789",
  "status": "queued",
  "trigger": "manual"
}
```

---

### GET `/train/jobs/{training_job_id}`

Get training job status.

**Authentication:** Viewer role required

**Response:**
```json
{
  "training_job_id": "train_20260208_120000_xyz789",
  "trigger": "manual",
  "status": "completed",
  "requested_by": "token:admin",
  "error_message": null,
  "queued_at": "2026-02-08T12:00:00Z",
  "started_at": "2026-02-08T12:00:01Z",
  "finished_at": "2026-02-08T12:05:30Z",
  "result": {
    "status": "completed",
    "mode": "builtin_feature_adapter",
    "adapter_path": "/data/models/feature_adapter_train_20260208_120000_xyz789.npz",
    "videos_used": 50,
    "clips_used": 3750,
    "embedding_dim": 768,
    "reindex": {
      "old_index_version": "v123",
      "new_index_version": "v124",
      "videos_refreshed": 50,
      "clips_indexed": 3750,
      "tasks_touched": ["assembly_line_1", "packaging_station_2"]
    }
  }
}
```

---

### GET `/train/nightly/status`

Get nightly training scheduler status.

**Authentication:** Viewer role required

**Response:**
```json
{
  "enabled": true,
  "next_run_local": "2026-02-09T02:00:00",
  "hour_local": 2,
  "min_new_videos": 10
}
```

---

## Audit Trail

### GET `/audit/trail`

Retrieve audit trail entries.

**Authentication:** Viewer role required

**Parameters:**
- `limit` (optional): Max entries (default: 100, max: 1000)

**Response:**
```json
{
  "items": [
    {
      "job_id": "score_20260208_120000_abc123",
      "job_type": "score",
      "requested_by": "token:operator",
      "task_id": "assembly_line_1",
      "subject": "1→42",
      "status": "completed",
      "model_name": null,
      "score": 87.5,
      "error_message": null,
      "queued_at": "2026-02-08T12:00:00Z",
      "started_at": "2026-02-08T12:00:01Z",
      "finished_at": "2026-02-08T12:00:15Z",
      "created_at": "2026-02-08T12:00:00Z"
    }
  ]
}
```

---

### POST `/audit/export`

Export signed audit trail for compliance.

**Authentication:** Admin role required

**Parameters:**
- `limit` (optional): Max entries (default: 500, max: 5000)

**Response:**
```json
{
  "export_id": "20260208_120000",
  "generated_at": "2026-02-08T12:00:00Z",
  "item_count": 150,
  "file_path": "/data/reports/audit_export_20260208_120000.json",
  "signature": {
    "algorithm": "hmac-sha256",
    "key_id": "prod",
    "payload_sha256": "abc123...",
    "signature_hex": "def456..."
  }
}
```

---

### GET `/audit/export/{export_id}/file`

Download signed audit export file.

**Authentication:** Admin role required

**Response:** JSON file with HMAC signature

---

## Operations

### GET `/ops/queue`

Get queue metrics and job statistics.

**Authentication:** Operator role required

**Response:**
```json
{
  "generated_at": "2026-02-08T12:00:00Z",
  "runtime_mode": "api",
  "queue": {
    "backend": "rq",
    "redis_ok": true,
    "error": null,
    "queues": [
      {
        "key": "sopilot_ingest",
        "name": "ingest",
        "queued": 5,
        "started": 2,
        "failed": 0,
        "finished": 123,
        "deferred": 0,
        "scheduled": 0
      },
      {
        "key": "sopilot_score",
        "name": "score",
        "queued": 10,
        "started": 4,
        "failed": 1,
        "finished": 456,
        "deferred": 0,
        "scheduled": 0
      }
    ]
  },
  "jobs": {
    "ingest": {
      "queued": 5,
      "running": 2,
      "completed": 123,
      "failed": 0
    },
    "score": {
      "queued": 10,
      "running": 4,
      "completed": 456,
      "failed": 1
    }
  }
}
```

---

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common Status Codes:**
- `400 Bad Request`: Invalid input parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions for operation
- `404 Not Found`: Resource not found
- `413 Payload Too Large`: Upload exceeds size limit
- `500 Internal Server Error`: Unexpected server error
- `503 Service Unavailable`: Service temporarily unavailable

---

## Authentication

SOPilot supports multiple authentication methods:

### Bearer Token
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/videos
```

### Basic Auth
```bash
curl -u username:password http://localhost:8000/videos
```

### Role-Based Tokens
Configure via `SOPILOT_API_ROLE_TOKENS`:
```bash
export SOPILOT_API_ROLE_TOKENS="admin:secret123,operator:secret456,viewer:secret789"
```

### Role Hierarchy
- **Admin**: Full access (all operations)
- **Operator**: Can upload videos, run scores
- **Viewer**: Read-only access

---

## Rate Limiting

Currently no built-in rate limiting. Recommend using nginx/Caddy reverse proxy with rate limiting rules.

---

## Versioning

API version: `v0.1.0`

Breaking changes will increment major version. Monitor `/health` endpoint for service status.
