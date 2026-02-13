# VIGIL-RAG Benchmark Data

## Format (JSONL)

Each line is a JSON object with these fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query_id` | string | yes | Unique identifier (e.g., "v01", "a03", "m02") |
| `video_id` | string | yes | Video identifier (SHA-256 or synthetic ID) |
| `query_text` | string | yes | Natural language query |
| `query_type` | string | yes | `"visual"` / `"audio"` / `"mixed"` |
| `relevant_clip_ids` | list[str] | yes | Ground-truth clip IDs |
| `relevant_time_ranges` | list[obj] | yes | `[{"start_sec": float, "end_sec": float}, ...]` |
| `event_detection` | bool | no | If true, this is an event detection query |
| `event_type` | string | no | Event type label (for event detection queries) |
| `notes` | string | no | Human-readable description |

## Files

- **`vigil_benchmark_v1.jsonl`** — Full benchmark (20 queries: 5 visual, 5 audio, 5 mixed, 5 event detection)
- **`smoke_benchmark.jsonl`** — CI smoke benchmark (6 queries: 2 visual, 2 audio, 2 mixed)

## Query Type Convention

- `visual`: Answer is determinable from video frames alone
- `audio`: Answer requires speech/sound (hybrid search should outperform visual-only)
- `mixed`: Both visual and audio cues are needed for best retrieval

## Usage

```bash
# Full evaluation with comparison
python scripts/evaluate_vigil_benchmark.py --benchmark benchmarks/vigil_benchmark_v1.jsonl

# Quick smoke test
python scripts/evaluate_vigil_benchmark.py --benchmark benchmarks/smoke_benchmark.jsonl

# Alpha sweep
python scripts/evaluate_vigil_benchmark.py --benchmark benchmarks/vigil_benchmark_v1.jsonl --alpha-sweep 0.0,0.3,0.5,0.7,1.0
```
