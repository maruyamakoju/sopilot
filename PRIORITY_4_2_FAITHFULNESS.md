# Priority 4.2: RAG Faithfulness - Clip-Only QA

**Date**: 2026-02-12
**Status**: ✅ **COMPLETE** - Video-LLM constrained to retrieved clips only

## Critical Problem Solved

**Before**: Video-LLM could see the entire video, breaking RAG faithfulness
**After**: Video-LLM sees ONLY the retrieved clip [start_sec, end_sec]

### Why This Matters

RAG (Retrieval-Augmented Generation) fundamentally requires:
1. **Retrieval**: Find relevant evidence clips
2. **Grounded Generation**: Answer based ONLY on retrieved evidence

Without clip constraint, the model could:
- See the full video and answer from anywhere
- Ignore retrieval results completely
- Break the entire RAG pipeline integrity

## Implementation

### Key Changes

**1. Version Pinning (`pyproject.toml`)**

```toml
vigil = [
    "transformers>=4.51.3",        # Qwen2.5-VL requires 4.51+ for proper model doc
    "qwen-vl-utils[decord]>=0.0.10",  # Video kwargs stability (return_video_kwargs)
    ...
]
```

**Why**:
- `transformers>=4.45.0` → ImportError / KeyError with Qwen2.5-VL
- `qwen-vl-utils>=0.0.8` → video kwargs issues in earlier versions
- Fixed versions prevent "model doesn't load" failures

**2. Clip-Only Inference (`video_llm_service.py`)**

**Old approach (BROKEN)**:
```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": f"file:///{video_path.absolute().as_posix()}",  # Full video!
                ...
            },
            {"type": "text", "text": question},
        ],
    }
]
```

**New approach (FIXED)**:
```python
# Extract frames ONLY from [start_sec, end_sec]
frame_indices = uniform_sample(start_frame, end_frame, target_frames)

# Save frames to temp directory
for idx, frame_idx in enumerate(frame_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    frame_path = temp_dir / f"frame_{idx:04d}.jpg"
    save_frame(frame_path)
    frame_paths.append(f"file:///{frame_path.absolute().as_posix()}")

# Pass frame list instead of full video
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": frame_paths,  # Clip frames only!
                ...
            },
            {"type": "text", "text": question},
        ],
    }
]
```

### Frame Sampling Strategy

**Constraint**: Cap at `max_frames` (default: 32) regardless of clip duration

```python
duration_sec = (end_frame - start_frame) / fps
target_frames = min(
    max_frames,  # Hard cap
    max(8, int(duration_sec * fps_target)),  # At least 8 frames
)
```

**Benefits**:
- 10-second clip: ~10 frames (1 FPS)
- 1-minute clip: 32 frames (capped)
- 1-hour clip: 32 frames (capped)

**Result**: Constant memory footprint, scales to any clip length

### Cleanup

**Problem**: Temporary frames accumulate if not cleaned up

**Solution**: Always cleanup in `finally` block
```python
finally:
    import shutil
    if temp_dir and Path(temp_dir).exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
```

## Validation

### Faithfulness Test Script

**Purpose**: Verify model answers based on clip content, not full video

**Test setup**:
```python
# Video structure:
# 0-2 sec: Black (intro)
# 2-4 sec: Red scene
# 4-6 sec: Green scene
# 6-8 sec: Blue scene
# 8-10 sec: Yellow scene

# Test 1: Ask "What color?" for Red clip [2-4 sec]
# Test 2: Ask "What color?" for Blue clip [6-8 sec]
```

**Expected (faithfulness working)**:
- Red clip → Answer mentions "red"
- Blue clip → Answer mentions "blue"
- Answers are different

**Failure mode (faithfulness broken)**:
- Both clips → Same answer (model sees full video)
- Red clip → Answer mentions "blue" (cross-contamination)

### Running the Test

```bash
# Generate test video + run validation
python scripts/validate_faithfulness.py --device cuda --llm-model qwen2.5-vl-7b

# Expected output:
# ✅ Answers are different: True
# ✅ Red clip mentions 'red': True
# ✅ Blue clip mentions 'blue': True
# 🎉 FAITHFULNESS TEST PASSED
```

## Done Condition (Priority 4.2)

✅ **Version pinning**: transformers>=4.51.3, qwen-vl-utils>=0.0.10
✅ **Clip extraction**: Frames extracted only from [start_sec, end_sec]
✅ **Frame list mode**: Messages use frame paths, not full video path
✅ **Cleanup**: Temporary frames cleaned up after inference
✅ **Validation script**: `validate_faithfulness.py` verifies clip-only behavior
✅ **E2E logging**: Clear indication of mock vs. real model loading

## Impact

### Before (Broken RAG)

```
User: "What happens in the video?"
Retrieval: Top-1 clip [5:00-5:30] about "safety violation"
Video-LLM: Sees full video (0:00-10:00)
Answer: "The video shows various activities including lunch break, machine operation, and safety training"
                                                          ^^^^^^^^^^^^^^^^^^^
                                                  (from other parts of video!)
```

### After (Faithful RAG)

```
User: "What happens in the video?"
Retrieval: Top-1 clip [5:00-5:30] about "safety violation"
Video-LLM: Sees ONLY clip [5:00-5:30]
Answer: "A worker is operating the machine without wearing safety goggles, which is a violation of protocol"
                                                                              ^^^^^^^^^^^^^^^^^^^^
                                                                        (from retrieved clip only!)
```

## Architecture

### RAG Pipeline Flow

```
1. Query → OpenCLIP → query_embedding
2. Qdrant/FAISS → top_k clips with [start_sec, end_sec]
3. For each clip:
   a. Extract frames [start_sec, end_sec] ONLY
   b. Save to temp_dir/frame_XXXX.jpg
   c. Pass frame_paths to Qwen2.5-VL
   d. Get answer based on clip frames ONLY
4. Return answer + evidence (clip timestamps)
```

### Key Invariant

**CRITICAL**: At no point does the Video-LLM see frames outside [start_sec, end_sec]

This ensures:
- Retrieval controls what the model sees
- Answers are grounded in evidence
- RAG faithfulness is maintained

## Limitations & Trade-offs

### 1. Temporary Disk I/O

**Cost**: Frames written to disk for each clip
**Mitigation**: Use tmpfs (RAM disk) if available
**Alternative**: In-memory frame passing (requires Qwen API support)

### 2. Frame Sampling Granularity

**Issue**: 32 frames may miss fine-grained events in long clips
**Mitigation**: Retrieval uses smaller chunks (micro: 2-4s) for granularity
**Future**: Adaptive frame sampling based on clip complexity

### 3. Multi-Clip Context

**Current**: Top-1 clip only
**Future** (Priority 4.3): Aggregate top-K clips for richer context

## Next Steps

**Priority 4.3**: Top-K Composite Answer
- Don't rely on single clip (top-1)
- Extract summaries from top-5 clips
- Synthesize final answer from multiple pieces of evidence

**Priority 5**: 2-Stage Event Detection
- Stage 1: Retrieval (fast, high recall)
- Stage 2: LLM verification (slow, high precision)
- Faithfulness guarantees false positives are grounded

## Files Changed

```
pyproject.toml                      # Version pinning: transformers>=4.51.3
src/sopilot/video_llm_service.py   # Clip-only inference (+50 lines)
scripts/vigil_smoke_e2e.py          # Model loading verification (+10 lines)
scripts/validate_faithfulness.py   # Faithfulness validation test (new, 250 lines)
PRIORITY_4_2_FAITHFULNESS.md       # This document
```

## Technical Details

### Qwen2.5-VL Video Input Modes

**Mode 1: Full video path** (AVOID for RAG)
```python
{"type": "video", "video": "file:///path/to/video.mp4"}
```
→ Model sees entire video

**Mode 2: Frame list** (USE for RAG)
```python
{"type": "video", "video": ["file:///frame_0.jpg", "file:///frame_1.jpg", ...]}
```
→ Model sees only specified frames

### Frame Extraction Implementation

```python
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)

start_frame = int(start_sec * fps)
end_frame = int(end_sec * fps)

# Uniform sampling
num_frames = end_frame - start_frame
step = num_frames / target_frames
frame_indices = [int(start_frame + i * step) for i in range(target_frames)]

# Extract
for idx, frame_idx in enumerate(frame_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    # Save to temp_dir/frame_{idx:04d}.jpg
```

## References

- **Original Issue**: "retrieval で見つけた [start_sec, end_sec] が QA 側に効いてない"
- **Solution**: Frame list mode + temporal extraction
- **Validation**: Multicolor video test (different clips → different answers)

---

**🎯 RAG Faithfulness Established**: Video-LLM answers are now guaranteed to be grounded in retrieved evidence, not the full video.
