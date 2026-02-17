# Mining Pipeline Implementation Notes

## Design Decisions

### 1. Frame Skipping Strategy

**Decision**: Default `frame_skip=5` for motion and proximity analysis.

**Rationale**:
- 30fps video → 6 samples/second is sufficient for hazard detection
- 5x speedup with minimal accuracy loss
- Most hazards last > 1 second, so 6 samples/sec captures them
- Audio runs at 1 sample/sec (no frame skipping)

**Trade-offs**:
- Higher frame_skip = faster but may miss brief events
- Lower frame_skip = slower but more complete coverage
- Configurable per use case

### 2. Signal Normalization

**Decision**: Percentile-based normalization (95th percentile).

**Rationale**:
- Robust to outliers (vs max normalization)
- Prevents single extreme value from saturating entire signal
- 95th percentile balances sensitivity and stability

**Implementation**:
```python
percentile_val = np.percentile(values, 95.0)
normalized = values / percentile_val
normalized = np.clip(normalized, 0.0, 1.0)
```

### 3. Audio Extraction

**Decision**: Use subprocess call to ffmpeg.

**Rationale**:
- OpenCV's cv2.VideoCapture doesn't expose audio API
- ffmpeg is industry standard for audio/video processing
- Graceful fallback to silent audio if ffmpeg unavailable

**Alternative considered**: moviepy, pydub
- Rejected: Additional heavy dependencies

### 4. Optical Flow Algorithm

**Decision**: Farneback dense optical flow.

**Rationale**:
- Dense flow (every pixel) vs sparse flow (feature points)
- Good balance of speed and accuracy
- Built into OpenCV (no extra dependencies)
- Well-suited for dashcam footage (structured scenes)

**Alternative considered**: Lucas-Kanade sparse flow
- Rejected: Misses motion in featureless regions (sky, road)

### 5. Object Detection Model

**Decision**: YOLOv8n (nano).

**Rationale**:
- Smallest/fastest YOLOv8 variant (~6MB)
- Real-time capable on GPU (>100 FPS)
- Adequate accuracy for insurance use case
- Supports 80 COCO classes (includes vehicles, pedestrians)

**Alternatives**:
- YOLOv8s/m/l/x: Higher accuracy but slower
- Faster-RCNN: Higher accuracy but much slower
- MobileNet-SSD: Faster but lower accuracy

### 6. Signal Fusion Strategy

**Decision**: Weighted sum with fixed weights.

**Rationale**:
- Simple, interpretable, deterministic
- Works well across diverse scenarios
- No training required
- Weights: audio=0.3, motion=0.4, proximity=0.3

**Weights rationale**:
- Motion highest (0.4): Most reliable signal across scenarios
- Audio medium (0.3): Important but not always present
- Proximity medium (0.3): Important but may have false positives

**Alternatives considered**:
- Learned weights: Rejected (requires training data)
- Max fusion: Rejected (too sensitive to noise)
- Product fusion: Rejected (requires all signals, kills missing)

### 7. Peak Detection

**Decision**: scipy.signal.find_peaks with distance constraint.

**Rationale**:
- Proven algorithm for time series peak detection
- `distance` parameter prevents duplicate peaks
- `height` parameter filters noise

**Parameters**:
- `height=0.3`: Minimum score threshold (configurable)
- `distance=3`: Minimum 3 seconds between peaks (avoids duplicates)

### 8. Clip Merging

**Decision**: Merge clips within 3-second gap.

**Rationale**:
- Multi-phase events (e.g., brake → crash → aftermath)
- Avoid fragmenting single incident into multiple clips
- 3 seconds chosen empirically (typical reaction time)

**Alternative**: No merging
- Rejected: Creates too many overlapping clips

### 9. Error Handling Philosophy

**Decision**: Fail gracefully with logging, don't crash.

**Examples**:
- Missing audio → return zeros, log warning
- No peaks found → return empty list
- Detection failure → skip frame, continue

**Rationale**:
- Production system must handle diverse inputs
- Partial results better than complete failure
- Log warnings for debugging

## Performance Characteristics

### Bottlenecks

1. **Proximity Analysis (YOLOv8)**: Slowest component
   - Solution: frame_skip, GPU acceleration, or skip entirely

2. **Optical Flow**: Second slowest
   - Solution: frame_skip, reduce pyramid levels

3. **Audio Extraction (ffmpeg)**: I/O bound
   - Minor bottleneck, hard to optimize

### Memory Usage

- Audio samples: ~60 MB for 60-second video at 16kHz mono
- Optical flow: ~2.5 MB per frame (640x480x2 float32)
- YOLOv8 model: ~12 MB GPU memory
- Total: <200 MB for typical video

## Edge Case Handling

### Missing Audio Track

```python
# Detected via ffmpeg failure or subprocess timeout
logger.warning("Failed to extract audio, returning silent audio")
return np.zeros(n_samples), sample_rate, duration_sec
```

### Very Short Video (<2 seconds)

```python
# Clips rejected by min_clip_duration check
if end_sec - start_sec < self.config.min_clip_duration:
    logger.debug(f"Skipping short clip at {peak_sec:.1f}s")
    continue
```

### No Objects Detected

```python
# Returns zero proximity scores
if result.boxes is None or len(result.boxes) == 0:
    return 0.0
```

### Corrupted Frames

```python
# Skip frame, log warning, continue
try:
    results = self.model.predict(frame, ...)
except Exception as e:
    logger.warning(f"Detection failed at frame {frame_idx}: {e}")
    proximity_per_frame.append(0.0)
```

## Testing Strategy

### Unit Tests

- Config validation (weights sum to 1.0, valid ranges)
- Normalization functions (percentile, clipping)
- Aggregation functions (frame→second conversion)

### Integration Tests

- Synthetic video generation (known motion patterns)
- End-to-end pipeline (all components together)
- Edge cases (no audio, short video, no peaks)

### Smoke Tests

- Real dashcam footage (manual verification)
- Performance benchmarks (processing speed)

## Future Enhancements

### Potential Improvements

1. **Learned Fusion Weights**
   - Train on labeled data to optimize weights per scenario
   - Risk: Requires training dataset

2. **Temporal Consistency**
   - Smooth signals with temporal filtering (moving average)
   - Risk: May blur sharp peaks

3. **Multi-Scale Peak Detection**
   - Detect peaks at multiple time scales (seconds, minutes)
   - Risk: Increased complexity

4. **Adaptive Thresholds**
   - Per-video threshold based on signal statistics
   - Risk: May miss events in low-activity videos

5. **Audio Event Classification**
   - Classify horn vs brake vs crash (not just detect)
   - Risk: Requires trained audio classifier

6. **Optical Flow Tracking**
   - Track specific objects across frames
   - Risk: Slower, more complex

### Known Limitations

1. **No Audio Classification**: Detects "loud event" but not "what kind"
2. **No Object Tracking**: Detects "car present" but not "car trajectory"
3. **Fixed Fusion Weights**: Not adaptive to scenario type
4. **No Temporal Context**: Each second scored independently
5. **No Video Quality Assessment**: Assumes good quality input

## Dependencies

### Required

- opencv-python (opencv-headless also works)
- numpy
- scipy

### Optional

- ultralytics (YOLOv8, for proximity analysis)
- matplotlib (for visualization)
- ffmpeg (binary, for audio extraction)

### Versions

Tested with:
- Python 3.10+
- opencv-python 4.8.0+
- numpy 1.24.0+
- scipy 1.11.0+
- ultralytics 8.0.0+

## Code Quality

### Compliance

- Type hints on all functions
- Docstrings (Google style)
- Logging (Python logging module)
- Error handling (try/except with meaningful messages)
- No silent failures (log warnings)

### Patterns

- Dataclasses for configuration (immutable, validated)
- Factory pattern for analyzers (config → analyzer)
- Lazy loading for heavy models (YOLOv8)
- Percentile normalization (robust to outliers)
- Max pooling for aggregation (keep strongest signal)

## Maintenance

### Adding New Signal

1. Create new analyzer module (e.g., `gps.py`)
2. Implement `analyze(video_path) -> np.ndarray` method
3. Return per-second scores [0.0, 1.0]
4. Add weight to FusionConfig
5. Update fuse_and_extract() to include new signal

### Tuning Parameters

Priority order for tuning:
1. `frame_skip`: Speed vs accuracy trade-off
2. `min_peak_score`: Sensitivity vs noise
3. Signal weights: Balance modalities
4. `merge_gap_sec`: Clip granularity
5. `clip_padding_sec`: Clip duration

## License

Proprietary - SOPilot Inc.
