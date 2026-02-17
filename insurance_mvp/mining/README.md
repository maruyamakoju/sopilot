# Multimodal Hazard Mining Pipeline

Production-ready system for detecting dangerous events in dashcam footage using multimodal signal analysis.

## Overview

The mining pipeline combines three complementary signals to identify hazardous events:

1. **Audio Analysis** - Detects brake sounds, horns, crashes via:
   - RMS (root mean square) volume level
   - Delta-RMS for sudden volume changes
   - Horn-band FFT (300-1000Hz frequency analysis)

2. **Motion Analysis** - Detects sudden/irregular movements via:
   - Optical flow (cv2.calcOpticalFlowFarneback)
   - Flow magnitude (sudden movements like hard braking)
   - Flow variance (chaotic multi-vehicle scenarios)

3. **Proximity Analysis** - Detects nearby hazards via:
   - YOLOv8n object detection (person, bicycle, car, bus, truck)
   - Proximity score (large bbox = close = dangerous)
   - Center distance (center of frame = front = dangerous)

4. **Signal Fusion** - Combines signals and extracts clips:
   - Weighted fusion (audio=0.3, motion=0.4, proximity=0.3)
   - Peak detection (scipy.signal.find_peaks)
   - Nearby peak merging (3-second gap)
   - Clip extraction (±5 seconds around peaks)

## Architecture

```
insurance_mvp/mining/
├── audio.py          # Audio signal analysis
├── motion.py         # Optical flow analysis
├── proximity.py      # YOLOv8 object detection
├── fuse.py           # Signal fusion and peak detection
├── __init__.py       # Module exports
└── README.md         # This file
```

## Installation

```bash
# Core dependencies (required)
pip install opencv-python numpy scipy

# YOLOv8 for proximity analysis (optional, recommended)
pip install ultralytics

# Visualization (optional)
pip install matplotlib
```

## Quick Start

### Basic Usage

```python
from insurance_mvp.mining import (
    AudioAnalyzer,
    MotionAnalyzer,
    ProximityAnalyzer,
    SignalFuser,
)

# Initialize analyzers
audio_analyzer = AudioAnalyzer()
motion_analyzer = MotionAnalyzer()
proximity_analyzer = ProximityAnalyzer()
fuser = SignalFuser()

# Analyze video
video_path = "dashcam001.mp4"

audio_scores = audio_analyzer.analyze(video_path)
motion_scores = motion_analyzer.analyze(video_path)
proximity_scores = proximity_analyzer.analyze(video_path)

# Extract hazard clips
clips = fuser.fuse_and_extract(
    audio_scores,
    motion_scores,
    proximity_scores,
    video_duration_sec=60.0
)

# Process clips
for clip in clips:
    print(f"Hazard at {clip.peak_sec:.1f}s: score={clip.score:.3f}")
    print(f"  Clip: [{clip.start_sec:.1f}s, {clip.end_sec:.1f}s]")
```

### Demo Script

```bash
# Full pipeline with visualization
python -m insurance_mvp.scripts.demo_mining_pipeline \
    --video-path data/dashcam001.mp4 \
    --output-dir results/ \
    --visualize \
    --extract-clips

# Fast mode (skip proximity analysis)
python -m insurance_mvp.scripts.demo_mining_pipeline \
    --video-path data/dashcam001.mp4 \
    --skip-proximity \
    --frame-skip 10
```

## Configuration

### Audio Analysis

```python
from insurance_mvp.mining import AudioAnalyzer, AudioConfig

config = AudioConfig(
    rms_window_sec=1.0,           # RMS window size
    rms_percentile=90.0,          # Normalization percentile
    rms_delta_threshold=2.0,      # Delta-RMS threshold
    horn_freq_min=300,            # Horn frequency band (Hz)
    horn_freq_max=1000,
    horn_power_threshold=0.7,     # Horn detection threshold
    sample_rate=16000,            # Audio sample rate
)

analyzer = AudioAnalyzer(config)
```

### Motion Analysis

```python
from insurance_mvp.mining import MotionAnalyzer, MotionConfig

config = MotionConfig(
    pyr_scale=0.5,                # Optical flow pyramid scale
    levels=3,                     # Pyramid levels
    winsize=15,                   # Window size
    iterations=3,                 # Iterations per level
    poly_n=5,                     # Polynomial neighborhood
    poly_sigma=1.2,               # Gaussian sigma
    frame_skip=5,                 # Process every N frames
    magnitude_percentile=95.0,    # Normalization percentile
    variance_percentile=95.0,
)

analyzer = MotionAnalyzer(config)
```

### Proximity Analysis

```python
from insurance_mvp.mining import ProximityAnalyzer, ProximityConfig

config = ProximityConfig(
    model_name="yolov8n.pt",      # YOLOv8 model (n/s/m/l/x)
    confidence_threshold=0.25,    # Detection confidence
    device="auto",                # "auto", "cuda", "cpu"
    target_classes=[0, 1, 2, 3, 5, 7],  # COCO classes to detect
    class_weights={               # Object type weights
        0: 1.5,  # person (high priority)
        1: 1.3,  # bicycle
        2: 1.0,  # car
        3: 1.2,  # motorcycle
        5: 1.0,  # bus
        7: 1.0,  # truck
    },
    frame_skip=5,                 # Process every N frames
    proximity_percentile=95.0,    # Normalization percentile
)

analyzer = ProximityAnalyzer(config)
```

### Signal Fusion

```python
from insurance_mvp.mining import SignalFuser, FusionConfig

config = FusionConfig(
    audio_weight=0.3,             # Audio signal weight
    motion_weight=0.4,            # Motion signal weight
    proximity_weight=0.3,         # Proximity signal weight
    top_k_peaks=20,               # Number of top peaks
    min_peak_score=0.3,           # Minimum peak score
    min_peak_distance=3,          # Min distance between peaks (sec)
    clip_padding_sec=5.0,         # Clip padding around peaks
    min_clip_duration=2.0,        # Minimum clip duration
    max_clip_duration=15.0,       # Maximum clip duration
    merge_gap_sec=3.0,            # Merge peaks within N seconds
)

fuser = SignalFuser(config)
```

## Performance Optimization

### Frame Skipping

Processing every frame is slow. Use `frame_skip` to sample frames:

```python
# Fast: Process every 10 frames (30fps → 3 samples/sec)
motion_config = MotionConfig(frame_skip=10)

# Balanced: Process every 5 frames (30fps → 6 samples/sec)
motion_config = MotionConfig(frame_skip=5)

# Slow: Process every frame (30fps → 30 samples/sec)
motion_config = MotionConfig(frame_skip=1)
```

**Recommendation**: `frame_skip=5` for 30fps video (6 samples/sec) balances speed and accuracy.

### GPU Acceleration

YOLOv8 automatically uses GPU if available:

```python
# Auto-detect GPU
config = ProximityConfig(device="auto")

# Force CPU (slower)
config = ProximityConfig(device="cpu")

# Force GPU
config = ProximityConfig(device="cuda")
```

### Skip Proximity for Speed

Proximity analysis (YOLOv8) is the slowest component. Skip it for faster processing:

```python
# Skip proximity, use audio + motion only
proximity_scores = np.zeros_like(motion_scores)

clips = fuser.fuse_and_extract(
    audio_scores,
    motion_scores,
    proximity_scores,  # All zeros
    video_duration_sec=60.0
)
```

## Output Format

### HazardClip Object

```python
@dataclass
class HazardClip:
    start_sec: float       # Clip start time (seconds)
    end_sec: float         # Clip end time (seconds)
    peak_sec: float        # Peak timestamp within clip
    score: float           # Danger score [0.0, 1.0]

    # Signal breakdown
    audio_score: float
    motion_score: float
    proximity_score: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec
```

### JSON Export

```python
import json

clips_data = [
    {
        "clip_id": i + 1,
        "start_sec": clip.start_sec,
        "end_sec": clip.end_sec,
        "peak_sec": clip.peak_sec,
        "score": clip.score,
        "breakdown": {
            "audio": clip.audio_score,
            "motion": clip.motion_score,
            "proximity": clip.proximity_score,
        }
    }
    for i, clip in enumerate(clips)
]

with open("hazard_clips.json", "w") as f:
    json.dump(clips_data, f, indent=2)
```

## Edge Cases

The pipeline handles common edge cases gracefully:

### Missing Audio Track

```python
# Returns zero audio scores, no crash
audio_scores = audio_analyzer.analyze("silent_video.mp4")
# Output: np.zeros(n_seconds)
```

### Very Short Video

```python
# Returns empty list if no valid clips found
clips = fuser.fuse_and_extract(audio, motion, proximity, video_duration_sec=1.0)
# Output: []
```

### No Peaks Found

```python
# Returns empty list if all scores below threshold
clips = fuser.fuse_and_extract(
    low_audio, low_motion, low_proximity,
    video_duration_sec=60.0
)
# Output: []
```

### Corrupt Video File

```python
# Raises RuntimeError with clear message
try:
    audio_scores = audio_analyzer.analyze("corrupt.mp4")
except RuntimeError as e:
    print(f"Error: {e}")
```

## Visualization

Generate signal plots for debugging:

```python
# Generate visualization
fuser.visualize_signals(
    audio_scores,
    motion_scores,
    proximity_scores,
    clips,
    output_path="signal_viz.png"
)
```

Output shows:
- Individual signals (audio, motion, proximity)
- Fused signal with peak markers
- Extracted clip ranges with scores

## Testing

```bash
# Run unit tests
pytest tests/test_mining_pipeline.py -v

# Run integration tests (slow, requires YOLOv8)
pytest tests/test_mining_pipeline.py -v --slow

# Test with coverage
pytest tests/test_mining_pipeline.py --cov=insurance_mvp.mining
```

## Performance Benchmarks

Typical performance on RTX 5090 (1080p 30fps video):

| Component | Processing Speed | Notes |
|-----------|-----------------|-------|
| Audio     | 100-200x realtime | Fast (CPU only) |
| Motion    | 5-10x realtime  | Medium (frame_skip=5) |
| Proximity | 2-5x realtime   | Slow (YOLOv8, GPU) |
| Fusion    | >1000x realtime | Fast (numpy) |

**Total**: ~3-8x realtime for full pipeline

Example: 60-second video processes in 8-20 seconds.

## Troubleshooting

### ffmpeg not found

Audio extraction requires ffmpeg:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### YOLOv8 model download fails

First run downloads ~6MB model:

```python
# Pre-download model
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Downloads on first run
```

### Out of memory (GPU)

Reduce batch size or use CPU:

```python
# Use CPU for proximity
config = ProximityConfig(device="cpu")

# Or increase frame_skip
config = ProximityConfig(frame_skip=10)  # Process fewer frames
```

### Slow processing

```python
# Speed optimization checklist:
# 1. Increase frame skip
config = MotionConfig(frame_skip=10)

# 2. Skip proximity analysis
proximity_scores = np.zeros_like(motion_scores)

# 3. Use smaller YOLOv8 model (already using yolov8n - smallest)

# 4. Reduce top-K peaks
config = FusionConfig(top_k_peaks=10)
```

## Advanced Usage

### Custom Signal Weights

Adjust weights based on your use case:

```python
# Audio-heavy scenario (urban driving with horn sounds)
config = FusionConfig(
    audio_weight=0.5,
    motion_weight=0.3,
    proximity_weight=0.2,
)

# Motion-heavy scenario (highway hard braking)
config = FusionConfig(
    audio_weight=0.2,
    motion_weight=0.6,
    proximity_weight=0.2,
)

# Proximity-heavy scenario (pedestrian detection)
config = FusionConfig(
    audio_weight=0.2,
    motion_weight=0.2,
    proximity_weight=0.6,
)
```

### Extract Specific Clips

```python
import cv2

def extract_clip(video_path, start_sec, end_sec, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()

# Extract top clip
if clips:
    top_clip = clips[0]
    extract_clip(
        "dashcam.mp4",
        top_clip.start_sec,
        top_clip.end_sec,
        "top_hazard.mp4"
    )
```

## License

Proprietary - Internal use only

## Support

For questions or issues, contact the SOPilot development team.
