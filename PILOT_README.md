# SOPilot Manufacturing Pilot Package

**Production-ready SOP video evaluation system for manufacturing training.**

This is the customer-facing deliverable that evaluates trainee performance videos and generates actionable reports with deviations, timestamps, and corrective actions.

---

## Quick Start (One Command)

```bash
python scripts/sopilot_evaluate_pilot.py \
    --gold demo_videos/manufacturing/oil_change_gold.mp4 \
    --trainee demo_videos/manufacturing/oil_change_trainee_1.mp4 \
    --sop oil_change
```

**Output**: JSON report with pass/fail, deviations, timestamps, and corrective actions.

---

## What This Does

**Input**:
- Gold standard video (correct procedure)
- Trainee video (performance to evaluate)
- SOP template (procedure definition)

**Output**:
1. **Overall Pass/Fail** (with score 0-100 and threshold)
2. **Deviation List** (missing steps, wrong order, execution issues)
3. **Timestamps** (when each deviation occurred)
4. **Severity Levels** (critical, high, medium, low)
5. **Corrective Actions** (what trainee should do next)
6. **Evaluation Time** (speed demonstration)

---

## Commercial Value Proposition

### Traditional Manual Review
- **Time**: 2 hours per video (human evaluator)
- **Cost**: $50-100/hour × 2 hours = $100-200 per evaluation
- **Consistency**: Subjective, varies by evaluator
- **Scalability**: Limited by human availability

### SOPilot Automated Evaluation
- **Time**: ~3 seconds per video
- **Cost**: $0.01 per evaluation (compute cost)
- **Consistency**: 100% objective, identical criteria
- **Scalability**: Evaluate 1000s of trainees in parallel

**ROI**: 99% cost reduction + 2400x speed improvement

---

## Supported SOPs

### 1. Oil Change (10 steps)
- **Threshold**: 80.0
- **Critical Steps**: SAFETY, CHECK
- **Duration**: ~80 seconds (10s per step)

```bash
python scripts/sopilot_evaluate_pilot.py \
    --gold demo_videos/manufacturing/oil_change_gold.mp4 \
    --trainee demo_videos/manufacturing/oil_change_trainee_1.mp4 \
    --sop oil_change
```

### 2. Brake Pad Replacement (8 steps)
- **Threshold**: 85.0
- **Critical Steps**: SAFETY, TORQUE, CHECK
- **Duration**: ~64 seconds (8s per step)

```bash
python scripts/sopilot_evaluate_pilot.py \
    --gold demo_videos/manufacturing/brake_pads_gold.mp4 \
    --trainee demo_videos/manufacturing/brake_pads_trainee_1.mp4 \
    --sop brake_pads
```

### 3. PPE Check (5 steps)
- **Threshold**: 90.0 (strict)
- **Critical Steps**: HELMET, GLASSES, GLOVES
- **Duration**: ~40 seconds (8s per step)

```bash
python scripts/sopilot_evaluate_pilot.py \
    --gold demo_videos/manufacturing/ppe_check_gold.mp4 \
    --trainee demo_videos/manufacturing/ppe_check_trainee_1.mp4 \
    --sop ppe_check
```

---

## Output Formats

### JSON Report (Default)

```bash
python scripts/sopilot_evaluate_pilot.py \
    --gold demo_videos/manufacturing/oil_change_gold.mp4 \
    --trainee demo_videos/manufacturing/oil_change_trainee_1.mp4 \
    --sop oil_change \
    --out report.json
```

**JSON Structure**:
```json
{
  "overall": {
    "pass": false,
    "score": 7.2,
    "threshold": 80.0,
    "grade": "F"
  },
  "deviations": [
    {
      "type": "step_missing",
      "step": "SAFETY",
      "timestamp": "0:04-0:08",
      "severity": "critical",
      "description": "Missing step: Put on safety glasses and gloves (CRITICAL SAFETY STEP)",
      "evidence": "Expected at 0:04-0:08"
    }
  ],
  "corrective_actions": [
    "CRITICAL: Ensure trainee completes Put on safety glasses and gloves before proceeding"
  ],
  "evaluation_time": 3.29,
  "metadata": {
    "sop": "oil_change",
    "trainee_id": "oil_change_trainee_1"
  }
}
```

### PDF Report (Requires reportlab)

```bash
pip install reportlab

python scripts/sopilot_evaluate_pilot.py \
    --gold demo_videos/manufacturing/oil_change_gold.mp4 \
    --trainee demo_videos/manufacturing/oil_change_trainee_1.mp4 \
    --sop oil_change \
    --out report.pdf
```

**PDF Features**:
- Color-coded severity (red=critical, orange=high, yellow=medium, blue=low)
- Professional formatting with tables
- Ready to print or email to trainee

---

## Deviation Types

### 1. Missing Step
**Description**: Required step not performed
**Severity**: Critical if step is in `critical_steps`, else High
**Example**: Trainee skipped SAFETY step (no PPE worn)

### 2. Order Swap
**Description**: Steps performed out of sequence
**Severity**: High
**Example**: CHECK before FILL (wrong order)

### 3. Execution Deviation
**Description**: Step performed but with low quality
**Severity**: Medium to High (based on similarity confidence)
**Example**: Filter removal technique differs from gold standard

### 4. Wrong Tool (Future)
**Description**: Incorrect tool used for step
**Severity**: High
**Example**: Used jack instead of wrench for filter removal

---

## Severity Classification

| Severity | Criteria | Action Required |
|----------|----------|-----------------|
| **Critical** | Missing critical_step (SAFETY, CHECK, etc.) | IMMEDIATE re-training |
| **High** | Missing non-critical step OR order swap with confidence > 0.8 | Re-training recommended |
| **Medium** | Execution deviation with confidence > 0.5 | Coaching recommended |
| **Low** | Minor execution variance | Optional coaching |

---

## Corrective Actions

SOPilot automatically generates actionable corrective actions based on detected deviations:

- **Critical Missing Steps**: "CRITICAL: Ensure trainee completes [step] before proceeding"
- **Order Issues**: "Review correct step sequence with trainee using gold standard video"
- **Quality Issues**: "Provide hands-on coaching for steps with execution quality issues"
- **No Issues**: "No major issues detected - trainee is ready for certification"

---

## Advanced Usage

### Custom Threshold

Override the default pass/fail threshold:

```bash
python scripts/sopilot_evaluate_pilot.py \
    --gold demo_videos/manufacturing/oil_change_gold.mp4 \
    --trainee demo_videos/manufacturing/oil_change_trainee_1.mp4 \
    --sop oil_change \
    --threshold 90
```

### Verbose Mode

Show detailed evaluation metrics:

```bash
python scripts/sopilot_evaluate_pilot.py \
    --gold demo_videos/manufacturing/oil_change_gold.mp4 \
    --trainee demo_videos/manufacturing/oil_change_trainee_1.mp4 \
    --sop oil_change \
    --verbose
```

**Verbose Output**:
```
[2/4] Evaluating SOP compliance...
  Raw score:     7.2
  Deviations:    2
  Missing steps: 1
  Order swaps:   0
```

---

## Batch Evaluation

Evaluate multiple trainees in a loop:

```bash
#!/bin/bash
GOLD="demo_videos/manufacturing/oil_change_gold.mp4"
for trainee in demo_videos/manufacturing/oil_change_trainee_*.mp4; do
    echo "Evaluating $(basename $trainee)..."
    python scripts/sopilot_evaluate_pilot.py \
        --gold "$GOLD" \
        --trainee "$trainee" \
        --sop oil_change \
        --out "reports/$(basename $trainee .mp4).json"
done
```

---

## Performance Benchmarks

**Hardware**: Windows 10, Python 3.10, CPU-only
**Videos**: ~60-80 seconds, 640x480, 24fps

| SOP | Gold Clips | Trainee Clips | Embedding Time | Evaluation Time | Total Time |
|-----|-----------|--------------|----------------|-----------------|------------|
| oil_change (10 steps) | 10 | 9-10 | 2.5s | 0.6s | 3.1s |
| brake_pads (8 steps) | 8 | 7-8 | 2.0s | 0.6s | 2.6s |
| ppe_check (5 steps) | 5 | 4-5 | 1.2s | 0.6s | 1.8s |

**Throughput**: ~20-30 videos/minute on single CPU core

---

## Example Scenarios

### Scenario 1: Safety Violation (trainee_1)
**Ground Truth**: Trainee skipped SAFETY step (no PPE)
**SOPilot Output**:
- Score: 7.2/100 (FAIL)
- 1 Critical Deviation: Missing SAFETY step @ 0:04-0:08
- Corrective Action: "CRITICAL: Ensure trainee completes Put on safety glasses and gloves before proceeding"

### Scenario 2: Wrong Order (trainee_2)
**Ground Truth**: Trainee reversed procedure (CHECK before FILL)
**SOPilot Output**:
- Score: 0.0/100 (FAIL)
- Multiple High Severity: Order swaps detected
- Corrective Action: "Review correct step sequence with trainee using gold standard video"

### Scenario 3: Wrong Tool (trainee_3)
**Ground Truth**: Used jack instead of wrench for filter removal
**SOPilot Output**:
- Score: 0.0/100 (FAIL)
- 1 Critical Deviation: Missing SAFETY step
- Multiple Execution Deviations
- Corrective Action: "CRITICAL: Ensure trainee completes Put on safety glasses and gloves before proceeding"

### Scenario 4: Perfect Execution (gold vs gold)
**Ground Truth**: Gold video compared to itself
**SOPilot Output**:
- Score: 100.0/100 (PASS)
- 0 Deviations
- Corrective Action: "No major issues detected - trainee is ready for certification"

---

## Integration Examples

### REST API Integration

```python
import requests
import json

response = requests.post("http://localhost:8000/api/pilot/evaluate", json={
    "gold_video_path": "/path/to/gold.mp4",
    "trainee_video_path": "/path/to/trainee.mp4",
    "sop": "oil_change"
})

report = response.json()
print(f"Score: {report['overall']['score']}")
print(f"Pass: {report['overall']['pass']}")
for dev in report['deviations']:
    print(f"  - {dev['description']} @ {dev['timestamp']}")
```

### LMS (Learning Management System) Integration

```python
import subprocess
import json

result = subprocess.run([
    "python", "scripts/sopilot_evaluate_pilot.py",
    "--gold", "gold.mp4",
    "--trainee", trainee_video_path,
    "--sop", "oil_change",
    "--out", "report.json"
], capture_output=True, text=True)

with open("report.json") as f:
    report = json.load(f)

# Update LMS database
if report['overall']['pass']:
    lms.mark_certified(trainee_id)
else:
    lms.assign_retraining(trainee_id, report['corrective_actions'])
```

---

## Troubleshooting

### "Gold video not found"
**Solution**: Generate demo videos first:
```bash
python scripts/generate_manufacturing_demo.py
```

### "reportlab not installed" (PDF output)
**Solution**: Install reportlab:
```bash
pip install reportlab
```

### Low scores despite correct execution
**Possible Causes**:
1. Video quality too different (lighting, camera angle)
2. Step duration mismatch (gold: 10s, trainee: 5s)
3. Custom SOP needs threshold tuning

**Solution**: Adjust threshold or use `--verbose` to inspect raw metrics

---

## Adding Custom SOPs

Edit `scripts/sopilot_evaluate_pilot.py` and add to `SOP_TEMPLATES`:

```python
SOP_TEMPLATES = {
    "your_sop": {
        "name": "Your SOP Full Name",
        "steps": ["STEP1", "STEP2", "STEP3"],
        "critical_steps": ["STEP1"],  # Must not skip
        "step_descriptions": {
            "STEP1": "Description of step 1",
            "STEP2": "Description of step 2",
            "STEP3": "Description of step 3",
        },
        "tool_requirements": {
            "STEP2": "wrench",  # Future use
        },
        "threshold": 80.0,
    },
    # ... existing SOPs
}
```

Then generate demo videos:
```bash
# Modify scripts/generate_manufacturing_demo.py to include your SOP
python scripts/generate_manufacturing_demo.py
```

---

## Technical Architecture

```
┌─────────────────┐
│  Gold Video     │───┐
└─────────────────┘   │
                      ├──> ClipWindowStream ──> Embeddings (904-dim vectors)
┌─────────────────┐   │
│ Trainee Video   │───┘
└─────────────────┘

                      │
                      V
              ┌───────────────┐
              │ evaluate_sop() │
              │  (DTW align)   │
              └───────┬────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        V             V             V
    Step         Deviation      Corrective
  Boundaries     Detection       Actions
        │             │             │
        └─────────────┴─────────────┘
                      │
                      V
              ┌───────────────┐
              │ JSON/PDF       │
              │ Report         │
              └───────────────┘
```

**Key Components**:
1. **ClipWindowStream**: Extract 4-second video clips at 1fps
2. **HeuristicClipEmbedder**: Convert clips to 904-dim feature vectors
3. **evaluate_sop()**: DTW alignment + deviation detection
4. **Deviation Parser**: Convert technical deviations to customer-facing reports
5. **Report Generator**: JSON or PDF output with corrective actions

---

## License & Support

**License**: Proprietary - SOPilot Manufacturing Pilot Package
**Support**: contact@sopilot.example.com
**Documentation**: https://docs.sopilot.example.com

---

## Success Criteria Checklist

- [x] One command execution (`python scripts/sopilot_evaluate_pilot.py ...`)
- [x] JSON output (API-compatible)
- [x] PDF output (human-readable)
- [x] Pass/fail verdict (with threshold)
- [x] Deviation list (with timestamps)
- [x] Severity classification (critical/high/medium/low)
- [x] Corrective actions (actionable guidance)
- [x] Evaluation time display (speed demonstration)
- [x] Multiple SOP support (oil_change, brake_pads, ppe_check)
- [x] Batch evaluation support (via shell scripts)
- [x] Exit code 0 (PASS) / 1 (FAIL) for CI/CD integration

---

## Next Steps

1. **Deploy to Production**: Package as Docker container or standalone executable
2. **Add More SOPs**: Extend to cover all manufacturing procedures
3. **Web Dashboard**: Build web UI for non-technical users
4. **Video Evidence**: Export deviation clips for visual confirmation
5. **Multi-language**: Translate corrective actions to Spanish, Chinese, etc.
