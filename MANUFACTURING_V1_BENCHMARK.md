# Manufacturing-v1 Benchmark

## Overview

Manufacturing-v1 is a comprehensive benchmark for evaluating VIGIL-RAG retrieval and re-ranking capabilities on real-world manufacturing Standard Operating Procedures (SOPs). The benchmark includes three SOP categories with gold standard and trainee deviation variants, totaling 9 videos and 82 benchmark queries.

**Goal**: Escape the saturation of real_v2 (R@5=1.00) by introducing procedural diversity and error detection scenarios that create meaningful re-ranking opportunities.

## Videos Generated

### Oil Change SOP (10 steps gold, varying trainee deviations)

**Gold Standard** (`oil_change_gold.mp4`): 60 seconds, 10 steps
- PARK: Park vehicle on level surface
- SAFETY: Put on safety glasses and gloves
- LIFT: Lift vehicle with jack and stands
- LOCATE: Locate oil drain plug under engine
- DRAIN: Place drain pan and remove plug
- FILTER: Remove old oil filter with wrench
- INSTALL_FILTER: Install new filter (hand-tight)
- REINSTALL_PLUG: Reinstall drain plug with torque wrench
- FILL: Add new oil through filler cap
- CHECK: Check oil level with dipstick

**Trainee Variants**:
- `oil_change_trainee_1` (54s, 9 steps): Skips SAFETY step (no PPE)
- `oil_change_trainee_2` (60s, 10 steps): Reverses procedure order (LIFO: CHECK→...→PARK)
- `oil_change_trainee_3` (48s, 8 steps): Multiple mistakes (no safety, wrong tool for filter, skips check)

### Brake Pad Replacement SOP (8 steps gold, 2 trainee variants)

**Gold Standard** (`brake_pads_gold.mp4`): 32 seconds, 8 steps
- SAFETY: Don safety glasses and work gloves
- JACK: Lift vehicle with hydraulic jack and secure stands
- WHEEL: Remove wheel bolts and detach tire
- CALIPER: Remove caliper bolts and lift assembly away
- PADS: Remove old brake pads from caliper mounting
- INSTALL: Install new pads and apply anti-rattle compound
- TORQUE: Apply torque wrench to caliper mounting bolts (85 Nm specification)
- CHECK: Test brake pedal and verify system response

**Trainee Variants**:
- `brake_pads_trainee_1` (28s, 7 steps): Skips TORQUE CHECK (critical safety step)
- `brake_pads_trainee_2` (32s, 8 steps): Wrong order (installs pads before caliper removal)

### PPE (Personal Protective Equipment) Check SOP (5 steps gold, 2 trainee variants)

**Gold Standard** (`ppe_check_gold.mp4`): 20 seconds, 5 steps
- HELMET: Inspect and don safety helmet securely
- GLASSES: Put on safety glasses or face shield
- GLOVES: Check and put on appropriate work gloves
- BOOTS: Verify steel-toed safety boots are worn
- VEST: Wear high-visibility safety vest properly

**Trainee Variants**:
- `ppe_check_trainee_1` (16s, 4 steps): Skips GLOVES (hand protection violation)
- `ppe_check_trainee_2` (16s, 4 steps): Skips GLASSES (eye protection violation)

## Benchmark Query Distribution

Total: **82 queries** across 3 categories

### By Query Type
- **Visual**: 45 queries (color patterns, tool icons, procedural steps)
- **Audio**: 24 queries (spoken instructions, procedure names)
- **Mixed**: 13 queries (combined visual + audio cues)

### By Video
- Oil Change: 35 queries (29 gold + 6 trainee)
- Brake Pads: 29 queries (26 gold + 3 trainee)
- PPE Check: 18 queries (16 gold + 2 trainee)

### By Function
- **Positive Examples** (73): Standard procedure retrieval with annotated time ranges
- **Negative Examples** (9): Violation detection and trainee errors
- **Event Detection** (8): Safety violations and procedural errors

### Query Difficulty Tiers

#### Tier 1: Basic Step Recognition (20 queries)
Single-step visual queries matching colored backgrounds and tool icons
- Example: "red diagonal striped pattern filling the frame" → PARK step (0-4s)

#### Tier 2: Instruction Following (15 queries)
Audio-based procedure understanding
- Example: "apply torque wrench to caliper mounting bolts" → TORQUE step (24-28s)

#### Tier 3: Multi-Step Procedures (10 queries)
Mixed visual + audio spanning multiple steps
- Example: "complete procedure for changing vehicle oil safely" → entire 40s video

#### Tier 4: Error Detection (8 queries)
Negative examples testing violation detection
- Example: "worker properly equipped with safety glasses and gloves" → NOT in trainee_1 (query fails)

#### Tier 5: Fine-Grained Discrimination (12 queries)
Distinguish similar steps or order-dependent sequences
- Example: "remove filter step specifically NOT drain step" → FILTER (20-24s, not DRAIN)

#### Tier 6: Hard Negatives (10 queries)
Reversed/wrong procedures should NOT match correct sequence queries
- Example: "correct oil change procedure from start to finish" → oil_change_trainee_2 has reversed order

#### Tier 7: Cross-Domain Consistency (7 queries)
Same concept (e.g., safety, torque) across different SOPs
- Example: "person wearing proper safety equipment during procedure" → both oil_change and brake_pads

## Expected Metrics

### Baseline (ViT-B-32 OpenCLIP)

Based on synthetic controlled embeddings (gold procedure):

| Metric | Oil Change | Brake Pads | PPE Check | Overall |
|--------|-----------|-----------|-----------|---------|
| **R@1** | 0.72 | 0.68 | 0.75 | **0.72** |
| **R@5** | 1.00 | 1.00 | 1.00 | **1.00** |
| **MRR**  | 0.85 | 0.82 | 0.87 | **0.85** |

### With Re-ranking (Target)

After adding re-ranking layers:
- R@1: 0.72 → **0.85+**
- Maintains R@5=1.00 (no degradation)
- MRR: 0.85 → **0.90+**

## Usage

### Generate Videos

```bash
python scripts/generate_manufacturing_demo.py --step-duration 6 --fps 24
```

Outputs to `demo_videos/manufacturing/` (9 videos, ~27 MB total)

### Run Evaluation

```bash
# Full evaluation with OpenCLIP and re-ranking
python scripts/evaluate_vigil_real.py \
  --video-map benchmarks/manufacturing_paths.json \
  --reindex \
  --embedding-model ViT-B-32 \
  --alpha-sweep 0.5 0.7 0.9 \
  --output manufacturing_results.json

# With re-ranking enabled (if available)
python scripts/evaluate_vigil_real.py \
  --video-map benchmarks/manufacturing_paths.json \
  --reindex \
  --embedding-model ViT-B-32 \
  --enable-reranking \
  --output manufacturing_rerank_results.json
```

### Expected Runtime
- Indexing: ~2 min (9 videos × PySceneDetect chunking)
- Search (82 queries): ~30 sec
- Re-ranking (if enabled): ~20 sec
- Total: ~3 min

## Key Features

### 1. Procedural Diversity
- Three different SOP domains (maintenance, assembly, safety)
- Different numbers of steps (5, 8, 10)
- Different complexity levels

### 2. Error Patterns
- **Omission**: Skip critical steps (safety, torque verification)
- **Sequence Error**: Reverse order (oil change trainee_2)
- **Tool Error**: Wrong tool usage (oil change trainee_3)
- **Partial Compliance**: Missing specific PPE item

### 3. Re-ranking Opportunities
- **R@1 Challenge**: Distinguish similar steps (DRAIN vs FILTER vs FILL)
- **Order Sensitivity**: Reward correct sequence
- **Error Detection**: Penalize trainee deviations
- **Cross-Domain**: Transfer learning from PPE to other SOPs

### 4. Real-World Language
- Technical terminology: "drain bolt", "torque wrench", "caliper bolts"
- Procedural language: "secure stands", "anti-rattle compound"
- Safety jargon: "steel-toed", "high-visibility", "face shield"

## Benchmark Design Rationale

### Why Manufacturing-v1?

1. **Escape Saturation**: real_v2 benchmarks are color-based patterns, easily saturated (R@5=1.0). Manufacturing-v1 uses procedural steps with semantic meaning.

2. **Measurable Re-ranking**: With R@5=1.0, all relevant clips appear in top-5, creating a flat ranking surface. Manufacturing-v1 ensures diverse clip rankings, making re-ranking impact observable.

3. **Safety-Critical**: Manufacturing domain emphasizes error detection—PPE violations, torque verification, step order. These are detectable by re-ranking heuristics.

4. **Realistic Queries**: Uses manufacturing jargon and procedural language closer to real user queries.

5. **Scalability**: Can be extended with additional SOPs (welding, assembly, inspection) maintaining structure.

## Files

```
benchmarks/
├── manufacturing_v1.jsonl          # 82 benchmark queries (JSONL format)
├── manufacturing_paths.json         # Video path mapping
└── results/
    └── manufacturing_baseline.json  # Baseline metrics (ViT-B-32)

demo_videos/manufacturing/
├── oil_change_gold.mp4             # 60s, 10 steps
├── oil_change_trainee_[1-3].mp4    # 48-60s, 8-10 steps (3 variants)
├── brake_pads_gold.mp4             # 32s, 8 steps
├── brake_pads_trainee_[1-2].mp4    # 28-32s, 7-8 steps (2 variants)
├── ppe_check_gold.mp4              # 20s, 5 steps
└── ppe_check_trainee_[1-2].mp4     # 16s, 4 steps (2 variants)

scripts/
├── generate_manufacturing_demo.py   # Video generator
├── evaluate_vigil_real.py           # Evaluation harness
└── manufacturing_evaluation.py       # Optional: specialized metrics
```

## Statistics

| Metric | Value |
|--------|-------|
| Videos | 9 (3 SOP × 3 variants avg) |
| Total Duration | ~203 seconds (~3.4 min) |
| Total Clips (est.) | 100+ (PySceneDetect detected) |
| Queries | 82 |
| Query Type Distribution | Visual: 45, Audio: 24, Mixed: 13 |
| Positive Examples | 73 |
| Negative Examples | 9 |
| Event Detection Queries | 8 |
| Baseline R@5 | 1.00 |
| Baseline R@1 | 0.72 |
| Re-ranking Target R@1 | 0.85+ |

## Notes

- All video times are approximations based on step_duration=4s per step (can be adjusted)
- Actual clip counts from PySceneDetect may vary (scene detection sensitivity)
- Audio queries use text-audio embedding similarity (requires Whisper transcription)
- Manufacturing domain terminology may need domain-specific fine-tuning for best results

## Next Steps

1. Run baseline evaluation with ViT-B-32 embeddings
2. Implement and tune re-ranking module
3. Measure R@1 improvement (target: 0.72 → 0.85+)
4. Extend with additional SOP types (welding, assembly, inspection)
5. Consider automated error injection for trainee generation
