Archive of Redundant / Legacy Scripts
======================================
Archived: 2026-02-24
Reason: These scripts are superseded by canonical scripts or are no longer
        part of the active Insurance MVP / VIGIL-RAG evaluation workflow.

Archived scripts and their canonical replacements:
---------------------------------------------------

1. LEGACY MANUFACTURING / PILOT SCRIPTS (SOPilot v0 era)
   - evaluate_manufacturing.py      → No direct replacement; manufacturing use-case
                                       is superseded by the Insurance MVP pipeline.
   - generate_manufacturing_demo.py → Synthetic video generation for manufacturing
                                       SOPs; not needed for dashcam evaluation.
   - demo_pilot_package.py          → Wrapper that calls evaluate_manufacturing.py;
                                       superseded by insurance_e2e_benchmark.py.
   - sopilot_evaluate_pilot.py      → Legacy customer-facing pilot evaluation tool;
                                       superseded by the Insurance MVP pipeline CLI
                                       (insurance_mvp/pipeline/cli.py).

2. REDUNDANT EVAL WRAPPER
   - eval_real_vlm.py               → Thin subprocess wrapper around
                                       real_data_benchmark.py. Use
                                       real_data_benchmark.py directly with the
                                       same --source / --input arguments.

3. OLD VIGIL-RAG SCRIPTS (no longer canonical)
   - evaluate_vigil_benchmark.py    → Replaced by the VIGILBenchmarkRunner class
                                       in sopilot.evaluation.vigil_benchmark.
                                       The canonical entry-point is now
                                       scripts/smoke_e2e.py (vigil path).
   - evaluate_vigil_real.py         → Real-data VIGIL evaluation; retained here
                                       for reference but not part of the standard
                                       CI/CD run. See scripts/vigil_smoke_e2e.py.
   - generate_vigil_benchmark_v2.py → Used to generate the gold.mp4 fixture for
                                       VIGIL v2 benchmarks. The fixture already
                                       exists; this generator is no longer needed
                                       in routine runs.

4. REDUNDANT NEURAL DEMO SCRIPTS
   - demo_e2e_pipeline.py           → Standalone matplotlib demo of the SOP neural
                                       pipeline stages. Superseded by
                                       train_benchmark.py for full pipeline runs
                                       and demo_neural_pipeline.py (kept here too).
   - demo_neural_pipeline.py        → Publication-quality figure generator for the
                                       neural pipeline components. Useful for paper
                                       figures but not part of the evaluation suite.

Canonical scripts that remain in scripts/:
-------------------------------------------
  vlm_accuracy_benchmark.py    - VLM eval on labelled demo videos (10 scenarios)
  real_vlm_eval_10.py          - Direct per-video VLM evaluation
  expanded_video_eval.py       - Multi-source evaluation (demo/jp/nexar)
  real_data_benchmark.py       - Real dashcam accuracy benchmark
  insurance_e2e_benchmark.py   - Full E2E pipeline benchmark (9/9 checks)
  batch_process.py             - Batch processing CLI
  train_benchmark.py           - Training benchmark (synthetic data)
  smoke_e2e.py                 - E2E smoke test (CI)
  download_nexar.py            - Nexar dataset downloader
  nexar_to_insurance_format.py - Nexar → Insurance label converter
