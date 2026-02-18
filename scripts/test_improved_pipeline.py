#!/usr/bin/env python
"""Quick test of improved pipeline with smart mock VLM + real fault/fraud detection"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "insurance_mvp"))

import json
from config import PipelineConfig, CosmosBackend, DeviceType
from demo_pipeline import run_demo

if __name__ == "__main__":
    print("="*70)
    print(" Testing Improved Insurance MVP Pipeline")
    print("="*70)
    print("\nUpgrades:")
    print("  ✓ Smart mock VLM (realistic scenario-aware reasoning)")
    print("  ✓ Real fault assessment (rule-based engine)")
    print("  ✓ Real fraud detection (multi-signal heuristics)")
    print("\nRunning demo...\n")

    # Run demo pipeline
    run_demo(output_dir="demo_results_upgraded", verbose=True)

    # Check outputs
    output_dir = Path("demo_results_upgraded")
    results_files = list(output_dir.glob("*/results.json"))

    print(f"\n{'='*70}")
    print(" RESULTS SUMMARY")
    print(f"{'='*70}\n")

    for results_file in results_files:
        with open(results_file) as f:
            data = json.load(f)

        video_id = data['video_id']
        print(f"\n{video_id}:")
        print(f"{'─'*70}")

        if data['assessments']:
            assessment = data['assessments'][0]  # First clip

            print(f"Severity:          {assessment['severity']}")
            print(f"Confidence:        {assessment['confidence']:.2%}")
            print(f"Fault Ratio:       {assessment['fault_assessment']['fault_ratio']:.1f}%")
            print(f"Fraud Score:       {assessment['fraud_risk']['risk_score']:.2f}")

            print(f"\nCausal Reasoning:")
            reasoning = assessment['causal_reasoning']
            # Wrap reasoning text
            import textwrap
            wrapped = textwrap.fill(reasoning, width=66, initial_indent="  ", subsequent_indent="  ")
            print(wrapped)

            print(f"\nFault Reasoning:")
            fault_reasoning = assessment['fault_assessment']['reasoning']
            wrapped_fault = textwrap.fill(fault_reasoning, width=66, initial_indent="  ", subsequent_indent="  ")
            print(wrapped_fault)

    print(f"\n{'='*70}")
    print(" SUCCESS - Check demo_results_upgraded/ for full HTML reports")
    print(f"{'='*70}\n")
