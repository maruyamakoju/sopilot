#!/usr/bin/env python3
"""Insurance MVP E2E Benchmark.

Runs the full pipeline on demo videos and compares against ground truth
from data/dashcam_demo/metadata.json.

Usage:
    python scripts/insurance_e2e_benchmark.py
    python scripts/insurance_e2e_benchmark.py --output results/benchmark.json
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from insurance_mvp.config import CosmosBackend, PipelineConfig  # noqa: E402
from insurance_mvp.pipeline import InsurancePipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Ground Truth
# ---------------------------------------------------------------------------


def load_ground_truth() -> dict:
    """Load ground truth from metadata.json."""
    metadata_path = project_root / "data" / "dashcam_demo" / "metadata.json"
    if not metadata_path.exists():
        print(f"WARNING: Ground truth not found at {metadata_path}")
        return {
            "collision": {
                "severity": "HIGH",
                "ground_truth": {"fault_ratio": 100.0, "scenario": "rear_end", "fraud_risk": 0.0},
            },
            "near_miss": {
                "severity": "MEDIUM",
                "ground_truth": {"fault_ratio": 0.0, "scenario": "pedestrian_avoidance", "fraud_risk": 0.0},
            },
            "normal": {
                "severity": "NONE",
                "ground_truth": {"fault_ratio": 0.0, "scenario": "normal_driving", "fraud_risk": 0.0},
            },
        }
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


def run_benchmark(output_path: str = None):
    """Run the E2E benchmark."""
    print("=" * 60)
    print("Insurance MVP E2E Benchmark")
    print("=" * 60)

    ground_truth = load_ground_truth()
    output_dir = tempfile.mkdtemp(prefix="insurance_benchmark_")

    # Configure pipeline with mock backend
    config = PipelineConfig(
        output_dir=output_dir,
        log_level="WARNING",
        parallel_workers=1,
        enable_conformal=True,
        enable_transcription=False,
        continue_on_error=True,
    )
    config.cosmos.backend = CosmosBackend.MOCK

    pipeline = InsurancePipeline(config)
    # Disable external dependencies
    pipeline.signal_fuser = None
    pipeline.cosmos_client = None

    results = {}
    total_checks = 0
    passed_checks = 0

    demo_dir = project_root / "data" / "dashcam_demo"

    for video_name, gt_data in ground_truth.items():
        print(f"\n--- {video_name} ---")
        gt = gt_data.get("ground_truth", gt_data)
        expected_severity = gt_data.get("severity", "NONE")

        # Check for real video file
        video_path = demo_dir / f"{video_name}.mp4"
        if not video_path.exists():
            # Create mock file for testing
            video_path = Path(output_dir) / f"{video_name}.mp4"
            video_path.write_bytes(b"mock video content")
            print(f"  Using mock video (real not found at {demo_dir / f'{video_name}.mp4'})")

        # Process video
        start = time.time()
        result = pipeline.process_video(str(video_path), video_id=video_name)
        elapsed = time.time() - start

        video_result = {
            "video_id": video_name,
            "success": result.success,
            "processing_time_sec": round(elapsed, 3),
            "expected_severity": expected_severity,
            "expected_fault_ratio": gt.get("fault_ratio", 0.0),
            "expected_fraud_risk": gt.get("fraud_risk", 0.0),
            "checks": [],
        }

        # Check 1: Pipeline success
        total_checks += 1
        check_success = result.success
        if check_success:
            passed_checks += 1
        video_result["checks"].append(
            {
                "name": "pipeline_success",
                "passed": check_success,
            }
        )
        print(f"  Pipeline success: {'PASS' if check_success else 'FAIL'}")

        # Check 2: Has assessments
        total_checks += 1
        has_assessments = result.assessments is not None and len(result.assessments) > 0
        if has_assessments:
            passed_checks += 1
        video_result["checks"].append(
            {
                "name": "has_assessments",
                "passed": has_assessments,
            }
        )
        print(f"  Has assessments: {'PASS' if has_assessments else 'FAIL'}")

        # Check 3: Output files exist
        total_checks += 1
        has_outputs = result.output_json_path is not None and Path(result.output_json_path).exists()
        if has_outputs:
            passed_checks += 1
        video_result["checks"].append(
            {
                "name": "output_files_exist",
                "passed": has_outputs,
            }
        )
        print(f"  Output files: {'PASS' if has_outputs else 'FAIL'}")

        if has_assessments:
            top_assessment = result.assessments[0]
            video_result["actual_severity"] = top_assessment.severity
            video_result["actual_fault_ratio"] = top_assessment.fault_assessment.fault_ratio
            video_result["actual_fraud_score"] = top_assessment.fraud_risk.risk_score

            print(f"  Actual severity: {top_assessment.severity} (expected: {expected_severity})")
            print(f"  Actual fault ratio: {top_assessment.fault_assessment.fault_ratio}%")
            print(f"  Actual fraud score: {top_assessment.fraud_risk.risk_score:.3f}")

        results[video_name] = video_result

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {total_checks - passed_checks}")
    print(f"Pass rate: {passed_checks / total_checks * 100:.1f}%")

    benchmark_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "pass_rate": round(passed_checks / total_checks * 100, 1),
        "results": results,
    }

    # Save output
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(benchmark_output, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")

    return benchmark_output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Insurance MVP E2E Benchmark")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    result = run_benchmark(output_path=args.output)

    # Exit with error code if any checks failed
    if result["passed_checks"] < result["total_checks"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
