#!/usr/bin/env python3
"""Insurance MVP Real VLM Benchmark.

Runs the 3 demo videos through the real Qwen2.5-VL-7B-Instruct backend
and compares results against ground truth. This is the definitive accuracy
test — no mocks involved.

Requirements:
    - GPU with >= 14 GB VRAM (RTX 5090 confirmed)
    - Qwen2.5-VL-7B-Instruct downloaded
    - pip install -e ".[vigil]"  (transformers, qwen-vl-utils)

Usage:
    python scripts/insurance_real_benchmark.py
    python scripts/insurance_real_benchmark.py --json
    python scripts/insurance_real_benchmark.py --quick  (severity-only prompt)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("real_benchmark")

# Ground truth from metadata.json
GROUND_TRUTH = {
    "collision": {
        "severity": "HIGH",
        "fault_ratio": 100.0,
        "fraud_risk": 0.0,
        "scenario": "rear_end",
    },
    "near_miss": {
        "severity": "MEDIUM",
        "fault_ratio": 0.0,
        "fraud_risk": 0.0,
        "scenario": "pedestrian_avoidance",
    },
    "normal": {
        "severity": "NONE",
        "fault_ratio": 0.0,
        "fraud_risk": 0.0,
        "scenario": "normal_driving",
    },
}

# Severity ordering for distance calculation
SEVERITY_ORDER = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


def severity_distance(predicted: str, expected: str) -> int:
    """Ordinal distance between severity levels (0 = exact match)."""
    return abs(SEVERITY_ORDER.get(predicted, -1) - SEVERITY_ORDER.get(expected, -1))


def check_gpu_ready() -> bool:
    """Quick GPU readiness check."""
    try:
        import torch

        if not torch.cuda.is_available():
            logger.error("CUDA not available. Real VLM benchmark requires GPU.")
            return False
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB")
        if vram_gb < 14.0:
            logger.error(f"Insufficient VRAM ({vram_gb:.1f} GB < 14 GB required)")
            return False
        return True
    except ImportError:
        logger.error("PyTorch not installed")
        return False


def run_benchmark(as_json: bool = False, quick_mode: bool = False, quantize: str | None = None) -> dict:
    """Run real VLM benchmark on demo videos.

    Args:
        as_json: Output results as JSON
        quick_mode: Use simplified severity-only prompt

    Returns:
        Benchmark results dictionary
    """
    from insurance_mvp.cosmos.client import VideoLLMClient, VLMConfig

    demo_dir = project_root / "data" / "dashcam_demo"
    videos = {
        "collision": demo_dir / "collision.mp4",
        "near_miss": demo_dir / "near_miss.mp4",
        "normal": demo_dir / "normal.mp4",
    }

    # Verify videos exist
    for _name, path in videos.items():
        if not path.exists():
            logger.error(f"Demo video not found: {path}")
            return {"error": f"Missing video: {path}"}

    # Initialize real VLM client (optimized defaults: 16 frames, 512 tokens)
    logger.info("Initializing Qwen2.5-VL-7B-Instruct (this may take 30-60s on first load)...")
    init_start = time.time()
    config = VLMConfig(
        model_name="qwen2.5-vl-7b",
        device="cuda",
        dtype="bfloat16",
        quantize=quantize,
    )
    client = VideoLLMClient(config)
    init_time = time.time() - init_start
    logger.info(f"Model loaded in {init_time:.1f}s")

    # Health check
    health = client.health_check()
    logger.info(f"Health check: {health['status']} (mock={health['is_mock']})")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Qwen2.5-VL-7B-Instruct",
        "backend": "real",
        "mode": "quick" if quick_mode else "full",
        "model_load_time_sec": round(init_time, 2),
        "videos": {},
        "summary": {},
    }

    total_severity_match = 0
    total_severity_distance = 0
    total_inference_time = 0

    for video_name, video_path in videos.items():
        gt = GROUND_TRUTH[video_name]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {video_name}.mp4 (expected severity={gt['severity']})")
        logger.info(f"{'=' * 60}")

        inference_start = time.time()
        try:
            assessment = client.assess_claim(
                video_path=video_path,
                video_id=video_name,
            )
            inference_time = time.time() - inference_start
            total_inference_time += inference_time

            # Compare with ground truth
            sev_match = assessment.severity == gt["severity"]
            sev_dist = severity_distance(assessment.severity, gt["severity"])

            if sev_match:
                total_severity_match += 1
            total_severity_distance += sev_dist

            video_result = {
                "video_id": video_name,
                "success": True,
                "inference_time_sec": round(inference_time, 2),
                "expected": {
                    "severity": gt["severity"],
                    "fault_ratio": gt["fault_ratio"],
                    "fraud_risk": gt["fraud_risk"],
                },
                "actual": {
                    "severity": assessment.severity,
                    "confidence": round(assessment.confidence, 3),
                    "fault_ratio": assessment.fault_assessment.fault_ratio,
                    "fault_reasoning": assessment.fault_assessment.reasoning[:200],
                    "scenario_type": assessment.fault_assessment.scenario_type,
                    "fraud_score": round(assessment.fraud_risk.risk_score, 3),
                    "fraud_indicators": assessment.fraud_risk.indicators,
                    "prediction_set": sorted(assessment.prediction_set),
                    "review_priority": assessment.review_priority,
                    "recommended_action": assessment.recommended_action,
                    "causal_reasoning": assessment.causal_reasoning[:300],
                },
                "checks": {
                    "severity_match": sev_match,
                    "severity_distance": sev_dist,
                    "fraud_below_threshold": assessment.fraud_risk.risk_score <= 0.3,
                },
            }

            if not as_json:
                print(f"\n--- {video_name} ---")
                print(f"  Inference time: {inference_time:.2f}s")
                print(
                    f"  Severity: {assessment.severity} (expected: {gt['severity']}) {'MATCH' if sev_match else 'MISMATCH'}"
                )
                print(f"  Confidence: {assessment.confidence:.3f}")
                print(
                    f"  Fault ratio: {assessment.fault_assessment.fault_ratio:.1f}% (expected: {gt['fault_ratio']:.1f}%)"
                )
                print(f"  Fraud score: {assessment.fraud_risk.risk_score:.3f} (expected: <= 0.3)")
                print(f"  Scenario: {assessment.fault_assessment.scenario_type}")
                print(f"  Prediction set: {sorted(assessment.prediction_set)}")
                print(f"  Review priority: {assessment.review_priority}")
                print(f"  Reasoning: {assessment.causal_reasoning[:200]}...")

        except Exception as e:
            inference_time = time.time() - inference_start
            logger.error(f"Failed to process {video_name}: {e}")
            video_result = {
                "video_id": video_name,
                "success": False,
                "error": str(e),
                "inference_time_sec": round(inference_time, 2),
            }

            if not as_json:
                print(f"\n--- {video_name} ---")
                print(f"  ERROR: {e}")

        results["videos"][video_name] = video_result

    # Summary
    n_videos = len(videos)
    results["summary"] = {
        "total_videos": n_videos,
        "severity_exact_match": total_severity_match,
        "severity_accuracy": round(total_severity_match / n_videos * 100, 1),
        "mean_severity_distance": round(total_severity_distance / n_videos, 2),
        "total_inference_time_sec": round(total_inference_time, 2),
        "mean_inference_time_sec": round(total_inference_time / n_videos, 2),
    }

    if not as_json:
        print(f"\n{'=' * 60}")
        print("REAL VLM BENCHMARK SUMMARY")
        print(f"{'=' * 60}")
        print("  Model: Qwen2.5-VL-7B-Instruct (real inference)")
        print(f"  Model load time: {init_time:.1f}s")
        print(f"  Severity accuracy: {results['summary']['severity_accuracy']}% ({total_severity_match}/{n_videos})")
        print(f"  Mean severity distance: {results['summary']['mean_severity_distance']}")
        print(f"  Total inference time: {total_inference_time:.1f}s")
        print(f"  Mean inference time: {total_inference_time / n_videos:.1f}s per video")

    # Save results
    output_dir = project_root / "reports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "real_benchmark.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    if as_json:
        print(json.dumps(results, indent=2, default=str))

    return results


def main():
    parser = argparse.ArgumentParser(description="Insurance MVP Real VLM Benchmark")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--quick", action="store_true", help="Use quick severity-only prompt")
    parser.add_argument(
        "--quantize", choices=["int4", "int8"], default=None, help="Quantize model (requires bitsandbytes)"
    )
    args = parser.parse_args()

    # Pre-flight checks
    if not check_gpu_ready():
        print("\nGPU not ready for real benchmark. Run:")
        print("  python scripts/insurance_gpu_check.py")
        sys.exit(1)

    results = run_benchmark(as_json=args.json, quick_mode=args.quick, quantize=args.quantize)

    accuracy = results.get("summary", {}).get("severity_accuracy", 0)
    if accuracy < 66.7:
        logger.warning(f"Severity accuracy {accuracy}% below threshold (67%)")
        sys.exit(1)

    logger.info("Benchmark complete")


if __name__ == "__main__":
    main()
