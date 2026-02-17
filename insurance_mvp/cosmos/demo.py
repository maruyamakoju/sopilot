"""Demo script for insurance claim assessment with Video-LLM.

Usage:
    # Mock mode (no GPU required)
    python -m insurance_mvp.cosmos.demo --video dashcam.mp4 --mock

    # Real inference with Qwen2.5-VL
    python -m insurance_mvp.cosmos.demo --video dashcam.mp4

    # NVIDIA Cosmos Reason 2 (when available)
    python -m insurance_mvp.cosmos.demo --video dashcam.mp4 --model nvidia-cosmos-reason-2
"""

import argparse
import json
import logging
from pathlib import Path

from .client import create_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Insurance claim assessment demo")
    parser.add_argument("--video", required=True, help="Path to dashcam video file")
    parser.add_argument("--video-id", default="demo_001", help="Video ID for metadata")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end-sec", type=float, default=None, help="End time in seconds")
    parser.add_argument(
        "--model",
        choices=["nvidia-cosmos-reason-2", "qwen2.5-vl-7b", "mock"],
        default="qwen2.5-vl-7b",
        help="Model to use",
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--mock", action="store_true", help="Use mock mode (shortcut for --model mock)")
    parser.add_argument("--output", help="Output JSON file path")

    args = parser.parse_args()

    # Validate video exists
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        return 1

    # Override model if --mock flag used
    model_name = "mock" if args.mock else args.model

    logger.info("=" * 80)
    logger.info("Insurance Claim Assessment Demo")
    logger.info("=" * 80)
    logger.info("Video: %s", video_path)
    logger.info("Video ID: %s", args.video_id)
    logger.info("Time range: %.2f - %.2f sec", args.start_sec, args.end_sec or 0)
    logger.info("Model: %s", model_name)
    logger.info("Device: %s", args.device)
    logger.info("=" * 80)

    # Create client
    try:
        logger.info("Creating Video-LLM client...")
        client = create_client(model_name=model_name, device=args.device)
        logger.info("Client created successfully")
    except Exception as exc:
        logger.error("Failed to create client: %s", exc)
        return 1

    # Assess claim
    try:
        logger.info("Running claim assessment...")
        assessment = client.assess_claim(
            video_path=video_path, video_id=args.video_id, start_sec=args.start_sec, end_sec=args.end_sec
        )
        logger.info("Assessment complete!")
    except Exception as exc:
        logger.error("Assessment failed: %s", exc)
        return 1

    # Display results
    print("\n" + "=" * 80)
    print("CLAIM ASSESSMENT RESULTS")
    print("=" * 80)
    print(f"\nVideo ID: {assessment.video_id}")
    print(f"Processing Time: {assessment.processing_time_sec:.2f} seconds")
    print(f"\n--- SEVERITY ---")
    print(f"Severity: {assessment.severity}")
    print(f"Confidence: {assessment.confidence:.2%}")
    print(f"Prediction Set: {', '.join(assessment.prediction_set)}")
    print(f"Review Priority: {assessment.review_priority}")
    print(f"\n--- FAULT ASSESSMENT ---")
    print(f"Fault Ratio: {assessment.fault_assessment.fault_ratio:.1f}%")
    print(f"Scenario Type: {assessment.fault_assessment.scenario_type}")
    print(f"Reasoning: {assessment.fault_assessment.reasoning}")
    if assessment.fault_assessment.applicable_rules:
        print(f"Applicable Rules: {', '.join(assessment.fault_assessment.applicable_rules)}")
    print(f"\n--- FRAUD RISK ---")
    print(f"Risk Score: {assessment.fraud_risk.risk_score:.2%}")
    if assessment.fraud_risk.indicators:
        print(f"Indicators: {', '.join(assessment.fraud_risk.indicators)}")
    print(f"Reasoning: {assessment.fraud_risk.reasoning}")
    print(f"\n--- CAUSAL REASONING ---")
    print(assessment.causal_reasoning)
    print(f"\n--- RECOMMENDED ACTION ---")
    print(assessment.recommended_action)

    if assessment.hazards:
        print(f"\n--- HAZARDS ({len(assessment.hazards)}) ---")
        for i, hazard in enumerate(assessment.hazards, 1):
            print(
                f"{i}. {hazard.type} @ {hazard.timestamp_sec:.1f}s: "
                f"{', '.join(hazard.actors)} ({hazard.spatial_relation})"
            )

    if assessment.evidence:
        print(f"\n--- EVIDENCE ({len(assessment.evidence)}) ---")
        for i, ev in enumerate(assessment.evidence, 1):
            print(f"{i}. [{ev.timestamp_sec:.1f}s] {ev.description}")

    print("=" * 80 + "\n")

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        with output_path.open("w") as f:
            json.dump(assessment.model_dump(mode="json"), f, indent=2, default=str)
        logger.info("Results saved to: %s", output_path)

    return 0


if __name__ == "__main__":
    exit(main())
