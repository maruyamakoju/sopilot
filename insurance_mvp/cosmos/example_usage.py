"""Example usage of Video-LLM client for insurance claim assessment.

This demonstrates the complete workflow from video to structured assessment.
"""

import cv2
import numpy as np
from pathlib import Path

from insurance_mvp.cosmos import create_client


def example_basic_usage():
    """Basic usage example."""
    print("=" * 80)
    print("Example 1: Basic Usage (Mock Mode)")
    print("=" * 80)

    # Create client in mock mode (no GPU required, instant results)
    client = create_client(model_name="mock", device="cpu")

    # Create a dummy video file for the example
    import tempfile
    import cv2
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = Path(f.name)

    # Create simple test video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    for i in range(90):  # 3 seconds
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    assessment = client.assess_claim(video_path=str(video_path), video_id="example_001", start_sec=0.0)

    # Cleanup
    video_path.unlink(missing_ok=True)

    # Display results
    print(f"\n[OK] Assessment complete!")
    print(f"  Video ID: {assessment.video_id}")
    print(f"  Severity: {assessment.severity}")
    print(f"  Confidence: {assessment.confidence:.2%}")
    print(f"  Fault Ratio: {assessment.fault_assessment.fault_ratio}%")
    print(f"  Fraud Risk: {assessment.fraud_risk.risk_score:.2%}")
    print(f"  Recommended Action: {assessment.recommended_action}")
    print(f"  Processing Time: {assessment.processing_time_sec:.2f}s")


def example_batch_processing():
    """Batch processing example (reuse client for multiple videos)."""
    print("\n" + "=" * 80)
    print("Example 2: Batch Processing (Model Caching)")
    print("=" * 80)

    # Create client ONCE (model stays loaded)
    client = create_client(model_name="mock", device="cpu")

    # Process multiple claims
    claims = [
        {"video_id": "batch_001", "video_path": "claim1.mp4", "start": 10.0, "end": 30.0},
        {"video_id": "batch_002", "video_path": "claim2.mp4", "start": 5.0, "end": 25.0},
        {"video_id": "batch_003", "video_path": "claim3.mp4", "start": 0.0, "end": 20.0},
    ]

    results = []
    for claim in claims:
        assessment = client.assess_claim(
            video_path=claim["video_path"],
            video_id=claim["video_id"],
            start_sec=claim["start"],
            end_sec=claim["end"],
        )
        results.append(assessment)
        print(f"  [OK] {claim['video_id']}: {assessment.severity} ({assessment.confidence:.2%})")

    print(f"\n[OK] Processed {len(results)} claims")


def example_custom_config():
    """Custom configuration example."""
    print("\n" + "=" * 80)
    print("Example 3: Custom Configuration")
    print("=" * 80)

    from insurance_mvp.cosmos import VLMConfig, VideoLLMClient

    # Create custom configuration
    config = VLMConfig(
        model_name="mock",
        device="cpu",
        fps=2.0,  # Lower FPS for faster processing
        max_frames=16,  # Fewer frames
        temperature=0.1,  # More conservative predictions
        timeout_sec=600.0,  # 10 minute timeout
    )

    client = VideoLLMClient(config)

    assessment = client.assess_claim(video_path="example.mp4", video_id="custom_001")

    print(f"  [OK] Config: {config.fps} FPS, {config.max_frames} max frames")
    print(f"  [OK] Result: {assessment.severity} ({assessment.confidence:.2%})")


def example_error_handling():
    """Error handling example."""
    print("\n" + "=" * 80)
    print("Example 4: Graceful Error Handling")
    print("=" * 80)

    client = create_client(model_name="mock", device="cpu")

    # Even if something goes wrong, we get a valid assessment (never crashes)
    # This would fail in real mode, but mock mode handles it gracefully
    assessment = client.assess_claim(video_path="nonexistent.mp4", video_id="error_001")

    print(f"  [OK] Got valid assessment even on error: {assessment.severity}")
    print(f"  [OK] Review priority: {assessment.review_priority}")
    print(f"  [OK] Recommended action: {assessment.recommended_action}")


def example_structured_output():
    """Structured output example."""
    print("\n" + "=" * 80)
    print("Example 5: Structured Output Access")
    print("=" * 80)

    client = create_client(model_name="mock", device="cpu")
    assessment = client.assess_claim(video_path="example.mp4", video_id="structured_001")

    # Access all structured fields
    print("\n  --- Severity Assessment ---")
    print(f"  Severity: {assessment.severity}")
    print(f"  Confidence: {assessment.confidence:.2%}")
    print(f"  Prediction Set: {', '.join(assessment.prediction_set)}")
    print(f"  Review Priority: {assessment.review_priority}")

    print("\n  --- Fault Assessment ---")
    print(f"  Fault Ratio: {assessment.fault_assessment.fault_ratio}%")
    print(f"  Scenario Type: {assessment.fault_assessment.scenario_type}")
    print(f"  Reasoning: {assessment.fault_assessment.reasoning}")

    print("\n  --- Fraud Risk ---")
    print(f"  Risk Score: {assessment.fraud_risk.risk_score:.2%}")
    print(f"  Indicators: {', '.join(assessment.fraud_risk.indicators) or 'None'}")
    print(f"  Reasoning: {assessment.fraud_risk.reasoning}")

    print("\n  --- Reasoning & Action ---")
    print(f"  Causal Reasoning: {assessment.causal_reasoning}")
    print(f"  Recommended Action: {assessment.recommended_action}")

    print("\n  --- Evidence ---")
    print(f"  Hazards: {len(assessment.hazards)}")
    print(f"  Evidence Items: {len(assessment.evidence)}")


def example_json_export():
    """JSON export example."""
    print("\n" + "=" * 80)
    print("Example 6: JSON Export")
    print("=" * 80)

    import json

    client = create_client(model_name="mock", device="cpu")
    assessment = client.assess_claim(video_path="example.mp4", video_id="export_001")

    # Export to JSON
    json_data = assessment.model_dump(mode="json")
    json_str = json.dumps(json_data, indent=2, default=str)

    print(f"  [OK] Exported to JSON ({len(json_str)} bytes)")
    print(f"\n{json_str[:500]}...")  # Show first 500 chars


def main():
    """Run all examples."""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  Insurance MVP - Video-LLM Client Usage Examples".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print("\n")

    example_basic_usage()
    example_batch_processing()
    example_custom_config()
    example_error_handling()
    example_structured_output()
    example_json_export()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run real inference: Replace 'mock' with 'qwen2.5-vl-7b'")
    print("  2. Use real video files: Provide actual dashcam footage paths")
    print("  3. Integrate with pipeline: Connect to mining + conformal modules")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
