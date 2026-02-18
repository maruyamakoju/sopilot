"""Insurance MVP Pipeline Demo

Demonstrates end-to-end pipeline with mock data for testing without real videos.

Usage:
    python demo_pipeline.py
    python demo_pipeline.py --output-dir demo_results/
    python demo_pipeline.py --verbose
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from insurance_mvp.config import CosmosBackend, DeviceType, PipelineConfig
from insurance_mvp.insurance.schema import ClaimAssessment
from insurance_mvp.pipeline import InsurancePipeline


def create_mock_video(video_path: Path, duration_sec: int = 30):
    """Create a minimal mock video file for testing"""
    # Just create a placeholder file
    # In production, this would be a real video file
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with open(video_path, "wb") as f:
        # Write a minimal file header (not a real video, just for testing)
        f.write(b"MOCK_VIDEO_FILE\n")
        f.write(f"Duration: {duration_sec}s\n".encode())
    return str(video_path)


def print_section(title: str, symbol: str = "="):
    """Print formatted section header"""
    print(f"\n{symbol * 60}")
    print(f"  {title}")
    print(f"{symbol * 60}\n")


def print_assessment_summary(assessment: ClaimAssessment, idx: int):
    """Print formatted assessment summary"""
    print(f"\n{'-' * 60}")
    print(f"Clip #{idx}")
    print(f"{'-' * 60}")
    print(f"Severity:         {assessment.severity} (confidence: {assessment.confidence:.2f})")
    print(f"Prediction Set:   {{{', '.join(sorted(assessment.prediction_set))}}}")
    print(f"Review Priority:  {assessment.review_priority}")
    print(f"Fault Ratio:      {assessment.fault_assessment.fault_ratio:.1f}%")
    print(f"Fraud Score:      {assessment.fraud_risk.risk_score:.2f}")
    print(f"Recommended:      {assessment.recommended_action}")
    print("\nReasoning:")
    print(f"  {assessment.causal_reasoning}")
    if assessment.fault_assessment.reasoning:
        print("\nFault Assessment:")
        print(f"  {assessment.fault_assessment.reasoning}")
    if assessment.fraud_risk.reasoning:
        print("\nFraud Analysis:")
        print(f"  {assessment.fraud_risk.reasoning}")


def run_demo(output_dir: str = "demo_results", verbose: bool = False):
    """Run pipeline demo with mock data"""

    print_section("Insurance MVP Pipeline Demo", "=")

    # Create temporary directory for mock videos
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    try:
        # Step 1: Create mock videos
        print_section("Step 1: Creating Mock Videos", "-")

        mock_videos = []
        video_scenarios = [
            ("dashcam_collision.mp4", "High-speed rear-end collision"),
            ("dashcam_near_miss.mp4", "Near-miss with pedestrian"),
            ("dashcam_normal.mp4", "Normal driving, no incidents"),
        ]

        for filename, description in video_scenarios:
            video_path = Path(temp_dir) / filename
            create_mock_video(video_path)
            mock_videos.append((str(video_path), description))
            print(f"[OK] Created: {filename} - {description}")

        # Step 2: Configure pipeline
        print_section("Step 2: Configuring Pipeline", "-")

        config = PipelineConfig(
            output_dir=output_dir,
            log_level="DEBUG" if verbose else "INFO",
            parallel_workers=1,  # Sequential for demo clarity
            enable_conformal=True,
            enable_transcription=False,  # Disable for demo (no audio in mock videos)
            enable_fraud_detection=True,
            enable_fault_assessment=True,
            continue_on_error=True,
            save_intermediate_results=True,
        )

        # Use mock backend for demo (no GPU required)
        config.cosmos.backend = CosmosBackend.MOCK
        config.cosmos.device = DeviceType.CPU

        # Reduce clip count for demo
        config.mining.top_k_clips = 5

        print("Configuration:")
        print(f"  Output directory: {config.output_dir}")
        print(f"  Cosmos backend:   {config.cosmos.backend.value}")
        print(f"  Top-K clips:      {config.mining.top_k_clips}")
        print(f"  Conformal:        {'Enabled' if config.enable_conformal else 'Disabled'}")
        print(f"  Parallel workers: {config.parallel_workers}")

        # Step 3: Initialize pipeline
        print_section("Step 3: Initializing Pipeline", "-")

        pipeline = InsurancePipeline(config)
        print("[OK] Pipeline initialized")
        print("  Components loaded:")
        print(f"    - Signal fuser:     {'[OK]' if pipeline.signal_fuser else '[X] (mock mode)'}")
        print(f"    - Cosmos client:    {'[OK]' if pipeline.cosmos_client else '[X] (mock mode)'}")
        print(f"    - Fault assessor:   {'[OK]' if pipeline.fault_assessor else '[X] (mock mode)'}")
        print(f"    - Fraud detector:   {'[OK]' if pipeline.fraud_detector else '[X] (mock mode)'}")
        print(f"    - Conformal:        {'[OK]' if pipeline.conformal_predictor else '[X]'}")

        # Step 4: Process videos
        print_section("Step 4: Processing Videos", "-")

        results = []
        for video_path, description in mock_videos:
            video_id = Path(video_path).stem
            print(f"\nProcessing: {video_id}")
            print(f"Description: {description}")

            result = pipeline.process_video(video_path, video_id=video_id)
            results.append(result)

            if result.success:
                print(f"[OK] Success ({result.processing_time_sec:.2f}s)")
                print(f"  Danger clips: {len(result.danger_clips) if result.danger_clips else 0}")
                print(f"  Assessments:  {len(result.assessments) if result.assessments else 0}")
            else:
                print(f"[X] Failed: {result.error_message}")

        # Step 5: Display results
        print_section("Step 5: Results Summary", "=")

        for result in results:
            if not result.success:
                continue

            print(f"\n{'=' * 60}")
            print(f"Video: {result.video_id}")
            print(f"{'=' * 60}")

            if not result.assessments:
                print("No danger clips detected (safe driving)")
                continue

            # Show top 3 assessments
            for idx, assessment in enumerate(result.assessments[:3], 1):
                print_assessment_summary(assessment, idx)

            if len(result.assessments) > 3:
                print(f"\n... and {len(result.assessments) - 3} more clips")

            # Show output files
            print(f"\n{'-' * 60}")
            print("Output Files:")
            print(f"  JSON:  {result.output_json_path}")
            print(f"  HTML:  {result.output_html_path}")

        # Step 6: Pipeline metrics
        print_section("Step 6: Pipeline Metrics", "=")

        metrics = pipeline.metrics
        print(f"Total Videos:        {metrics.total_videos}")
        print(f"Successful:          {metrics.successful_videos}")
        print(f"Failed:              {metrics.failed_videos}")
        print(f"Total Clips Mined:   {metrics.total_clips_mined}")
        print(f"Total Clips Analyzed: {metrics.total_clips_analyzed}")
        print(f"\nProcessing Time:     {metrics.total_processing_time_sec:.2f}s")

        if metrics.successful_videos > 0:
            avg_time = metrics.total_processing_time_sec / metrics.successful_videos
            print(f"Avg Time per Video:  {avg_time:.2f}s")

        # Step 7: Summary statistics
        print_section("Step 7: Severity Distribution", "-")

        all_assessments = []
        for result in results:
            if result.success and result.assessments:
                all_assessments.extend(result.assessments)

        if all_assessments:
            severity_counts = {}
            priority_counts = {}

            for assessment in all_assessments:
                severity_counts[assessment.severity] = severity_counts.get(assessment.severity, 0) + 1
                priority_counts[assessment.review_priority] = priority_counts.get(assessment.review_priority, 0) + 1

            print("Severity Distribution:")
            for severity in ["HIGH", "MEDIUM", "LOW", "NONE"]:
                count = severity_counts.get(severity, 0)
                if count > 0:
                    pct = 100 * count / len(all_assessments)
                    print(f"  {severity:8s}: {count:2d} ({pct:5.1f}%)")

            print("\nReview Priority Distribution:")
            for priority in ["URGENT", "STANDARD", "LOW_PRIORITY"]:
                count = priority_counts.get(priority, 0)
                if count > 0:
                    pct = 100 * count / len(all_assessments)
                    print(f"  {priority:12s}: {count:2d} ({pct:5.1f}%)")

            # Average scores
            avg_confidence = sum(a.confidence for a in all_assessments) / len(all_assessments)
            avg_fault = sum(a.fault_assessment.fault_ratio for a in all_assessments) / len(all_assessments)
            avg_fraud = sum(a.fraud_risk.risk_score for a in all_assessments) / len(all_assessments)

            print("\nAverage Scores:")
            print(f"  Confidence:   {avg_confidence:.2f}")
            print(f"  Fault Ratio:  {avg_fault:.1f}%")
            print(f"  Fraud Score:  {avg_fraud:.2f}")

        # Final summary
        print_section("Demo Complete!", "=")
        print(f"Results saved to: {output_dir}/")
        print("\nTo view HTML reports:")
        print("  # Windows")
        print(f"  start {output_dir}\\<video_id>\\report.html")
        print("\n  # Linux/Mac")
        print(f"  open {output_dir}/<video_id>/report.html")
        print("\nTo view JSON results:")
        print(f"  cat {output_dir}/<video_id>/results.json | jq .")

    finally:
        # Cleanup temporary directory
        print(f"\nCleaning up temporary files: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Insurance MVP Pipeline Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_pipeline.py
  python demo_pipeline.py --output-dir my_demo/
  python demo_pipeline.py --verbose
""",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="demo_results",
        help="Output directory for demo results (default: demo_results)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Run demo
    run_demo(output_dir=args.output_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
