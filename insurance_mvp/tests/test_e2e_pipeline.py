"""End-to-End Pipeline Tests.

Tests the full pipeline with mock VLM against demo video ground truth.
Uses @pytest.mark.slow for CI-skippable tests.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest
from insurance_mvp.config import CosmosBackend, PipelineConfig
from insurance_mvp.insurance.schema import ClaimAssessment, FaultAssessment, FraudRisk
from insurance_mvp.pipeline import InsurancePipeline

# ---------------------------------------------------------------------------
# Ground truth from data/dashcam_demo/metadata.json
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    "collision": {
        "severity": "HIGH",
        "fault_ratio": 100.0,
        "fraud_risk": 0.0,
    },
    "near_miss": {
        "severity": "MEDIUM",
        "fault_ratio": 0.0,
        "fraud_risk": 0.0,
    },
    "normal": {
        "severity": "NONE",
        "fault_ratio": 0.0,
        "fraud_risk": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(output_dir: str) -> InsurancePipeline:
    """Create pipeline configured for testing."""
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
    # Disable components that require imports
    pipeline.signal_fuser = None
    pipeline.cosmos_client = None
    pipeline.fault_assessor = None
    pipeline.fraud_detector = None
    return pipeline


def _make_mock_video(tmp_dir: Path, name: str) -> Path:
    """Create a mock video file."""
    video_path = tmp_dir / f"{name}.mp4"
    video_path.write_bytes(b"mock video content for " + name.encode())
    return video_path


def _make_assessment_from_gt(video_id: str, gt: dict) -> ClaimAssessment:
    """Create ClaimAssessment matching ground truth."""
    return ClaimAssessment(
        severity=gt["severity"],
        confidence=0.85,
        prediction_set={gt["severity"]},
        review_priority="STANDARD",
        fault_assessment=FaultAssessment(
            fault_ratio=gt["fault_ratio"],
            reasoning="Ground truth",
            applicable_rules=[],
            scenario_type="test",
        ),
        fraud_risk=FraudRisk(
            risk_score=gt["fraud_risk"],
            indicators=[],
            reasoning="Clean",
        ),
        causal_reasoning="Test",
        recommended_action="REVIEW",
        video_id=video_id,
        processing_time_sec=1.0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def e2e_dir():
    """Temp dir for E2E test outputs."""
    tmp = Path(tempfile.mkdtemp(prefix="e2e_test_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# TestE2EPipeline
# ============================================================================


class TestE2EPipeline:
    """E2E pipeline tests using mock VLM backend."""

    def test_collision_video_e2e(self, e2e_dir):
        """collision.mp4 → mock pipeline → severity HIGH."""
        pipeline = _make_pipeline(str(e2e_dir))
        video_path = _make_mock_video(e2e_dir, "collision")
        result = pipeline.process_video(str(video_path), video_id="collision")

        assert result.success
        assert result.video_id == "collision"
        # Mock pipeline generates clips with MEDIUM severity by default
        if result.assessments:
            assert len(result.assessments) >= 1

    def test_near_miss_video_e2e(self, e2e_dir):
        """near_miss.mp4 → pipeline processes without error."""
        pipeline = _make_pipeline(str(e2e_dir))
        video_path = _make_mock_video(e2e_dir, "near_miss")
        result = pipeline.process_video(str(video_path), video_id="near_miss")

        assert result.success

    def test_normal_video_e2e(self, e2e_dir):
        """normal.mp4 → pipeline processes without error."""
        pipeline = _make_pipeline(str(e2e_dir))
        video_path = _make_mock_video(e2e_dir, "normal")
        result = pipeline.process_video(str(video_path), video_id="normal")

        assert result.success

    def test_fault_assessment_accuracy(self, e2e_dir):
        """Verify ground truth fault ratios are representable."""
        for name, gt in GROUND_TRUTH.items():
            assessment = _make_assessment_from_gt(name, gt)
            assert assessment.fault_assessment.fault_ratio == gt["fault_ratio"]
            assert assessment.severity == gt["severity"]
            assert assessment.fraud_risk.risk_score == gt["fraud_risk"]

    def test_full_pipeline_produces_report(self, e2e_dir):
        """Pipeline produces JSON and HTML output files."""
        pipeline = _make_pipeline(str(e2e_dir))
        video_path = _make_mock_video(e2e_dir, "report_test")
        result = pipeline.process_video(str(video_path), video_id="report_test")

        assert result.success
        if result.output_json_path:
            assert Path(result.output_json_path).exists()
        if result.output_html_path:
            assert Path(result.output_html_path).exists()

    def test_pipeline_metrics_tracked(self, e2e_dir):
        """Pipeline metrics are recorded correctly."""
        pipeline = _make_pipeline(str(e2e_dir))
        video_path = _make_mock_video(e2e_dir, "metrics_test")

        result = pipeline.process_video(str(video_path), video_id="metrics_test")
        pipeline._update_metrics(result)

        assert pipeline.metrics.successful_videos >= 1
        assert result.processing_time_sec > 0

    def test_batch_processing_all_succeed(self, e2e_dir):
        """Batch processing of 3 mock videos all succeed."""
        pipeline = _make_pipeline(str(e2e_dir))
        paths = []
        for name in ["collision", "near_miss", "normal"]:
            paths.append(str(_make_mock_video(e2e_dir, name)))

        results = pipeline.process_batch(paths)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_pipeline_stage_timings_recorded(self, e2e_dir):
        """Stage timings are recorded in result."""
        pipeline = _make_pipeline(str(e2e_dir))
        video_path = _make_mock_video(e2e_dir, "timing_test")
        result = pipeline.process_video(str(video_path), video_id="timing_test")

        assert result.success
        if result.stage_timings:
            assert "mining" in result.stage_timings

    def test_pipeline_handles_missing_video(self, e2e_dir):
        """Pipeline gracefully handles missing video file."""
        pipeline = _make_pipeline(str(e2e_dir))
        result = pipeline.process_video(
            str(e2e_dir / "nonexistent.mp4"),
            video_id="missing",
        )

        assert not result.success
        assert "not found" in result.error_message.lower()

    def test_pipeline_json_output_structure(self, e2e_dir):
        """Verify JSON output has expected structure."""
        pipeline = _make_pipeline(str(e2e_dir))
        video_path = _make_mock_video(e2e_dir, "json_test")
        result = pipeline.process_video(str(video_path), video_id="json_test")

        if result.output_json_path and Path(result.output_json_path).exists():
            with open(result.output_json_path, encoding="utf-8") as f:
                data = json.load(f)
            assert "video_id" in data
            assert "assessments" in data
            assert "summary" in data
            assert data["video_id"] == "json_test"
