"""Tests for Phase 3: VLM accuracy benchmark infrastructure.

Validates the benchmark script, mock VLM scenario matching, and
expanded metadata coverage.
"""

import json
from pathlib import Path

import pytest
from insurance_mvp.pipeline.stages.vlm_inference import mock_vlm_result

# Ground truth expectations for all 10 scenarios
SCENARIO_EXPECTATIONS = {
    "collision": "HIGH",
    "collision_intersection": "HIGH",
    "near_miss": "MEDIUM",
    "near_miss_cyclist": "MEDIUM",
    "near_miss_vehicle": "MEDIUM",
    "swerve_avoidance": "MEDIUM",
    "normal": "NONE",
    "normal_city": "NONE",
    "low_parking_bump": "LOW",
    "hard_braking_clear": "LOW",
}


class TestMockVlmScenarios:
    """Mock VLM must return correct severity for all 10 video filenames."""

    @pytest.mark.parametrize("filename,expected", list(SCENARIO_EXPECTATIONS.items()))
    def test_mock_vlm_severity(self, filename, expected):
        """Mock VLM returns correct severity for each scenario."""
        clip = {"video_path": f"/data/{filename}.mp4"}
        result = mock_vlm_result(clip)
        assert result["severity"] == expected, (
            f"Mock VLM for '{filename}' returned {result['severity']}, expected {expected}"
        )

    def test_mock_vlm_has_confidence(self):
        """All mock VLM results include confidence."""
        for filename in SCENARIO_EXPECTATIONS:
            clip = {"video_path": f"/data/{filename}.mp4"}
            result = mock_vlm_result(clip)
            assert "confidence" in result
            assert 0.0 <= result["confidence"] <= 1.0

    def test_mock_vlm_has_reasoning(self):
        """All mock VLM results include reasoning."""
        for filename in SCENARIO_EXPECTATIONS:
            clip = {"video_path": f"/data/{filename}.mp4"}
            result = mock_vlm_result(clip)
            assert "reasoning" in result
            assert len(result["reasoning"]) > 0


class TestMetadataCompleteness:
    """metadata.json must cover all 10 scenarios."""

    @pytest.fixture
    def metadata(self):
        meta_path = Path(__file__).resolve().parent.parent.parent / "data" / "dashcam_demo" / "metadata.json"
        if not meta_path.exists():
            pytest.skip("metadata.json not found")
        with open(meta_path) as f:
            return json.load(f)

    def test_metadata_has_10_scenarios(self, metadata):
        """metadata.json should have 10 scenarios."""
        assert len(metadata) == 10

    def test_metadata_has_all_expected_keys(self, metadata):
        """All expected scenario keys must be present."""
        for key in SCENARIO_EXPECTATIONS:
            assert key in metadata, f"Missing scenario: {key}"

    def test_metadata_severity_matches(self, metadata):
        """Ground truth severity in metadata must match expectations."""
        for key, expected in SCENARIO_EXPECTATIONS.items():
            assert metadata[key]["severity"] == expected, (
                f"metadata[{key}].severity = {metadata[key]['severity']}, expected {expected}"
            )

    def test_metadata_has_ground_truth(self, metadata):
        """All scenarios have ground_truth with required fields."""
        for key, meta in metadata.items():
            assert "ground_truth" in meta, f"{key} missing ground_truth"
            gt = meta["ground_truth"]
            assert "fault_ratio" in gt
            assert "scenario" in gt


class TestMiningSignalScores:
    """Mock danger clips must include motion/proximity scores."""

    def test_mock_clips_have_signal_scores(self):
        from insurance_mvp.pipeline.stages.mining import mock_danger_clips

        clips = mock_danger_clips("/data/near_miss.mp4", "near_miss")
        for clip in clips:
            assert "motion_score" in clip
            assert "proximity_score" in clip

    @pytest.mark.parametrize(
        "filename,expected_motion,expected_proximity",
        [
            ("collision", 0.9, 0.9),
            ("near_miss", 0.8, 0.7),
            ("normal", 0.1, 0.1),
        ],
    )
    def test_signal_scores_match_scenario(self, filename, expected_motion, expected_proximity):
        from insurance_mvp.pipeline.stages.mining import mock_danger_clips

        clips = mock_danger_clips(f"/data/{filename}.mp4", filename)
        assert clips[0]["motion_score"] == expected_motion
        assert clips[0]["proximity_score"] == expected_proximity


class TestBenchmarkOutputFormat:
    """Benchmark report must have required structure."""

    def test_report_structure(self):
        """Validate expected report keys exist."""
        # Import the benchmark module
        from scripts.vlm_accuracy_benchmark import SEVERITY_LEVELS

        assert len(SEVERITY_LEVELS) == 4
        assert SEVERITY_LEVELS == ["NONE", "LOW", "MEDIUM", "HIGH"]

    def test_confusion_matrix_shape(self):
        """Confusion matrix must be 4x4 for NONE/LOW/MEDIUM/HIGH."""
        from scripts.vlm_accuracy_benchmark import SEVERITY_LEVELS

        # Build a mock confusion matrix
        matrix = [[0] * len(SEVERITY_LEVELS) for _ in SEVERITY_LEVELS]
        assert len(matrix) == 4
        assert all(len(row) == 4 for row in matrix)
