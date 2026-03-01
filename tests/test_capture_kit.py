"""Tests for Capture Kit modules: video_quality and deviation_templates.

Covers:
- VideoQualityChecker with synthetic videos (valid, dark, overexposed, short)
- VideoQualityReport serialization (to_dict)
- Custom threshold support
- Non-existent file handling
- annotate_deviation / annotate_deviations for all deviation types and severities
- generate_summary_comment for all decision branches
- Integration: apply_task_policy + make_decision with template annotation
"""
from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from sopilot.core.video_quality import VideoQualityChecker, VideoQualityReport
from sopilot.core.deviation_templates import (
    annotate_deviation,
    annotate_deviations,
    generate_summary_comment,
)
from sopilot.core.score_pipeline import apply_task_policy, make_decision


# ── Helpers ───────────────────────────────────────────────────────


def make_video(colors, fps=8, size=(96, 96), frames_per_color=24):
    """Create a synthetic AVI video file, return path."""
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as f:
        tmp = f.name
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(tmp, fourcc, fps, size)
    for color in colors:
        frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        for _ in range(frames_per_color):
            writer.write(frame)
    writer.release()
    return tmp


# ═══════════════════════════════════════════════════════════════════
# Part 1: VideoQualityChecker tests
# ═══════════════════════════════════════════════════════════════════


class TestVideoQualityValid:
    """A synthetic video with good brightness, sharpness, and stability."""

    @pytest.fixture(autouse=True)
    def setup(self):
        # Create a multi-frame video with moderate brightness (128).
        # Use a textured pattern so Laplacian variance (sharpness) is high.
        self.path = self._make_textured_video()
        yield
        os.unlink(self.path)

    @staticmethod
    def _make_textured_video(fps=8, size=(96, 96), num_frames=48):
        """Create a video with high-contrast texture for realistic sharpness."""
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as f:
            tmp = f.name
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(tmp, fourcc, fps, size)
        rng = np.random.RandomState(42)
        for _ in range(num_frames):
            # Checkerboard-like pattern with noise gives high Laplacian variance
            frame = rng.randint(60, 200, (size[1], size[0], 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return tmp

    def test_overall_pass(self):
        checker = VideoQualityChecker()
        report = checker.check(self.path)
        assert report.overall_pass is True

    def test_all_checks_pass(self):
        checker = VideoQualityChecker()
        report = checker.check(self.path)
        for c in report.checks:
            assert c.passed is True, f"Check '{c.name}' unexpectedly failed"

    def test_no_recommendations(self):
        checker = VideoQualityChecker()
        report = checker.check(self.path)
        assert report.recommendations_ja == []
        assert report.recommendations_en == []

    def test_frame_count_sampled_positive(self):
        checker = VideoQualityChecker()
        report = checker.check(self.path)
        assert report.frame_count_sampled > 0

    def test_duration_and_resolution_present(self):
        checker = VideoQualityChecker()
        report = checker.check(self.path)
        assert report.duration_sec > 0
        assert report.resolution[0] > 0
        assert report.resolution[1] > 0


class TestVideoQualityDark:
    """A very dark video should fail brightness check."""

    @pytest.fixture(autouse=True)
    def setup(self):
        # Mean brightness ~10 (well below default 40)
        self.path = make_video([(10, 10, 10)], frames_per_color=48)
        yield
        os.unlink(self.path)

    def test_overall_fail(self):
        checker = VideoQualityChecker()
        report = checker.check(self.path)
        assert report.overall_pass is False

    def test_brightness_check_fails(self):
        checker = VideoQualityChecker()
        report = checker.check(self.path)
        brightness_checks = [c for c in report.checks if c.name == "brightness"]
        assert len(brightness_checks) == 1
        assert brightness_checks[0].passed is False
        assert brightness_checks[0].value < 40

    def test_recommendations_populated(self):
        checker = VideoQualityChecker()
        report = checker.check(self.path)
        assert len(report.recommendations_ja) > 0
        assert len(report.recommendations_en) > 0


class TestVideoQualityOverexposed:
    """An overexposed (very bright) video should fail brightness check."""

    @pytest.fixture(autouse=True)
    def setup(self):
        # Mean brightness ~245 (well above default 220)
        self.path = make_video([(245, 245, 245)], frames_per_color=48)
        yield
        os.unlink(self.path)

    def test_overall_fail(self):
        checker = VideoQualityChecker()
        report = checker.check(self.path)
        assert report.overall_pass is False

    def test_brightness_check_fails_overexposed(self):
        checker = VideoQualityChecker()
        report = checker.check(self.path)
        brightness_checks = [c for c in report.checks if c.name == "brightness"]
        assert len(brightness_checks) == 1
        assert brightness_checks[0].passed is False
        assert brightness_checks[0].value > 220


class TestVideoQualityTooShort:
    """A video shorter than 3 seconds should fail duration check."""

    @pytest.fixture(autouse=True)
    def setup(self):
        # 8 fps * 8 frames = 1 second (well below default 3s)
        self.path = make_video([(128, 128, 128)], fps=8, frames_per_color=8)
        yield
        os.unlink(self.path)

    def test_overall_fail(self):
        checker = VideoQualityChecker()
        report = checker.check(self.path)
        assert report.overall_pass is False

    def test_duration_check_fails(self):
        checker = VideoQualityChecker()
        report = checker.check(self.path)
        duration_checks = [c for c in report.checks if c.name == "duration"]
        assert len(duration_checks) == 1
        assert duration_checks[0].passed is False
        assert duration_checks[0].value < 3.0


class TestVideoQualityNonExistent:
    """A non-existent file should return a report with readable=False."""

    def test_non_existent_file(self):
        checker = VideoQualityChecker()
        report = checker.check("/this/path/does/not/exist_video.avi")
        assert report.overall_pass is False
        assert len(report.checks) == 1
        assert report.checks[0].name == "readable"
        assert report.checks[0].passed is False
        assert report.frame_count_sampled == 0

    def test_non_existent_recommendations(self):
        checker = VideoQualityChecker()
        report = checker.check("/this/path/does/not/exist_video.avi")
        assert len(report.recommendations_ja) > 0
        assert len(report.recommendations_en) > 0


class TestVideoQualityReportToDict:
    """Verify to_dict() returns all expected keys."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.path = make_video([(128, 128, 128)], frames_per_color=48)
        yield
        os.unlink(self.path)

    def test_top_level_keys(self):
        checker = VideoQualityChecker()
        report = checker.check(self.path)
        d = report.to_dict()
        expected_keys = {
            "overall_pass",
            "checks",
            "recommendations_ja",
            "recommendations_en",
            "frame_count_sampled",
            "duration_sec",
            "resolution",
        }
        assert set(d.keys()) == expected_keys

    def test_check_item_keys(self):
        checker = VideoQualityChecker()
        report = checker.check(self.path)
        d = report.to_dict()
        assert len(d["checks"]) > 0
        check_keys = {"name", "passed", "value", "threshold", "message_ja", "message_en"}
        for item in d["checks"]:
            assert set(item.keys()) == check_keys

    def test_duration_sec_is_float(self):
        checker = VideoQualityChecker()
        report = checker.check(self.path)
        d = report.to_dict()
        assert isinstance(d["duration_sec"], float)

    def test_resolution_is_list(self):
        checker = VideoQualityChecker()
        report = checker.check(self.path)
        d = report.to_dict()
        assert isinstance(d["resolution"], list)
        assert len(d["resolution"]) == 2


class TestVideoQualityCustomThresholds:
    """Verify custom thresholds override defaults."""

    @pytest.fixture(autouse=True)
    def setup(self):
        # A solid gray (128,128,128) video: brightness=128, low sharpness (uniform)
        self.path = make_video([(128, 128, 128)], frames_per_color=48)
        yield
        os.unlink(self.path)

    def test_custom_min_brightness_makes_pass(self):
        # Set min_brightness very high so it fails
        checker = VideoQualityChecker(min_brightness=200.0)
        report = checker.check(self.path)
        brightness_checks = [c for c in report.checks if c.name == "brightness"]
        assert len(brightness_checks) == 1
        assert brightness_checks[0].passed is False

    def test_custom_max_brightness_makes_fail(self):
        # Set max_brightness very low so overexposure triggers
        checker = VideoQualityChecker(max_brightness=100.0)
        report = checker.check(self.path)
        brightness_checks = [c for c in report.checks if c.name == "brightness"]
        assert len(brightness_checks) == 1
        assert brightness_checks[0].passed is False

    def test_custom_min_sharpness_zero_passes(self):
        # Set sharpness threshold to 0 so the uniform video passes
        checker = VideoQualityChecker(min_sharpness=0.0)
        report = checker.check(self.path)
        sharpness_checks = [c for c in report.checks if c.name == "sharpness"]
        assert len(sharpness_checks) == 1
        assert sharpness_checks[0].passed is True

    def test_custom_min_duration_long(self):
        # Set min_duration high so the video fails
        checker = VideoQualityChecker(min_duration_sec=999.0)
        report = checker.check(self.path)
        duration_checks = [c for c in report.checks if c.name == "duration"]
        assert len(duration_checks) == 1
        assert duration_checks[0].passed is False

    def test_custom_min_resolution_high(self):
        # Video is 96x96; set min_resolution > 96 to trigger failure
        checker = VideoQualityChecker(min_resolution=200)
        report = checker.check(self.path)
        res_checks = [c for c in report.checks if c.name == "resolution"]
        assert len(res_checks) == 1
        assert res_checks[0].passed is False


# ═══════════════════════════════════════════════════════════════════
# Part 2: deviation_templates tests
# ═══════════════════════════════════════════════════════════════════


class TestAnnotateDeviationByType:
    """Test annotate_deviation produces correct templates for each type."""

    def test_missing_step(self):
        dev = {"type": "missing_step", "step_index": 0, "severity": "critical"}
        result = annotate_deviation(dev)
        assert "手順1" in result["comment_ja"]
        assert "実施されていません" in result["comment_ja"]
        assert "Step 1" in result["comment_en"]
        assert "not performed" in result["comment_en"]

    def test_step_deviation(self):
        dev = {"type": "step_deviation", "step_index": 2, "severity": "quality"}
        result = annotate_deviation(dev)
        assert "手順3" in result["comment_ja"]
        assert "基準と異なります" in result["comment_ja"]
        assert "Step 3" in result["comment_en"]
        assert "deviates" in result["comment_en"]

    def test_order_swap(self):
        dev = {"type": "order_swap", "step_index": 1, "severity": "quality"}
        result = annotate_deviation(dev)
        assert "手順2" in result["comment_ja"]
        assert "順序が入れ替わ" in result["comment_ja"]
        assert "Step 2" in result["comment_en"]
        assert "out of order" in result["comment_en"]

    def test_over_time(self):
        dev = {"type": "over_time", "step_index": None, "severity": "efficiency", "over_time_ratio": 0.35}
        result = annotate_deviation(dev)
        assert "35%" in result["comment_ja"]
        assert "超過" in result["comment_ja"]
        assert "35%" in result["comment_en"]
        assert "exceeds" in result["comment_en"]

    def test_unknown_type(self):
        dev = {"type": "something_new", "step_index": 0, "severity": "quality"}
        result = annotate_deviation(dev)
        assert "不明な逸脱" in result["comment_ja"]
        assert "Unknown deviation" in result["comment_en"]


class TestAnnotateDeviationWithStepDefs:
    """Verify step names from step_definitions are used."""

    def test_step_name_from_definitions(self):
        dev = {"type": "missing_step", "step_index": 1, "severity": "critical"}
        step_defs = [
            {"step_index": 0, "name_ja": "準備", "name_en": "Preparation"},
            {"step_index": 1, "name_ja": "消毒", "name_en": "Disinfection"},
            {"step_index": 2, "name_ja": "確認", "name_en": "Verification"},
        ]
        result = annotate_deviation(dev, step_definitions=step_defs)
        assert "消毒" in result["comment_ja"]
        assert "Disinfection" in result["comment_en"]

    def test_fallback_when_step_index_not_in_defs(self):
        dev = {"type": "missing_step", "step_index": 99, "severity": "critical"}
        step_defs = [{"step_index": 0, "name_ja": "準備", "name_en": "Preparation"}]
        result = annotate_deviation(dev, step_definitions=step_defs)
        # Falls back to "手順100" since step_index=99 is not in defs
        assert "手順100" in result["comment_ja"]
        assert "Step 100" in result["comment_en"]

    def test_no_step_definitions(self):
        dev = {"type": "step_deviation", "step_index": 0, "severity": "quality"}
        result = annotate_deviation(dev, step_definitions=None)
        assert "手順1" in result["comment_ja"]
        assert "Step 1" in result["comment_en"]


class TestAnnotateDeviationSeverities:
    """Verify severity labels and descriptions for each severity level."""

    @pytest.mark.parametrize("severity,expected_ja,expected_en_sub", [
        ("critical", "重大", "Critical deviation"),
        ("quality", "品質", "Quality-impacting"),
        ("efficiency", "効率", "Efficiency-related"),
    ])
    def test_severity_labels(self, severity, expected_ja, expected_en_sub):
        dev = {"type": "missing_step", "step_index": 0, "severity": severity}
        result = annotate_deviation(dev)
        assert result["severity_ja"] == expected_ja
        assert expected_en_sub in result["severity_description_en"]

    def test_unknown_severity_passthrough(self):
        dev = {"type": "missing_step", "step_index": 0, "severity": "custom_sev"}
        result = annotate_deviation(dev)
        assert result["severity_ja"] == "custom_sev"
        assert result["severity_description_ja"] == ""
        assert result["severity_description_en"] == ""


class TestAnnotateDeviationKeyPresence:
    """Verify all annotation keys are always present."""

    REQUIRED_KEYS = {"comment_ja", "comment_en", "severity_ja", "severity_description_ja", "severity_description_en"}

    def test_keys_present_after_annotation(self):
        dev = {"type": "missing_step", "step_index": 0, "severity": "critical"}
        annotate_deviation(dev)
        assert self.REQUIRED_KEYS.issubset(dev.keys())

    def test_keys_present_for_unknown_type(self):
        dev = {"type": "unknown_xyz", "severity": "quality"}
        annotate_deviation(dev)
        assert self.REQUIRED_KEYS.issubset(dev.keys())

    def test_keys_present_with_no_step_index(self):
        dev = {"type": "over_time", "severity": "efficiency"}
        annotate_deviation(dev)
        assert self.REQUIRED_KEYS.issubset(dev.keys())


class TestAnnotateDeviations:
    """Test bulk annotation via annotate_deviations."""

    def test_processes_list(self):
        devs = [
            {"type": "missing_step", "step_index": 0, "severity": "critical"},
            {"type": "step_deviation", "step_index": 1, "severity": "quality"},
            {"type": "order_swap", "step_index": 2, "severity": "quality"},
        ]
        result = annotate_deviations(devs)
        assert len(result) == 3
        for dev in result:
            assert "comment_ja" in dev
            assert "comment_en" in dev

    def test_empty_list(self):
        result = annotate_deviations([])
        assert result == []

    def test_in_place_modification(self):
        devs = [{"type": "missing_step", "step_index": 0, "severity": "critical"}]
        returned = annotate_deviations(devs)
        # Should be the same list object (in-place)
        assert returned is devs
        assert "comment_ja" in devs[0]

    def test_with_step_definitions(self):
        devs = [{"type": "missing_step", "step_index": 0, "severity": "critical"}]
        step_defs = [{"step_index": 0, "name_ja": "手洗い", "name_en": "Handwashing"}]
        annotate_deviations(devs, step_definitions=step_defs)
        assert "手洗い" in devs[0]["comment_ja"]
        assert "Handwashing" in devs[0]["comment_en"]


class TestGenerateSummaryComment:
    """Test generate_summary_comment for all decision branches."""

    def test_pass_clean(self):
        """Pass with no quality or critical deviations."""
        result = generate_summary_comment(
            score=85.0,
            decision="pass",
            severity_counts={"critical": 0, "quality": 0, "efficiency": 0},
        )
        assert "85.0" in result["ja"]
        assert "基準を満たしています" in result["ja"]
        assert "Meets the standard" in result["en"]

    def test_pass_with_quality_deviations(self):
        """Pass but with quality deviations noted."""
        result = generate_summary_comment(
            score=72.0,
            decision="pass",
            severity_counts={"critical": 0, "quality": 2, "efficiency": 1},
        )
        assert "合格" in result["ja"]
        assert "品質逸脱2件" in result["ja"]
        assert "効率逸脱1件" in result["ja"]
        assert "Pass" in result["en"]
        assert "2 quality deviation(s)" in result["en"]

    def test_fail_critical(self):
        """Fail due to critical deviations."""
        result = generate_summary_comment(
            score=40.0,
            decision="fail",
            severity_counts={"critical": 3, "quality": 0, "efficiency": 0},
        )
        assert "重大逸脱3件" in result["ja"]
        assert "再教育" in result["ja"]
        assert "3 critical deviation(s)" in result["en"]
        assert "Retraining required" in result["en"]

    def test_fail_without_critical(self):
        """Fail without critical deviations (score-only fail)."""
        result = generate_summary_comment(
            score=30.0,
            decision="fail",
            severity_counts={"critical": 0, "quality": 0, "efficiency": 0},
        )
        assert "不合格" in result["ja"]
        assert "Fail" in result["en"]

    def test_needs_review(self):
        result = generate_summary_comment(
            score=55.0,
            decision="needs_review",
            severity_counts={"critical": 0, "quality": 1, "efficiency": 0},
        )
        assert "管理者レビュー" in result["ja"]
        assert "Supervisor review" in result["en"]

    def test_retrain(self):
        result = generate_summary_comment(
            score=35.0,
            decision="retrain",
            severity_counts={"critical": 0, "quality": 0, "efficiency": 0},
        )
        assert "再訓練" in result["ja"]
        assert "Retraining required" in result["en"]

    def test_returns_ja_and_en_keys(self):
        result = generate_summary_comment(score=50.0, decision="pass")
        assert "ja" in result
        assert "en" in result

    def test_unknown_decision(self):
        """Unknown decision should still return a score line."""
        result = generate_summary_comment(score=50.0, decision="something_else")
        assert "50.0" in result["ja"]
        assert "50.0" in result["en"]


# ═══════════════════════════════════════════════════════════════════
# Part 3: Integration — score pipeline + templates
# ═══════════════════════════════════════════════════════════════════


class TestApplyTaskPolicyAnnotation:
    """After apply_task_policy(), deviations should have comment_ja fields."""

    @staticmethod
    def _base_result(deviations):
        return {
            "score": 70.0,
            "deviations": deviations,
            "metrics": {"over_time_ratio": 0.0},
        }

    @staticmethod
    def _base_profile():
        return {
            "pass_score": 60.0,
            "retrain_score": 50.0,
            "deviation_policy": {
                "missing_step": "critical",
                "step_deviation": "quality",
                "order_swap": "quality",
                "over_time": "efficiency",
            },
        }

    def test_deviations_annotated_after_apply(self):
        result = self._base_result([
            {"type": "missing_step", "step_index": 0},
            {"type": "step_deviation", "step_index": 1},
        ])
        profile = self._base_profile()
        apply_task_policy(
            result,
            profile=profile,
            efficiency_over_time_threshold=0.3,
            default_pass_score=60.0,
            default_retrain_score=50.0,
        )
        for dev in result["deviations"]:
            assert "comment_ja" in dev, f"Deviation missing comment_ja: {dev}"
            assert "comment_en" in dev, f"Deviation missing comment_en: {dev}"
            assert "severity_ja" in dev, f"Deviation missing severity_ja: {dev}"

    def test_step_definitions_from_profile(self):
        result = self._base_result([
            {"type": "missing_step", "step_index": 0},
        ])
        profile = self._base_profile()
        profile["step_definitions"] = [
            {"step_index": 0, "name_ja": "検体採取", "name_en": "Sample Collection"},
        ]
        apply_task_policy(
            result,
            profile=profile,
            efficiency_over_time_threshold=0.3,
            default_pass_score=60.0,
            default_retrain_score=50.0,
        )
        assert "検体採取" in result["deviations"][0]["comment_ja"]
        assert "Sample Collection" in result["deviations"][0]["comment_en"]

    def test_over_time_deviation_annotated(self):
        """When over_time_ratio exceeds threshold, a new deviation is added and annotated."""
        result = self._base_result([])
        result["metrics"]["over_time_ratio"] = 0.5
        profile = self._base_profile()
        apply_task_policy(
            result,
            profile=profile,
            efficiency_over_time_threshold=0.3,
            default_pass_score=60.0,
            default_retrain_score=50.0,
        )
        over_time_devs = [d for d in result["deviations"] if d["type"] == "over_time"]
        assert len(over_time_devs) == 1
        assert "comment_ja" in over_time_devs[0]


class TestMakeDecisionComments:
    """make_decision() returns comment_ja and comment_en in the summary."""

    def test_pass_decision_has_comments(self):
        summary = make_decision(
            score=80.0,
            deviations=[],
            pass_score=60.0,
            retrain_score=50.0,
        )
        assert "comment_ja" in summary
        assert "comment_en" in summary
        assert summary["decision"] == "pass"
        assert "80.0" in summary["comment_ja"]

    def test_fail_critical_decision_has_comments(self):
        summary = make_decision(
            score=80.0,
            deviations=[
                {"type": "missing_step", "step_index": 0, "severity": "critical"},
            ],
            pass_score=60.0,
            retrain_score=50.0,
        )
        assert summary["decision"] == "fail"
        assert "comment_ja" in summary
        assert "comment_en" in summary
        assert "重大逸脱" in summary["comment_ja"]

    def test_needs_review_decision_has_comments(self):
        summary = make_decision(
            score=55.0,
            deviations=[],
            pass_score=60.0,
            retrain_score=50.0,
        )
        assert summary["decision"] == "needs_review"
        assert "管理者レビュー" in summary["comment_ja"]
        assert "Supervisor review" in summary["comment_en"]

    def test_retrain_decision_has_comments(self):
        summary = make_decision(
            score=40.0,
            deviations=[],
            pass_score=60.0,
            retrain_score=50.0,
        )
        assert summary["decision"] == "retrain"
        assert "再訓練" in summary["comment_ja"]
        assert "Retraining required" in summary["comment_en"]
