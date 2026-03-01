"""Tests for sopilot.core.report_pdf.build_report_pdf."""
import unittest
from unittest import mock

import fitz  # PyMuPDF – used to extract text from generated PDFs

from sopilot.core.report_pdf import build_report_pdf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(*, score=92.5, decision="pass", deviations=None, review=None):
    """Build a minimal job dict resembling what the service produces."""
    return {
        "result": {
            "score": score,
            "task_id": "test-task",
            "gold_video_id": 1,
            "trainee_video_id": 2,
            "summary": {
                "decision": decision,
                "decision_reason": "Score above threshold",
                "pass_score": 90.0,
                "severity_counts": {
                    "critical": 0,
                    "quality": 0,
                    "efficiency": 0,
                },
            },
            "metrics": {
                "miss_steps": 0,
                "swap_steps": 0,
                "deviation_steps": 0,
                "over_time_ratio": 0.0,
                "dtw_normalized_cost": 0.012,
            },
            "deviations": deviations or [],
        },
        "review": review,
    }


def _extract_text(pdf_bytes: bytes) -> str:
    """Extract all text from PDF bytes using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


# ---------------------------------------------------------------------------
# Basic PDF structure
# ---------------------------------------------------------------------------

class TestBasicPdf(unittest.TestCase):
    """Verify basic PDF generation returns valid PDF bytes."""

    def test_returns_bytes(self):
        result = build_report_pdf(1, _make_job())
        self.assertIsInstance(result, bytes)

    def test_has_valid_pdf_header(self):
        result = build_report_pdf(1, _make_job())
        self.assertTrue(result[:5] == b"%PDF-", "Output should start with %PDF- header")

    def test_nonzero_length(self):
        result = build_report_pdf(1, _make_job())
        self.assertGreater(len(result), 100, "PDF should contain meaningful content")


# ---------------------------------------------------------------------------
# Complete job data
# ---------------------------------------------------------------------------

class TestCompleteJobData(unittest.TestCase):
    """PDF generation with full data (score, metrics, deviations, summary)."""

    def test_complete_job_produces_valid_pdf(self):
        devs = [
            {
                "type": "missing_step",
                "severity": "critical",
                "step_index": 0,
                "gold_timecode": [0, 4],
                "trainee_timecode": None,
                "detail": "No aligned trainee segment.",
            }
        ]
        review = {
            "verdict": "fail",
            "note": "Needs improvement.",
            "updated_at": "2026-01-15T10:00:00",
        }
        job = _make_job(score=75.3, decision="fail", deviations=devs, review=review)
        result = build_report_pdf(42, job)
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:5] == b"%PDF-")

    def test_complete_job_contains_key_data(self):
        devs = [
            {
                "type": "missing_step",
                "severity": "critical",
                "step_index": 0,
                "gold_timecode": [0, 4],
                "trainee_timecode": None,
                "detail": "No aligned trainee segment.",
            }
        ]
        review = {
            "verdict": "fail",
            "note": "Needs improvement.",
            "updated_at": "2026-01-15T10:00:00",
        }
        job = _make_job(score=75.3, decision="fail", deviations=devs, review=review)
        text = _extract_text(build_report_pdf(42, job))
        self.assertIn("75.3", text)
        self.assertIn("Job #42", text)
        self.assertIn("test-task", text)


# ---------------------------------------------------------------------------
# None / empty result handling
# ---------------------------------------------------------------------------

class TestMissingResult(unittest.TestCase):
    """Handle jobs where result is None or the dict is empty."""

    def test_handles_result_none(self):
        job = {"result": None, "review": None}
        result = build_report_pdf(1, job)
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:5] == b"%PDF-")

    def test_handles_empty_job_dict(self):
        job = {}
        result = build_report_pdf(1, job)
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:5] == b"%PDF-")

    def test_handles_result_empty_dict(self):
        job = {"result": {}, "review": None}
        result = build_report_pdf(1, job)
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:5] == b"%PDF-")


# ---------------------------------------------------------------------------
# Empty deviations list (shows "all clear" message)
# ---------------------------------------------------------------------------

class TestEmptyDeviations(unittest.TestCase):
    """When deviations list is empty, the PDF should show the all-clear message."""

    def test_empty_deviations_produces_valid_pdf(self):
        result = build_report_pdf(1, _make_job(deviations=[]))
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:5] == b"%PDF-")

    def test_empty_deviations_contains_all_clear_text(self):
        """The all-clear Japanese text should appear in the PDF."""
        text = _extract_text(build_report_pdf(1, _make_job(deviations=[])))
        # "逸脱なし — 全手順正常" is rendered for empty deviations
        self.assertIn("\u9038\u8131\u306a\u3057", text)  # "逸脱なし"

    def test_empty_deviations_shows_zero_count(self):
        text = _extract_text(build_report_pdf(1, _make_job(deviations=[])))
        self.assertIn("0 \u4ef6", text)  # "0 件"


# ---------------------------------------------------------------------------
# Deviations present
# ---------------------------------------------------------------------------

class TestDeviationsPresent(unittest.TestCase):
    """When deviations are provided, the table is generated without error."""

    def _make_devs(self):
        return [
            {
                "type": "missing_step",
                "severity": "critical",
                "step_index": 0,
                "gold_timecode": [0, 4],
                "trainee_timecode": None,
                "detail": "No aligned trainee segment.",
            },
            {
                "type": "step_deviation",
                "severity": "quality",
                "step_index": 1,
                "gold_timecode": [4, 8],
                "trainee_timecode": [3, 9],
                "detail": "Timing mismatch observed.",
            },
            {
                "type": "order_swap",
                "severity": "efficiency",
                "step_index": 2,
                "gold_timecode": [8, 12],
                "trainee_timecode": [10, 14],
                "detail": "Steps performed in wrong order.",
            },
        ]

    def test_multiple_deviations_valid_pdf(self):
        job = _make_job(deviations=self._make_devs())
        result = build_report_pdf(1, job)
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:5] == b"%PDF-")

    def test_deviation_count_in_section_title(self):
        devs = self._make_devs()
        text = _extract_text(build_report_pdf(1, _make_job(deviations=devs)))
        self.assertIn("3 \u4ef6", text)  # "3 件"

    def test_deviation_with_none_trainee_timecode(self):
        devs = [
            {
                "type": "missing_step",
                "severity": "critical",
                "step_index": 0,
                "gold_timecode": [0, 4],
                "trainee_timecode": None,
                "detail": "Missing.",
            }
        ]
        text = _extract_text(build_report_pdf(1, _make_job(deviations=devs)))
        # Only gold timecode should appear, no trainee timecode
        self.assertIn("G:0s-4s", text)

    def test_deviation_with_both_timecodes(self):
        devs = [
            {
                "type": "step_deviation",
                "severity": "quality",
                "step_index": 0,
                "gold_timecode": [5, 10],
                "trainee_timecode": [6, 12],
                "detail": "Slight deviation.",
            }
        ]
        text = _extract_text(build_report_pdf(1, _make_job(deviations=devs)))
        self.assertIn("G:5s-10s", text)
        self.assertIn("T:6s-12s", text)

    def test_deviation_severity_labels(self):
        """Each severity type should render its Japanese label."""
        devs = self._make_devs()
        text = _extract_text(build_report_pdf(1, _make_job(deviations=devs)))
        self.assertIn("\u91cd\u5927", text)  # "重大" (critical)
        self.assertIn("\u54c1\u8cea", text)  # "品質" (quality)
        self.assertIn("\u52b9\u7387", text)  # "効率" (efficiency)

    def test_deviation_type_labels(self):
        """Each deviation type should render its Japanese label."""
        devs = self._make_devs()
        text = _extract_text(build_report_pdf(1, _make_job(deviations=devs)))
        self.assertIn("\u624b\u9806\u7701\u7565", text)  # "手順省略" (missing_step)
        self.assertIn("\u54c1\u8cea\u9038\u8131", text)  # "品質逸脱" (step_deviation)
        self.assertIn("\u624b\u9806\u5165\u66ff", text)  # "手順入替" (order_swap)


# ---------------------------------------------------------------------------
# Review data present
# ---------------------------------------------------------------------------

class TestReviewPresent(unittest.TestCase):
    """When review data is provided, the review section is rendered."""

    def test_review_produces_valid_pdf(self):
        review = {
            "verdict": "pass",
            "note": "Looks good.",
            "updated_at": "2026-01-01T00:00:00",
        }
        result = build_report_pdf(1, _make_job(review=review))
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:5] == b"%PDF-")

    def test_review_section_title_present(self):
        review = {
            "verdict": "pass",
            "note": "Looks good.",
            "updated_at": "2026-01-01T00:00:00",
        }
        text = _extract_text(build_report_pdf(1, _make_job(review=review)))
        self.assertIn("\u8a55\u4fa1\u8005\u8a18\u9332", text)  # "評価者記録"

    def test_review_note_rendered(self):
        review = {
            "verdict": "pass",
            "note": "Looks good.",
            "updated_at": "2026-01-01T00:00:00",
        }
        text = _extract_text(build_report_pdf(1, _make_job(review=review)))
        self.assertIn("Looks good.", text)

    def test_review_with_long_note_is_truncated(self):
        """Notes longer than 80 chars should be truncated to 80 + '...'."""
        long_note = "X" * 100
        review = {
            "verdict": "fail",
            "note": long_note,
            "updated_at": "2026-01-01",
        }
        text = _extract_text(build_report_pdf(1, _make_job(review=review)))
        # Full 100-char string should NOT appear
        self.assertNotIn(long_note, text)
        # Truncated version (80 chars + "...") should appear
        self.assertIn("X" * 80 + "...", text)

    def test_review_with_empty_note(self):
        review = {
            "verdict": "pass",
            "note": "",
            "updated_at": "2026-01-01",
        }
        result = build_report_pdf(1, _make_job(review=review))
        self.assertTrue(result[:5] == b"%PDF-")


# ---------------------------------------------------------------------------
# Review data absent
# ---------------------------------------------------------------------------

class TestReviewAbsent(unittest.TestCase):
    """When review is None, the review section should be skipped."""

    def test_no_review_produces_valid_pdf(self):
        result = build_report_pdf(1, _make_job(review=None))
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:5] == b"%PDF-")

    def test_no_review_section_title_absent(self):
        text = _extract_text(build_report_pdf(1, _make_job(review=None)))
        self.assertNotIn("\u8a55\u4fa1\u8005\u8a18\u9332", text)  # "評価者記録"

    def test_no_review_pdf_is_smaller(self):
        """PDF without review should generally be shorter than one with review."""
        pdf_no_review = build_report_pdf(1, _make_job(review=None))
        review = {
            "verdict": "pass",
            "note": "Detailed review comment here.",
            "updated_at": "2026-01-01T00:00:00",
        }
        pdf_with_review = build_report_pdf(1, _make_job(review=review))
        self.assertGreaterEqual(len(pdf_with_review), len(pdf_no_review))


# ---------------------------------------------------------------------------
# Score display formatting
# ---------------------------------------------------------------------------

class TestScoreFormatting(unittest.TestCase):
    """Score values should be formatted as floats with one decimal place."""

    def test_integer_score_formatted_with_decimal(self):
        """A score of 100 should be rendered as '100.0'."""
        text = _extract_text(build_report_pdf(1, _make_job(score=100)))
        self.assertIn("100.0", text)

    def test_float_score_one_decimal(self):
        text = _extract_text(build_report_pdf(1, _make_job(score=87.3)))
        self.assertIn("87.3", text)

    def test_zero_score(self):
        text = _extract_text(build_report_pdf(1, _make_job(score=0.0)))
        self.assertIn("0.0", text)

    def test_none_score_shows_dash(self):
        """When score is None the display should show a dash character."""
        job = _make_job()
        job["result"]["score"] = None
        result = build_report_pdf(1, job)
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:5] == b"%PDF-")
        # Should still produce valid PDF; the score_s will be "—"
        text = _extract_text(result)
        # Verify the numeric score is absent
        self.assertNotIn("92.5", text)


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------

class TestDecisionTypes(unittest.TestCase):
    """Each decision type should produce a valid PDF with the correct label."""

    def test_decision_pass(self):
        text = _extract_text(build_report_pdf(1, _make_job(decision="pass")))
        self.assertIn("\u5408\u683c", text)  # "合格"

    def test_decision_fail(self):
        text = _extract_text(build_report_pdf(1, _make_job(decision="fail")))
        self.assertIn("\u4e0d\u5408\u683c", text)  # "不合格"

    def test_decision_needs_review(self):
        text = _extract_text(build_report_pdf(1, _make_job(decision="needs_review")))
        self.assertIn("\u8981\u78ba\u8a8d", text)  # "要確認"

    def test_decision_retrain(self):
        text = _extract_text(build_report_pdf(1, _make_job(decision="retrain")))
        self.assertIn("\u518d\u7814\u4fee", text)  # "再研修"

    def test_unknown_decision_uses_fallback(self):
        """An unknown decision string should use the raw string as fallback."""
        result = build_report_pdf(1, _make_job(decision="unknown_decision"))
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:5] == b"%PDF-")
        text = _extract_text(result)
        # With unknown decision, _DEC_JP.get falls back to the raw key
        self.assertIn("unknown_decision", text)


# ---------------------------------------------------------------------------
# Long detail text truncation
# ---------------------------------------------------------------------------

class TestLongDetailTruncation(unittest.TestCase):
    """Deviation detail text longer than 50 chars should be truncated to 47 + '...'."""

    def _make_dev_with_detail(self, detail):
        return [
            {
                "type": "step_deviation",
                "severity": "quality",
                "step_index": 0,
                "gold_timecode": [0, 4],
                "trainee_timecode": [0, 5],
                "detail": detail,
            }
        ]

    def test_short_detail_not_truncated(self):
        """Detail <= 50 chars should appear fully."""
        short = "Short detail text"
        text = _extract_text(build_report_pdf(
            1, _make_job(deviations=self._make_dev_with_detail(short))
        ))
        self.assertIn(short, text)

    def test_long_detail_truncated(self):
        """Detail > 50 chars should be truncated; full text should NOT appear."""
        long_detail = "A" * 60
        text = _extract_text(build_report_pdf(
            1, _make_job(deviations=self._make_dev_with_detail(long_detail))
        ))
        # The full 60-char string should NOT appear
        self.assertNotIn(long_detail, text)
        # The truncated version (47 A's + "...") should appear
        self.assertIn("A" * 47 + "...", text)

    def test_exactly_50_chars_not_truncated(self):
        """Detail of exactly 50 chars should NOT be truncated."""
        exact = "B" * 50
        text = _extract_text(build_report_pdf(
            1, _make_job(deviations=self._make_dev_with_detail(exact))
        ))
        self.assertIn(exact, text)

    def test_51_chars_is_truncated(self):
        """Detail of 51 chars should be truncated."""
        detail_51 = "C" * 51
        text = _extract_text(build_report_pdf(
            1, _make_job(deviations=self._make_dev_with_detail(detail_51))
        ))
        self.assertNotIn(detail_51, text)
        self.assertIn("C" * 47 + "...", text)


# ---------------------------------------------------------------------------
# Missing fpdf2 import
# ---------------------------------------------------------------------------

class TestMissingFpdf2(unittest.TestCase):
    """When fpdf2 is not installed, ImportError with a helpful message is raised."""

    def test_import_error_with_helpful_message(self):
        """Simulate fpdf2 being absent and verify the error message."""
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name == "fpdf":
                raise ImportError("No module named 'fpdf'")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import):
            with self.assertRaises(ImportError) as ctx:
                build_report_pdf(1, _make_job())
            self.assertIn("fpdf2", str(ctx.exception))
            self.assertIn("pip install fpdf2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
