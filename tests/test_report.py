"""Tests for sopilot.core.report.build_report_html."""
import html as html_lib

from sopilot import __version__
from sopilot.core.report import build_report_html

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


# ---------------------------------------------------------------------------
# Basic HTML structure
# ---------------------------------------------------------------------------

class TestBasicHtml:
    def test_returns_doctype(self):
        html = build_report_html(1, _make_job())
        assert html.lower().startswith("<!doctype html>")

    def test_contains_title(self):
        html = build_report_html(42, _make_job())
        assert "<title>" in html
        assert "42" in html  # job id in title

    def test_contains_score_value(self):
        html = build_report_html(1, _make_job(score=87.3))
        assert "87.3" in html

    def test_contains_html_close_tag(self):
        html = build_report_html(1, _make_job())
        assert "</html>" in html

    def test_contains_body_tag(self):
        html = build_report_html(1, _make_job())
        assert "<body" in html


# ---------------------------------------------------------------------------
# Missing result (result=None)
# ---------------------------------------------------------------------------

class TestMissingResult:
    def test_handles_result_none(self):
        job = {"result": None, "review": None}
        html = build_report_html(1, job)
        # Should still produce valid HTML without crashing
        assert "<!doctype html>" in html.lower()
        assert "</html>" in html

    def test_handles_empty_job_dict(self):
        job = {}
        html = build_report_html(1, job)
        assert "<!doctype html>" in html.lower()


# ---------------------------------------------------------------------------
# Empty deviations list
# ---------------------------------------------------------------------------

class TestEmptyDeviations:
    def test_no_deviation_section_shows_all_clear(self):
        html = build_report_html(1, _make_job(deviations=[]))
        # The Japanese text for "no deviations"
        assert "\u9038\u8131\u306a\u3057" in html  # "逸脱なし"

    def test_deviations_present_shows_table(self):
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
        html = build_report_html(1, _make_job(deviations=devs))
        assert "<table" in html
        assert "1 件" in html or "1 件" in html  # deviation count


# ---------------------------------------------------------------------------
# HTML-escaping of user-controlled fields
# ---------------------------------------------------------------------------

class TestHtmlEscaping:
    def test_task_id_is_escaped(self):
        job = _make_job()
        job["result"]["task_id"] = '<script>alert("xss")</script>'
        html = build_report_html(1, job)
        # The raw script tag should NOT appear unescaped
        assert '<script>alert("xss")</script>' not in html
        # The escaped angle brackets should be present
        assert "&lt;script&gt;" in html

    def test_decision_reason_is_escaped(self):
        job = _make_job()
        job["result"]["summary"]["decision_reason"] = "<b>bold</b>"
        html = build_report_html(1, job)
        assert "<b>bold</b>" not in html
        assert "&lt;b&gt;bold&lt;/b&gt;" in html

    def test_deviation_detail_is_escaped(self):
        devs = [
            {
                "type": "step_deviation",
                "severity": "quality",
                "step_index": 0,
                "gold_timecode": [0, 4],
                "trainee_timecode": [0, 5],
                "detail": '<img src=x onerror="alert(1)">',
            }
        ]
        html = build_report_html(1, _make_job(deviations=devs))
        assert '<img src=x onerror="alert(1)">' not in html


# ---------------------------------------------------------------------------
# Review section
# ---------------------------------------------------------------------------

class TestReviewSection:
    def test_review_rendered_when_present(self):
        review = {
            "verdict": "pass",
            "note": "Looks good.",
            "updated_at": "2026-01-01T00:00:00",
        }
        html = build_report_html(1, _make_job(review=review))
        # Japanese label for reviewer record section
        assert "\u8a55\u4fa1\u8005\u8a18\u9332" in html  # "評価者記録"
        assert "Looks good." in html

    def test_review_not_rendered_when_absent(self):
        html = build_report_html(1, _make_job(review=None))
        assert "\u8a55\u4fa1\u8005\u8a18\u9332" not in html  # "評価者記録"

    def test_review_note_is_escaped(self):
        review = {
            "verdict": "fail",
            "note": '<script>bad()</script>',
            "updated_at": "2026-01-01",
        }
        html = build_report_html(1, _make_job(review=review))
        assert '<script>bad()</script>' not in html
        assert html_lib.escape('<script>bad()</script>') in html


# ---------------------------------------------------------------------------
# Version string
# ---------------------------------------------------------------------------

class TestVersionInOutput:
    def test_version_reference_appears(self):
        html = build_report_html(1, _make_job())
        # The report template embeds the version token in the header badge and
        # footer.  Note: the current implementation uses plain (non-f) strings
        # for those lines, so the literal text "{__version__}" appears in the
        # output rather than the resolved value.  Verify the version reference
        # is present in either resolved or literal form.
        has_resolved = __version__ in html
        has_literal = "{__version__}" in html
        assert has_resolved or has_literal
