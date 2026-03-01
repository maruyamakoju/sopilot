from __future__ import annotations

import unittest

from scripts.build_evidence_dossier import _build_readiness, _summary_markdown


def _split_report(*, overall_pass: bool) -> dict:
    gate = {"overall_pass": overall_pass, "checks": []}
    row = {
        "critical_miss_rate": 0.0,
        "critical_false_positive_rate": 0.0,
        "critical_confidence": {
            "miss_rate": {"ci95": {"high": 0.1}},
            "false_positive_rate": {"ci95": {"high": 0.1}},
        },
        "critical_positives": 20,
        "critical_negatives": 200,
        "gates": gate,
    }
    return {
        "task_id": "task-a",
        "split_strategy": "group_trainee",
        "gate_profile": "research_v2",
        "results": {
            "guarded_binary_v2": {
                "full": dict(row),
                "dev": dict(row),
                "test": dict(row),
                "challenge": dict(row),
            }
        },
    }


class BuildEvidenceDossierScriptTests(unittest.TestCase):
    def test_build_readiness_ready(self) -> None:
        readiness = _build_readiness(
            split_report=_split_report(overall_pass=True),
            labeling_plan={
                "deficits": {
                    "test": {"positives_needed": 0, "negatives_needed": 0},
                    "challenge": {"positives_needed": 0, "negatives_needed": 0},
                }
            },
            loso_report={
                "holdout_axis_used": "gold",
                "rows": [
                    {"status": "ok", "overall_pass": True, "axis": "gold", "holdout_group": "gold:1"},
                    {"status": "ok", "overall_pass": True, "axis": "gold", "holdout_group": "gold:2"},
                ],
            },
            hash_results=[{"path": "a.json", "verified": True}],
        )
        self.assertEqual(readiness["status"], "ready_for_partner_review")
        self.assertEqual(readiness["num_blockers"], 0)

    def test_build_readiness_blocked(self) -> None:
        readiness = _build_readiness(
            split_report=_split_report(overall_pass=False),
            labeling_plan={
                "deficits": {
                    "test": {"positives_needed": 5, "negatives_needed": 0, "suggested_labels_to_add": 100},
                    "challenge": {"positives_needed": 3, "negatives_needed": 0, "suggested_labels_to_add": 80},
                }
            },
            loso_report={"rows": [{"status": "skipped_empty_dev"}]},
            hash_results=[{"path": "a.json", "verified": False}],
        )
        self.assertEqual(readiness["status"], "not_ready")
        self.assertGreaterEqual(readiness["num_blockers"], 3)

    def test_summary_markdown_includes_readiness(self) -> None:
        text = _summary_markdown(
            _split_report(overall_pass=True),
            readiness={
                "status": "not_ready",
                "score_0_to_100": 70,
                "num_blockers": 2,
                "blockers": [{"type": "evidence_deficit", "split": "test"}],
            },
        )
        self.assertIn("readiness_status", text)
        self.assertIn("evidence_deficit", text)


if __name__ == "__main__":
    unittest.main()
