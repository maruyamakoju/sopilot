from __future__ import annotations

import unittest

from scripts.evaluate_loso_sweep import _parse_axis_order, _row_markdown, _select_axis


class EvaluateLosoSweepScriptTests(unittest.TestCase):
    def test_parse_axis_order(self) -> None:
        self.assertEqual(_parse_axis_order("site,gold,site"), ["site", "gold"])

    def test_select_axis_fallback(self) -> None:
        axis, meta = _select_axis(
            requested_axis="site",
            auto_fallback_axis=True,
            axis_order=["site", "gold", "trainee"],
            axis_summaries={
                "site": {"groups_total": 1, "groups_holdout_ready": 1},
                "gold": {"groups_total": 4, "groups_holdout_ready": 3},
                "trainee": {"groups_total": 3, "groups_holdout_ready": 1},
            },
            min_groups_for_generalization=2,
        )
        self.assertEqual(axis, "gold")
        self.assertTrue(meta["fallback_used"])

    def test_row_markdown(self) -> None:
        text = _row_markdown(
            [
                {
                    "axis": "site",
                    "holdout_group": "site-a",
                    "site": "site-a",
                    "dev_jobs": 100,
                    "holdout_jobs": 30,
                    "holdout_positives": 10,
                    "holdout_negatives": 20,
                    "miss_rate": 0.0,
                    "fpr": 0.0,
                    "miss_ci95_high": 0.2,
                    "fpr_ci95_high": 0.1,
                    "constraint_violation": 0.0,
                    "status": "ok",
                    "overall_pass": False,
                    "policy_id": "loso-policy-abc",
                }
            ]
        )
        self.assertIn("site-a", text)
        self.assertIn("loso-policy-abc", text)


if __name__ == "__main__":
    unittest.main()
