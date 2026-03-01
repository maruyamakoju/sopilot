from __future__ import annotations

import unittest

from scripts.evaluate_shadow_candidate import _shadow_diff


class EvaluateShadowCandidateScriptTests(unittest.TestCase):
    def test_shadow_diff_counts(self) -> None:
        completed_jobs = [
            {
                "id": 1,
                "score": {
                    "metrics": {"dtw_normalized_cost": 0.01},
                    "deviations": [{"type": "missing_step", "severity": "critical"}],
                },
            },
            {
                "id": 2,
                "score": {
                    "metrics": {"dtw_normalized_cost": 0.08},
                    "deviations": [{"type": "missing_step", "severity": "critical"}],
                },
            },
            {
                "id": 3,
                "score": {
                    "metrics": {"dtw_normalized_cost": 0.03},
                    "deviations": [],
                },
            },
        ]

        diff = _shadow_diff(
            completed_jobs=completed_jobs,
            baseline_mode="legacy_binary",
            baseline_threshold=0.5,
            baseline_policy=None,
            candidate_mode="guarded_binary_v2",
            candidate_threshold=0.5,
            candidate_policy=None,
        )
        self.assertEqual(diff["both_detected"], 1)
        self.assertEqual(diff["baseline_only"], 1)
        self.assertEqual(diff["candidate_only"], 0)
        self.assertEqual(diff["neither_detected"], 1)
        self.assertEqual(diff["changed_job_ids"], [1])


if __name__ == "__main__":
    unittest.main()
