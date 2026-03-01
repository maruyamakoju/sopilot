from __future__ import annotations

import unittest

from scripts.extract_error_cases import extract_error_cases


class ExtractErrorCasesScriptTests(unittest.TestCase):
    def test_extract_error_cases_legacy_mode(self) -> None:
        completed_jobs = [
            {
                "id": 1,
                "task_id": "task-a",
                "gold_video_id": 100,
                "trainee_video_id": 201,
                "trainee_site_id": "site-a",
                "score": {
                    "score": 70.0,
                    "metrics": {"dtw_normalized_cost": 0.2, "miss_steps": 1},
                    "deviations": [{"type": "missing_step", "severity": "critical"}],
                    "summary": {"decision": "fail", "decision_reason": "critical deviation detected"},
                },
            },
            {
                "id": 2,
                "task_id": "task-a",
                "gold_video_id": 100,
                "trainee_video_id": 202,
                "trainee_site_id": "site-a",
                "score": {
                    "score": 95.0,
                    "metrics": {"dtw_normalized_cost": 0.01},
                    "deviations": [],
                    "summary": {"decision": "pass"},
                },
            },
            {
                "id": 3,
                "task_id": "task-a",
                "gold_video_id": 100,
                "trainee_video_id": 203,
                "trainee_site_id": "site-b",
                "score": {
                    "score": 92.0,
                    "metrics": {"dtw_normalized_cost": 0.02},
                    "deviations": [],
                    "summary": {"decision": "pass"},
                },
            },
        ]
        labels = {1: False, 2: True, 3: False}
        video_map = {100: "/g/gold.mp4", 201: "/t/a.mp4", 202: "/t/b.mp4", 203: "/t/c.mp4"}

        fp_rows, fn_rows, breakdown = extract_error_cases(
            completed_jobs=completed_jobs,
            labels=labels,
            video_path_by_id=video_map,
            scoring_mode="legacy_binary",
            critical_threshold=0.5,
            backend="color-motion",
        )

        self.assertEqual(len(fp_rows), 1)
        self.assertEqual(len(fn_rows), 1)
        self.assertEqual(fp_rows[0]["job_id"], 1)
        self.assertEqual(fn_rows[0]["job_id"], 2)
        self.assertEqual(breakdown["overall"]["false_positive_jobs"], 1)
        self.assertEqual(breakdown["overall"]["false_negative_jobs"], 1)
        self.assertEqual(breakdown["overall"]["false_positive_rate"], 0.5)
        self.assertEqual(breakdown["overall"]["miss_rate"], 1.0)

    def test_extract_error_cases_continuous_breakdown(self) -> None:
        completed_jobs = [
            {
                "id": 11,
                "task_id": "task-a",
                "gold_video_id": 100,
                "trainee_video_id": 301,
                "trainee_site_id": "site-a",
                "score": {
                    "score": 88.0,
                    "metrics": {"dtw_normalized_cost": 0.02},
                    "deviations": [{"type": "step_deviation", "severity": "quality"}],
                },
            },
            {
                "id": 12,
                "task_id": "task-a",
                "gold_video_id": 100,
                "trainee_video_id": 302,
                "trainee_site_id": "site-a",
                "score": {
                    "score": 90.0,
                    "metrics": {"dtw_normalized_cost": 0.02},
                    "deviations": [{"type": "step_deviation", "severity": "quality"}],
                },
            },
            {
                "id": 13,
                "task_id": "task-a",
                "gold_video_id": 101,
                "trainee_video_id": 303,
                "trainee_site_id": "site-b",
                "score": {
                    "score": 98.0,
                    "metrics": {"dtw_normalized_cost": 0.01},
                    "deviations": [],
                },
            },
        ]
        labels = {11: False, 12: False, 13: False}
        video_map = {100: "/g/100.mp4", 101: "/g/101.mp4", 301: "/t/1.mp4", 302: "/t/2.mp4", 303: "/t/3.mp4"}

        fp_rows, fn_rows, breakdown = extract_error_cases(
            completed_jobs=completed_jobs,
            labels=labels,
            video_path_by_id=video_map,
            scoring_mode="continuous_v1",
            critical_threshold=0.2,
            backend="vjepa2",
        )

        self.assertEqual(len(fn_rows), 0)
        self.assertEqual(len(fp_rows), 2)
        self.assertEqual(breakdown["overall"]["false_positive_rate"], round(2 / 3, 6))

        by_dev = {row["deviation_type"]: row for row in breakdown["by_deviation_type"]}
        self.assertIn("step_deviation", by_dev)
        self.assertEqual(by_dev["step_deviation"]["false_positive_rate"], 1.0)
        by_site = {row["site_id"]: row for row in breakdown["by_site_id"]}
        self.assertEqual(by_site["site-a"]["false_positive_rate"], 1.0)
        self.assertEqual(by_site["site-b"]["false_positive_rate"], 0.0)
        by_reason = {row["fp_reason"]: row for row in breakdown["by_fp_reason"]}
        self.assertIn("non_missing_critical", by_reason)

    def test_extract_error_cases_assigns_reason_tags(self) -> None:
        completed_jobs = [
            {
                "id": 21,
                "task_id": "task-a",
                "gold_video_id": 100,
                "trainee_video_id": 401,
                "trainee_site_id": "site-a",
                "score": {
                    "score": 70.0,
                    "metrics": {"dtw_normalized_cost": 0.14, "miss_steps": 1},
                    "deviations": [
                        {
                            "type": "missing_step",
                            "severity": "critical",
                            "mean_distance": 0.13,
                            "expected_span_len": 1,
                        }
                    ],
                },
            }
        ]
        labels = {21: False}
        fp_rows, fn_rows, breakdown = extract_error_cases(
            completed_jobs=completed_jobs,
            labels=labels,
            video_path_by_id={100: "/g/100.mp4", 401: "/t/401.mp4"},
            scoring_mode="guarded_binary_v1",
            critical_threshold=0.5,
            backend="color-motion",
        )
        self.assertEqual(len(fp_rows), 1)
        self.assertEqual(len(fn_rows), 0)
        self.assertEqual(fp_rows[0]["fp_reason_auto"], "high_distance_alignment_mismatch")
        self.assertIn("mean_distance_gt_0_11", fp_rows[0]["fp_reason_tags"])
        reason_rows = {row["fp_reason"]: row for row in breakdown["by_fp_reason"]}
        self.assertEqual(reason_rows["high_distance_alignment_mismatch"]["false_positive_jobs"], 1)


if __name__ == "__main__":
    unittest.main()
