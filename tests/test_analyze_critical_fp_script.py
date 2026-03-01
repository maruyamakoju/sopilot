from __future__ import annotations

import unittest

from scripts.analyze_critical_fp import analyze_fp


class AnalyzeCriticalFpScriptTests(unittest.TestCase):
    def test_analyze_fp_buckets(self) -> None:
        labels_payload = {
            "task_id": "task-a",
            "jobs": [
                {"job_id": 1, "critical_expected": False, "predicted_critical": True},
                {"job_id": 2, "critical_expected": False, "predicted_critical": True},
                {"job_id": 3, "critical_expected": True, "predicted_critical": True},
            ],
        }
        summary_payload = {
            "all_score_jobs": [
                {"job_id": 1, "trainee_video_id": 101},
                {"job_id": 2, "trainee_video_id": 102},
                {"job_id": 3, "trainee_video_id": 103},
            ],
            "video_path_by_id": {
                "101": "/tmp/trainee_001_bad_freeze.mp4",
                "102": "/tmp/trainee_001.mp4",
                "103": "/tmp/trainee_002_bad_cut_tail.mp4",
            },
        }
        report = analyze_fp(labels_payload=labels_payload, summary_payload=summary_payload, sample_limit=10)
        self.assertEqual(report["fp_jobs"], 2)
        self.assertEqual(report["fp_bucket_counts"]["bad_freeze"], 1)
        self.assertEqual(report["fp_bucket_counts"]["normal"], 1)


if __name__ == "__main__":
    unittest.main()

