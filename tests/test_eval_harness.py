import json
import tempfile
import unittest
from pathlib import Path

from sopilot.eval.harness import (
    available_critical_scoring_modes,
    compute_critical_score,
    compute_critical_threshold_sweep,
    compute_poc_metrics,
    load_critical_labels,
    parse_critical_labels,
    recommend_threshold_from_sweep,
)


class EvalHarnessTests(unittest.TestCase):
    def test_available_scoring_modes(self) -> None:
        modes = available_critical_scoring_modes()
        self.assertIn("legacy_binary", modes)
        self.assertIn("continuous_v1", modes)
        self.assertIn("guarded_binary_v1", modes)
        self.assertIn("guarded_binary_v2", modes)

    def test_compute_metrics_with_labels(self) -> None:
        completed = [
            {
                "id": 1,
                "gold_video_id": 10,
                "trainee_video_id": 20,
                "score": {
                    "score": 72.0,
                    "metrics": {"dtw_normalized_cost": 0.41},
                    "deviations": [{"type": "missing_step", "severity": "critical"}],
                },
            },
            {
                "id": 2,
                "gold_video_id": 10,
                "trainee_video_id": 21,
                "score": {
                    "score": 94.0,
                    "metrics": {"dtw_normalized_cost": 0.12},
                    "deviations": [],
                },
            },
            {
                "id": 3,
                "gold_video_id": 10,
                "trainee_video_id": 21,
                "score": {
                    "score": 91.0,
                    "metrics": {"dtw_normalized_cost": 0.15},
                    "deviations": [],
                },
            },
        ]
        labels = {1: True, 2: False, 3: False}
        report = compute_poc_metrics(completed, labels)
        self.assertEqual(report["num_completed_jobs"], 3)
        self.assertEqual(report["critical_confusion"]["tp"], 1)
        self.assertEqual(report["critical_confusion"]["fp"], 0)
        self.assertEqual(report["critical_confusion"]["fn"], 0)
        self.assertEqual(report["rescore_jitter"]["num_pairs_with_repeats"], 1)
        self.assertGreater(report["rescore_jitter"]["max_delta"], 0.0)

    def test_missing_step_noncritical_is_not_detected_critical(self) -> None:
        completed = [
            {
                "id": 1,
                "gold_video_id": 10,
                "trainee_video_id": 20,
                "score": {
                    "score": 80.0,
                    "metrics": {"dtw_normalized_cost": 0.2},
                    "deviations": [{"type": "missing_step", "severity": "quality"}],
                },
            }
        ]
        labels = {1: False}
        report = compute_poc_metrics(completed, labels)
        self.assertEqual(report["critical_confusion"]["fp"], 0)

    def test_parse_critical_labels_skips_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "labels.json"
            path.write_text(
                json.dumps(
                    {
                        "jobs": [
                            {"job_id": 1, "critical_expected": True},
                            {"job_id": 2, "critical_expected": None},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            labels = parse_critical_labels(path)
            self.assertIn(1, labels)
            self.assertTrue(labels[1])
            self.assertNotIn(2, labels)

    def test_load_critical_labels_returns_scope_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "labels.json"
            path.write_text(
                json.dumps(
                    {
                        "jobs": [
                            {"job_id": 1, "critical_expected": True},
                            {"job_id": 2, "critical_expected": None},
                            {"job_id": 3, "critical_expected": False},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            label_set = load_critical_labels(path)
            self.assertEqual(label_set.total_jobs, 3)
            self.assertEqual(label_set.labeled_jobs, 2)
            self.assertEqual(label_set.unknown_jobs, 1)
            self.assertTrue(label_set.labels[1])
            self.assertFalse(label_set.labels[3])

    def test_compute_metrics_reports_coverage_and_confidence(self) -> None:
        completed = [
            {
                "id": 1,
                "gold_video_id": 10,
                "trainee_video_id": 20,
                "score": {
                    "score": 70.0,
                    "metrics": {"dtw_normalized_cost": 0.4},
                    "deviations": [{"type": "missing_step", "severity": "critical"}],
                },
            },
            {
                "id": 2,
                "gold_video_id": 10,
                "trainee_video_id": 21,
                "score": {
                    "score": 90.0,
                    "metrics": {"dtw_normalized_cost": 0.1},
                    "deviations": [],
                },
            },
        ]
        labels = {1: True, 2: False}
        report = compute_poc_metrics(
            completed,
            labels,
            label_scope={"labels_total_jobs": 3, "labels_labeled_jobs": 2, "labels_unknown_jobs": 1},
        )
        self.assertEqual(report["labels_total_jobs"], 3)
        self.assertEqual(report["labels_labeled_jobs"], 2)
        self.assertEqual(report["labels_unknown_jobs"], 1)
        self.assertEqual(report["completed_labeled_jobs"], 2)
        self.assertEqual(report["coverage_rate"], 1.0)
        self.assertEqual(report["critical_positives"], 1)
        self.assertEqual(report["critical_negatives"], 1)
        ci = report["critical_confidence"]["false_positive_rate"]["ci95"]
        self.assertIsNotNone(ci)
        self.assertGreaterEqual(ci["high"], ci["low"])

    def test_compute_critical_score_is_high_for_explicit_critical(self) -> None:
        high = compute_critical_score(
            {
                "deviations": [{"type": "missing_step", "severity": "critical"}],
                "metrics": {"miss_steps": 1, "dtw_normalized_cost": 0.1},
            }
        )
        low = compute_critical_score(
            {
                "deviations": [{"type": "step_deviation", "severity": "quality"}],
                "metrics": {"miss_steps": 0, "dtw_normalized_cost": 0.01},
            }
        )
        self.assertGreater(high, 0.8)
        self.assertLess(low, 0.5)

    def test_compute_metrics_continuous_mode(self) -> None:
        completed = [
            {
                "id": 1,
                "gold_video_id": 10,
                "trainee_video_id": 20,
                "score": {
                    "score": 70.0,
                    "metrics": {"dtw_normalized_cost": 0.4, "miss_steps": 1},
                    "deviations": [{"type": "missing_step", "severity": "critical"}],
                },
            },
            {
                "id": 2,
                "gold_video_id": 10,
                "trainee_video_id": 21,
                "score": {
                    "score": 90.0,
                    "metrics": {"dtw_normalized_cost": 0.1},
                    "deviations": [{"type": "step_deviation", "severity": "quality"}],
                },
            },
        ]
        labels = {1: True, 2: False}
        report = compute_poc_metrics(
            completed,
            labels,
            critical_scoring_mode="continuous_v1",
            critical_threshold=0.9,
        )
        self.assertEqual(report["critical_scoring_mode"], "continuous_v1")
        self.assertEqual(report["critical_threshold"], 0.9)
        self.assertIsNotNone(report["critical_score_stats"]["mean"])
        self.assertEqual(report["critical_confusion"]["tp"], 1)
        self.assertEqual(report["critical_confusion"]["fp"], 0)

    def test_threshold_sweep_and_recommendation(self) -> None:
        completed = [
            {
                "id": 1,
                "gold_video_id": 10,
                "trainee_video_id": 20,
                "score": {
                    "score": 60.0,
                    "metrics": {"dtw_normalized_cost": 0.2, "miss_steps": 1},
                    "deviations": [{"type": "missing_step", "severity": "critical"}],
                },
            },
            {
                "id": 2,
                "gold_video_id": 10,
                "trainee_video_id": 21,
                "score": {
                    "score": 90.0,
                    "metrics": {"dtw_normalized_cost": 0.02},
                    "deviations": [{"type": "step_deviation", "severity": "quality"}],
                },
            },
            {
                "id": 3,
                "gold_video_id": 10,
                "trainee_video_id": 22,
                "score": {
                    "score": 98.0,
                    "metrics": {"dtw_normalized_cost": 0.01},
                    "deviations": [],
                },
            },
        ]
        labels = {1: True, 2: False, 3: False}
        rows = compute_critical_threshold_sweep(
            completed,
            labels,
            thresholds=[0.2, 0.9],
            scoring_mode="continuous_v1",
        )
        self.assertEqual(len(rows), 2)
        low_th = rows[0]
        high_th = rows[1]
        self.assertEqual(low_th["threshold"], 0.2)
        self.assertEqual(high_th["threshold"], 0.9)
        self.assertGreater(low_th["critical_false_positive_rate"], high_th["critical_false_positive_rate"])

        recommended = recommend_threshold_from_sweep(
            rows,
            max_miss_rate=0.05,
            max_false_positive_rate=0.1,
        )
        self.assertIsNotNone(recommended)
        self.assertEqual(recommended["threshold"], 0.9)

    def test_guarded_binary_mode_filters_low_dtw_critical(self) -> None:
        completed = [
            {
                "id": 1,
                "gold_video_id": 10,
                "trainee_video_id": 20,
                "score": {
                    "score": 70.0,
                    "metrics": {"dtw_normalized_cost": 0.02},
                    "deviations": [{"type": "missing_step", "severity": "critical"}],
                },
            },
            {
                "id": 2,
                "gold_video_id": 10,
                "trainee_video_id": 21,
                "score": {
                    "score": 60.0,
                    "metrics": {"dtw_normalized_cost": 0.05},
                    "deviations": [{"type": "missing_step", "severity": "critical"}],
                },
            },
        ]
        labels = {1: False, 2: True}
        report = compute_poc_metrics(
            completed,
            labels,
            critical_scoring_mode="guarded_binary_v1",
            critical_threshold=0.5,
        )
        self.assertEqual(report["critical_confusion"]["fp"], 0)
        self.assertEqual(report["critical_confusion"]["tp"], 1)

    def test_guarded_binary_v2_filters_high_distance_missing_step(self) -> None:
        completed = [
            {
                "id": 1,
                "gold_video_id": 10,
                "trainee_video_id": 20,
                "score": {
                    "score": 60.0,
                    "metrics": {"dtw_normalized_cost": 0.09, "miss_steps": 1},
                    "deviations": [
                        {
                            "type": "missing_step",
                            "severity": "critical",
                            "mean_distance": 0.12,
                            "expected_span_len": 1,
                        }
                    ],
                },
            },
            {
                "id": 2,
                "gold_video_id": 10,
                "trainee_video_id": 21,
                "score": {
                    "score": 60.0,
                    "metrics": {"dtw_normalized_cost": 0.05, "miss_steps": 1},
                    "deviations": [
                        {
                            "type": "missing_step",
                            "severity": "critical",
                            "mean_distance": 0.09,
                            "expected_span_len": 1,
                        }
                    ],
                },
            },
        ]
        labels = {1: False, 2: True}
        report = compute_poc_metrics(
            completed,
            labels,
            critical_scoring_mode="guarded_binary_v2",
            critical_threshold=0.5,
        )
        self.assertEqual(report["critical_confusion"]["fp"], 0)
        self.assertEqual(report["critical_confusion"]["tp"], 1)

    def test_guarded_binary_v2_respects_policy_override(self) -> None:
        completed = [
            {
                "id": 1,
                "gold_video_id": 10,
                "trainee_video_id": 20,
                "score": {
                    "score": 60.0,
                    "metrics": {"dtw_normalized_cost": 0.08, "miss_steps": 1},
                    "deviations": [
                        {
                            "type": "missing_step",
                            "severity": "critical",
                            "mean_distance": 0.12,
                            "expected_span_len": 1,
                        }
                    ],
                },
            }
        ]
        labels = {1: True}
        report_default = compute_poc_metrics(
            completed,
            labels,
            critical_scoring_mode="guarded_binary_v2",
            critical_threshold=0.5,
        )
        self.assertEqual(report_default["critical_confusion"]["fn"], 1)

        report_policy = compute_poc_metrics(
            completed,
            labels,
            critical_scoring_mode="guarded_binary_v2",
            critical_threshold=0.5,
            critical_policy={
                "scoring_mode": "guarded_binary_v2",
                "critical_threshold": 0.5,
                "guardrails": {
                    "guarded_binary_v2": {
                        "min_dtw": 0.025,
                        "max_critical_missing_mean_distance": 0.13,
                        "max_critical_missing_expected_span": 1.5,
                    }
                },
            },
        )
        self.assertEqual(report_policy["critical_confusion"]["tp"], 1)

    def test_compute_metrics_reports_drift_summary(self) -> None:
        completed = []
        for idx in range(60):
            has_critical = idx >= 30
            completed.append(
                {
                    "id": idx + 1,
                    "gold_video_id": 100,
                    "trainee_video_id": 200 + idx,
                    "score": {
                        "score": 95.0 if not has_critical else 70.0,
                        "metrics": {"dtw_normalized_cost": 0.02 if not has_critical else 0.12},
                        "deviations": (
                            []
                            if not has_critical
                            else [{"type": "missing_step", "severity": "critical"}]
                        ),
                    },
                }
            )
        report = compute_poc_metrics(completed, None, critical_scoring_mode="legacy_binary")
        drift = report["drift"]
        self.assertTrue(drift["enabled"])
        self.assertIsNotNone(drift["critical_score_psi"])
        self.assertIsNotNone(drift["score_psi"])
        self.assertIsNotNone(drift["critical_detected_rate_shift_abs"])


if __name__ == "__main__":
    unittest.main()
