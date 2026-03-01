from __future__ import annotations

import unittest

from scripts.run_backend_ablation import (
    _extract_coverage_and_confusion,
    _resolve_report_controls,
    build_markdown_report,
    summarize_runs,
)


class BackendAblationScriptTests(unittest.TestCase):
    def test_summarize_only_controls_use_existing_and_run_repeats(self) -> None:
        runs = [
            {"backend": "color-motion", "repeat": 1},
            {"backend": "color-motion", "repeat": 2},
            {"backend": "color-motion", "repeat": 5},
        ]
        existing = {
            "repeats": 5,
            "base_seed": 1729,
            "vjepa2_pooling": "first_token",
            "critical_patterns": ["_bad_freeze"],
            "skip_generate_bad": True,
            "skip_prefill_labels": True,
            "disable_embedder_fallback": True,
            "fail_on_gate": False,
        }
        controls = _resolve_report_controls(
            summarize_only=True,
            runs=runs,
            existing_report=existing,
            args_repeats=1,
            args_base_seed=9999,
            args_vjepa2_pooling="mean_tokens",
            args_critical_patterns=["_bad_skip_start"],
            args_skip_generate_bad=False,
            args_skip_prefill_labels=False,
            args_disable_embedder_fallback=False,
            args_fail_on_gate=True,
        )
        self.assertEqual(controls["repeats"], 5)
        self.assertEqual(controls["base_seed"], 1729)
        self.assertEqual(controls["vjepa2_pooling"], "first_token")
        self.assertEqual(controls["critical_patterns"], ["_bad_freeze"])
        self.assertTrue(controls["skip_generate_bad"])
        self.assertTrue(controls["skip_prefill_labels"])
        self.assertTrue(controls["disable_embedder_fallback"])
        self.assertFalse(controls["fail_on_gate"])

    def test_coverage_uses_confusion_counts_even_with_rescore_total(self) -> None:
        gate_payload = {
            "num_completed_jobs": 200,  # includes rescore duplicates
            "critical_confusion": {"tp": 2, "tn": 8, "fp": 1, "fn": 2},
        }
        local_summary = {"scored_total": 13, "scored_completed": 200}
        label_scope = {"labels_total_jobs": 13, "labels_labeled_jobs": 13, "labels_unknown_jobs": 0}

        coverage = _extract_coverage_and_confusion(
            gate_payload=gate_payload,
            local_summary=local_summary,
            label_scope=label_scope,
        )

        self.assertEqual(coverage["expected_job_count"], 13)
        self.assertEqual(coverage["completed_job_count"], 13)
        self.assertEqual(coverage["completed_score_jobs_total"], 200)
        self.assertEqual(coverage["coverage_rate"], 1.0)

    def test_summarize_runs_and_pairwise(self) -> None:
        runs = [
            {
                "backend": "color-motion",
                "repeat": 1,
                "returncode": 0,
                "duration_sec": 10.0,
                "gate_metrics": {
                    "overall_pass": True,
                    "critical_false_positive_rate": 0.20,
                    "critical_miss_rate": 0.0,
                    "dtw_p90": 0.1,
                },
            },
            {
                "backend": "color-motion",
                "repeat": 2,
                "returncode": 0,
                "duration_sec": 11.0,
                "gate_metrics": {
                    "overall_pass": True,
                    "critical_false_positive_rate": 0.18,
                    "critical_miss_rate": 0.0,
                    "dtw_p90": 0.11,
                },
            },
            {
                "backend": "vjepa2",
                "repeat": 1,
                "returncode": 0,
                "duration_sec": 14.0,
                "gate_metrics": {
                    "overall_pass": False,
                    "critical_false_positive_rate": 0.30,
                    "critical_miss_rate": 0.0,
                    "dtw_p90": 0.2,
                },
            },
            {
                "backend": "vjepa2",
                "repeat": 2,
                "returncode": 0,
                "duration_sec": 15.0,
                "gate_metrics": {
                    "overall_pass": False,
                    "critical_false_positive_rate": 0.35,
                    "critical_miss_rate": 0.0,
                    "dtw_p90": 0.25,
                },
            },
        ]
        summary = summarize_runs(runs, ["color-motion", "vjepa2"])
        self.assertEqual(summary["by_backend"]["color-motion"]["runs"], 2)
        self.assertEqual(summary["by_backend"]["color-motion"]["pass_runs"], 2)
        self.assertEqual(summary["by_backend"]["vjepa2"]["pass_runs"], 0)
        self.assertIsNotNone(summary["by_backend"]["color-motion"]["std_duration_sec"])
        self.assertIsNotNone(summary["by_backend"]["vjepa2"]["critical_false_positive_rate_ci95"])
        self.assertAlmostEqual(
            summary["pairwise"][0]["mean_delta_fpr_right_minus_left"],
            0.135,
            places=6,
        )
        self.assertAlmostEqual(
            summary["pairwise"][0]["mean_delta_duration_sec_right_minus_left"],
            4.0,
            places=6,
        )
        self.assertEqual(summary["pairwise"][0]["right_wins_fpr_rate"], 0.0)
        self.assertIsNotNone(summary["pairwise"][0]["delta_fpr_ci95"])
        self.assertIsNotNone(summary["pairwise"][0]["sign_test_pvalue_fpr"])

    def test_strict_metrics_exclude_contaminated_runs(self) -> None:
        runs = [
            {
                "backend": "color-motion",
                "repeat": 1,
                "returncode": 0,
                "duration_sec": 10.0,
                "contaminated_by_fallback": False,
                "gate_metrics": {
                    "overall_pass": True,
                    "critical_false_positive_rate": 0.1,
                    "critical_miss_rate": 0.0,
                    "dtw_p90": 0.1,
                },
            },
            {
                "backend": "vjepa2",
                "repeat": 1,
                "returncode": 0,
                "duration_sec": 14.0,
                "contaminated_by_fallback": True,
                "gate_metrics": {
                    "overall_pass": False,
                    "critical_false_positive_rate": 0.5,
                    "critical_miss_rate": 0.0,
                    "dtw_p90": 0.9,
                },
            },
        ]
        summary = summarize_runs(runs, ["color-motion", "vjepa2"])
        self.assertEqual(summary["by_backend"]["vjepa2"]["contaminated_runs"], 1)
        self.assertEqual(summary["by_backend"]["vjepa2"]["strict_runs_with_metrics"], 0)
        self.assertEqual(summary["pairwise"][0]["strict_paired_repeats"], [])

    def test_markdown_contains_backend_rows(self) -> None:
        report = {
            "generated_at_utc": "2026-02-13T00:00:00+00:00",
            "task_id": "task-a",
            "runs": [],
            "backends": ["color-motion", "vjepa2"],
            "summary": {
                "by_backend": {
                    "color-motion": {
                        "runs": 1,
                        "runs_with_metrics": 1,
                        "pass_runs": 1,
                        "nonzero_return_runs": 0,
                        "pass_rate": 1.0,
                        "mean_duration_sec": 10.0,
                        "std_duration_sec": 0.0,
                        "mean_critical_false_positive_rate": 0.2,
                        "std_critical_false_positive_rate": 0.0,
                        "mean_critical_miss_rate": 0.0,
                        "mean_dtw_p90": 0.1,
                        "std_dtw_p90": 0.0,
                    },
                    "vjepa2": {
                        "runs": 1,
                        "runs_with_metrics": 1,
                        "pass_runs": 0,
                        "nonzero_return_runs": 0,
                        "pass_rate": 0.0,
                        "mean_duration_sec": 14.0,
                        "std_duration_sec": 0.0,
                        "mean_critical_false_positive_rate": 0.3,
                        "std_critical_false_positive_rate": 0.0,
                        "mean_critical_miss_rate": 0.0,
                        "mean_dtw_p90": 0.2,
                        "std_dtw_p90": 0.0,
                    },
                },
                "pairwise": [
                    {
                        "left_backend": "color-motion",
                        "right_backend": "vjepa2",
                        "paired_repeats": [1],
                        "mean_delta_fpr_right_minus_left": 0.1,
                        "delta_fpr_ci95": {"low": 0.1, "high": 0.1},
                        "right_wins_fpr_rate": 0.0,
                        "sign_test_pvalue_fpr": 1.0,
                        "mean_delta_dtw_p90_right_minus_left": 0.1,
                        "delta_dtw_p90_ci95": {"low": 0.1, "high": 0.1},
                        "right_wins_dtw_rate": 0.0,
                        "sign_test_pvalue_dtw_p90": 1.0,
                        "mean_delta_duration_sec_right_minus_left": 4.0,
                        "delta_duration_sec_ci95": {"low": 4.0, "high": 4.0},
                        "right_wins_duration_rate": 0.0,
                        "sign_test_pvalue_duration_sec": 1.0,
                    }
                ],
            },
        }
        md = build_markdown_report(report)
        self.assertIn("| color-motion |", md)
        self.assertIn("| vjepa2 |", md)
        self.assertIn("Backend Ablation Report", md)
        self.assertIn("delta_fpr_ci95", md)


if __name__ == "__main__":
    unittest.main()
