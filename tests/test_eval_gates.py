import unittest

from sopilot.eval.gates import (
    GateConfig,
    available_gate_profiles,
    evaluate_gates,
    get_gate_profile,
    is_gate_profile_locked,
    merge_gate_config,
)


class EvalGatesTests(unittest.TestCase):
    def test_gate_pass(self) -> None:
        report = {
            "critical_miss_rate": 0.02,
            "critical_false_positive_rate": 0.1,
            "rescore_jitter": {"max_delta": 1.2},
            "dtw_normalized_cost_stats": {"p90": 0.3},
        }
        gates = evaluate_gates(
            report,
            GateConfig(
                max_critical_miss_rate=0.05,
                max_critical_false_positive_rate=0.2,
                max_rescore_jitter=2.0,
                max_dtw_p90=0.4,
            ),
        )
        self.assertTrue(gates["overall_pass"])

    def test_gate_fail(self) -> None:
        report = {
            "critical_miss_rate": 0.2,
            "critical_false_positive_rate": 0.1,
            "rescore_jitter": {"max_delta": 1.2},
            "dtw_normalized_cost_stats": {"p90": 0.3},
        }
        gates = evaluate_gates(report, GateConfig(max_critical_miss_rate=0.05))
        self.assertFalse(gates["overall_pass"])
        failures = [x for x in gates["checks"] if x["enabled"] and not x["pass"]]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["name"], "critical_miss_rate")

    def test_profile_registry(self) -> None:
        profiles = available_gate_profiles()
        self.assertIn("legacy_poc", profiles)
        self.assertIn("research_v1", profiles)
        self.assertIn("research_v2", profiles)
        self.assertIn("ops_v1", profiles)
        legacy = get_gate_profile("legacy_poc")
        self.assertEqual(legacy.max_critical_false_positive_rate, 0.30)
        self.assertTrue(is_gate_profile_locked("research_v1"))
        self.assertFalse(is_gate_profile_locked("research_v2"))
        self.assertFalse(is_gate_profile_locked("ops_v1"))
        ops = get_gate_profile("ops_v1")
        self.assertIsNotNone(ops.max_drift_critical_score_psi)
        self.assertIsNotNone(ops.max_critical_detected_rate_shift_abs)

    def test_merge_gate_config_overrides_profile(self) -> None:
        base = get_gate_profile("legacy_poc")
        merged = merge_gate_config(base, max_critical_false_positive_rate=0.25)
        self.assertEqual(merged.max_critical_false_positive_rate, 0.25)
        self.assertEqual(merged.max_dtw_p90, 0.60)

    def test_research_gate_fails_when_coverage_evidence_is_insufficient(self) -> None:
        report = {
            "critical_miss_rate": 0.0,
            "critical_false_positive_rate": 0.05,
            "critical_confidence": {
                "miss_rate": {"ci95": {"high": 0.05}},
                "false_positive_rate": {"ci95": {"high": 0.08}},
            },
            "rescore_jitter": {"max_delta": 0.1, "num_pairs_with_repeats": 20},
            "dtw_normalized_cost_stats": {"p90": 0.05},
            "num_completed_jobs": 150,
            "labels_total_jobs": 150,
            "labels_labeled_jobs": 150,
            "critical_positives": 10,
            "critical_negatives": 90,
            "coverage_rate": 0.90,
        }
        gates = evaluate_gates(report, get_gate_profile("research_v1"))
        self.assertFalse(gates["overall_pass"])
        failed_names = {item["name"] for item in gates["checks"] if item["enabled"] and not item["pass"]}
        self.assertIn("num_completed_jobs", failed_names)
        self.assertIn("coverage_rate", failed_names)
        self.assertIn("rescore_pairs", failed_names)

    def test_gate_fails_when_ci95_upper_bound_is_too_high(self) -> None:
        report = {
            "critical_miss_rate": 0.01,
            "critical_false_positive_rate": 0.1,
            "critical_confidence": {
                "miss_rate": {"ci95": {"high": 0.04}},
                "false_positive_rate": {"ci95": {"high": 0.35}},
            },
            "rescore_jitter": {"max_delta": 0.2, "num_pairs_with_repeats": 200},
            "dtw_normalized_cost_stats": {"p90": 0.1},
            "num_completed_jobs": 1000,
            "labels_total_jobs": 300,
            "labels_labeled_jobs": 300,
            "critical_positives": 60,
            "critical_negatives": 200,
            "coverage_rate": 1.0,
        }
        config = GateConfig(max_critical_false_positive_rate_ci95_high=0.2)
        gates = evaluate_gates(report, config)
        self.assertFalse(gates["overall_pass"])
        failed = [item for item in gates["checks"] if item["enabled"] and not item["pass"]]
        self.assertEqual(failed[0]["name"], "critical_false_positive_rate_ci95_high")

    def test_gate_fails_on_drift_threshold_violation(self) -> None:
        report = {
            "rescore_jitter": {"max_delta": 0.1, "num_pairs_with_repeats": 100},
            "dtw_normalized_cost_stats": {"p90": 0.1},
            "drift": {
                "critical_score_psi": 0.4,
                "score_psi": 0.1,
                "dtw_normalized_cost_psi": 0.1,
                "critical_detected_rate_shift_abs": 0.05,
            },
        }
        config = GateConfig(max_drift_critical_score_psi=0.25)
        gates = evaluate_gates(report, config)
        self.assertFalse(gates["overall_pass"])
        failed = [item for item in gates["checks"] if item["enabled"] and not item["pass"]]
        self.assertEqual(failed[0]["name"], "drift_critical_score_psi")


if __name__ == "__main__":
    unittest.main()
