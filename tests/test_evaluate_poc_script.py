from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.evaluate_poc import _build_gate_config, _parse_sweep_thresholds


def _args(**kwargs):
    base = {
        "gate_profile": None,
        "allow_profile_overrides": False,
        "critical_sweep_auto": False,
        "critical_sweep_values": None,
        "critical_sweep_start": None,
        "critical_sweep_stop": None,
        "critical_sweep_step": 0.05,
        "max_critical_miss_rate": None,
        "max_critical_fpr": None,
        "max_critical_miss_ci95_high": None,
        "max_critical_fpr_ci95_high": None,
        "max_rescore_jitter": None,
        "max_dtw_p90": None,
        "max_drift_critical_score_psi": None,
        "max_drift_score_psi": None,
        "max_drift_dtw_psi": None,
        "max_critical_detected_rate_shift_abs": None,
        "min_completed_jobs": None,
        "min_labels_total_jobs": None,
        "min_labeled_jobs": None,
        "min_critical_positives": None,
        "min_critical_negatives": None,
        "min_coverage_rate": None,
        "min_rescore_pairs": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


class EvaluatePocScriptTests(unittest.TestCase):
    def test_parse_sweep_thresholds_from_values_and_range(self) -> None:
        args = _args(
            critical_sweep_values="0.1, 0.5",
            critical_sweep_start=0.2,
            critical_sweep_stop=0.4,
            critical_sweep_step=0.1,
        )
        values = _parse_sweep_thresholds(args)
        self.assertEqual(values, [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_parse_sweep_thresholds_auto(self) -> None:
        args = _args(critical_sweep_auto=True)
        values = _parse_sweep_thresholds(args)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 1.0)
        self.assertEqual(len(values), 21)

    def test_build_gate_config_blocks_locked_profile_override(self) -> None:
        args = _args(
            gate_profile="research_v1",
            max_critical_fpr=0.25,
            allow_profile_overrides=False,
        )
        with self.assertRaises(SystemExit):
            _build_gate_config(args)

    def test_build_gate_config_allows_locked_profile_override_with_flag(self) -> None:
        args = _args(
            gate_profile="research_v1",
            max_critical_fpr=0.25,
            allow_profile_overrides=True,
        )
        cfg = _build_gate_config(args)
        self.assertEqual(cfg.max_critical_false_positive_rate, 0.25)

    def test_build_gate_config_accepts_drift_overrides(self) -> None:
        args = _args(
            gate_profile="ops_v1",
            max_drift_critical_score_psi=0.2,
            max_drift_score_psi=0.2,
            max_drift_dtw_psi=0.2,
            max_critical_detected_rate_shift_abs=0.1,
        )
        cfg = _build_gate_config(args)
        self.assertEqual(cfg.max_drift_critical_score_psi, 0.2)
        self.assertEqual(cfg.max_critical_detected_rate_shift_abs, 0.1)


if __name__ == "__main__":
    unittest.main()
