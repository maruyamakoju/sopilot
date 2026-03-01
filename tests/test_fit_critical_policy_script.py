from __future__ import annotations

import unittest

from scripts.fit_critical_policy import _fit_constraints, _parse_grid
from sopilot.eval.gates import GateConfig


class FitCriticalPolicyScriptTests(unittest.TestCase):
    def test_parse_grid(self) -> None:
        values = _parse_grid("0.1, 0.2, 0.1, -1, 0.3")
        self.assertEqual(values, [0.1, 0.2, 0.3])

    def test_fit_constraints(self) -> None:
        report = {
            "critical_miss_rate": 0.01,
            "critical_false_positive_rate": 0.05,
            "critical_confidence": {
                "miss_rate": {"ci95": {"high": 0.04}},
                "false_positive_rate": {"ci95": {"high": 0.09}},
            },
        }
        config = GateConfig(
            max_critical_miss_rate=0.05,
            max_critical_false_positive_rate=0.10,
            max_critical_miss_rate_ci95_high=0.10,
            max_critical_false_positive_rate_ci95_high=0.20,
        )
        feasible, violation = _fit_constraints(report=report, gate_config=config, use_ci_constraints=True)
        self.assertTrue(feasible)
        self.assertEqual(violation, 0.0)

        bad_report = {
            "critical_miss_rate": 0.2,
            "critical_false_positive_rate": 0.3,
            "critical_confidence": {
                "miss_rate": {"ci95": {"high": 0.4}},
                "false_positive_rate": {"ci95": {"high": 0.5}},
            },
        }
        feasible_bad, violation_bad = _fit_constraints(
            report=bad_report,
            gate_config=config,
            use_ci_constraints=True,
        )
        self.assertFalse(feasible_bad)
        self.assertGreater(violation_bad, 0.0)

    def test_fit_constraints_no_ci(self) -> None:
        report = {
            "critical_miss_rate": 0.03,
            "critical_false_positive_rate": 0.04,
        }
        config = GateConfig(
            max_critical_miss_rate=0.05,
            max_critical_false_positive_rate=0.10,
            max_critical_miss_rate_ci95_high=0.02,
            max_critical_false_positive_rate_ci95_high=0.02,
        )
        feasible, _ = _fit_constraints(report=report, gate_config=config, use_ci_constraints=False)
        self.assertTrue(feasible)


if __name__ == "__main__":
    unittest.main()
