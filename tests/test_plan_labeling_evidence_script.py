from __future__ import annotations

import unittest

from scripts.plan_labeling_evidence import (
    _binomial_tail_at_least,
    _min_labels_for_positive_target,
    _min_n_for_zero_error_ci,
    _wilson_upper_zero_errors,
)


class PlanLabelingEvidenceScriptTests(unittest.TestCase):
    def test_wilson_upper_zero_errors_monotonic(self) -> None:
        u10 = _wilson_upper_zero_errors(10)
        u20 = _wilson_upper_zero_errors(20)
        self.assertIsNotNone(u10)
        self.assertIsNotNone(u20)
        self.assertGreater(float(u10), float(u20))

    def test_min_n_for_target(self) -> None:
        n = _min_n_for_zero_error_ci(0.2)
        self.assertGreaterEqual(n, 16)
        self.assertLessEqual(n, 20)

    def test_binomial_tail_at_least(self) -> None:
        p = _binomial_tail_at_least(n=10, k=3, p=0.2)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_min_labels_for_positive_target(self) -> None:
        n = _min_labels_for_positive_target(
            positives_needed=10,
            prevalence=0.2,
            hit_probability=0.9,
            max_n=1000,
        )
        self.assertGreaterEqual(n, 10)
        p = _binomial_tail_at_least(n=n, k=10, p=0.2)
        self.assertGreaterEqual(p, 0.9)


if __name__ == "__main__":
    unittest.main()
