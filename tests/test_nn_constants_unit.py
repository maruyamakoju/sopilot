"""Unit tests for nn/constants.py."""

from sopilot.nn.constants import GAMMA_MIN, INF


class TestConstants:
    """Verify constant values and invariants."""

    def test_inf_is_large_finite(self):
        assert INF == 1e9
        assert float("inf") > INF

    def test_gamma_min_positive(self):
        assert GAMMA_MIN > 0
        assert GAMMA_MIN == 1e-4

    def test_gamma_min_smaller_than_inf(self):
        assert GAMMA_MIN < INF
