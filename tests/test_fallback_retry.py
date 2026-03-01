"""Tests for FallbackEmbedder retry-after-interval logic."""

import unittest
from unittest.mock import patch

import numpy as np

from sopilot.services.embedder import FallbackEmbedder


class _FailingEmbedder:
    name = "primary"

    def embed(self, frames):
        raise RuntimeError("boom")


class _CountingEmbedder:
    name = "primary"

    def __init__(self):
        self.calls = 0

    def embed(self, frames):
        self.calls += 1
        return np.ones(3, dtype=np.float32)


class _WorkingFallback:
    name = "fallback"

    def __init__(self):
        self.calls = 0

    def embed(self, frames):
        self.calls += 1
        return np.zeros(3, dtype=np.float32)


class FallbackRetryTests(unittest.TestCase):
    def test_retries_primary_after_interval(self) -> None:
        """After failover, primary should be retried once interval elapses."""
        primary = _CountingEmbedder()
        fallback = _WorkingFallback()
        embedder = FallbackEmbedder(primary, fallback)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

        # Primary works on first call
        result = embedder.embed(frames)
        self.assertEqual(primary.calls, 1)
        self.assertFalse(embedder._failed_over)
        np.testing.assert_array_equal(result, np.ones(3, dtype=np.float32))

    def test_stays_on_fallback_before_interval(self) -> None:
        """Within the retry interval, fallback should be used."""
        fallback = _WorkingFallback()
        embedder = FallbackEmbedder(_FailingEmbedder(), fallback)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

        # First call triggers failover
        embedder.embed(frames)
        self.assertTrue(embedder._failed_over)
        self.assertEqual(fallback.calls, 1)

        # Second call within interval should use fallback directly
        embedder.embed(frames)
        self.assertEqual(fallback.calls, 2)

    def test_retries_primary_after_interval_elapses(self) -> None:
        """After interval elapses, primary is retried."""
        fallback = _WorkingFallback()
        embedder = FallbackEmbedder(_FailingEmbedder(), fallback)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

        # Trigger failover
        embedder.embed(frames)
        self.assertTrue(embedder._failed_over)

        # Simulate time passing beyond retry interval
        with patch("time.monotonic", return_value=embedder._failed_at + embedder.RETRY_INTERVAL + 1):
            # Primary still fails, so fallback is used, but primary was retried
            embedder.embed(frames)
            self.assertEqual(fallback.calls, 2)
            self.assertTrue(embedder._failed_over)

    def test_recovers_when_primary_heals(self) -> None:
        """If primary recovers after interval, switch back to it."""

        call_count = 0

        class _HealingEmbedder:
            name = "primary"

            def embed(self, frames):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("temporary failure")
                return np.ones(3, dtype=np.float32) * 2.0

        primary = _HealingEmbedder()
        fallback = _WorkingFallback()
        embedder = FallbackEmbedder(primary, fallback)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

        # First call fails, uses fallback
        result1 = embedder.embed(frames)
        self.assertTrue(embedder._failed_over)
        self.assertEqual(embedder.name, "primary->fallback:fallback")
        np.testing.assert_array_equal(result1, np.zeros(3, dtype=np.float32))

        # Simulate time passing
        with patch("time.monotonic", return_value=embedder._failed_at + embedder.RETRY_INTERVAL + 1):
            # Primary heals, should recover
            result2 = embedder.embed(frames)
            self.assertFalse(embedder._failed_over)
            self.assertEqual(embedder.name, "primary")
            np.testing.assert_array_equal(result2, np.ones(3, dtype=np.float32) * 2.0)

    def test_name_updates_on_failover_and_recovery(self) -> None:
        call_count = 0

        class _OnceFailer:
            name = "vjepa2"

            def embed(self, frames):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("fail")
                return np.ones(3, dtype=np.float32)

        primary = _OnceFailer()
        fallback = _WorkingFallback()
        embedder = FallbackEmbedder(primary, fallback)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

        self.assertEqual(embedder.name, "vjepa2")

        embedder.embed(frames)
        self.assertEqual(embedder.name, "vjepa2->fallback:fallback")

        with patch("time.monotonic", return_value=embedder._failed_at + embedder.RETRY_INTERVAL + 1):
            embedder.embed(frames)
            self.assertEqual(embedder.name, "vjepa2")


    def test_exponential_backoff_increases_interval(self) -> None:
        """Each consecutive failure should double the retry interval."""
        fallback = _WorkingFallback()
        embedder = FallbackEmbedder(_FailingEmbedder(), fallback)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

        # First failure: interval = 60s
        embedder.embed(frames)
        self.assertEqual(embedder._consecutive_failures, 1)
        self.assertAlmostEqual(embedder._current_interval, 60.0)

        # Simulate time passing, retry -> fails again: interval = 120s
        with patch("time.monotonic", return_value=embedder._failed_at + 61):
            embedder.embed(frames)
        self.assertEqual(embedder._consecutive_failures, 2)
        self.assertAlmostEqual(embedder._current_interval, 120.0)

        # Retry again -> fails: interval = 240s
        with patch("time.monotonic", return_value=embedder._failed_at + 121):
            embedder.embed(frames)
        self.assertEqual(embedder._consecutive_failures, 3)
        self.assertAlmostEqual(embedder._current_interval, 240.0)

    def test_interval_capped_at_max(self) -> None:
        """Retry interval should not exceed MAX_RETRY_INTERVAL."""
        fallback = _WorkingFallback()
        embedder = FallbackEmbedder(_FailingEmbedder(), fallback)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

        # Trigger many failures
        for _ in range(8):
            with patch("time.monotonic", return_value=embedder._failed_at + embedder._current_interval + 1):
                embedder.embed(frames)
        self.assertLessEqual(embedder._current_interval, embedder.MAX_RETRY_INTERVAL)

    def test_permanent_failure_after_max_retries(self) -> None:
        """After MAX_CONSECUTIVE_FAILURES, should stop retrying primary."""
        fallback = _WorkingFallback()
        embedder = FallbackEmbedder(_FailingEmbedder(), fallback)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

        # Trigger MAX_CONSECUTIVE_FAILURES failures
        for _ in range(embedder.MAX_CONSECUTIVE_FAILURES):
            with patch("time.monotonic", return_value=embedder._failed_at + embedder._current_interval + 1):
                embedder.embed(frames)

        self.assertTrue(embedder._permanently_failed)
        self.assertEqual(embedder._consecutive_failures, embedder.MAX_CONSECUTIVE_FAILURES)

        # Even after a long time, should NOT retry primary
        with patch("time.monotonic", return_value=embedder._failed_at + 999999):
            self.assertFalse(embedder._should_retry())

    def test_recovery_resets_backoff_state(self) -> None:
        """When primary recovers, all backoff state should reset."""
        call_count = 0

        class _EventuallyHeals:
            name = "primary"

            def embed(self, frames):
                nonlocal call_count
                call_count += 1
                if call_count <= 3:
                    raise RuntimeError("failing")
                return np.ones(3, dtype=np.float32)

        primary = _EventuallyHeals()
        fallback = _WorkingFallback()
        embedder = FallbackEmbedder(primary, fallback)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

        # Fail 3 times
        for _ in range(3):
            with patch("time.monotonic", return_value=embedder._failed_at + embedder._current_interval + 1):
                embedder.embed(frames)
        self.assertEqual(embedder._consecutive_failures, 3)
        self.assertGreater(embedder._current_interval, embedder.BASE_RETRY_INTERVAL)

        # Now primary heals
        with patch("time.monotonic", return_value=embedder._failed_at + embedder._current_interval + 1):
            embedder.embed(frames)
        self.assertFalse(embedder._failed_over)
        self.assertEqual(embedder._consecutive_failures, 0)
        self.assertAlmostEqual(embedder._current_interval, embedder.BASE_RETRY_INTERVAL)
        self.assertFalse(embedder._permanently_failed)


if __name__ == "__main__":
    unittest.main()
