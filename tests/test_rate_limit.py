"""Tests for sopilot.api.rate_limit middleware."""


from sopilot.api.rate_limit import _SlidingWindow


class TestSlidingWindow:
    def test_allows_within_limit(self):
        w = _SlidingWindow(60.0, 5)
        for _ in range(5):
            ok, _ = w.allow("client1")
            assert ok

    def test_blocks_over_limit(self):
        w = _SlidingWindow(60.0, 3)
        for _ in range(3):
            w.allow("client1")
        ok, remaining = w.allow("client1")
        assert not ok
        assert remaining == 0

    def test_separate_clients(self):
        w = _SlidingWindow(60.0, 2)
        w.allow("a")
        w.allow("a")
        ok_a, _ = w.allow("a")
        ok_b, _ = w.allow("b")
        assert not ok_a
        assert ok_b

    def test_remaining_count(self):
        w = _SlidingWindow(60.0, 5)
        _, rem = w.allow("x")
        assert rem == 4
        _, rem = w.allow("x")
        assert rem == 3

    def test_cleanup_removes_expired(self):
        import time
        w = _SlidingWindow(0.01, 100)
        w.allow("a")
        time.sleep(0.05)  # generous margin for Windows timer resolution
        w.cleanup()
        assert "a" not in w._clients or len(w._clients["a"]) == 0
