"""Tests for insurance_mvp.metrics module."""

from insurance_mvp.metrics import InMemoryMetrics


class TestInMemoryMetrics:
    def test_counter_inc(self):
        m = InMemoryMetrics()
        m.inc("requests")
        m.inc("requests")
        snap = m.snapshot()
        assert snap["counters"]["requests"] == 2.0

    def test_counter_with_labels(self):
        m = InMemoryMetrics()
        m.inc("requests", labels={"status": "ok"})
        m.inc("requests", labels={"status": "error"})
        snap = m.snapshot()
        assert snap["counters"]['requests{status="ok"}'] == 1.0
        assert snap["counters"]['requests{status="error"}'] == 1.0

    def test_gauge_set_and_inc_dec(self):
        m = InMemoryMetrics()
        m.set_gauge("active", 5.0)
        m.inc_gauge("active")
        m.dec_gauge("active", 2.0)
        snap = m.snapshot()
        assert snap["gauges"]["active"] == 4.0

    def test_histogram_observe(self):
        m = InMemoryMetrics()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            m.observe("latency", v)
        snap = m.snapshot()
        h = snap["histograms"]["latency"]
        assert h["count"] == 5
        assert h["sum"] == 15.0
        assert h["mean"] == 3.0
        assert h["p50"] == 3.0

    def test_timer_context_manager(self):
        import time

        m = InMemoryMetrics()
        with m.timer("op_duration"):
            time.sleep(0.01)
        snap = m.snapshot()
        assert snap["histograms"]["op_duration"]["count"] == 1
        assert snap["histograms"]["op_duration"]["sum"] > 0.0

    def test_reset(self):
        m = InMemoryMetrics()
        m.inc("x")
        m.set_gauge("y", 1)
        m.observe("z", 1)
        m.reset()
        snap = m.snapshot()
        assert snap["counters"] == {}
        assert snap["gauges"] == {}
        assert snap["histograms"] == {}

    def test_empty_histogram_summary(self):
        m = InMemoryMetrics()
        snap = m.snapshot()
        assert snap["histograms"] == {}
