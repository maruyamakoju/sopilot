"""Phase 48: SLA Observatory — SLAMonitor unit tests.

55 tests covering SLODefinition, SLOStatus, _percentile helper,
SLAMonitor record/query/SLO-check/Prometheus-export, edge cases, and
thread-safety.
"""
from __future__ import annotations

import threading
import time

import pytest

from sopilot.perception.sla_monitor import (
    SLAMonitor,
    SLODefinition,
    SLOStatus,
    _DEFAULT_SLOS,
    _percentile,
)


# ---------------------------------------------------------------------------
# _percentile helper (6 tests)
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty_returns_zero(self):
        assert _percentile([], 0.95) == 0.0

    def test_single_element(self):
        assert _percentile([42.0], 0.50) == 42.0

    def test_two_elements_median(self):
        result = _percentile([10.0, 20.0], 0.50)
        assert result == pytest.approx(15.0)

    def test_p0_returns_minimum(self):
        data = sorted([5.0, 10.0, 15.0, 20.0])
        assert _percentile(data, 0.0) == pytest.approx(5.0)

    def test_p100_returns_maximum(self):
        data = sorted([5.0, 10.0, 15.0, 20.0])
        assert _percentile(data, 1.0) == pytest.approx(20.0)

    def test_p95_large_dataset(self):
        data = sorted(float(i) for i in range(1, 101))  # 1..100
        # P95 of 100 equally-spaced values: index = 0.95*99 = 94.05 → 95.05
        result = _percentile(data, 0.95)
        assert 95.0 <= result <= 96.0


# ---------------------------------------------------------------------------
# SLODefinition (5 tests)
# ---------------------------------------------------------------------------


class TestSLODefinition:
    def test_fields(self):
        slo = SLODefinition("frame_p95", "frame_total", 0.95, 200.0, "desc")
        assert slo.name == "frame_p95"
        assert slo.stage == "frame_total"
        assert slo.percentile == 0.95
        assert slo.threshold_ms == 200.0

    def test_to_dict_keys(self):
        slo = SLODefinition("x", "y", 0.99, 100.0)
        d = slo.to_dict()
        assert set(d.keys()) == {"name", "stage", "percentile", "threshold_ms", "description"}

    def test_default_description_empty(self):
        slo = SLODefinition("a", "b", 0.5, 50.0)
        assert slo.description == ""

    def test_to_dict_values(self):
        slo = SLODefinition("n", "s", 0.95, 75.0, "hello")
        d = slo.to_dict()
        assert d["name"] == "n"
        assert d["threshold_ms"] == 75.0
        assert d["description"] == "hello"

    def test_default_slos_count(self):
        assert len(_DEFAULT_SLOS) == 5


# ---------------------------------------------------------------------------
# SLOStatus (4 tests)
# ---------------------------------------------------------------------------


class TestSLOStatus:
    def _make_status(self, compliant=True):
        slo = SLODefinition("x", "y", 0.95, 100.0)
        return SLOStatus(slo=slo, current_value_ms=80.0, compliant=compliant,
                         sample_count=50, violation_count=2)

    def test_to_dict_compliant(self):
        s = self._make_status(True)
        d = s.to_dict()
        assert d["compliant"] is True
        assert d["current_value_ms"] == pytest.approx(80.0)

    def test_to_dict_violated(self):
        s = self._make_status(False)
        d = s.to_dict()
        assert d["compliant"] is False

    def test_to_dict_keys(self):
        s = self._make_status()
        d = s.to_dict()
        assert "name" in d and "violation_count" in d and "sample_count" in d

    def test_violation_count(self):
        s = self._make_status()
        assert s.violation_count == 2


# ---------------------------------------------------------------------------
# SLAMonitor — construction (3 tests)
# ---------------------------------------------------------------------------


class TestSLAMonitorConstruction:
    def test_default_slos(self):
        m = SLAMonitor()
        assert len(m._slos) == 5

    def test_custom_window_size(self):
        m = SLAMonitor(window_size=50)
        assert m._window_size == 50

    def test_custom_slos(self):
        slo = SLODefinition("custom", "stage_x", 0.95, 100.0)
        m = SLAMonitor(slos=[slo])
        assert len(m._slos) == 1
        assert m._slos[0].name == "custom"


# ---------------------------------------------------------------------------
# SLAMonitor.record (6 tests)
# ---------------------------------------------------------------------------


class TestSLAMonitorRecord:
    def test_record_creates_stage(self):
        m = SLAMonitor()
        m.record("myStage", 10.0)
        assert "myStage" in m._windows

    def test_record_increments_total(self):
        m = SLAMonitor()
        m.record("a", 1.0)
        m.record("a", 2.0)
        assert m._total_recorded == 2

    def test_record_window_fifo_eviction(self):
        m = SLAMonitor(window_size=10)
        for i in range(20):
            m.record("s", float(i))
        assert len(m._windows["s"]) == 10
        assert m._windows["s"][-1] == 19.0

    def test_record_multiple_stages(self):
        m = SLAMonitor()
        m.record("detection_ms", 15.0)
        m.record("tracking_ms", 5.0)
        assert "detection_ms" in m._windows
        assert "tracking_ms" in m._windows

    def test_record_float_coercion(self):
        m = SLAMonitor()
        m.record("s", 42)  # int
        assert isinstance(m._windows["s"][0], float)

    def test_window_minimum_size_clamped(self):
        m = SLAMonitor(window_size=2)
        assert m._window_size >= 10  # clamped to max(10, window_size)


# ---------------------------------------------------------------------------
# SLAMonitor.get_percentiles (6 tests)
# ---------------------------------------------------------------------------


class TestGetPercentiles:
    def test_no_samples_empty(self):
        m = SLAMonitor()
        assert m.get_percentiles("nonexistent") == {}

    def test_keys_present(self):
        m = SLAMonitor()
        for v in [10.0, 20.0, 30.0]:
            m.record("s", v)
        p = m.get_percentiles("s")
        assert set(p.keys()) == {"p50_ms", "p95_ms", "p99_ms", "mean_ms", "max_ms", "min_ms", "count"}

    def test_count_correct(self):
        m = SLAMonitor()
        for _ in range(7):
            m.record("s", 10.0)
        assert m.get_percentiles("s")["count"] == 7

    def test_p50_reasonable(self):
        m = SLAMonitor()
        for v in range(1, 11):
            m.record("s", float(v))
        p = m.get_percentiles("s")
        assert 5.0 <= p["p50_ms"] <= 6.0

    def test_max_min(self):
        m = SLAMonitor()
        for v in [3.0, 1.0, 2.0]:
            m.record("s", v)
        p = m.get_percentiles("s")
        assert p["max_ms"] == pytest.approx(3.0)
        assert p["min_ms"] == pytest.approx(1.0)

    def test_rounded_to_3dp(self):
        m = SLAMonitor()
        m.record("s", 10.123456789)
        p = m.get_percentiles("s")
        # Values should be rounded to 3 decimal places
        assert round(p["p50_ms"], 3) == p["p50_ms"]


# ---------------------------------------------------------------------------
# SLAMonitor.get_all_stage_stats (3 tests)
# ---------------------------------------------------------------------------


class TestGetAllStageStats:
    def test_empty_no_stages(self):
        m = SLAMonitor()
        assert m.get_all_stage_stats() == {}

    def test_returns_all_stages(self):
        m = SLAMonitor()
        m.record("a", 10.0)
        m.record("b", 20.0)
        stats = m.get_all_stage_stats()
        assert "a" in stats and "b" in stats

    def test_values_not_empty(self):
        m = SLAMonitor()
        m.record("s", 5.0)
        stats = m.get_all_stage_stats()
        assert stats["s"]["count"] == 1


# ---------------------------------------------------------------------------
# SLAMonitor.check_slos (8 tests)
# ---------------------------------------------------------------------------


class TestCheckSlos:
    def test_no_data_compliant(self):
        m = SLAMonitor()
        statuses = m.check_slos()
        # No data means compliant by assumption
        assert all(s.compliant for s in statuses)

    def test_all_compliant_fast(self):
        m = SLAMonitor()
        for _ in range(100):
            m.record("frame_total", 10.0)
            m.record("detection_ms", 5.0)
            m.record("tracking_ms", 2.0)
            m.record("zone_analysis_ms", 1.0)
        statuses = m.check_slos()
        assert all(s.compliant for s in statuses)

    def test_frame_p95_violated(self):
        m = SLAMonitor()
        for _ in range(100):
            m.record("frame_total", 300.0)  # > 200ms threshold
        statuses = m.check_slos()
        frame_p95 = next(s for s in statuses if s.slo.name == "frame_total_p95")
        assert not frame_p95.compliant

    def test_violation_count_increments(self):
        m = SLAMonitor()
        for _ in range(50):
            m.record("frame_total", 1000.0)
        # Call check_slos twice
        m.check_slos()
        m.check_slos()
        statuses = m.check_slos()
        frame_p95 = next(s for s in statuses if s.slo.name == "frame_total_p95")
        assert frame_p95.violation_count >= 1

    def test_violated_sorted_first(self):
        m = SLAMonitor()
        for _ in range(100):
            m.record("frame_total", 300.0)  # violates p95 and p99
        statuses = m.check_slos()
        # First entries should be non-compliant
        non_compliant = [s for s in statuses if not s.compliant]
        if non_compliant:
            assert statuses[0].compliant is False

    def test_returns_list_of_slo_status(self):
        m = SLAMonitor()
        statuses = m.check_slos()
        assert isinstance(statuses, list)
        assert all(isinstance(s, SLOStatus) for s in statuses)

    def test_sample_count_correct(self):
        m = SLAMonitor()
        for _ in range(25):
            m.record("frame_total", 50.0)
        statuses = m.check_slos()
        ft = next(s for s in statuses if s.slo.stage == "frame_total")
        assert ft.sample_count == 25

    def test_custom_slo_violation(self):
        slo = SLODefinition("custom", "my_stage", 0.95, 10.0)
        m = SLAMonitor(slos=[slo])
        for _ in range(100):
            m.record("my_stage", 50.0)  # > 10ms threshold
        statuses = m.check_slos()
        assert not statuses[0].compliant


# ---------------------------------------------------------------------------
# SLAMonitor.get_stats (5 tests)
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_keys(self):
        m = SLAMonitor()
        s = m.get_stats()
        assert "slos_total" in s and "slos_compliant" in s and "slos_violated" in s

    def test_slos_total_matches_defaults(self):
        m = SLAMonitor()
        assert m.get_stats()["slos_total"] == 5

    def test_stages_tracked_empty_initially(self):
        m = SLAMonitor()
        assert m.get_stats()["stages_tracked"] == []

    def test_total_samples_counted(self):
        m = SLAMonitor()
        for _ in range(10):
            m.record("s", 1.0)
        assert m.get_stats()["total_samples_recorded"] == 10

    def test_uptime_positive(self):
        m = SLAMonitor()
        time.sleep(0.1)
        assert m.get_stats()["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# SLAMonitor.get_prometheus_lines (5 tests)
# ---------------------------------------------------------------------------


class TestGetPrometheusLines:
    def test_empty_returns_no_lines(self):
        m = SLAMonitor()
        lines = m.get_prometheus_lines()
        assert lines == []

    def test_has_help_and_type(self):
        m = SLAMonitor()
        m.record("frame_total", 50.0)
        lines = m.get_prometheus_lines()
        assert any("# HELP" in l for l in lines)
        assert any("# TYPE" in l for l in lines)

    def test_stage_latency_lines_present(self):
        m = SLAMonitor()
        for v in [10.0, 20.0, 30.0]:
            m.record("detection_ms", v)
        lines = m.get_prometheus_lines()
        assert any("sla_stage_latency_ms" in l and "detection_ms" in l for l in lines)

    def test_slo_compliance_lines_present(self):
        m = SLAMonitor()
        m.record("frame_total", 50.0)
        lines = m.get_prometheus_lines()
        assert any("sla_slo_compliant" in l for l in lines)

    def test_quantile_labels(self):
        m = SLAMonitor()
        for _ in range(20):
            m.record("frame_total", 10.0)
        lines = m.get_prometheus_lines()
        quantile_lines = [l for l in lines if "quantile=" in l]
        quantiles = {l.split('quantile="')[1].split('"')[0] for l in quantile_lines}
        assert "0.5" in quantiles
        assert "0.95" in quantiles
        assert "0.99" in quantiles


# ---------------------------------------------------------------------------
# SLAMonitor.reset (3 tests)
# ---------------------------------------------------------------------------


class TestReset:
    def test_clears_windows(self):
        m = SLAMonitor()
        m.record("s", 10.0)
        m.reset()
        assert m._windows == {}

    def test_clears_violation_counts(self):
        m = SLAMonitor()
        for _ in range(100):
            m.record("frame_total", 1000.0)
        m.check_slos()
        m.reset()
        assert m._violation_counts == {}

    def test_clears_total_recorded(self):
        m = SLAMonitor()
        for _ in range(5):
            m.record("s", 1.0)
        m.reset()
        assert m._total_recorded == 0


# ---------------------------------------------------------------------------
# Thread safety (4 tests)
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_record(self):
        m = SLAMonitor(window_size=1000)
        errors = []

        def worker():
            try:
                for _ in range(200):
                    m.record("frame_total", 10.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(m._windows.get("frame_total", [])) <= 1000

    def test_concurrent_check_slos(self):
        m = SLAMonitor()
        for _ in range(100):
            m.record("frame_total", 50.0)
        errors = []

        def reader():
            try:
                for _ in range(20):
                    m.check_slos()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_concurrent_record_and_read(self):
        m = SLAMonitor()
        errors = []

        def writer():
            try:
                for _ in range(100):
                    m.record("s", 5.0)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    m.get_percentiles("s")
            except Exception as e:
                errors.append(e)

        ts = [threading.Thread(target=writer) for _ in range(3)]
        ts += [threading.Thread(target=reader) for _ in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert errors == []

    def test_reset_thread_safe(self):
        m = SLAMonitor()
        errors = []

        def recorder():
            try:
                for _ in range(50):
                    m.record("s", 1.0)
            except Exception as e:
                errors.append(e)

        def resetter():
            try:
                for _ in range(5):
                    m.reset()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        ts = [threading.Thread(target=recorder) for _ in range(4)]
        ts.append(threading.Thread(target=resetter))
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert errors == []
