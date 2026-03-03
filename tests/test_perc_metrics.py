"""Tests for sopilot.perception.perc_metrics — PercMetricsRegistry + /metrics endpoint integration."""
from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_registry():
    """Return a brand-new PercMetricsRegistry (not the singleton)."""
    from sopilot.perception.perc_metrics import PercMetricsRegistry
    return PercMetricsRegistry()


# ---------------------------------------------------------------------------
# Unit tests — PercMetricsRegistry
# ---------------------------------------------------------------------------

class TestPercMetricsRegistryUnit(unittest.TestCase):

    def setUp(self):
        # Always start from a clean singleton so tests are independent.
        from sopilot.perception.perc_metrics import reset_registry
        reset_registry()

    # --- singleton -----------------------------------------------------------

    def test_get_registry_returns_instance(self):
        from sopilot.perception.perc_metrics import get_registry, PercMetricsRegistry
        reg = get_registry()
        self.assertIsInstance(reg, PercMetricsRegistry)

    def test_get_registry_is_singleton(self):
        from sopilot.perception.perc_metrics import get_registry
        r1 = get_registry()
        r2 = get_registry()
        self.assertIs(r1, r2)

    def test_reset_registry_zeroes_counters(self):
        from sopilot.perception.perc_metrics import get_registry, reset_registry
        reg = get_registry()
        reg.record_frame(processing_ms=50.0)
        reg.record_detection(count=3)
        reg.record_vlm_call()
        reset_registry()
        snap = get_registry().get_snapshot()
        self.assertEqual(snap["frames_total"], 0)
        self.assertEqual(snap["detections_total"], 0)
        self.assertEqual(snap["vlm_calls_total"], 0)

    # --- record_frame --------------------------------------------------------

    def test_record_frame_increments_frames_total(self):
        reg = _fresh_registry()
        reg.record_frame()
        self.assertEqual(reg.frames_total, 1)

    def test_record_frame_multiple(self):
        reg = _fresh_registry()
        for _ in range(5):
            reg.record_frame()
        self.assertEqual(reg.frames_total, 5)

    def test_record_frame_updates_processing_sum(self):
        reg = _fresh_registry()
        reg.record_frame(processing_ms=100.0)
        snap = reg.get_snapshot()
        self.assertAlmostEqual(snap["processing_seconds_sum"], 0.1, places=6)
        self.assertEqual(snap["processing_seconds_count"], 1)

    def test_record_frame_zero_ms(self):
        reg = _fresh_registry()
        reg.record_frame(processing_ms=0.0)
        snap = reg.get_snapshot()
        self.assertEqual(snap["processing_seconds_sum"], 0.0)
        self.assertEqual(snap["processing_seconds_count"], 1)

    # --- record_detection ----------------------------------------------------

    def test_record_detection_default(self):
        reg = _fresh_registry()
        reg.record_detection()
        self.assertEqual(reg.detections_total, 1)

    def test_record_detection_count_five(self):
        reg = _fresh_registry()
        reg.record_detection(count=5)
        self.assertEqual(reg.detections_total, 5)

    def test_record_detection_accumulates(self):
        reg = _fresh_registry()
        reg.record_detection(count=3)
        reg.record_detection(count=7)
        self.assertEqual(reg.detections_total, 10)

    # --- record_vlm_call -----------------------------------------------------

    def test_record_vlm_call_increments(self):
        reg = _fresh_registry()
        reg.record_vlm_call()
        self.assertEqual(reg.vlm_calls_total, 1)

    def test_record_vlm_call_multiple(self):
        reg = _fresh_registry()
        for _ in range(4):
            reg.record_vlm_call()
        self.assertEqual(reg.vlm_calls_total, 4)

    # --- record_violation ----------------------------------------------------

    def test_record_violation_critical(self):
        reg = _fresh_registry()
        reg.record_violation(severity="critical")
        self.assertEqual(reg.violations_by_severity["critical"], 1)

    def test_record_violation_default_severity(self):
        reg = _fresh_registry()
        reg.record_violation()
        self.assertIn("warning", reg.violations_by_severity)
        self.assertEqual(reg.violations_by_severity["warning"], 1)

    def test_record_violation_multiple_severities(self):
        reg = _fresh_registry()
        reg.record_violation(severity="info")
        reg.record_violation(severity="warning")
        reg.record_violation(severity="critical")
        reg.record_violation(severity="warning")
        self.assertEqual(reg.violations_by_severity["info"], 1)
        self.assertEqual(reg.violations_by_severity["warning"], 2)
        self.assertEqual(reg.violations_by_severity["critical"], 1)

    # --- record_anomaly ------------------------------------------------------

    def test_record_anomaly_behavioral(self):
        reg = _fresh_registry()
        reg.record_anomaly(detector="behavioral")
        self.assertEqual(reg.anomaly_events_by_detector["behavioral"], 1)

    def test_record_anomaly_default(self):
        reg = _fresh_registry()
        reg.record_anomaly()
        self.assertIn("unknown", reg.anomaly_events_by_detector)

    def test_record_anomaly_accumulates(self):
        reg = _fresh_registry()
        reg.record_anomaly(detector="spatial")
        reg.record_anomaly(detector="spatial")
        reg.record_anomaly(detector="temporal")
        self.assertEqual(reg.anomaly_events_by_detector["spatial"], 2)
        self.assertEqual(reg.anomaly_events_by_detector["temporal"], 1)

    # --- record_event --------------------------------------------------------

    def test_record_event_zone_entered(self):
        reg = _fresh_registry()
        reg.record_event(event_type="ZONE_ENTERED")
        self.assertEqual(reg.events_by_type["ZONE_ENTERED"], 1)

    def test_record_event_default(self):
        reg = _fresh_registry()
        reg.record_event()
        self.assertIn("unknown", reg.events_by_type)

    # --- get_processing_seconds_avg ------------------------------------------

    def test_avg_zero_before_records(self):
        reg = _fresh_registry()
        self.assertEqual(reg.get_processing_seconds_avg(), 0.0)

    def test_avg_correct_after_two_records(self):
        reg = _fresh_registry()
        reg.record_frame(processing_ms=100.0)  # 0.1 s
        reg.record_frame(processing_ms=300.0)  # 0.3 s
        avg = reg.get_processing_seconds_avg()
        self.assertAlmostEqual(avg, 0.2, places=6)

    # --- get_snapshot --------------------------------------------------------

    def test_snapshot_has_all_keys(self):
        reg = _fresh_registry()
        snap = reg.get_snapshot()
        expected_keys = {
            "frames_total",
            "detections_total",
            "vlm_calls_total",
            "processing_seconds_sum",
            "processing_seconds_count",
            "violations_by_severity",
            "anomaly_events_by_detector",
            "events_by_type",
        }
        self.assertEqual(set(snap.keys()), expected_keys)

    def test_snapshot_reflects_recorded_data(self):
        reg = _fresh_registry()
        reg.record_frame(processing_ms=200.0)
        reg.record_detection(count=4)
        reg.record_vlm_call()
        reg.record_violation(severity="warning")
        reg.record_anomaly(detector="interaction")
        snap = reg.get_snapshot()
        self.assertEqual(snap["frames_total"], 1)
        self.assertEqual(snap["detections_total"], 4)
        self.assertEqual(snap["vlm_calls_total"], 1)
        self.assertAlmostEqual(snap["processing_seconds_sum"], 0.2, places=6)
        self.assertEqual(snap["violations_by_severity"]["warning"], 1)
        self.assertEqual(snap["anomaly_events_by_detector"]["interaction"], 1)

    def test_snapshot_is_copy(self):
        """Mutating the returned dict should not affect the registry."""
        reg = _fresh_registry()
        reg.record_violation(severity="info")
        snap = reg.get_snapshot()
        snap["violations_by_severity"]["info"] = 999
        # Registry's internal dict must be unchanged
        snap2 = reg.get_snapshot()
        self.assertEqual(snap2["violations_by_severity"]["info"], 1)

    # --- reset ---------------------------------------------------------------

    def test_reset_clears_all_counters(self):
        reg = _fresh_registry()
        reg.record_frame(processing_ms=50.0)
        reg.record_detection(count=2)
        reg.record_vlm_call()
        reg.record_violation(severity="critical")
        reg.record_anomaly(detector="behavioral")
        reg.record_event(event_type="COLLISION")
        reg.reset()
        snap = reg.get_snapshot()
        self.assertEqual(snap["frames_total"], 0)
        self.assertEqual(snap["detections_total"], 0)
        self.assertEqual(snap["vlm_calls_total"], 0)
        self.assertEqual(snap["processing_seconds_sum"], 0.0)
        self.assertEqual(snap["processing_seconds_count"], 0)
        self.assertEqual(snap["violations_by_severity"], {})
        self.assertEqual(snap["anomaly_events_by_detector"], {})
        self.assertEqual(snap["events_by_type"], {})

    # --- thread safety -------------------------------------------------------

    def test_thread_safety_record_frame(self):
        """10 threads × 100 calls each → frames_total must equal 1000."""
        reg = _fresh_registry()
        n_threads = 10
        n_calls = 100

        def worker():
            for _ in range(n_calls):
                reg.record_frame(processing_ms=10.0)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(reg.frames_total, n_threads * n_calls)

    def test_thread_safety_mixed_operations(self):
        """Multiple threads calling different record_* methods concurrently."""
        reg = _fresh_registry()
        n_threads = 5
        n_calls = 50

        def worker():
            for _ in range(n_calls):
                reg.record_frame()
                reg.record_detection()
                reg.record_vlm_call()
                reg.record_violation(severity="warning")
                reg.record_anomaly(detector="spatial")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = n_threads * n_calls
        self.assertEqual(reg.frames_total, total)
        self.assertEqual(reg.detections_total, total)
        self.assertEqual(reg.vlm_calls_total, total)
        self.assertEqual(reg.violations_by_severity["warning"], total)
        self.assertEqual(reg.anomaly_events_by_detector["spatial"], total)


# ---------------------------------------------------------------------------
# Integration tests — /metrics endpoint via TestClient
# ---------------------------------------------------------------------------

class TestMetricsEndpointIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Isolate the registry before creating the app so metrics are clean.
        from sopilot.perception.perc_metrics import reset_registry, get_registry
        reset_registry()

        # Seed some metrics so labeled lines appear in the /metrics output.
        reg = get_registry()
        reg.record_frame(processing_ms=18.9)
        reg.record_detection(count=3)
        reg.record_vlm_call()
        reg.record_violation(severity="warning")
        reg.record_violation(severity="critical")
        reg.record_anomaly(detector="behavioral")

        cls._tmp = tempfile.mkdtemp()
        data_dir = Path(cls._tmp) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        os.environ["SOPILOT_DATA_DIR"] = str(data_dir)
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-percmetrics"

        from fastapi.testclient import TestClient
        from sopilot.main import create_app
        cls.client = TestClient(create_app())

    def _get_metrics_text(self) -> str:
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        return resp.text

    # --- key presence --------------------------------------------------------

    def test_perception_frames_total_present(self):
        self.assertIn("perception_frames_total", self._get_metrics_text())

    def test_perception_detections_total_present(self):
        self.assertIn("perception_detections_total", self._get_metrics_text())

    def test_perception_vlm_calls_total_present(self):
        self.assertIn("perception_vlm_calls_total", self._get_metrics_text())

    def test_perception_processing_seconds_avg_present(self):
        self.assertIn("perception_processing_seconds_avg", self._get_metrics_text())

    def test_perception_violations_total_present(self):
        text = self._get_metrics_text()
        self.assertIn("perception_violations_total", text)

    def test_perception_anomaly_events_total_present(self):
        text = self._get_metrics_text()
        self.assertIn("perception_anomaly_events_total", text)

    # --- label values --------------------------------------------------------

    def test_violation_severity_warning_label(self):
        text = self._get_metrics_text()
        self.assertIn('severity="warning"', text)

    def test_violation_severity_critical_label(self):
        text = self._get_metrics_text()
        self.assertIn('severity="critical"', text)

    def test_anomaly_detector_behavioral_label(self):
        text = self._get_metrics_text()
        self.assertIn('detector="behavioral"', text)

    # --- HELP / TYPE lines ---------------------------------------------------

    def test_help_line_frames_total(self):
        text = self._get_metrics_text()
        self.assertIn("# HELP perception_frames_total", text)

    def test_type_line_frames_total_counter(self):
        text = self._get_metrics_text()
        self.assertIn("# TYPE perception_frames_total counter", text)

    def test_help_line_processing_avg(self):
        text = self._get_metrics_text()
        self.assertIn("# HELP perception_processing_seconds_avg", text)

    def test_type_line_processing_avg_gauge(self):
        text = self._get_metrics_text()
        self.assertIn("# TYPE perception_processing_seconds_avg gauge", text)

    # --- existing metrics still present --------------------------------------

    def test_existing_queue_depth_still_present(self):
        self.assertIn("sopilot_queue_depth", self._get_metrics_text())

    def test_existing_sopilot_info_still_present(self):
        self.assertIn("sopilot_info", self._get_metrics_text())

    # --- content type --------------------------------------------------------

    def test_content_type_prometheus(self):
        resp = self.client.get("/metrics")
        ct = resp.headers.get("content-type", "")
        self.assertIn("text/plain", ct)


if __name__ == "__main__":
    unittest.main()
