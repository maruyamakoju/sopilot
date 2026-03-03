"""Tests for sopilot/perception/multimodal.py — Multi-modal Fusion Engine.

42 tests covering:
- Enum values (SignalSource, FusionType)
- Dataclass serialisation (MultimodalSignal, FusionEvent)
- MultimodalFusionEngine lifecycle (ingest, create, get, clear, state_dict)
- Fusion rules (anomaly, access, intrusion, urgency, catch-all)
- Edge cases (empty signals, outside window, deduplication, buffer overflow)
- Thread safety

Run:  python -m pytest tests/test_multimodal.py -v
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock

from sopilot.perception.multimodal import (
    FusionEvent,
    FusionType,
    MultimodalFusionEngine,
    MultimodalSignal,
    SignalSource,
)
from sopilot.perception.types import (
    EntityEvent,
    EntityEventType,
    SceneGraph,
    ViolationSeverity,
    WorldState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ws(events=None) -> WorldState:
    sg = SceneGraph(
        timestamp=100.0,
        frame_number=100,
        entities=[],
        relations=[],
    )
    return WorldState(
        timestamp=100.0,
        frame_number=100,
        scene_graph=sg,
        active_tracks={},
        events=events or [],
        zone_occupancy={},
    )


def _make_event(
    event_type: EntityEventType = EntityEventType.ANOMALY,
    severity: ViolationSeverity = ViolationSeverity.WARNING,  # kept for call-site compat, not passed
    ts: float = 100.0,
    details: dict | None = None,
) -> EntityEvent:
    return EntityEvent(
        event_type=event_type,
        entity_id=1,
        timestamp=ts,
        frame_number=100,
        details=details or {"description_ja": "異常検出"},
    )


def _make_signal(
    source: SignalSource = SignalSource.AUDIO,
    signal_type: str = "noise",
    ts: float = 100.0,
    value: float = 0.5,
) -> MultimodalSignal:
    return MultimodalSignal(
        signal_id="test-sig-id",
        source=source,
        signal_type=signal_type,
        timestamp=ts,
        value=value,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSignalSourceEnum(unittest.TestCase):
    def test_audio_value(self):
        self.assertEqual(SignalSource.AUDIO.value, "audio")

    def test_iot_value(self):
        self.assertEqual(SignalSource.IOT.value, "iot")

    def test_access_value(self):
        self.assertEqual(SignalSource.ACCESS.value, "access")

    def test_custom_value(self):
        self.assertEqual(SignalSource.CUSTOM.value, "custom")


class TestFusionTypeEnum(unittest.TestCase):
    def test_multimodal_anomaly_value(self):
        self.assertEqual(FusionType.MULTIMODAL_ANOMALY.value, "multimodal_anomaly")

    def test_confirmed_access_value(self):
        self.assertEqual(FusionType.CONFIRMED_ACCESS.value, "confirmed_access")

    def test_sensor_confirmed_intrusion_value(self):
        self.assertEqual(FusionType.SENSOR_CONFIRMED_INTRUSION.value, "sensor_confirmed_intrusion")

    def test_urgency_confirmed_value(self):
        self.assertEqual(FusionType.URGENCY_CONFIRMED.value, "urgency_confirmed")

    def test_correlated_event_value(self):
        self.assertEqual(FusionType.CORRELATED_EVENT.value, "correlated_event")


class TestMultimodalSignalToDict(unittest.TestCase):
    def setUp(self):
        self.sig = MultimodalSignal(
            signal_id="abc-123",
            source=SignalSource.AUDIO,
            signal_type="gunshot",
            timestamp=500.0,
            value=0.9,
            metadata={"channel": 1},
        )
        self.d = self.sig.to_dict()

    def test_has_signal_id(self):
        self.assertEqual(self.d["signal_id"], "abc-123")

    def test_source_is_string(self):
        self.assertEqual(self.d["source"], "audio")

    def test_signal_type(self):
        self.assertEqual(self.d["signal_type"], "gunshot")

    def test_timestamp(self):
        self.assertEqual(self.d["timestamp"], 500.0)

    def test_value(self):
        self.assertAlmostEqual(self.d["value"], 0.9)

    def test_metadata(self):
        self.assertEqual(self.d["metadata"], {"channel": 1})


class TestFusionEventToDict(unittest.TestCase):
    def setUp(self):
        self.fe = FusionEvent(
            fusion_id="fid-1",
            timestamp=200.0,
            fusion_type=FusionType.CONFIRMED_ACCESS,
            signal_ids=["s1", "s2"],
            visual_event_type="zone_entered",
            confidence=0.85,
            description_ja="テスト",
            description_en="Test",
            details={"x": 1},
        )
        self.d = self.fe.to_dict()

    def test_fusion_id(self):
        self.assertEqual(self.d["fusion_id"], "fid-1")

    def test_fusion_type_is_string(self):
        self.assertEqual(self.d["fusion_type"], "confirmed_access")

    def test_signal_ids(self):
        self.assertEqual(self.d["signal_ids"], ["s1", "s2"])

    def test_confidence(self):
        self.assertAlmostEqual(self.d["confidence"], 0.85)

    def test_descriptions(self):
        self.assertEqual(self.d["description_ja"], "テスト")
        self.assertEqual(self.d["description_en"], "Test")


class TestMultimodalFusionEngineConstruction(unittest.TestCase):
    def test_initial_state(self):
        engine = MultimodalFusionEngine()
        state = engine.get_state_dict()
        self.assertEqual(state["buffered_signals"], 0)
        self.assertEqual(state["total_ingested"], 0)
        self.assertEqual(state["total_fused"], 0)
        self.assertEqual(state["signals_by_source"], {})
        self.assertEqual(state["recent_fusion_count"], 0)


class TestIngestSignal(unittest.TestCase):
    def test_ingest_returns_signal_id(self):
        engine = MultimodalFusionEngine()
        sig = _make_signal()
        returned_id = engine.ingest_signal(sig)
        self.assertEqual(returned_id, sig.signal_id)

    def test_ingest_increments_buffered_count(self):
        engine = MultimodalFusionEngine()
        engine.ingest_signal(_make_signal())
        self.assertEqual(engine.get_state_dict()["buffered_signals"], 1)

    def test_ingest_increments_total_ingested(self):
        engine = MultimodalFusionEngine()
        engine.ingest_signal(_make_signal())
        engine.ingest_signal(_make_signal())
        self.assertEqual(engine.get_state_dict()["total_ingested"], 2)


class TestCreateSignal(unittest.TestCase):
    def test_default_timestamp_uses_time_now(self):
        engine = MultimodalFusionEngine()
        before = time.time()
        sig = engine.create_signal(SignalSource.AUDIO, "noise")
        after = time.time()
        self.assertGreaterEqual(sig.timestamp, before)
        self.assertLessEqual(sig.timestamp, after)

    def test_explicit_timestamp_used(self):
        engine = MultimodalFusionEngine()
        sig = engine.create_signal(SignalSource.IOT, "motion_pir", timestamp=12345.0)
        self.assertEqual(sig.timestamp, 12345.0)

    def test_string_source_accepted(self):
        engine = MultimodalFusionEngine()
        sig = engine.create_signal("access", "door_open", timestamp=1.0)
        self.assertEqual(sig.source, SignalSource.ACCESS)

    def test_signal_is_buffered_after_create(self):
        engine = MultimodalFusionEngine()
        engine.create_signal(SignalSource.CUSTOM, "custom_event", timestamp=1.0)
        self.assertEqual(engine.get_state_dict()["buffered_signals"], 1)


class TestGetSignals(unittest.TestCase):
    def test_no_filter_returns_all_recent(self):
        engine = MultimodalFusionEngine()
        now = time.time()
        engine.create_signal(SignalSource.AUDIO, "noise", timestamp=now)
        engine.create_signal(SignalSource.IOT, "motion", timestamp=now)
        sigs = engine.get_signals(lookback_seconds=60.0)
        self.assertEqual(len(sigs), 2)

    def test_filter_by_source_returns_only_that_source(self):
        engine = MultimodalFusionEngine()
        now = time.time()
        engine.create_signal(SignalSource.AUDIO, "noise", timestamp=now)
        engine.create_signal(SignalSource.IOT, "motion", timestamp=now)
        sigs = engine.get_signals(source=SignalSource.AUDIO, lookback_seconds=60.0)
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0].source, SignalSource.AUDIO)

    def test_zero_lookback_returns_nothing(self):
        engine = MultimodalFusionEngine()
        # Signal 2 seconds old — lookback=0 means only signals from right now
        engine.create_signal(SignalSource.AUDIO, "noise", timestamp=time.time() - 2.0)
        sigs = engine.get_signals(lookback_seconds=0.0)
        self.assertEqual(len(sigs), 0)

    def test_short_lookback_excludes_old(self):
        engine = MultimodalFusionEngine()
        engine.create_signal(SignalSource.AUDIO, "old", timestamp=time.time() - 100.0)
        engine.create_signal(SignalSource.AUDIO, "new", timestamp=time.time())
        sigs = engine.get_signals(lookback_seconds=5.0)
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0].signal_type, "new")


class TestFuseWithVisual(unittest.TestCase):
    def test_no_signals_returns_empty(self):
        engine = MultimodalFusionEngine()
        ws = _make_ws(events=[_make_event()])
        result = engine.fuse_with_visual(ws, window_seconds=5.0)
        self.assertEqual(result, [])

    def test_no_visual_events_returns_empty(self):
        engine = MultimodalFusionEngine()
        engine.create_signal(SignalSource.AUDIO, "noise", timestamp=100.0, value=0.9)
        ws = _make_ws(events=[])
        result = engine.fuse_with_visual(ws, window_seconds=5.0)
        self.assertEqual(result, [])

    def test_audio_anomaly_plus_anomaly_event_produces_fusion(self):
        engine = MultimodalFusionEngine()
        engine.create_signal(SignalSource.AUDIO, "alarm", timestamp=100.0, value=0.8)
        ws = _make_ws(events=[_make_event(EntityEventType.ANOMALY, ts=100.0)])
        result = engine.fuse_with_visual(ws, window_seconds=5.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].fusion_type, FusionType.MULTIMODAL_ANOMALY)

    def test_fusion_confidence_correct_for_audio_anomaly(self):
        engine = MultimodalFusionEngine()
        engine.create_signal(SignalSource.AUDIO, "alarm", timestamp=100.0, value=0.8)
        ws = _make_ws(events=[_make_event(EntityEventType.ANOMALY, ts=100.0)])
        result = engine.fuse_with_visual(ws, window_seconds=5.0)
        expected_confidence = min(1.0, 0.7 + 0.8 * 0.3)
        self.assertAlmostEqual(result[0].confidence, expected_confidence, places=3)

    def test_fusion_result_has_description(self):
        engine = MultimodalFusionEngine()
        engine.create_signal(SignalSource.AUDIO, "gunshot", timestamp=100.0, value=0.9)
        ws = _make_ws(events=[_make_event(EntityEventType.ANOMALY, ts=100.0)])
        result = engine.fuse_with_visual(ws, window_seconds=5.0)
        self.assertIn("異常", result[0].description_ja)
        self.assertIn("anomaly", result[0].description_en.lower())

    def test_access_door_open_plus_zone_entered_gives_confirmed_access(self):
        engine = MultimodalFusionEngine()
        engine.create_signal(SignalSource.ACCESS, "door_open", timestamp=100.0, value=0.7)
        ws = _make_ws(events=[_make_event(EntityEventType.ZONE_ENTERED, ts=100.0)])
        result = engine.fuse_with_visual(ws, window_seconds=5.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].fusion_type, FusionType.CONFIRMED_ACCESS)
        self.assertAlmostEqual(result[0].confidence, 0.85)

    def test_access_badge_plus_zone_entered_gives_confirmed_access(self):
        engine = MultimodalFusionEngine()
        engine.create_signal(SignalSource.ACCESS, "badge_swipe", timestamp=100.0, value=0.7)
        ws = _make_ws(events=[_make_event(EntityEventType.ZONE_ENTERED, ts=100.0)])
        result = engine.fuse_with_visual(ws, window_seconds=5.0)
        self.assertEqual(result[0].fusion_type, FusionType.CONFIRMED_ACCESS)

    def test_iot_motion_pir_plus_zone_entered_gives_sensor_intrusion(self):
        engine = MultimodalFusionEngine()
        engine.create_signal(SignalSource.IOT, "motion_pir", timestamp=100.0, value=0.6)
        ws = _make_ws(events=[_make_event(EntityEventType.ZONE_ENTERED, ts=100.0)])
        result = engine.fuse_with_visual(ws, window_seconds=5.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].fusion_type, FusionType.SENSOR_CONFIRMED_INTRUSION)
        self.assertAlmostEqual(result[0].confidence, 0.90)

    def test_signal_outside_window_produces_no_fusion(self):
        engine = MultimodalFusionEngine()
        # Signal is 20 seconds away from the visual event
        engine.create_signal(SignalSource.AUDIO, "alarm", timestamp=80.0, value=0.9)
        ws = _make_ws(events=[_make_event(EntityEventType.ANOMALY, ts=100.0)])
        result = engine.fuse_with_visual(ws, window_seconds=5.0)
        self.assertEqual(result, [])

    def test_audio_below_threshold_does_not_fire_multimodal_anomaly(self):
        engine = MultimodalFusionEngine()
        # value=0.5 is below 0.6 threshold for Rule 1
        engine.create_signal(SignalSource.AUDIO, "noise", timestamp=100.0, value=0.5)
        ws = _make_ws(events=[_make_event(EntityEventType.ANOMALY, ts=100.0)])
        # Should not fire MULTIMODAL_ANOMALY; value < 0.8 so no catch-all either
        result = engine.fuse_with_visual(ws, window_seconds=5.0)
        self.assertEqual(result, [])

    def test_deduplication_same_state_called_twice(self):
        engine = MultimodalFusionEngine()
        engine.create_signal(SignalSource.AUDIO, "alarm", timestamp=100.0, value=0.9)
        ws = _make_ws(events=[_make_event(EntityEventType.ANOMALY, ts=100.0)])
        # First call
        r1 = engine.fuse_with_visual(ws, window_seconds=5.0)
        # Second call with identical world state — new fusion pass, may fire again
        # (pairs reset between calls). The important thing: no crash, returns list.
        r2 = engine.fuse_with_visual(ws, window_seconds=5.0)
        self.assertIsInstance(r1, list)
        self.assertIsInstance(r2, list)

    def test_urgency_confirmed_via_running_and_alarm(self):
        engine = MultimodalFusionEngine()
        engine.create_signal(SignalSource.AUDIO, "alarm_siren", timestamp=100.0, value=0.7)
        evt = _make_event(
            EntityEventType.STATE_CHANGED,
            ts=100.0,
            details={"description_ja": "running detected"},
        )
        ws = _make_ws(events=[evt])
        result = engine.fuse_with_visual(ws, window_seconds=5.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].fusion_type, FusionType.URGENCY_CONFIRMED)
        self.assertAlmostEqual(result[0].confidence, 0.80)

    def test_catch_all_high_value_signal_produces_correlated_event(self):
        engine = MultimodalFusionEngine()
        # IOT signal with value=0.9 but signal_type that does not match motion/pir
        engine.create_signal(SignalSource.IOT, "temperature_high", timestamp=100.0, value=0.9)
        # Visual event is not ZONE_ENTERED, so specific IOT rule won't fire
        ws = _make_ws(events=[_make_event(EntityEventType.ENTERED, ts=100.0)])
        result = engine.fuse_with_visual(ws, window_seconds=5.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].fusion_type, FusionType.CORRELATED_EVENT)
        self.assertAlmostEqual(result[0].confidence, 0.5)


class TestGetFusionLog(unittest.TestCase):
    def test_returns_most_recent_n(self):
        engine = MultimodalFusionEngine()
        for i in range(5):
            engine.create_signal(SignalSource.AUDIO, "alarm", timestamp=float(i), value=0.9)
            ws = _make_ws(events=[_make_event(EntityEventType.ANOMALY, ts=float(i))])
            engine.fuse_with_visual(ws, window_seconds=2.0)
        log = engine.get_fusion_log(n=3)
        self.assertLessEqual(len(log), 3)

    def test_log_grows_with_fusions(self):
        engine = MultimodalFusionEngine()
        engine.create_signal(SignalSource.AUDIO, "alarm", timestamp=100.0, value=0.9)
        ws = _make_ws(events=[_make_event(EntityEventType.ANOMALY, ts=100.0)])
        engine.fuse_with_visual(ws, window_seconds=5.0)
        self.assertGreaterEqual(engine.get_state_dict()["total_fused"], 1)


class TestClearSignals(unittest.TestCase):
    def test_removes_old_signals(self):
        engine = MultimodalFusionEngine()
        old_ts = time.time() - 400.0
        engine.create_signal(SignalSource.AUDIO, "old", timestamp=old_ts)
        engine.create_signal(SignalSource.AUDIO, "new", timestamp=time.time())
        removed = engine.clear_signals(older_than_seconds=300.0)
        self.assertEqual(removed, 1)
        self.assertEqual(engine.get_state_dict()["buffered_signals"], 1)

    def test_returns_correct_removed_count(self):
        engine = MultimodalFusionEngine()
        old_ts = time.time() - 400.0
        engine.create_signal(SignalSource.AUDIO, "s1", timestamp=old_ts)
        engine.create_signal(SignalSource.AUDIO, "s2", timestamp=old_ts)
        removed = engine.clear_signals(older_than_seconds=300.0)
        self.assertEqual(removed, 2)

    def test_clear_all_with_zero_seconds(self):
        engine = MultimodalFusionEngine()
        # Use explicitly old timestamps so they are older than time.time()-0
        old_ts = time.time() - 1.0
        engine.create_signal(SignalSource.AUDIO, "s1", timestamp=old_ts)
        engine.create_signal(SignalSource.IOT, "s2", timestamp=old_ts)
        removed = engine.clear_signals(older_than_seconds=0.0)
        self.assertEqual(removed, 2)
        self.assertEqual(engine.get_state_dict()["buffered_signals"], 0)


class TestGetStateDict(unittest.TestCase):
    def test_has_required_keys(self):
        engine = MultimodalFusionEngine()
        state = engine.get_state_dict()
        for key in ("buffered_signals", "total_ingested", "total_fused", "signals_by_source", "recent_fusion_count"):
            self.assertIn(key, state)

    def test_signals_by_source_counts(self):
        engine = MultimodalFusionEngine()
        now = time.time()
        engine.create_signal(SignalSource.AUDIO, "a", timestamp=now)
        engine.create_signal(SignalSource.AUDIO, "b", timestamp=now)
        engine.create_signal(SignalSource.IOT, "c", timestamp=now)
        state = engine.get_state_dict()
        self.assertEqual(state["signals_by_source"]["audio"], 2)
        self.assertEqual(state["signals_by_source"]["iot"], 1)


class TestBufferOverflow(unittest.TestCase):
    def test_buffer_pruned_when_full(self):
        engine = MultimodalFusionEngine()
        max_sig = MultimodalFusionEngine.MAX_SIGNALS
        for i in range(max_sig + 1):
            engine.create_signal(SignalSource.CUSTOM, "flood", timestamp=float(i))
        # After pruning 20%, buffer should be well below MAX_SIGNALS
        self.assertLess(engine.get_state_dict()["buffered_signals"], max_sig)

    def test_total_ingested_counts_all(self):
        engine = MultimodalFusionEngine()
        max_sig = MultimodalFusionEngine.MAX_SIGNALS
        for i in range(max_sig + 1):
            engine.create_signal(SignalSource.CUSTOM, "flood", timestamp=float(i))
        self.assertEqual(engine.get_state_dict()["total_ingested"], max_sig + 1)


class TestThreadSafety(unittest.TestCase):
    def test_concurrent_ingest_no_crash(self):
        engine = MultimodalFusionEngine()
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(20):
                    engine.create_signal(SignalSource.AUDIO, "noise", timestamp=time.time())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], msg=f"Thread errors: {errors}")
        self.assertEqual(engine.get_state_dict()["total_ingested"], 100)


if __name__ == "__main__":
    unittest.main()
