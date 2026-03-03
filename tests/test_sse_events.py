"""Tests for sopilot.perception.sse_events — Thread-safe SSE event queue."""

import asyncio
import json
import threading
import time

import pytest

import sopilot.perception.sse_events as sse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_session(name: str = "test-session") -> str:
    """Remove a session from the registry so tests start clean."""
    sse.remove(name)
    return name


# ===========================================================================
# 1. TestPercEvent
# ===========================================================================

class TestPercEvent:
    def test_to_sse_starts_with_data(self):
        ev = sse.PercEvent(event_type="ANOMALY", session_id="s1")
        assert ev.to_sse().startswith("data: ")

    def test_to_sse_ends_with_double_newline(self):
        ev = sse.PercEvent(event_type="ANOMALY", session_id="s1")
        assert ev.to_sse().endswith("\n\n")

    def test_to_sse_json_parseable(self):
        ev = sse.PercEvent(event_type="GOAL_DETECTED", session_id="s2", payload={"score": 0.9})
        raw = ev.to_sse()
        # Strip "data: " prefix and trailing newlines
        body = raw[len("data: "):].strip()
        parsed = json.loads(body)
        assert isinstance(parsed, dict)

    def test_to_sse_contains_event_type(self):
        ev = sse.PercEvent(event_type="EPISODE_CLOSED", session_id="s3")
        raw = ev.to_sse()
        body = json.loads(raw[len("data: "):].strip())
        assert body["type"] == "EPISODE_CLOSED"

    def test_to_sse_contains_session_id(self):
        ev = sse.PercEvent(event_type="ENTITY_APPEARED", session_id="my-session")
        body = json.loads(ev.to_sse()[len("data: "):].strip())
        assert body["session_id"] == "my-session"

    def test_to_sse_payload_merged_at_top_level(self):
        ev = sse.PercEvent(
            event_type="DELIBERATION_RESULT",
            session_id="s4",
            payload={"confidence": 0.85, "label": "worker"},
        )
        body = json.loads(ev.to_sse()[len("data: "):].strip())
        assert body["confidence"] == 0.85
        assert body["label"] == "worker"

    def test_to_sse_contains_timestamp(self):
        ev = sse.PercEvent(event_type="ANOMALY", session_id="s5")
        body = json.loads(ev.to_sse()[len("data: "):].strip())
        assert "timestamp" in body
        assert isinstance(body["timestamp"], float)

    def test_timestamp_auto_set_when_not_provided(self):
        before = time.time()
        ev = sse.PercEvent(event_type="ANOMALY", session_id="s6")
        after = time.time()
        assert before <= ev.timestamp <= after

    def test_custom_timestamp_preserved(self):
        custom_ts = 1_700_000_000.0
        ev = sse.PercEvent(event_type="ANOMALY", session_id="s7", timestamp=custom_ts)
        assert ev.timestamp == custom_ts
        body = json.loads(ev.to_sse()[len("data: "):].strip())
        assert body["timestamp"] == custom_ts

    def test_empty_payload_produces_clean_json(self):
        ev = sse.PercEvent(event_type="NL_TASK_TRIGGERED", session_id="s8")
        body = json.loads(ev.to_sse()[len("data: "):].strip())
        assert "type" in body
        assert "session_id" in body
        assert "timestamp" in body


# ===========================================================================
# 2. TestEventQueue
# ===========================================================================

class TestEventQueue:
    def _make_queue(self, session_id: str = "q-test") -> sse.EventQueue:
        return sse.EventQueue(session_id)

    def _make_event(self, session_id: str = "q-test") -> sse.PercEvent:
        return sse.PercEvent(event_type="ANOMALY", session_id=session_id)

    def test_push_returns_true_normally(self):
        q = self._make_queue()
        ev = self._make_event()
        assert q.push(ev) is True

    def test_qsize_increments_after_push(self):
        q = self._make_queue()
        assert q.qsize() == 0
        q.push(self._make_event())
        assert q.qsize() == 1
        q.push(self._make_event())
        assert q.qsize() == 2

    def test_push_count_increments(self):
        q = self._make_queue()
        q.push(self._make_event())
        q.push(self._make_event())
        assert q.push_count == 2

    def test_push_to_full_queue_increments_drop_count(self):
        q = sse.EventQueue("full-q")
        # Fill to MAX_SIZE
        for _ in range(sse.EventQueue.MAX_SIZE):
            q.push(self._make_event("full-q"))
        initial_drop = q.drop_count
        # One more push on a full queue should drop oldest and increment drop_count
        result = q.push(self._make_event("full-q"))
        # drop_count should increase
        assert q.drop_count > initial_drop
        # Queue should not exceed MAX_SIZE
        assert q.qsize() <= sse.EventQueue.MAX_SIZE

    def test_push_returns_false_on_full(self):
        q = sse.EventQueue("full-q2")
        for _ in range(sse.EventQueue.MAX_SIZE):
            q.push(self._make_event("full-q2"))
        result = q.push(self._make_event("full-q2"))
        assert result is False

    def test_clear_empties_queue(self):
        q = self._make_queue()
        for _ in range(10):
            q.push(self._make_event())
        assert q.qsize() == 10
        q.clear()
        assert q.qsize() == 0

    def test_get_stats_returns_dict(self):
        q = self._make_queue("stats-q")
        stats = q.get_stats()
        assert isinstance(stats, dict)

    def test_get_stats_has_required_keys(self):
        q = self._make_queue("stats-q2")
        stats = q.get_stats()
        assert "session_id" in stats
        assert "queued" in stats
        assert "pushed" in stats
        assert "dropped" in stats

    def test_get_stats_session_id_matches(self):
        q = self._make_queue("stats-sid")
        assert q.get_stats()["session_id"] == "stats-sid"

    def test_get_stats_reflects_push_count(self):
        q = self._make_queue("stats-count")
        q.push(self._make_event("stats-count"))
        q.push(self._make_event("stats-count"))
        assert q.get_stats()["pushed"] == 2

    def test_get_stats_queued_reflects_qsize(self):
        q = self._make_queue("stats-queued")
        q.push(self._make_event("stats-queued"))
        assert q.get_stats()["queued"] == 1


# ===========================================================================
# 3. TestEventQueueAsync
# ===========================================================================

class TestEventQueueAsync:
    """Async tests using asyncio.run() wrapper for compatibility."""

    def test_next_event_returns_pushed_event(self):
        """Event pushed before calling next_event() is returned."""
        async def _run():
            q = sse.EventQueue("async-q1")
            ev = sse.PercEvent(event_type="ANOMALY", session_id="async-q1")
            q.push(ev)
            result = await q.next_event(timeout=1.0)
            assert result is not None
            assert result.event_type == "ANOMALY"

        asyncio.run(_run())

    def test_next_event_returns_none_on_timeout(self):
        """Empty queue returns None after timeout expires."""
        async def _run():
            q = sse.EventQueue("async-timeout")
            result = await q.next_event(timeout=0.2)
            assert result is None

        asyncio.run(_run())

    def test_multiple_sequential_events_in_order(self):
        """Multiple events are returned in FIFO order."""
        async def _run():
            q = sse.EventQueue("async-fifo")
            events = [
                sse.PercEvent(event_type="ANOMALY", session_id="async-fifo", payload={"n": i})
                for i in range(3)
            ]
            for ev in events:
                q.push(ev)

            results = []
            for _ in range(3):
                r = await q.next_event(timeout=1.0)
                assert r is not None
                results.append(r)

            for i, r in enumerate(results):
                assert r.payload["n"] == i

        asyncio.run(_run())

    def test_next_event_session_id_preserved(self):
        """Returned event has the correct session_id."""
        async def _run():
            q = sse.EventQueue("async-sid")
            ev = sse.PercEvent(event_type="ENTITY_APPEARED", session_id="async-sid")
            q.push(ev)
            result = await q.next_event(timeout=1.0)
            assert result is not None
            assert result.session_id == "async-sid"

        asyncio.run(_run())

    def test_next_event_after_clear_returns_none(self):
        """After clearing, next_event times out and returns None."""
        async def _run():
            q = sse.EventQueue("async-clear")
            ev = sse.PercEvent(event_type="ANOMALY", session_id="async-clear")
            q.push(ev)
            q.clear()
            result = await q.next_event(timeout=0.2)
            assert result is None

        asyncio.run(_run())


# ===========================================================================
# 4. TestRegistry
# ===========================================================================

class TestRegistry:
    def setup_method(self):
        # Clean up known test session IDs before each test
        for sid in ["reg-new", "reg-same", "reg-get", "reg-unknown",
                    "reg-remove", "reg-push-false", "reg-push-true",
                    "reg-list", "reg-list2"]:
            sse.remove(sid)

    def test_get_or_create_creates_new_queue(self):
        q = sse.get_or_create("reg-new")
        assert isinstance(q, sse.EventQueue)

    def test_get_or_create_returns_same_object_on_second_call(self):
        q1 = sse.get_or_create("reg-same")
        q2 = sse.get_or_create("reg-same")
        assert q1 is q2

    def test_get_returns_none_for_unknown_session(self):
        result = sse.get("reg-unknown")
        assert result is None

    def test_get_returns_queue_after_get_or_create(self):
        sse.get_or_create("reg-get")
        result = sse.get("reg-get")
        assert isinstance(result, sse.EventQueue)

    def test_remove_removes_session(self):
        sse.get_or_create("reg-remove")
        sse.remove("reg-remove")
        assert sse.get("reg-remove") is None

    def test_remove_nonexistent_session_no_error(self):
        # Should not raise
        sse.remove("reg-nonexistent-xyz")

    def test_push_event_returns_false_for_unknown_session(self):
        result = sse.push_event("reg-push-false", "ANOMALY", {})
        assert result is False

    def test_push_event_returns_true_for_known_session(self):
        sse.get_or_create("reg-push-true")
        result = sse.push_event("reg-push-true", "ANOMALY", {"x": 1})
        assert result is True

    def test_push_event_event_is_in_queue(self):
        sse.get_or_create("reg-push-true")
        sse.push_event("reg-push-true", "GOAL_DETECTED", {"score": 0.7})
        q = sse.get("reg-push-true")
        assert q is not None
        assert q.qsize() >= 1

    def test_list_sessions_includes_registered_session(self):
        sse.get_or_create("reg-list")
        sessions = sse.list_sessions()
        assert "reg-list" in sessions

    def test_list_sessions_includes_multiple_sessions(self):
        sse.get_or_create("reg-list")
        sse.get_or_create("reg-list2")
        sessions = sse.list_sessions()
        assert "reg-list" in sessions
        assert "reg-list2" in sessions

    def test_list_sessions_does_not_include_removed_session(self):
        sse.get_or_create("reg-remove")
        sse.remove("reg-remove")
        sessions = sse.list_sessions()
        assert "reg-remove" not in sessions


# ===========================================================================
# 5. TestEventGenerator
# ===========================================================================

class TestEventGenerator:
    def setup_method(self):
        sse.remove("gen-session")
        sse.remove("gen-keepalive")

    def test_event_generator_yields_data_sse(self):
        """Generator yields SSE string starting with 'data:' for a queued event."""
        async def _run():
            sid = "gen-session"
            sse.remove(sid)
            # Pre-populate the queue before creating generator
            eq = sse.get_or_create(sid)
            eq.push(sse.PercEvent(event_type="ANOMALY", session_id=sid, payload={"v": 42}))

            gen = sse.event_generator(sid)
            result = await gen.__anext__()
            assert result.startswith("data: ")
            # Verify it is valid JSON
            body = json.loads(result[len("data: "):].strip())
            assert body["type"] == "ANOMALY"

        asyncio.run(_run())

    def test_event_generator_yields_keepalive_when_empty(self):
        """Generator yields keepalive comment when queue is empty (short timeout)."""
        async def _run():
            sid = "gen-keepalive"
            sse.remove(sid)
            # Override next_event to use a very short timeout by patching the queue
            eq = sse.get_or_create(sid)
            # Monkey-patch next_event on this instance to use a short timeout
            original = eq.next_event

            async def short_timeout(timeout=15.0):
                return await original(timeout=0.2)

            eq.next_event = short_timeout  # type: ignore[method-assign]

            gen = sse.event_generator(sid)
            result = await gen.__anext__()
            assert "keepalive" in result

        asyncio.run(_run())

    def test_event_generator_data_contains_session_id(self):
        """SSE data body contains the session_id field."""
        async def _run():
            sid = "gen-session"
            sse.remove(sid)
            eq = sse.get_or_create(sid)
            eq.push(sse.PercEvent(event_type="ENTITY_APPEARED", session_id=sid))

            gen = sse.event_generator(sid)
            result = await gen.__anext__()
            body = json.loads(result[len("data: "):].strip())
            assert body["session_id"] == sid

        asyncio.run(_run())


# ===========================================================================
# 6. TestConcurrency
# ===========================================================================

class TestConcurrency:
    def test_concurrent_pushes_push_plus_drop_equals_total_attempts(self):
        """5 threads each push 50 events. push_count + drop_count == 250. No exceptions."""
        sid = "concurrency-test"
        sse.remove(sid)
        eq = sse.get_or_create(sid)

        num_threads = 5
        pushes_per_thread = 50
        total = num_threads * pushes_per_thread

        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(pushes_per_thread):
                    eq.push(sse.PercEvent(event_type="ANOMALY", session_id=sid))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Exceptions in threads: {errors}"
        assert eq.push_count + eq.drop_count == total

    def test_concurrent_pushes_no_data_corruption(self):
        """Concurrent pushes do not corrupt push_count (type remains int)."""
        sid = "concurrency-corruption"
        sse.remove(sid)
        eq = sse.get_or_create(sid)

        def worker():
            for _ in range(20):
                eq.push(sse.PercEvent(event_type="ANOMALY", session_id=sid))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert isinstance(eq.push_count, int)
        assert isinstance(eq.drop_count, int)
        assert eq.push_count + eq.drop_count == 100

    def test_concurrent_registry_access_no_race(self):
        """Concurrent get_or_create and push_event calls do not raise."""
        errors: list[Exception] = []

        def worker(i: int):
            try:
                sid = f"concurrent-reg-{i % 3}"
                sse.get_or_create(sid)
                sse.push_event(sid, "ANOMALY", {"i": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Cleanup
        for i in range(3):
            sse.remove(f"concurrent-reg-{i}")
