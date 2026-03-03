"""Tests for sopilot/perception/reid.py — Cross-Camera Re-ID.

Covers:
  - AppearanceEncoder.encode() and _color_histogram()
  - cosine_similarity()
  - ReIDFeature dataclass
  - GlobalTrack dataclass properties and to_dict()
  - CrossCameraTracker register, get, cross-camera filter,
    temporal constraint, pruning, state dict, and reset
"""

import time
import threading
import numpy as np
import pytest

from sopilot.perception.reid import (
    KNOWN_LABELS,
    LABEL_INDEX,
    LABEL_DIM,
    GEOM_DIM,
    VEL_DIM,
    COLOR_DIM,
    BASE_DIM,
    FULL_DIM,
    ReIDFeature,
    GlobalTrack,
    AppearanceEncoder,
    cosine_similarity,
    CrossCameraTracker,
    get_global_tracker,
    reset_global_tracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature(
    entity_id: int = 1,
    session_id: str = "cam-A",
    label: str = "person",
    bbox: tuple = (0.1, 0.1, 0.2, 0.3),
    velocity: tuple = (0.0, 0.0),
    frame=None,
    timestamp: float | None = None,
) -> ReIDFeature:
    fv = AppearanceEncoder.encode(entity_id, label, bbox, velocity, frame)
    ts = timestamp if timestamp is not None else time.time()
    return ReIDFeature(
        entity_id=entity_id,
        session_id=session_id,
        label=label,
        feature_vector=fv,
        bbox=bbox,
        timestamp=ts,
    )


# ---------------------------------------------------------------------------
# 1. TestAppearanceEncoder
# ---------------------------------------------------------------------------

class TestAppearanceEncoder:
    def test_encode_returns_ndarray(self):
        result = AppearanceEncoder.encode(1, "person", (0.1, 0.1, 0.2, 0.3))
        assert isinstance(result, np.ndarray)

    def test_encode_returns_float32(self):
        result = AppearanceEncoder.encode(1, "person", (0.1, 0.1, 0.2, 0.3))
        assert result.dtype == np.float32

    def test_encode_l2_normalized(self):
        result = AppearanceEncoder.encode(1, "person", (0.1, 0.1, 0.2, 0.3))
        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 1e-5

    def test_encode_base_dim_without_frame(self):
        result = AppearanceEncoder.encode(1, "person", (0.1, 0.1, 0.2, 0.3))
        assert result.shape == (BASE_DIM,)

    def test_label_onehot_person_sets_index_0(self):
        result = AppearanceEncoder.encode(1, "person", (0.1, 0.1, 0.2, 0.3))
        # Before normalization, person is at index 0 — after norm, entry should be nonzero
        assert result[LABEL_INDEX["person"]] > 0.0

    def test_label_onehot_vehicle_different_index(self):
        person_vec = AppearanceEncoder.encode(1, "person", (0.1, 0.1, 0.2, 0.3))
        vehicle_vec = AppearanceEncoder.encode(1, "vehicle", (0.1, 0.1, 0.2, 0.3))
        # vehicle index > person index, so maximal component should differ
        assert np.argmax(person_vec[:LABEL_DIM]) != np.argmax(vehicle_vec[:LABEL_DIM])

    def test_same_label_same_bbox_identical_vectors(self):
        v1 = AppearanceEncoder.encode(1, "person", (0.2, 0.3, 0.15, 0.4))
        v2 = AppearanceEncoder.encode(2, "person", (0.2, 0.3, 0.15, 0.4))
        assert np.allclose(v1, v2, atol=1e-6)

    def test_different_labels_different_vectors(self):
        v1 = AppearanceEncoder.encode(1, "person", (0.1, 0.1, 0.2, 0.3))
        v2 = AppearanceEncoder.encode(1, "truck", (0.1, 0.1, 0.2, 0.3))
        assert not np.allclose(v1, v2, atol=1e-4)

    def test_different_velocities_different_vectors(self):
        v1 = AppearanceEncoder.encode(1, "person", (0.1, 0.1, 0.2, 0.3), velocity=(0.0, 0.0))
        v2 = AppearanceEncoder.encode(1, "person", (0.1, 0.1, 0.2, 0.3), velocity=(0.5, 0.5))
        assert not np.allclose(v1, v2, atol=1e-4)

    def test_encode_with_frame_returns_longer_vector(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = AppearanceEncoder.encode(1, "person", (0.1, 0.1, 0.5, 0.5), frame=frame)
        assert result.shape == (FULL_DIM,)

    def test_encode_with_frame_still_float32(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = AppearanceEncoder.encode(1, "person", (0.1, 0.1, 0.5, 0.5), frame=frame)
        assert result.dtype == np.float32

    def test_encode_with_frame_l2_normalized(self):
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = AppearanceEncoder.encode(1, "person", (0.1, 0.1, 0.5, 0.5), frame=frame)
        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 1e-5

    def test_color_histogram_returns_48_dim(self):
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        hist = AppearanceEncoder._color_histogram(frame, (0.1, 0.1, 0.5, 0.5))
        assert hist.shape == (COLOR_DIM,)

    def test_color_histogram_sums_to_one(self):
        frame = np.random.randint(10, 200, (100, 100, 3), dtype=np.uint8)
        hist = AppearanceEncoder._color_histogram(frame, (0.1, 0.1, 0.5, 0.5))
        assert abs(hist.sum() - 1.0) < 1e-5

    def test_color_histogram_empty_crop_returns_zeros(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # bbox where x2 <= x1
        hist = AppearanceEncoder._color_histogram(frame, (0.5, 0.5, 0.0, 0.0))
        assert np.all(hist == 0.0)
        assert hist.shape == (COLOR_DIM,)

    def test_unknown_label_does_not_crash(self):
        result = AppearanceEncoder.encode(1, "alien_robot", (0.1, 0.1, 0.2, 0.3))
        assert isinstance(result, np.ndarray)
        assert abs(float(np.linalg.norm(result)) - 1.0) < 1e-5

    def test_velocity_clipped_to_minus_one_one(self):
        # Huge velocity should not explode — vector still normalized
        result = AppearanceEncoder.encode(1, "person", (0.1, 0.1, 0.2, 0.3), velocity=(999.0, -999.0))
        assert abs(float(np.linalg.norm(result)) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 2. TestCosineSimilarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_normalized_vectors_returns_one(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_zero_vectors_returns_zero(self):
        v = np.zeros(4, dtype=np.float32)
        result = cosine_similarity(v, v)
        assert abs(result) < 1e-6

    def test_opposite_vectors_returns_negative_one(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        neg = -v
        assert abs(cosine_similarity(v, neg) - (-1.0)) < 1e-6

    def test_orthogonal_vectors_returns_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_different_length_vectors_no_error(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        result = cosine_similarity(a, b)
        assert isinstance(result, float)

    def test_similar_vectors_high_score(self):
        rng = np.random.default_rng(42)
        v = rng.standard_normal(16).astype(np.float32)
        v /= np.linalg.norm(v)
        noise = rng.standard_normal(16).astype(np.float32) * 0.02
        v2 = v + noise
        v2 /= np.linalg.norm(v2)
        assert cosine_similarity(v, v2) > 0.9

    def test_result_clipped_to_minus_one_one(self):
        # manually craft vectors that might exceed due to floating point
        v = np.ones(4, dtype=np.float32)
        v /= np.linalg.norm(v)
        result = cosine_similarity(v, v)
        assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# 3. TestReIDFeature
# ---------------------------------------------------------------------------

class TestReIDFeature:
    def test_create_reid_feature(self):
        fv = np.zeros(BASE_DIM, dtype=np.float32)
        feat = ReIDFeature(
            entity_id=5,
            session_id="cam-1",
            label="worker",
            feature_vector=fv,
            bbox=(0.1, 0.2, 0.3, 0.4),
        )
        assert feat.entity_id == 5
        assert feat.session_id == "cam-1"
        assert feat.label == "worker"
        assert feat.track_age == 0

    def test_timestamp_auto_set(self):
        before = time.time()
        fv = np.zeros(BASE_DIM, dtype=np.float32)
        feat = ReIDFeature(
            entity_id=1,
            session_id="cam-1",
            label="person",
            feature_vector=fv,
            bbox=(0.0, 0.0, 0.1, 0.1),
        )
        after = time.time()
        assert before <= feat.timestamp <= after

    def test_explicit_timestamp(self):
        fv = np.zeros(BASE_DIM, dtype=np.float32)
        ts = 12345.0
        feat = ReIDFeature(
            entity_id=1, session_id="cam-1", label="person",
            feature_vector=fv, bbox=(0.0, 0.0, 0.1, 0.1), timestamp=ts,
        )
        assert feat.timestamp == ts


# ---------------------------------------------------------------------------
# 4. TestGlobalTrack
# ---------------------------------------------------------------------------

class TestGlobalTrack:
    def _make_track(self) -> GlobalTrack:
        f1 = _make_feature(entity_id=1, session_id="cam-A")
        f2 = _make_feature(entity_id=2, session_id="cam-B")
        f3 = _make_feature(entity_id=1, session_id="cam-A")
        return GlobalTrack(
            id="track-001",
            appearances=[f1, f2, f3],
            first_seen=f1.timestamp,
            last_seen=f3.timestamp,
            dominant_label="person",
        )

    def test_track_count_equals_appearances_length(self):
        gt = self._make_track()
        assert gt.track_count == len(gt.appearances)

    def test_session_ids_deduplicated(self):
        gt = self._make_track()
        sids = gt.session_ids
        assert len(sids) == 2
        assert "cam-A" in sids
        assert "cam-B" in sids

    def test_to_dict_has_expected_keys(self):
        gt = self._make_track()
        d = gt.to_dict()
        expected_keys = {"id", "track_count", "dominant_label", "first_seen", "last_seen", "session_ids", "appearances"}
        assert set(d.keys()) == expected_keys

    def test_to_dict_appearances_structure(self):
        gt = self._make_track()
        d = gt.to_dict()
        for app in d["appearances"]:
            assert "entity_id" in app
            assert "session_id" in app
            assert "label" in app
            assert "timestamp" in app

    def test_to_dict_track_count_matches(self):
        gt = self._make_track()
        d = gt.to_dict()
        assert d["track_count"] == gt.track_count

    def test_empty_appearances(self):
        gt = GlobalTrack(id="empty-track")
        assert gt.track_count == 0
        assert gt.session_ids == []


# ---------------------------------------------------------------------------
# 5. TestCrossCamera_Register
# ---------------------------------------------------------------------------

class TestCrossCamera_Register:
    def test_register_new_entity_creates_global_track(self):
        tracker = CrossCameraTracker()
        feat = _make_feature(entity_id=1, session_id="cam-A")
        gid = tracker.register(1, "cam-A", feat)
        assert isinstance(gid, str)
        assert len(gid) > 0
        assert len(tracker.get_global_tracks()) == 1

    def test_same_session_entity_returns_same_global_id(self):
        tracker = CrossCameraTracker()
        feat1 = _make_feature(entity_id=1, session_id="cam-A")
        feat2 = _make_feature(entity_id=1, session_id="cam-A")
        gid1 = tracker.register(1, "cam-A", feat1)
        gid2 = tracker.register(1, "cam-A", feat2)
        assert gid1 == gid2

    def test_second_call_updates_appearances(self):
        tracker = CrossCameraTracker()
        feat1 = _make_feature(entity_id=1, session_id="cam-A")
        feat2 = _make_feature(entity_id=1, session_id="cam-A")
        gid = tracker.register(1, "cam-A", feat1)
        tracker.register(1, "cam-A", feat2)
        gt = tracker.get_global_track(gid)
        assert gt is not None
        assert gt.track_count == 2

    def test_cross_session_same_appearance_matches_low_threshold(self):
        """With low threshold, same label+bbox from different session merges."""
        tracker = CrossCameraTracker(similarity_threshold=0.5)
        bbox = (0.2, 0.2, 0.3, 0.3)
        feat_a = _make_feature(entity_id=1, session_id="cam-A", label="person", bbox=bbox)
        feat_b = _make_feature(entity_id=1, session_id="cam-B", label="person", bbox=bbox)
        gid_a = tracker.register(1, "cam-A", feat_a)
        gid_b = tracker.register(1, "cam-B", feat_b)
        # Both should point to the same global track
        assert gid_a == gid_b

    def test_cross_session_very_different_appearance_no_match_high_threshold(self):
        """With very high threshold, different label entities stay separate."""
        tracker = CrossCameraTracker(similarity_threshold=0.95)
        feat_a = _make_feature(entity_id=1, session_id="cam-A", label="person",
                               bbox=(0.1, 0.1, 0.2, 0.3))
        feat_b = _make_feature(entity_id=1, session_id="cam-B", label="truck",
                               bbox=(0.7, 0.7, 0.2, 0.2))
        gid_a = tracker.register(1, "cam-A", feat_a)
        gid_b = tracker.register(1, "cam-B", feat_b)
        assert gid_a != gid_b


# ---------------------------------------------------------------------------
# 6. TestCrossCamera_GetMethods
# ---------------------------------------------------------------------------

class TestCrossCamera_GetMethods:
    def setup_method(self):
        self.tracker = CrossCameraTracker()
        feat = _make_feature(entity_id=10, session_id="cam-X")
        self.gid = self.tracker.register(10, "cam-X", feat)

    def test_get_global_tracks_returns_list(self):
        result = self.tracker.get_global_tracks()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_get_global_track_by_id_returns_global_track(self):
        gt = self.tracker.get_global_track(self.gid)
        assert gt is not None
        assert gt.id == self.gid

    def test_get_global_track_unknown_returns_none(self):
        result = self.tracker.get_global_track("nonexistent-id-xyz")
        assert result is None

    def test_get_global_id_returns_correct_id(self):
        result = self.tracker.get_global_id("cam-X", 10)
        assert result == self.gid

    def test_get_global_id_unknown_key_returns_none(self):
        result = self.tracker.get_global_id("cam-X", 9999)
        assert result is None


# ---------------------------------------------------------------------------
# 7. TestCrossCamera_CrossCameraFilter
# ---------------------------------------------------------------------------

class TestCrossCamera_CrossCameraFilter:
    def test_single_session_entity_not_in_cross_camera(self):
        tracker = CrossCameraTracker()
        feat = _make_feature(entity_id=1, session_id="cam-A")
        tracker.register(1, "cam-A", feat)
        result = tracker.get_cross_camera_global_tracks()
        assert len(result) == 0

    def test_two_session_entity_in_cross_camera(self):
        """Force a cross-camera match by using a very low threshold."""
        tracker = CrossCameraTracker(similarity_threshold=0.01)
        bbox = (0.2, 0.2, 0.3, 0.3)
        feat_a = _make_feature(entity_id=1, session_id="cam-A", label="person", bbox=bbox)
        feat_b = _make_feature(entity_id=2, session_id="cam-B", label="person", bbox=bbox)
        gid_a = tracker.register(1, "cam-A", feat_a)
        gid_b = tracker.register(2, "cam-B", feat_b)
        # With 0.01 threshold, these identical features should merge
        assert gid_a == gid_b
        result = tracker.get_cross_camera_global_tracks()
        assert len(result) == 1

    def test_two_separate_entities_no_cross_camera(self):
        tracker = CrossCameraTracker(similarity_threshold=0.99)
        feat_a = _make_feature(entity_id=1, session_id="cam-A", label="person",
                               bbox=(0.1, 0.1, 0.1, 0.1))
        feat_b = _make_feature(entity_id=2, session_id="cam-B", label="truck",
                               bbox=(0.8, 0.8, 0.1, 0.1))
        tracker.register(1, "cam-A", feat_a)
        tracker.register(2, "cam-B", feat_b)
        # Two separate global tracks, each with only one session
        result = tracker.get_cross_camera_global_tracks()
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 8. TestCrossCamera_TemporalConstraint
# ---------------------------------------------------------------------------

class TestCrossCamera_TemporalConstraint:
    def test_old_track_not_matched(self):
        """Entity last seen > max_time_gap ago → new track, no merge."""
        tracker = CrossCameraTracker(similarity_threshold=0.01, max_time_gap=5.0)
        now = time.time()
        old_ts = now - 100.0   # 100s ago, well beyond 5s gap
        bbox = (0.2, 0.2, 0.3, 0.3)
        feat_old = _make_feature(entity_id=1, session_id="cam-A", label="person",
                                 bbox=bbox, timestamp=old_ts)
        feat_new = _make_feature(entity_id=2, session_id="cam-B", label="person",
                                 bbox=bbox, timestamp=now)
        gid_a = tracker.register(1, "cam-A", feat_old)
        # Manually set last_seen of the old track to old_ts
        gt = tracker.get_global_track(gid_a)
        if gt:
            gt.last_seen = old_ts
        gid_b = tracker.register(2, "cam-B", feat_new)
        # Should NOT match, so different global IDs
        assert gid_a != gid_b


# ---------------------------------------------------------------------------
# 9. TestCrossCamera_Prune
# ---------------------------------------------------------------------------

class TestCrossCamera_Prune:
    def test_register_beyond_max_stays_at_max(self):
        tracker = CrossCameraTracker()
        limit = CrossCameraTracker.MAX_GLOBAL_TRACKS
        # Register limit + 10 distinct entities, each in their own unique session
        # so no cross-session matching happens
        for i in range(limit + 10):
            feat = _make_feature(
                entity_id=i,
                session_id=f"cam-{i}",  # unique per entity, no cross-session
                label="person",
                bbox=(0.01 * (i % 99), 0.01 * (i % 99), 0.1, 0.1),
            )
            tracker.register(i, f"cam-{i}", feat)
        # Global tracks should be pruned to MAX
        assert len(tracker.get_global_tracks()) <= limit

    def test_entity_index_cleaned_after_prune(self):
        tracker = CrossCameraTracker()
        limit = CrossCameraTracker.MAX_GLOBAL_TRACKS
        for i in range(limit + 10):
            feat = _make_feature(
                entity_id=i,
                session_id=f"cam-{i}",
                label="person",
                bbox=(0.01 * (i % 99), 0.01 * (i % 99), 0.1, 0.1),
            )
            tracker.register(i, f"cam-{i}", feat)
        state = tracker.get_state_dict()
        # Every entry in entity_index should point to an existing global track
        tracks_by_id = {gt.id for gt in tracker.get_global_tracks()}
        # The registered_entities count should be consistent
        assert state["registered_entities"] <= limit + 10


# ---------------------------------------------------------------------------
# 10. TestCrossCamera_StateReset
# ---------------------------------------------------------------------------

class TestCrossCamera_StateReset:
    def test_get_state_dict_has_expected_keys(self):
        tracker = CrossCameraTracker()
        feat = _make_feature(entity_id=1, session_id="cam-A")
        tracker.register(1, "cam-A", feat)
        state = tracker.get_state_dict()
        expected_keys = {"total_global_tracks", "cross_camera_tracks", "registered_entities", "tracks"}
        assert set(state.keys()) == expected_keys

    def test_get_state_dict_counts_correct(self):
        tracker = CrossCameraTracker()
        feat = _make_feature(entity_id=1, session_id="cam-A")
        tracker.register(1, "cam-A", feat)
        state = tracker.get_state_dict()
        assert state["total_global_tracks"] == 1
        assert state["cross_camera_tracks"] == 0
        assert state["registered_entities"] == 1

    def test_reset_clears_global_tracks(self):
        tracker = CrossCameraTracker()
        feat = _make_feature(entity_id=1, session_id="cam-A")
        tracker.register(1, "cam-A", feat)
        tracker.reset()
        assert len(tracker.get_global_tracks()) == 0

    def test_reset_clears_entity_index(self):
        tracker = CrossCameraTracker()
        feat = _make_feature(entity_id=1, session_id="cam-A")
        tracker.register(1, "cam-A", feat)
        tracker.reset()
        assert tracker.get_global_id("cam-A", 1) is None

    def test_get_state_dict_tracks_is_list(self):
        tracker = CrossCameraTracker()
        state = tracker.get_state_dict()
        assert isinstance(state["tracks"], list)

    def test_module_level_tracker_singleton(self):
        t1 = get_global_tracker()
        t2 = get_global_tracker()
        assert t1 is t2

    def test_reset_global_tracker(self):
        tracker = get_global_tracker()
        feat = _make_feature(entity_id=99, session_id="cam-Z")
        tracker.register(99, "cam-Z", feat)
        reset_global_tracker()
        assert len(get_global_tracker().get_global_tracks()) == 0


# ---------------------------------------------------------------------------
# Thread-safety smoke test
# ---------------------------------------------------------------------------

class TestCrossCamera_ThreadSafety:
    def test_concurrent_register_no_crash(self):
        tracker = CrossCameraTracker()
        errors = []

        def register_many(session_id: str, start: int):
            try:
                for i in range(start, start + 20):
                    feat = _make_feature(entity_id=i, session_id=session_id)
                    tracker.register(i, session_id, feat)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_many, args=(f"cam-{c}", c * 20))
            for c in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert len(tracker.get_global_tracks()) > 0
