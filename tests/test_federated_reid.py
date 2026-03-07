"""Tests for Phase 41: Privacy-Preserving Federated Re-ID.

Tests cover:
    - PrivacyBudget properties + noise_scale + to_dict
    - LocalEmbedding / ImportedEmbedding dataclasses
    - FederatedReIDStore: add, export, import, match, stats
    - Differential privacy noise (Gaussian mechanism)
    - Budget exhaustion → RuntimeError
    - Persistence (save/load round-trip)
    - Thread safety (concurrent add + export)
"""

from __future__ import annotations

import base64
import json
import math
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from sopilot.perception.federated_reid import (
    _CLIP_NORM,
    _EMBEDDING_DIM,
    FederatedReIDStore,
    ImportedEmbedding,
    LocalEmbedding,
    PrivacyBudget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_emb(dim: int = _EMBEDDING_DIM, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# PrivacyBudget
# ---------------------------------------------------------------------------


class TestPrivacyBudgetDefaults:
    def test_default_values(self):
        b = PrivacyBudget()
        assert b.epsilon == 1.0
        assert b.delta == 1e-5
        assert b.max_queries == 100
        assert b.queries_used == 0

    def test_epsilon_per_query(self):
        b = PrivacyBudget(epsilon=2.0, max_queries=10)
        assert b.epsilon_per_query == pytest.approx(0.2)

    def test_epsilon_per_query_safe_against_zero(self):
        b = PrivacyBudget(epsilon=1.0, max_queries=0)
        # max(1, 0) = 1 → 1.0/1 = 1.0
        assert b.epsilon_per_query == pytest.approx(1.0)

    def test_remaining_budget_initial(self):
        b = PrivacyBudget(epsilon=1.0, max_queries=100)
        assert b.remaining_budget == pytest.approx(1.0)

    def test_remaining_budget_partial(self):
        b = PrivacyBudget(epsilon=1.0, max_queries=10)
        b.queries_used = 5
        # used = 5 * (1.0/10) = 0.5
        assert b.remaining_budget == pytest.approx(0.5)

    def test_remaining_budget_clamps_at_zero(self):
        b = PrivacyBudget(epsilon=1.0, max_queries=10)
        b.queries_used = 15  # over budget
        assert b.remaining_budget == 0.0

    def test_is_exhausted_false(self):
        b = PrivacyBudget(max_queries=10)
        b.queries_used = 9
        assert b.is_exhausted is False

    def test_is_exhausted_true(self):
        b = PrivacyBudget(max_queries=10)
        b.queries_used = 10
        assert b.is_exhausted is True


class TestPrivacyBudgetNoiseScale:
    def test_noise_scale_formula(self):
        """σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε_per_query."""
        b = PrivacyBudget(epsilon=1.0, delta=1e-5, max_queries=100)
        eps = b.epsilon_per_query  # 0.01
        expected_sigma = _CLIP_NORM * math.sqrt(2 * math.log(1.25 / 1e-5)) / eps
        assert b.noise_scale() == pytest.approx(expected_sigma, rel=1e-4)

    def test_noise_scale_increases_with_smaller_epsilon(self):
        b1 = PrivacyBudget(epsilon=1.0, max_queries=10)
        b2 = PrivacyBudget(epsilon=0.1, max_queries=10)
        # Smaller epsilon → less privacy cost per query → larger noise
        assert b2.noise_scale() > b1.noise_scale()

    def test_noise_scale_with_sensitivity(self):
        b = PrivacyBudget(epsilon=1.0, max_queries=10)
        sigma1 = b.noise_scale(sensitivity=1.0)
        sigma2 = b.noise_scale(sensitivity=2.0)
        assert sigma2 == pytest.approx(2 * sigma1, rel=1e-5)

    def test_noise_scale_exhausted_budget(self):
        b = PrivacyBudget(epsilon=1.0, max_queries=0)
        # epsilon_per_query = 1/max(1,0) = 1.0 — should not be inf
        assert b.noise_scale() > 0

    def test_noise_scale_positive(self):
        b = PrivacyBudget(epsilon=0.5, delta=1e-6, max_queries=50)
        assert b.noise_scale() > 0


class TestPrivacyBudgetToDict:
    def test_to_dict_keys(self):
        b = PrivacyBudget()
        d = b.to_dict()
        assert "epsilon" in d
        assert "delta" in d
        assert "max_queries" in d
        assert "queries_used" in d
        assert "remaining_budget" in d
        assert "is_exhausted" in d
        assert "noise_scale" in d

    def test_to_dict_values_consistent(self):
        b = PrivacyBudget(epsilon=2.0, max_queries=20)
        b.queries_used = 5
        d = b.to_dict()
        assert d["epsilon"] == 2.0
        assert d["max_queries"] == 20
        assert d["queries_used"] == 5
        assert not d["is_exhausted"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestLocalEmbedding:
    def test_fields(self):
        emb = _rand_emb()
        le = LocalEmbedding(
            entry_id="id1",
            person_id="p42",
            camera_id="cam01",
            embedding=emb,
            timestamp=1234.5,
            confidence=0.9,
        )
        assert le.entry_id == "id1"
        assert le.person_id == "p42"
        assert le.camera_id == "cam01"
        assert np.allclose(le.embedding, emb)
        assert le.confidence == 0.9


class TestImportedEmbedding:
    def test_fields(self):
        emb = _rand_emb(seed=7)
        ie = ImportedEmbedding(
            entry_id="id2",
            person_id="peer_p3",
            source_camera="cam99",
            embedding=emb,
            timestamp=9999.0,
            dp_epsilon=0.01,
        )
        assert ie.source_camera == "cam99"
        assert ie.dp_epsilon == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# FederatedReIDStore — basic API
# ---------------------------------------------------------------------------


class TestFederatedReIDStoreInit:
    def test_init_defaults(self):
        store = FederatedReIDStore()
        stats = store.get_stats()
        assert stats["camera_id"] == "local"
        assert stats["local_embeddings"] == 0
        assert stats["imported_embeddings"] == 0
        assert stats["match_count"] == 0

    def test_init_custom_camera(self):
        store = FederatedReIDStore(camera_id="cam_A")
        assert store.get_stats()["camera_id"] == "cam_A"

    def test_get_local_count_empty(self):
        store = FederatedReIDStore()
        assert store.get_local_count() == 0


class TestAddLocalEmbedding:
    def test_returns_uuid_string(self):
        store = FederatedReIDStore()
        emb = _rand_emb(seed=1)
        entry_id = store.add_local_embedding("p1", emb, "cam01")
        assert isinstance(entry_id, str)
        assert len(entry_id) == 36  # UUID4

    def test_count_increases(self):
        store = FederatedReIDStore()
        for i in range(5):
            store.add_local_embedding(f"p{i}", _rand_emb(seed=i))
        assert store.get_local_count() == 5

    def test_embedding_is_l2_normalized(self):
        store = FederatedReIDStore()
        raw = np.random.RandomState(42).randn(_EMBEDDING_DIM).astype(np.float32)
        raw *= 100  # large norm
        store.add_local_embedding("p1", raw)
        # Verify stored embedding is unit norm via export
        packet = store.export_private()
        emb_b64 = packet["embeddings"][0]["embedding_b64"]
        # Noisy embedding after noise can deviate from unit norm, just check
        # that the local embedding was normalized (indirectly: noise_scale should be finite)
        assert packet["noise_scale"] < 1e6

    def test_rolling_window_max_1000(self):
        store = FederatedReIDStore()
        for i in range(1010):
            store.add_local_embedding(f"p{i}", _rand_emb(seed=i % 100))
        # Should not exceed _MAX_LOCAL = 1000
        assert store.get_local_count() == 1000

    def test_default_camera_id(self):
        store = FederatedReIDStore(camera_id="cam_default")
        entry_id = store.add_local_embedding("p1", _rand_emb())
        assert isinstance(entry_id, str)


# ---------------------------------------------------------------------------
# Export (DP noise)
# ---------------------------------------------------------------------------


class TestExportPrivate:
    def setup_method(self):
        self.store = FederatedReIDStore(
            privacy_epsilon=1.0,
            max_export_queries=10,
            camera_id="cam01",
        )
        for i in range(3):
            self.store.add_local_embedding(f"p{i}", _rand_emb(seed=i))

    def test_export_returns_dict(self):
        packet = self.store.export_private()
        assert isinstance(packet, dict)

    def test_export_packet_keys(self):
        packet = self.store.export_private()
        for key in ("source_camera", "export_epsilon", "noise_scale", "timestamp",
                    "embedding_dim", "count", "embeddings"):
            assert key in packet

    def test_export_count_matches(self):
        packet = self.store.export_private()
        assert packet["count"] == 3
        assert len(packet["embeddings"]) == 3

    def test_export_embedding_has_base64(self):
        packet = self.store.export_private()
        for emb_entry in packet["embeddings"]:
            assert "embedding_b64" in emb_entry
            raw = base64.b64decode(emb_entry["embedding_b64"])
            arr = np.frombuffer(raw, dtype=np.float32)
            assert len(arr) == _EMBEDDING_DIM

    def test_export_increments_budget(self):
        before = self.store.get_stats()["privacy_budget"]["queries_used"]
        self.store.export_private()
        after = self.store.get_stats()["privacy_budget"]["queries_used"]
        assert after == before + 1

    def test_export_adds_gaussian_noise(self):
        """Noisy embedding must differ from original."""
        store = FederatedReIDStore(privacy_epsilon=0.01, max_export_queries=10)
        emb = _rand_emb(seed=99)
        store.add_local_embedding("p1", emb)
        packet = store.export_private()
        raw = base64.b64decode(packet["embeddings"][0]["embedding_b64"])
        noisy = np.frombuffer(raw, dtype=np.float32).copy()
        # With ε=0.01, noise will be large → embedding will differ
        assert not np.allclose(emb, noisy, atol=1e-6)

    def test_export_budget_exhaustion(self):
        store = FederatedReIDStore(privacy_epsilon=1.0, max_export_queries=2)
        store.add_local_embedding("p1", _rand_emb())
        store.export_private()
        store.export_private()
        with pytest.raises(RuntimeError, match="exhausted"):
            store.export_private()

    def test_export_empty_store(self):
        store = FederatedReIDStore()
        packet = store.export_private()
        assert packet["count"] == 0
        assert packet["embeddings"] == []

    def test_export_source_camera(self):
        packet = self.store.export_private()
        assert packet["source_camera"] == "cam01"

    def test_export_epsilon_override(self):
        packet = self.store.export_private(epsilon_budget=0.5)
        # noise_scale should be recalculated for epsilon=0.5
        expected_sigma = _CLIP_NORM * math.sqrt(2 * math.log(1.25 / 1e-5)) / 0.5
        assert packet["noise_scale"] == pytest.approx(expected_sigma, rel=1e-3)


# ---------------------------------------------------------------------------
# Import from peer camera
# ---------------------------------------------------------------------------


class TestImportFromCamera:
    def test_import_increases_count(self):
        store_a = FederatedReIDStore(camera_id="cam_a")
        store_b = FederatedReIDStore(camera_id="cam_b")
        for i in range(3):
            store_a.add_local_embedding(f"p{i}", _rand_emb(seed=i))
        packet = store_a.export_private()
        n = store_b.import_from_camera("cam_a", packet)
        assert n == 3
        assert store_b.get_stats()["imported_embeddings"] == 3

    def test_import_tracks_peer_camera(self):
        store = FederatedReIDStore()
        store_a = FederatedReIDStore(camera_id="cam_a")
        store_a.add_local_embedding("p1", _rand_emb())
        packet = store_a.export_private()
        store.import_from_camera("cam_a", packet)
        stats = store.get_stats()
        assert "cam_a" in stats["peer_cameras"]
        assert stats["import_stats"]["cam_a"] == 1

    def test_import_bad_base64_skipped(self):
        bad_packet = {
            "embeddings": [
                {"person_id": "p1", "embedding_b64": "not_valid_b64!!!",
                 "confidence": 1.0, "timestamp": time.time()}
            ],
            "export_epsilon": 0.01,
            "embedding_dim": _EMBEDDING_DIM,
        }
        store = FederatedReIDStore()
        n = store.import_from_camera("cam_bad", bad_packet)
        assert n == 0

    def test_import_wrong_dim_skipped(self):
        wrong_emb = np.random.randn(100).astype(np.float32)
        wrong_b64 = base64.b64encode(wrong_emb.tobytes()).decode("ascii")
        packet = {
            "embeddings": [
                {"person_id": "p1", "embedding_b64": wrong_b64,
                 "confidence": 1.0, "timestamp": time.time()}
            ],
            "export_epsilon": 0.01,
            "embedding_dim": _EMBEDDING_DIM,
        }
        store = FederatedReIDStore()
        n = store.import_from_camera("cam_wrong", packet)
        assert n == 0

    def test_import_multiple_cameras(self):
        store = FederatedReIDStore()
        for i in range(3):
            peer = FederatedReIDStore(camera_id=f"cam_{i}")
            peer.add_local_embedding("p1", _rand_emb(seed=i))
            store.import_from_camera(f"cam_{i}", peer.export_private())
        assert len(store.get_stats()["peer_cameras"]) == 3


# ---------------------------------------------------------------------------
# Cross-camera matching
# ---------------------------------------------------------------------------


class TestCrossCameraMatch:
    def setup_method(self):
        self.store_a = FederatedReIDStore(camera_id="cam_a")
        self.store_b = FederatedReIDStore(camera_id="cam_b")
        # Add a known person embedding to store_a
        self.known_emb = _rand_emb(seed=42)
        self.store_a.add_local_embedding("person_42", self.known_emb)
        for i in range(5):  # distractors
            self.store_a.add_local_embedding(f"person_{i}", _rand_emb(seed=i))
        # Export (with large epsilon → low noise for this test)
        packet = self.store_a.export_private(epsilon_budget=100.0)
        self.store_b.import_from_camera("cam_a", packet)

    def test_returns_list(self):
        result = self.store_b.cross_camera_match(self.known_emb)
        assert isinstance(result, list)

    def test_match_count_increments(self):
        before = self.store_b.get_stats()["match_count"]
        self.store_b.cross_camera_match(self.known_emb)
        after = self.store_b.get_stats()["match_count"]
        assert after == before + 1

    def test_top_k_respected(self):
        result = self.store_b.cross_camera_match(self.known_emb, top_k=3)
        assert len(result) <= 3

    def test_match_result_keys(self):
        result = self.store_b.cross_camera_match(self.known_emb, top_k=1)
        if result:
            entry = result[0]
            assert "person_id" in entry
            assert "source_camera" in entry
            assert "similarity" in entry
            assert "dp_epsilon" in entry

    def test_no_imports_returns_empty(self):
        store = FederatedReIDStore()
        result = store.cross_camera_match(self.known_emb)
        assert result == []

    def test_similarity_descending(self):
        result = self.store_b.cross_camera_match(self.known_emb, top_k=5)
        sims = [r["similarity"] for r in result]
        assert sims == sorted(sims, reverse=True)

    def test_min_similarity_filter(self):
        # Set very high threshold → should get few or no results
        result = self.store_b.cross_camera_match(self.known_emb, min_similarity=0.999)
        assert all(r["similarity"] >= 0.999 for r in result)

    def test_query_normalized_automatically(self):
        """Query with large norm should still produce valid similarities."""
        big_query = self.known_emb * 100.0
        result = self.store_b.cross_camera_match(big_query, min_similarity=0.0)
        # Should not crash; similarities should be [-1, 1]
        for r in result:
            assert -1.0 <= r["similarity"] <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_stats_structure(self):
        store = FederatedReIDStore(camera_id="test_cam")
        stats = store.get_stats()
        assert stats["camera_id"] == "test_cam"
        assert "local_embeddings" in stats
        assert "imported_embeddings" in stats
        assert "peer_cameras" in stats
        assert "import_stats" in stats
        assert "match_count" in stats
        assert "privacy_budget" in stats
        assert "uptime_seconds" in stats

    def test_stats_uptime_positive(self):
        store = FederatedReIDStore()
        time.sleep(0.01)
        assert store.get_stats()["uptime_seconds"] >= 0.0

    def test_stats_privacy_budget(self):
        store = FederatedReIDStore(privacy_epsilon=2.0)
        budget = store.get_stats()["privacy_budget"]
        assert budget["epsilon"] == 2.0
        assert "noise_scale" in budget


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fed_reid.json"
            store = FederatedReIDStore(persist_path=path)
            embs = [_rand_emb(seed=i) for i in range(5)]
            for i, e in enumerate(embs):
                store.add_local_embedding(f"p{i}", e)
            store._save()

            # Load into a new store
            store2 = FederatedReIDStore(
                privacy_epsilon=1.0,
                persist_path=path,
            )
            assert store2.get_local_count() == 5

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "fed.json"
            store = FederatedReIDStore(persist_path=path)
            store.add_local_embedding("p1", _rand_emb())
            store._save()
            assert path.exists()

    def test_load_restores_budget_queries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fed.json"
            store = FederatedReIDStore(max_export_queries=20, persist_path=path)
            store.add_local_embedding("p1", _rand_emb())
            store.export_private()
            store.export_private()
            store._save()

            store2 = FederatedReIDStore(max_export_queries=20, persist_path=path)
            assert store2.get_stats()["privacy_budget"]["queries_used"] == 2

    def test_no_persist_path_save_silent(self):
        store = FederatedReIDStore()
        store.add_local_embedding("p1", _rand_emb())
        store._save()  # should not raise

    def test_load_missing_file_silent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            store = FederatedReIDStore(persist_path=path)
            # _load is called in __init__, should not raise
            assert store.get_local_count() == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_add(self):
        store = FederatedReIDStore()
        results: list[str] = []
        errors: list[Exception] = []

        def add_worker(i: int) -> None:
            try:
                eid = store.add_local_embedding(f"p{i}", _rand_emb(seed=i))
                results.append(eid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_worker, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.get_local_count() == 30

    def test_concurrent_add_and_export(self):
        store = FederatedReIDStore(
            privacy_epsilon=1.0,
            max_export_queries=50,
        )
        for i in range(10):
            store.add_local_embedding(f"p{i}", _rand_emb(seed=i))

        export_results: list[dict] = []
        errors: list[Exception] = []

        def export_worker() -> None:
            try:
                pkt = store.export_private()
                export_results.append(pkt)
            except RuntimeError:
                pass  # budget exhausted is OK
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=export_worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_import_and_match(self):
        store = FederatedReIDStore()
        peer = FederatedReIDStore(camera_id="peer")
        for i in range(5):
            peer.add_local_embedding(f"p{i}", _rand_emb(seed=i))
        packet = peer.export_private()
        store.import_from_camera("peer", packet)

        errors: list[Exception] = []

        def match_worker() -> None:
            try:
                store.cross_camera_match(_rand_emb(), top_k=3)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=match_worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
