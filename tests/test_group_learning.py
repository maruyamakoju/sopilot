"""Tests for Phase 14B: GroupLearningStore + cross-camera learning endpoints.

Coverage:
    - GroupLearningStore: save, load, delete, list_all, compare
    - SigmaTuner.apply_detector_sigmas()
    - API endpoints: GET/POST /vigil/camera-groups/{id}/learning/...
                     GET /vigil/camera-groups/learning/compare
"""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigma_state(base=2.0, behavioral=None, spatial=None, temporal=None, interaction=None):
    dets = {}
    for det, val in [("behavioral", behavioral), ("spatial", spatial),
                     ("temporal", temporal), ("interaction", interaction)]:
        cur = val if val is not None else base
        dets[det] = {"current_sigma": cur, "base_sigma": base,
                     "delta": round(cur - base, 4), "adjusted": val is not None}
    return {"base_sigma": base, "total_adjustments": 0 if behavioral is None else 1,
            "target_fp_rate": 0.3, "detector_sigmas": dets, "recent_adjustments": []}


def _tuner_stats(total=0, confirmed=0, suppressed=None, trusted=None):
    return {
        "total_feedback": total,
        "confirmed": confirmed,
        "denied": total - confirmed,
        "overall_confirm_rate": confirmed / total if total else 0.0,
        "pairs_tracked": 0, "pairs_suppressed": 0, "pairs_trusted": 0,
        "last_tuning": time.time(),
        "pair_stats": [],
        "suppressed_pairs": suppressed or [],
        "trusted_pairs": trusted or [],
        "min_samples_for_tuning": 10,
    }


# ===========================================================================
# TestGroupLearningStore
# ===========================================================================


class TestGroupLearningStore(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self._store_dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _store(self):
        from sopilot.vigil.group_learning import GroupLearningStore
        return GroupLearningStore(self._store_dir)

    def test_save_creates_json_file(self):
        store = self._store()
        store.save(1, "入口", _sigma_state(), _tuner_stats())
        self.assertTrue(Path(self._store_dir, "group_1.json").exists())

    def test_load_returns_snapshot(self):
        store = self._store()
        store.save(1, "入口", _sigma_state(behavioral=3.2), _tuner_stats(total=10, confirmed=7))
        snap = store.load(1)
        self.assertIsNotNone(snap)
        self.assertEqual(snap["group_id"], 1)
        self.assertEqual(snap["group_name"], "入口")
        self.assertAlmostEqual(snap["detector_sigmas"]["behavioral"], 3.2, places=3)

    def test_load_returns_none_for_nonexistent(self):
        store = self._store()
        self.assertIsNone(store.load(999))

    def test_delete_removes_file(self):
        store = self._store()
        store.save(2, "製造", _sigma_state(), _tuner_stats())
        self.assertTrue(store.delete(2))
        self.assertFalse(Path(self._store_dir, "group_2.json").exists())

    def test_list_all_returns_all_saved_groups(self):
        store = self._store()
        store.save(1, "入口", _sigma_state(), _tuner_stats())
        store.save(2, "製造", _sigma_state(), _tuner_stats())
        result = store.list_all()
        self.assertEqual(len(result), 2)

    def test_compare_returns_recommendations_when_sigma_differs(self):
        store = self._store()
        # behavioral: group1=3.5, group2=1.5 → delta=2.0 >= threshold(0.3)
        store.save(1, "入口", _sigma_state(behavioral=3.5), _tuner_stats())
        store.save(2, "製造", _sigma_state(behavioral=1.5), _tuner_stats())
        result = store.compare()
        recs = result["recommendations"]
        self.assertGreater(len(recs), 0)
        behavioral_rec = next(r for r in recs if r["detector"] == "behavioral")
        self.assertGreaterEqual(behavioral_rec["delta"], 0.3)
        self.assertIn("入口", behavioral_rec["note_ja"] + behavioral_rec["highest_sigma"]["group_name"])

    def test_compare_no_recommendation_for_small_diff(self):
        store = self._store()
        # delta = 0.1 < threshold(0.3) → no recommendation
        store.save(1, "A", _sigma_state(behavioral=2.1), _tuner_stats())
        store.save(2, "B", _sigma_state(behavioral=2.0), _tuner_stats())
        result = store.compare()
        behavioral_recs = [r for r in result["recommendations"] if r["detector"] == "behavioral"]
        self.assertEqual(len(behavioral_recs), 0)


# ===========================================================================
# TestSigmaTunerApplyDetectorSigmas
# ===========================================================================


class TestSigmaTunerApplyDetectorSigmas(unittest.TestCase):
    def _make(self, **kw):
        from sopilot.perception.sigma_tuner import SigmaTuner
        return SigmaTuner(**kw)

    def test_apply_sets_sigma_values(self):
        st = self._make(base_sigma=2.0)
        applied = st.apply_detector_sigmas({"behavioral": 3.5, "spatial": 1.5})
        self.assertAlmostEqual(st.get_sigma("behavioral"), 3.5, places=3)
        self.assertAlmostEqual(st.get_sigma("spatial"), 1.5, places=3)
        self.assertEqual(sorted(applied), sorted(["behavioral", "spatial"]))

    def test_apply_clamps_to_sigma_max(self):
        st = self._make(sigma_max=6.0)
        st.apply_detector_sigmas({"behavioral": 99.0})
        self.assertLessEqual(st.get_sigma("behavioral"), 6.0)

    def test_apply_clamps_to_sigma_min(self):
        st = self._make(sigma_min=1.0)
        st.apply_detector_sigmas({"behavioral": 0.1})
        self.assertGreaterEqual(st.get_sigma("behavioral"), 1.0)

    def test_apply_empty_dict_returns_empty(self):
        st = self._make()
        applied = st.apply_detector_sigmas({})
        self.assertEqual(applied, [])


# ===========================================================================
# TestGroupLearningEndpoints
# ===========================================================================


class _EndpointBase(unittest.TestCase):
    """Base class wiring app state for camera-group learning tests."""

    def setUp(self):
        import os
        import tempfile
        from pathlib import Path as _P
        from fastapi.testclient import TestClient
        from unittest.mock import MagicMock
        from sopilot.main import create_app
        from sopilot.vigil.camera_group_repository import CameraGroupRepository
        from sopilot.vigil.group_learning import GroupLearningStore
        from sopilot.perception.sigma_tuner import SigmaTuner

        self._tmp = tempfile.TemporaryDirectory()
        root = _P(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "cgl-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"

        self.app = create_app()
        self.client = TestClient(self.app)

        # Real SQLite repo (in-memory via tmp path)
        self.repo = self.app.state.camera_group_repo

        # Real GroupLearningStore
        self.store = GroupLearningStore(str(root / "group_learning"))
        self.app.state.group_learning_store = self.store

        # Create a group in the DB
        self.group = self.repo.create(name="テスト入口", description="entrance")
        self.group_id = self.group["id"]

        # SigmaTuner + engine mock
        self.sigma_tuner = SigmaTuner(base_sigma=2.0)
        engine = MagicMock()
        engine._sigma_tuner = self.sigma_tuner
        engine._anomaly_tuner = None
        vlm = MagicMock()
        vlm._engine = engine
        self.app.state.vigil_pipeline._vlm = vlm
        self.engine = engine

    def tearDown(self):
        import os
        self._tmp.cleanup()
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                  "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM"):
            os.environ.pop(k, None)


class TestGroupLearningEndpoints(_EndpointBase):
    def test_export_200(self):
        r = self.client.post(f"/vigil/camera-groups/{self.group_id}/learning/export")
        self.assertEqual(r.status_code, 200)

    def test_export_creates_snapshot(self):
        self.client.post(f"/vigil/camera-groups/{self.group_id}/learning/export")
        snap = self.store.load(self.group_id)
        self.assertIsNotNone(snap)
        self.assertEqual(snap["group_id"], self.group_id)

    def test_export_404_for_nonexistent_group(self):
        r = self.client.post("/vigil/camera-groups/99999/learning/export")
        self.assertEqual(r.status_code, 404)

    def test_get_group_learning_404_before_export(self):
        r = self.client.get(f"/vigil/camera-groups/{self.group_id}/learning")
        self.assertEqual(r.status_code, 404)

    def test_get_group_learning_200_after_export(self):
        self.client.post(f"/vigil/camera-groups/{self.group_id}/learning/export")
        r = self.client.get(f"/vigil/camera-groups/{self.group_id}/learning")
        self.assertEqual(r.status_code, 200)
        self.assertIn("detector_sigmas", r.json())

    def test_import_200(self):
        # Export first, then import
        self.client.post(f"/vigil/camera-groups/{self.group_id}/learning/export")
        r = self.client.post(f"/vigil/camera-groups/{self.group_id}/learning/import")
        self.assertEqual(r.status_code, 200)

    def test_import_applies_sigma_to_engine(self):
        # Set group snapshot with behavioral=4.0
        from sopilot.perception.sigma_tuner import SigmaTuner
        snap_sigma = _sigma_state(behavioral=4.0)
        from sopilot.vigil.group_learning import GroupLearningStore
        self.store.save(self.group_id, "入口", snap_sigma, _tuner_stats())

        self.client.post(f"/vigil/camera-groups/{self.group_id}/learning/import")
        self.assertAlmostEqual(self.sigma_tuner.get_sigma("behavioral"), 4.0, places=3)

    def test_import_404_when_no_snapshot(self):
        r = self.client.post(f"/vigil/camera-groups/{self.group_id}/learning/import")
        self.assertEqual(r.status_code, 404)

    def test_compare_200(self):
        r = self.client.get("/vigil/camera-groups/learning/compare")
        self.assertEqual(r.status_code, 200)
        self.assertIn("groups", r.json())
        self.assertIn("recommendations", r.json())

    def test_compare_empty_when_no_snapshots(self):
        r = self.client.get("/vigil/camera-groups/learning/compare")
        self.assertEqual(r.json()["groups"], [])


class TestGroupLearningCrossGroup(_EndpointBase):
    def test_compare_shows_multiple_groups(self):
        g2 = self.repo.create(name="製造フロア")
        # Export sigma for each group
        self.store.save(self.group_id, "テスト入口", _sigma_state(behavioral=3.5), _tuner_stats())
        self.store.save(g2["id"], "製造フロア", _sigma_state(behavioral=1.5), _tuner_stats())
        r = self.client.get("/vigil/camera-groups/learning/compare")
        self.assertEqual(len(r.json()["groups"]), 2)
        self.assertGreater(len(r.json()["recommendations"]), 0)

    def test_import_from_low_fp_group_lowers_sigma(self):
        """Importing from a low-sigma group should lower engine's sigma."""
        g2 = self.repo.create(name="低FP製造")
        # g2 has lower behavioral sigma than default
        self.store.save(g2["id"], "低FP製造", _sigma_state(behavioral=1.5), _tuner_stats())
        self.sigma_tuner._detector_sigmas["behavioral"] = 3.5  # current high sigma

        self.client.post(f"/vigil/camera-groups/{g2['id']}/learning/import")
        self.assertAlmostEqual(self.sigma_tuner.get_sigma("behavioral"), 1.5, places=3)


if __name__ == "__main__":
    unittest.main()
