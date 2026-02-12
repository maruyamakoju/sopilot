from __future__ import annotations

import hashlib
import hmac
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from conftest import make_test_settings
from sopilot.db import Database
from sopilot.service import SopilotService


def _build_service(data_dir: Path, runtime_mode: str = "api", **overrides) -> SopilotService:
    """Create a service with temp dirs, inline queue, no nightly."""
    defaults = dict(min_scoring_clips=1, max_side=160, audit_signing_key_id="test")
    defaults.update(overrides)
    settings = make_test_settings(data_dir, **defaults)
    db = Database(settings.db_path)
    return SopilotService(settings, db, runtime_mode=runtime_mode)


def _seed_video(svc: SopilotService, task_id: str, role: str, dim: int = 4, n_clips: int = 3) -> int:
    """Insert a video directly into DB and write embedding + meta files."""
    from sopilot.db import VideoCreateInput

    svc.settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
    svc.settings.raw_dir.mkdir(parents=True, exist_ok=True)

    dummy_file = svc.settings.raw_dir / f"dummy_{task_id}_{role}.mp4"
    dummy_file.write_bytes(b"\x00" * 64)

    video_id = svc.db.create_video(
        VideoCreateInput(
            task_id=task_id,
            role=role,
            file_path=str(dummy_file),
            embedding_model="heuristic-v1",
        )
    )

    embeddings = np.random.randn(n_clips, dim).astype(np.float32)
    raw_path = svc.settings.embeddings_dir / f"video_{video_id}.raw.npy"
    emb_path = svc.settings.embeddings_dir / f"video_{video_id}.npy"
    meta_path = svc.settings.embeddings_dir / f"video_{video_id}.json"

    np.save(raw_path, embeddings)
    np.save(emb_path, embeddings)
    meta = [{"clip_idx": i, "start_sec": float(i * 4), "end_sec": float((i + 1) * 4)} for i in range(n_clips)]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    svc.db.finalize_video(
        video_id=video_id,
        raw_embedding_path=str(raw_path),
        embedding_path=str(emb_path),
        clip_meta_path=str(meta_path),
        embedding_model="heuristic-v1",
        num_clips=n_clips,
    )
    svc.db.add_clips(video_id=video_id, task_id=task_id, role=role, rows=meta)

    index_rows = [
        {
            "video_id": video_id,
            "clip_idx": i,
            "start_sec": float(i * 4),
            "end_sec": float((i + 1) * 4),
            "role": role,
        }
        for i in range(n_clips)
    ]
    svc.index.add(task_id=task_id, vectors=embeddings, metadata=index_rows)
    return video_id


class TestAdapterLoading:
    def test_no_adapter_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            svc = _build_service(Path(td) / "data")
            try:
                result = svc._load_current_adapter()
                assert result is None
            finally:
                svc.shutdown()

    def test_adapter_loads_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            svc = _build_service(data_dir)
            try:
                models_dir = svc.settings.models_dir
                models_dir.mkdir(parents=True, exist_ok=True)

                mean = np.array([1.0, 2.0, 3.0], dtype=np.float32)
                std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
                adapter_path = models_dir / "test_adapter.npz"
                np.savez(adapter_path, mean=mean, std=std)

                pointer_path = models_dir / "current_adapter.json"
                pointer_path.write_text(json.dumps({"adapter_path": str(adapter_path)}), encoding="utf-8")

                result = svc._load_current_adapter()
                assert result is not None
                loaded_mean, loaded_std = result
                np.testing.assert_array_almost_equal(loaded_mean, mean)
                np.testing.assert_array_almost_equal(loaded_std, std)
            finally:
                svc.shutdown()

    def test_adapter_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            svc = _build_service(data_dir)
            try:
                models_dir = svc.settings.models_dir
                models_dir.mkdir(parents=True, exist_ok=True)

                mean = np.array([1.0], dtype=np.float32)
                std = np.array([1.0], dtype=np.float32)
                adapter_path = models_dir / "adapter.npz"
                np.savez(adapter_path, mean=mean, std=std)

                pointer_path = models_dir / "current_adapter.json"
                pointer_path.write_text(json.dumps({"adapter_path": str(adapter_path)}), encoding="utf-8")

                first = svc._load_current_adapter()
                second = svc._load_current_adapter()
                # Same object reference means cache hit
                assert first is second
            finally:
                svc.shutdown()

    def test_adapter_disabled_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            svc = _build_service(data_dir, enable_feature_adapter=False)
            try:
                result = svc._load_current_adapter()
                assert result is None
            finally:
                svc.shutdown()

    def test_corrupt_pointer_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            svc = _build_service(data_dir)
            try:
                models_dir = svc.settings.models_dir
                models_dir.mkdir(parents=True, exist_ok=True)
                pointer_path = models_dir / "current_adapter.json"
                pointer_path.write_text("not valid json{{{", encoding="utf-8")

                result = svc._load_current_adapter()
                assert result is None
            finally:
                svc.shutdown()


class TestApplyFeatureAdapter:
    def test_without_adapter_returns_input(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            svc = _build_service(Path(td) / "data", enable_feature_adapter=False)
            try:
                inp = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
                out = svc._apply_feature_adapter(inp)
                np.testing.assert_array_equal(out, inp)
            finally:
                svc.shutdown()

    def test_with_adapter_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            svc = _build_service(data_dir)
            try:
                models_dir = svc.settings.models_dir
                models_dir.mkdir(parents=True, exist_ok=True)

                mean = np.array([0.0, 0.0], dtype=np.float32)
                std = np.array([1.0, 1.0], dtype=np.float32)
                adapter_path = models_dir / "adapter.npz"
                np.savez(adapter_path, mean=mean, std=std)
                pointer_path = models_dir / "current_adapter.json"
                pointer_path.write_text(json.dumps({"adapter_path": str(adapter_path)}), encoding="utf-8")

                inp = np.array([[3.0, 4.0]], dtype=np.float32)
                out = svc._apply_feature_adapter(inp)
                # With mean=0, std=1, adapter is identity z-score -> then L2 normalize
                expected_norm = np.linalg.norm([3.0, 4.0])
                np.testing.assert_almost_equal(out[0, 0], 3.0 / expected_norm, decimal=4)
                np.testing.assert_almost_equal(out[0, 1], 4.0 / expected_norm, decimal=4)
            finally:
                svc.shutdown()

    def test_dimension_mismatch_returns_input(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            svc = _build_service(data_dir)
            try:
                models_dir = svc.settings.models_dir
                models_dir.mkdir(parents=True, exist_ok=True)

                mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                adapter_path = models_dir / "adapter.npz"
                np.savez(adapter_path, mean=mean, std=std)
                pointer_path = models_dir / "current_adapter.json"
                pointer_path.write_text(json.dumps({"adapter_path": str(adapter_path)}), encoding="utf-8")

                # Input has 2 dims, adapter has 3 -> mismatch, should return input as-is
                inp = np.array([[1.0, 2.0]], dtype=np.float32)
                out = svc._apply_feature_adapter(inp)
                np.testing.assert_array_equal(out, inp)
            finally:
                svc.shutdown()


class TestDeleteVideoAndReindex:
    def test_delete_removes_video_and_reindexes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            svc = _build_service(data_dir)
            try:
                vid1 = _seed_video(svc, "task1", "gold", dim=4, n_clips=3)
                vid2 = _seed_video(svc, "task1", "trainee", dim=4, n_clips=2)

                result = svc.delete_video(vid1)
                assert result["video_id"] == vid1
                assert result["task_id"] == "task1"
                # After deleting vid1, only vid2's clips should remain in the index
                assert result["reindexed_clips"] == 2

                # Confirm vid1 is gone from DB
                assert svc.get_video_info(vid1) is None
                # vid2 should still exist
                assert svc.get_video_info(vid2) is not None
            finally:
                svc.shutdown()

    def test_delete_nonexistent_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            svc = _build_service(Path(td) / "data")
            try:
                with pytest.raises(ValueError, match="video not found"):
                    svc.delete_video(99999)
            finally:
                svc.shutdown()


class TestSearch:
    def test_search_returns_results(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            svc = _build_service(data_dir)
            try:
                vid1 = _seed_video(svc, "task1", "gold", dim=4, n_clips=3)
                vid2 = _seed_video(svc, "task1", "trainee", dim=4, n_clips=3)

                result = svc.search("task1", vid1, clip_idx=0, k=5)
                assert result["task_id"] == "task1"
                assert result["query_video_id"] == vid1
                assert result["query_clip_idx"] == 0
                # Should exclude self from results
                assert all(not (item["video_id"] == vid1 and item["clip_idx"] == 0) for item in result["items"])
            finally:
                svc.shutdown()

    def test_search_clip_out_of_range(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            svc = _build_service(data_dir)
            try:
                vid = _seed_video(svc, "task1", "gold", dim=4, n_clips=3)
                with pytest.raises(ValueError, match="clip_idx out of range"):
                    svc.search("task1", vid, clip_idx=99, k=5)
            finally:
                svc.shutdown()

    def test_search_task_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            svc = _build_service(data_dir)
            try:
                vid = _seed_video(svc, "task1", "gold", dim=4, n_clips=3)
                with pytest.raises(ValueError, match="task_id does not match"):
                    svc.search("task_wrong", vid, clip_idx=0, k=5)
            finally:
                svc.shutdown()


class TestSignedAuditExport:
    def test_export_without_key_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            svc = _build_service(Path(td) / "data", audit_signing_key="")
            try:
                with pytest.raises(RuntimeError, match="signing key is not configured"):
                    svc.export_signed_audit_trail(limit=10)
            finally:
                svc.shutdown()

    def test_export_produces_valid_signature(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            secret = "test-secret-key-123"
            svc = _build_service(
                data_dir,
                audit_signing_key=secret,
                audit_signing_key_id="key-1",
            )
            try:
                result = svc.export_signed_audit_trail(limit=10)
                assert "export_id" in result
                assert "signature" in result
                assert result["signature"]["algorithm"] == "hmac-sha256"
                assert result["signature"]["key_id"] == "key-1"

                # Verify the file was written
                export_path = Path(result["file_path"])
                assert export_path.exists()

                # Read the file and verify signature
                saved = json.loads(export_path.read_text(encoding="utf-8"))
                sig_block = saved.pop("signature")
                canonical = json.dumps(saved, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                expected_sig = hmac.new(
                    secret.encode("utf-8"),
                    canonical.encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                assert sig_block["signature_hex"] == expected_sig
            finally:
                svc.shutdown()

    def test_get_audit_export_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            secret = "test-secret"
            svc = _build_service(data_dir, audit_signing_key=secret)
            try:
                result = svc.export_signed_audit_trail(limit=10)
                export_id = result["export_id"]

                path = svc.get_audit_export_path(export_id)
                assert path.exists()
            finally:
                svc.shutdown()

    def test_get_audit_export_invalid_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            svc = _build_service(Path(td) / "data")
            try:
                with pytest.raises(ValueError, match="invalid export_id"):
                    svc.get_audit_export_path("   ")
                with pytest.raises(ValueError, match="audit export not found"):
                    svc.get_audit_export_path("nonexistent_export_id")
            finally:
                svc.shutdown()


class TestQueueMetrics:
    def test_inline_queue_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            svc = _build_service(Path(td) / "data")
            try:
                metrics = svc.get_queue_metrics()
                assert "generated_at" in metrics
                assert metrics["runtime_mode"] == "api"
                assert metrics["queue"]["backend"] == "inline"
                assert "jobs" in metrics
            finally:
                svc.shutdown()

    def test_no_queue_in_worker_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            svc = _build_service(Path(td) / "data", runtime_mode="worker")
            try:
                metrics = svc.get_queue_metrics()
                assert metrics["queue"]["backend"] == "none"
                assert metrics["queue"]["redis_ok"] is None
            finally:
                svc.shutdown()


class TestListVideos:
    def test_list_videos_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            svc = _build_service(Path(td) / "data")
            try:
                result = svc.list_videos(task_id=None, limit=100)
                assert result == []
            finally:
                svc.shutdown()

    def test_list_videos_with_filter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            svc = _build_service(data_dir)
            try:
                _seed_video(svc, "task_a", "gold", dim=4, n_clips=2)
                _seed_video(svc, "task_b", "gold", dim=4, n_clips=2)

                all_videos = svc.list_videos(task_id=None, limit=100)
                assert len(all_videos) == 2

                filtered = svc.list_videos(task_id="task_a", limit=100)
                assert len(filtered) == 1
                assert filtered[0]["task_id"] == "task_a"
            finally:
                svc.shutdown()


class TestBuiltinFeatureAdaptation:
    def test_builtin_adapter_with_sufficient_data(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            svc = _build_service(data_dir)
            try:
                _seed_video(svc, "task1", "gold", dim=4, n_clips=5)
                _seed_video(svc, "task1", "trainee", dim=4, n_clips=5)
                svc.settings.models_dir.mkdir(parents=True, exist_ok=True)

                videos = svc.db.list_videos_for_training(since_created_at=None)
                result = svc._run_builtin_feature_adaptation("test-job-1", videos)
                assert result["status"] == "completed"
                assert result["mode"] == "builtin_feature_adapter"
                assert result["clips_used"] >= 10
                assert result["embedding_dim"] == 4

                # Adapter should now be saved
                pointer_path = svc.settings.models_dir / "current_adapter.json"
                assert pointer_path.exists()
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                assert Path(pointer["adapter_path"]).exists()
            finally:
                svc.shutdown()

    def test_builtin_adapter_skips_with_insufficient_data(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            svc = _build_service(data_dir)
            try:
                # No videos -> insufficient
                result = svc._run_builtin_feature_adaptation("test-job-2", [])
                assert result["status"] == "skipped"
                assert "insufficient" in result["reason"]
            finally:
                svc.shutdown()


class TestServiceLifecycle:
    def test_shutdown_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            svc = _build_service(Path(td) / "data")
            svc.shutdown()
            # Second shutdown should not raise
            svc.shutdown()

    def test_worker_mode_has_no_queue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            svc = _build_service(Path(td) / "data", runtime_mode="worker")
            try:
                assert svc._queue is None
                with pytest.raises(RuntimeError, match="queue manager is not available"):
                    svc.enqueue_score(gold_video_id=1, trainee_video_id=2)
            finally:
                svc.shutdown()

    def test_unsupported_queue_backend_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(ValueError, match="queue_backend"):
                _build_service(Path(td) / "data", queue_backend="nosuchbackend")


class TestNightlyScheduler:
    def test_nightly_status_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            svc = _build_service(Path(td) / "data", nightly_enabled=False)
            try:
                status = svc.get_nightly_status()
                assert status["enabled"] is False
                assert status["next_run_local"] is None
            finally:
                svc.shutdown()

    def test_compute_next_nightly_run(self) -> None:
        from datetime import datetime

        with tempfile.TemporaryDirectory() as td:
            svc = _build_service(Path(td) / "data", nightly_hour_local=3)
            try:
                now = datetime(2026, 2, 8, 10, 0, 0)
                next_run = svc._compute_next_nightly_run(now)
                # 10:00 is past 3:00, so should be next day at 03:00
                assert next_run.hour == 3
                assert next_run.day == 9

                early = datetime(2026, 2, 8, 1, 0, 0)
                next_run_early = svc._compute_next_nightly_run(early)
                # 01:00 is before 3:00, so should be same day at 03:00
                assert next_run_early.hour == 3
                assert next_run_early.day == 8
            finally:
                svc.shutdown()
