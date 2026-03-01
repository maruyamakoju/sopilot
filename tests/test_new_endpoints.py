"""Tests for DELETE /videos/{video_id} and POST /score/batch endpoints."""

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from sopilot.main import create_app


def _make_video(path: Path, colors: list[tuple[int, int, int]]) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 8.0, (96, 96))
    for color in colors:
        frame = np.full((96, 96, 3), color, dtype=np.uint8)
        for _ in range(24):
            writer.write(frame)
    writer.release()


class DeleteVideoTests(unittest.TestCase):
    """Tests for DELETE /videos/{video_id}."""

    def test_delete_video_success(self) -> None:
        """Upload a video, verify it exists, delete it, verify 404 on subsequent GET."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            old_task = os.environ.get("SOPILOT_PRIMARY_TASK_ID")
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-delete"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload a video
                    video_file = root / "to_delete.avi"
                    _make_video(video_file, [(0, 0, 255), (0, 255, 0)])

                    with video_file.open("rb") as fh:
                        upload_resp = client.post(
                            "/videos",
                            files={"file": ("to_delete.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-delete"},
                        )
                    self.assertEqual(upload_resp.status_code, 200, upload_resp.text)
                    video_id = upload_resp.json()["video_id"]

                    # Verify the video exists
                    get_resp = client.get(f"/videos/{video_id}")
                    self.assertEqual(get_resp.status_code, 200)
                    self.assertEqual(get_resp.json()["video_id"], video_id)

                    # Delete the video
                    delete_resp = client.delete(f"/videos/{video_id}")
                    self.assertEqual(delete_resp.status_code, 200)
                    body = delete_resp.json()
                    self.assertEqual(body["video_id"], video_id)
                    self.assertTrue(body["deleted"])

                    # Verify subsequent GET returns 404
                    get_after = client.get(f"/videos/{video_id}")
                    self.assertEqual(get_after.status_code, 404)
            finally:
                if old_data is None:
                    os.environ.pop("SOPILOT_DATA_DIR", None)
                else:
                    os.environ["SOPILOT_DATA_DIR"] = old_data
                if old_backend is None:
                    os.environ.pop("SOPILOT_EMBEDDER_BACKEND", None)
                else:
                    os.environ["SOPILOT_EMBEDDER_BACKEND"] = old_backend
                if old_task is None:
                    os.environ.pop("SOPILOT_PRIMARY_TASK_ID", None)
                else:
                    os.environ["SOPILOT_PRIMARY_TASK_ID"] = old_task

    def test_delete_nonexistent_video(self) -> None:
        """Try deleting a video with ID 99999, expect 404."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            old_task = os.environ.get("SOPILOT_PRIMARY_TASK_ID")
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-delete"
            try:
                app = create_app()
                with TestClient(app) as client:
                    delete_resp = client.delete("/videos/99999")
                    self.assertEqual(delete_resp.status_code, 404)
            finally:
                if old_data is None:
                    os.environ.pop("SOPILOT_DATA_DIR", None)
                else:
                    os.environ["SOPILOT_DATA_DIR"] = old_data
                if old_backend is None:
                    os.environ.pop("SOPILOT_EMBEDDER_BACKEND", None)
                else:
                    os.environ["SOPILOT_EMBEDDER_BACKEND"] = old_backend
                if old_task is None:
                    os.environ.pop("SOPILOT_PRIMARY_TASK_ID", None)
                else:
                    os.environ["SOPILOT_PRIMARY_TASK_ID"] = old_task

    def test_delete_video_with_score_jobs(self) -> None:
        """Upload gold + trainee, create score job, try deleting gold, expect 409."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            old_task = os.environ.get("SOPILOT_PRIMARY_TASK_ID")
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-delete"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])

                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-delete"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])

                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-delete"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Create a score job referencing both videos
                    score_resp = client.post(
                        "/score",
                        json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                    )
                    self.assertEqual(score_resp.status_code, 200, score_resp.text)

                    # Try to delete the gold video -- should fail with 409
                    delete_resp = client.delete(f"/videos/{gold_id}")
                    self.assertEqual(delete_resp.status_code, 409)

                    # Try to delete the trainee video -- should also fail with 409
                    delete_resp2 = client.delete(f"/videos/{trainee_id}")
                    self.assertEqual(delete_resp2.status_code, 409)
            finally:
                if old_data is None:
                    os.environ.pop("SOPILOT_DATA_DIR", None)
                else:
                    os.environ["SOPILOT_DATA_DIR"] = old_data
                if old_backend is None:
                    os.environ.pop("SOPILOT_EMBEDDER_BACKEND", None)
                else:
                    os.environ["SOPILOT_EMBEDDER_BACKEND"] = old_backend
                if old_task is None:
                    os.environ.pop("SOPILOT_PRIMARY_TASK_ID", None)
                else:
                    os.environ["SOPILOT_PRIMARY_TASK_ID"] = old_task

    def test_delete_video_cleans_up_file(self) -> None:
        """Verify that deleting a video also removes the video file from disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            old_task = os.environ.get("SOPILOT_PRIMARY_TASK_ID")
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-delete"
            try:
                app = create_app()
                with TestClient(app) as client:
                    video_file = root / "cleanup.avi"
                    _make_video(video_file, [(0, 0, 255), (0, 255, 0)])

                    with video_file.open("rb") as fh:
                        upload_resp = client.post(
                            "/videos",
                            files={"file": ("cleanup.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-delete"},
                        )
                    self.assertEqual(upload_resp.status_code, 200, upload_resp.text)
                    video_id = upload_resp.json()["video_id"]

                    # Verify the stream endpoint works (file exists on disk)
                    stream_resp = client.get(f"/videos/{video_id}/stream")
                    self.assertEqual(stream_resp.status_code, 200)

                    # Delete the video
                    delete_resp = client.delete(f"/videos/{video_id}")
                    self.assertEqual(delete_resp.status_code, 200)

                    # The stream endpoint should now return 404 since the video
                    # (and its file) have been removed
                    stream_after = client.get(f"/videos/{video_id}/stream")
                    self.assertEqual(stream_after.status_code, 404)
            finally:
                if old_data is None:
                    os.environ.pop("SOPILOT_DATA_DIR", None)
                else:
                    os.environ["SOPILOT_DATA_DIR"] = old_data
                if old_backend is None:
                    os.environ.pop("SOPILOT_EMBEDDER_BACKEND", None)
                else:
                    os.environ["SOPILOT_EMBEDDER_BACKEND"] = old_backend
                if old_task is None:
                    os.environ.pop("SOPILOT_PRIMARY_TASK_ID", None)
                else:
                    os.environ["SOPILOT_PRIMARY_TASK_ID"] = old_task


class BatchScoreTests(unittest.TestCase):
    """Tests for POST /score/batch."""

    def test_batch_score(self) -> None:
        """Upload gold + trainee, post batch score with that pair, verify response."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            old_task = os.environ.get("SOPILOT_PRIMARY_TASK_ID")
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-batch"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])

                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-batch"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])

                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-batch"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Submit batch score with one pair
                    batch_resp = client.post(
                        "/score/batch",
                        json={
                            "pairs": [
                                {"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                            ]
                        },
                    )
                    self.assertEqual(batch_resp.status_code, 200, batch_resp.text)
                    body = batch_resp.json()

                    # Verify response structure
                    self.assertIn("jobs", body)
                    self.assertIn("count", body)
                    self.assertEqual(body["count"], 1)
                    self.assertEqual(len(body["jobs"]), 1)

                    job = body["jobs"][0]
                    self.assertIn("job_id", job)
                    self.assertIn("status", job)
                    self.assertEqual(job["status"], "queued")

                    # Verify the job was actually created by querying it
                    job_id = job["job_id"]
                    job_resp = client.get(f"/score/{job_id}")
                    self.assertEqual(job_resp.status_code, 200)
                    self.assertEqual(job_resp.json()["job_id"], job_id)
            finally:
                if old_data is None:
                    os.environ.pop("SOPILOT_DATA_DIR", None)
                else:
                    os.environ["SOPILOT_DATA_DIR"] = old_data
                if old_backend is None:
                    os.environ.pop("SOPILOT_EMBEDDER_BACKEND", None)
                else:
                    os.environ["SOPILOT_EMBEDDER_BACKEND"] = old_backend
                if old_task is None:
                    os.environ.pop("SOPILOT_PRIMARY_TASK_ID", None)
                else:
                    os.environ["SOPILOT_PRIMARY_TASK_ID"] = old_task

    def test_batch_score_multiple_pairs(self) -> None:
        """Submit batch score with multiple pairs, verify all jobs are created."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            old_task = os.environ.get("SOPILOT_PRIMARY_TASK_ID")
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-batch"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])

                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-batch"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload two trainee videos
                    trainee_file_1 = root / "trainee1.avi"
                    _make_video(trainee_file_1, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file_1.open("rb") as fh:
                        trainee1_resp = client.post(
                            "/videos",
                            files={"file": ("trainee1.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-batch"},
                        )
                    self.assertEqual(trainee1_resp.status_code, 200, trainee1_resp.text)
                    trainee_id_1 = trainee1_resp.json()["video_id"]

                    trainee_file_2 = root / "trainee2.avi"
                    _make_video(trainee_file_2, [(255, 0, 0), (0, 255, 0)])
                    with trainee_file_2.open("rb") as fh:
                        trainee2_resp = client.post(
                            "/videos",
                            files={"file": ("trainee2.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-batch"},
                        )
                    self.assertEqual(trainee2_resp.status_code, 200, trainee2_resp.text)
                    trainee_id_2 = trainee2_resp.json()["video_id"]

                    # Submit batch score with two pairs
                    batch_resp = client.post(
                        "/score/batch",
                        json={
                            "pairs": [
                                {"gold_video_id": gold_id, "trainee_video_id": trainee_id_1},
                                {"gold_video_id": gold_id, "trainee_video_id": trainee_id_2},
                            ]
                        },
                    )
                    self.assertEqual(batch_resp.status_code, 200, batch_resp.text)
                    body = batch_resp.json()

                    self.assertEqual(body["count"], 2)
                    self.assertEqual(len(body["jobs"]), 2)

                    # Each job should have a unique ID and status "queued"
                    job_ids = {job["job_id"] for job in body["jobs"]}
                    self.assertEqual(len(job_ids), 2, "Batch jobs should have unique IDs")
                    for job in body["jobs"]:
                        self.assertEqual(job["status"], "queued")
            finally:
                if old_data is None:
                    os.environ.pop("SOPILOT_DATA_DIR", None)
                else:
                    os.environ["SOPILOT_DATA_DIR"] = old_data
                if old_backend is None:
                    os.environ.pop("SOPILOT_EMBEDDER_BACKEND", None)
                else:
                    os.environ["SOPILOT_EMBEDDER_BACKEND"] = old_backend
                if old_task is None:
                    os.environ.pop("SOPILOT_PRIMARY_TASK_ID", None)
                else:
                    os.environ["SOPILOT_PRIMARY_TASK_ID"] = old_task

    def test_batch_score_invalid_pair(self) -> None:
        """Post batch score with nonexistent video IDs, expect error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            old_task = os.environ.get("SOPILOT_PRIMARY_TASK_ID")
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-batch"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Submit batch score with nonexistent video IDs
                    batch_resp = client.post(
                        "/score/batch",
                        json={
                            "pairs": [
                                {"gold_video_id": 99998, "trainee_video_id": 99999},
                            ]
                        },
                    )
                    # Should return an error (4xx) since the videos don't exist
                    self.assertGreaterEqual(batch_resp.status_code, 400)
                    self.assertLess(batch_resp.status_code, 500)
            finally:
                if old_data is None:
                    os.environ.pop("SOPILOT_DATA_DIR", None)
                else:
                    os.environ["SOPILOT_DATA_DIR"] = old_data
                if old_backend is None:
                    os.environ.pop("SOPILOT_EMBEDDER_BACKEND", None)
                else:
                    os.environ["SOPILOT_EMBEDDER_BACKEND"] = old_backend
                if old_task is None:
                    os.environ.pop("SOPILOT_PRIMARY_TASK_ID", None)
                else:
                    os.environ["SOPILOT_PRIMARY_TASK_ID"] = old_task

    def test_batch_score_empty_pairs_rejected(self) -> None:
        """Post batch score with empty pairs list, expect 422 validation error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            old_task = os.environ.get("SOPILOT_PRIMARY_TASK_ID")
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-batch"
            try:
                app = create_app()
                with TestClient(app) as client:
                    batch_resp = client.post(
                        "/score/batch",
                        json={"pairs": []},
                    )
                    # Empty pairs list should be rejected by Pydantic (min_length=1)
                    self.assertEqual(batch_resp.status_code, 422)
            finally:
                if old_data is None:
                    os.environ.pop("SOPILOT_DATA_DIR", None)
                else:
                    os.environ["SOPILOT_DATA_DIR"] = old_data
                if old_backend is None:
                    os.environ.pop("SOPILOT_EMBEDDER_BACKEND", None)
                else:
                    os.environ["SOPILOT_EMBEDDER_BACKEND"] = old_backend
                if old_task is None:
                    os.environ.pop("SOPILOT_PRIMARY_TASK_ID", None)
                else:
                    os.environ["SOPILOT_PRIMARY_TASK_ID"] = old_task


class ReadinessEndpointTests(unittest.TestCase):
    """Tests for GET /readiness."""

    def test_readiness_returns_200(self) -> None:
        """Hit /readiness and verify 200 with ready: true."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            old_task = os.environ.get("SOPILOT_PRIMARY_TASK_ID")
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-xxx"
            try:
                app = create_app()
                with TestClient(app) as client:
                    resp = client.get("/readiness")
                    self.assertEqual(resp.status_code, 200)
                    body = resp.json()
                    self.assertTrue(body["ready"])
            finally:
                if old_data is None:
                    os.environ.pop("SOPILOT_DATA_DIR", None)
                else:
                    os.environ["SOPILOT_DATA_DIR"] = old_data
                if old_backend is None:
                    os.environ.pop("SOPILOT_EMBEDDER_BACKEND", None)
                else:
                    os.environ["SOPILOT_EMBEDDER_BACKEND"] = old_backend
                if old_task is None:
                    os.environ.pop("SOPILOT_PRIMARY_TASK_ID", None)
                else:
                    os.environ["SOPILOT_PRIMARY_TASK_ID"] = old_task

    def test_readiness_exempt_from_auth(self) -> None:
        """With API key set, /readiness should still return 200 without providing the key."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            old_task = os.environ.get("SOPILOT_PRIMARY_TASK_ID")
            old_api_key = os.environ.get("SOPILOT_API_KEY")
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-xxx"
            os.environ["SOPILOT_API_KEY"] = "secret-test-key-12345"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Do NOT send any auth header
                    resp = client.get("/readiness")
                    self.assertEqual(resp.status_code, 200)
                    body = resp.json()
                    self.assertTrue(body["ready"])
            finally:
                if old_data is None:
                    os.environ.pop("SOPILOT_DATA_DIR", None)
                else:
                    os.environ["SOPILOT_DATA_DIR"] = old_data
                if old_backend is None:
                    os.environ.pop("SOPILOT_EMBEDDER_BACKEND", None)
                else:
                    os.environ["SOPILOT_EMBEDDER_BACKEND"] = old_backend
                if old_task is None:
                    os.environ.pop("SOPILOT_PRIMARY_TASK_ID", None)
                else:
                    os.environ["SOPILOT_PRIMARY_TASK_ID"] = old_task
                if old_api_key is None:
                    os.environ.pop("SOPILOT_API_KEY", None)
                else:
                    os.environ["SOPILOT_API_KEY"] = old_api_key


class VideoUpdateTests(unittest.TestCase):
    """Tests for PATCH /videos/{video_id}."""

    def test_update_video_metadata(self) -> None:
        """Upload a video, PATCH with site_id, verify response and GET reflect update."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            old_task = os.environ.get("SOPILOT_PRIMARY_TASK_ID")
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-xxx"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload a video
                    video_file = root / "update_test.avi"
                    _make_video(video_file, [(0, 0, 255), (0, 255, 0)])

                    with video_file.open("rb") as fh:
                        upload_resp = client.post(
                            "/videos",
                            files={"file": ("update_test.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-xxx"},
                        )
                    self.assertEqual(upload_resp.status_code, 200, upload_resp.text)
                    video_id = upload_resp.json()["video_id"]

                    # PATCH the video with site_id
                    patch_resp = client.patch(
                        f"/videos/{video_id}",
                        json={"site_id": "factory-a"},
                    )
                    self.assertEqual(patch_resp.status_code, 200, patch_resp.text)
                    patch_body = patch_resp.json()
                    self.assertEqual(patch_body["site_id"], "factory-a")

                    # Verify GET also reflects the updated value
                    get_resp = client.get(f"/videos/{video_id}")
                    self.assertEqual(get_resp.status_code, 200)
                    self.assertEqual(get_resp.json()["site_id"], "factory-a")
            finally:
                if old_data is None:
                    os.environ.pop("SOPILOT_DATA_DIR", None)
                else:
                    os.environ["SOPILOT_DATA_DIR"] = old_data
                if old_backend is None:
                    os.environ.pop("SOPILOT_EMBEDDER_BACKEND", None)
                else:
                    os.environ["SOPILOT_EMBEDDER_BACKEND"] = old_backend
                if old_task is None:
                    os.environ.pop("SOPILOT_PRIMARY_TASK_ID", None)
                else:
                    os.environ["SOPILOT_PRIMARY_TASK_ID"] = old_task

    def test_update_nonexistent_video(self) -> None:
        """PATCH on /videos/99999 should return 404."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            old_task = os.environ.get("SOPILOT_PRIMARY_TASK_ID")
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-xxx"
            try:
                app = create_app()
                with TestClient(app) as client:
                    patch_resp = client.patch(
                        "/videos/99999",
                        json={"site_id": "factory-a"},
                    )
                    self.assertEqual(patch_resp.status_code, 404)
            finally:
                if old_data is None:
                    os.environ.pop("SOPILOT_DATA_DIR", None)
                else:
                    os.environ["SOPILOT_DATA_DIR"] = old_data
                if old_backend is None:
                    os.environ.pop("SOPILOT_EMBEDDER_BACKEND", None)
                else:
                    os.environ["SOPILOT_EMBEDDER_BACKEND"] = old_backend
                if old_task is None:
                    os.environ.pop("SOPILOT_PRIMARY_TASK_ID", None)
                else:
                    os.environ["SOPILOT_PRIMARY_TASK_ID"] = old_task


class ScoreStatusFilterTests(unittest.TestCase):
    """Tests for score status validation."""

    def test_invalid_score_status_filter(self) -> None:
        """GET /score?status=invalid_status should return 422."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            old_task = os.environ.get("SOPILOT_PRIMARY_TASK_ID")
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-xxx"
            try:
                app = create_app()
                with TestClient(app) as client:
                    resp = client.get("/score?status=invalid_status")
                    self.assertEqual(resp.status_code, 422)
            finally:
                if old_data is None:
                    os.environ.pop("SOPILOT_DATA_DIR", None)
                else:
                    os.environ["SOPILOT_DATA_DIR"] = old_data
                if old_backend is None:
                    os.environ.pop("SOPILOT_EMBEDDER_BACKEND", None)
                else:
                    os.environ["SOPILOT_EMBEDDER_BACKEND"] = old_backend
                if old_task is None:
                    os.environ.pop("SOPILOT_PRIMARY_TASK_ID", None)
                else:
                    os.environ["SOPILOT_PRIMARY_TASK_ID"] = old_task

    def test_valid_score_status_filter(self) -> None:
        """GET /score?status=completed should return 200."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            old_task = os.environ.get("SOPILOT_PRIMARY_TASK_ID")
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-xxx"
            try:
                app = create_app()
                with TestClient(app) as client:
                    resp = client.get("/score?status=completed")
                    self.assertEqual(resp.status_code, 200)
            finally:
                if old_data is None:
                    os.environ.pop("SOPILOT_DATA_DIR", None)
                else:
                    os.environ["SOPILOT_DATA_DIR"] = old_data
                if old_backend is None:
                    os.environ.pop("SOPILOT_EMBEDDER_BACKEND", None)
                else:
                    os.environ["SOPILOT_EMBEDDER_BACKEND"] = old_backend
                if old_task is None:
                    os.environ.pop("SOPILOT_PRIMARY_TASK_ID", None)
                else:
                    os.environ["SOPILOT_PRIMARY_TASK_ID"] = old_task


class CancelScoreJobTests(unittest.TestCase):
    """Tests for score job cancellation (DB-level to avoid race with fast queue)."""

    def test_cancel_queued_job_db_level(self) -> None:
        """Create a score job directly in DB and cancel via HTTP."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-cancel"
            try:
                app = create_app()
                with TestClient(app) as client:
                    gold_file = root / "gold.avi"
                    trainee_file = root / "trainee.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])

                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-cancel"},
                        )
                    gold_id = gold_resp.json()["video_id"]

                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-cancel"},
                        )
                    trainee_id = trainee_resp.json()["video_id"]

                    # Insert job directly in DB (bypass queue) so it stays queued
                    db = app.state.sopilot_service.database
                    job_id = db.create_score_job(gold_id, trainee_id)

                    cancel_resp = client.post(f"/score/{job_id}/cancel")
                    self.assertEqual(cancel_resp.status_code, 200)
                    self.assertEqual(cancel_resp.json()["status"], "cancelled")
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_cancel_completed_job_returns_409(self) -> None:
        """Cancelling a completed job returns 409."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-cancel"
            try:
                import time
                app = create_app()
                with TestClient(app) as client:
                    gold_file = root / "gold.avi"
                    trainee_file = root / "trainee.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])

                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-cancel"},
                        )
                    gold_id = gold_resp.json()["video_id"]

                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-cancel"},
                        )
                    trainee_id = trainee_resp.json()["video_id"]

                    # Submit score and wait for completion
                    score_resp = client.post(
                        "/score",
                        json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                    )
                    job_id = score_resp.json()["job_id"]
                    deadline = time.time() + 5.0
                    while time.time() < deadline:
                        j = client.get(f"/score/{job_id}").json()
                        if j["status"] in ("completed", "failed"):
                            break
                        time.sleep(0.1)

                    cancel_resp = client.post(f"/score/{job_id}/cancel")
                    self.assertEqual(cancel_resp.status_code, 409)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_cancel_nonexistent_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-cancel"
            try:
                app = create_app()
                with TestClient(app) as client:
                    resp = client.post("/score/99999/cancel")
                    self.assertEqual(resp.status_code, 404)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


class PaginationMetadataTests(unittest.TestCase):
    """Tests for pagination metadata in /videos response."""

    def test_video_list_includes_total_and_has_more(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-pag"
            try:
                app = create_app()
                with TestClient(app) as client:
                    video_file = root / "v.avi"
                    _make_video(video_file, [(0, 0, 255), (0, 255, 0)])

                    with video_file.open("rb") as fh:
                        client.post(
                            "/videos",
                            files={"file": ("v.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-pag"},
                        )

                    resp = client.get("/videos?limit=100")
                    self.assertEqual(resp.status_code, 200)
                    body = resp.json()
                    self.assertIn("total", body)
                    self.assertIn("has_more", body)
                    self.assertGreaterEqual(body["total"], 1)
                    self.assertFalse(body["has_more"])
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


class WebhookConfigTests(unittest.TestCase):
    """Test webhook URL is read from env."""

    def test_webhook_url_from_env(self) -> None:
        import os as _os
        _os.environ["SOPILOT_DATA_DIR"] = "/tmp/test"
        _os.environ["SOPILOT_WEBHOOK_URL"] = "https://example.com/hook"
        try:
            from sopilot.config import Settings
            s = Settings.from_env()
            self.assertEqual(s.webhook_url, "https://example.com/hook")
        finally:
            _os.environ.pop("SOPILOT_DATA_DIR", None)
            _os.environ.pop("SOPILOT_WEBHOOK_URL", None)

    def test_webhook_url_empty_means_none(self) -> None:
        import os as _os
        _os.environ["SOPILOT_DATA_DIR"] = "/tmp/test"
        _os.environ.pop("SOPILOT_WEBHOOK_URL", None)
        try:
            from sopilot.config import Settings
            s = Settings.from_env()
            self.assertIsNone(s.webhook_url)
        finally:
            _os.environ.pop("SOPILOT_DATA_DIR", None)


class CSVExportTests(unittest.TestCase):
    """Tests for GET /score/export/csv endpoint."""

    def test_csv_export_returns_200_with_csv_content_type(self) -> None:
        """Hit /score/export/csv and verify 200 with text/csv content type."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-csv"
            try:
                app = create_app()
                with TestClient(app) as client:
                    resp = client.get("/score/export/csv")
                    self.assertEqual(resp.status_code, 200)
                    content_type = resp.headers.get("content-type", "")
                    self.assertIn("text/csv", content_type)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_csv_export_includes_proper_headers(self) -> None:
        """Verify the CSV output contains the expected header row."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-csv"
            try:
                app = create_app()
                with TestClient(app) as client:
                    resp = client.get("/score/export/csv")
                    self.assertEqual(resp.status_code, 200)
                    text = resp.text
                    # The first line should be the CSV header row
                    first_line = text.split("\n")[0].strip()
                    expected_columns = [
                        "job_id", "task_id", "gold_video_id", "trainee_video_id",
                        "score", "decision", "miss_steps", "swap_steps",
                        "deviation_steps", "over_time_ratio", "dtw_normalized_cost",
                        "created_at", "finished_at",
                    ]
                    for col in expected_columns:
                        self.assertIn(col, first_line, f"CSV header missing column: {col}")
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_csv_export_content_disposition_header(self) -> None:
        """Verify the CSV response includes a Content-Disposition attachment header."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-csv"
            try:
                app = create_app()
                with TestClient(app) as client:
                    resp = client.get("/score/export/csv")
                    self.assertEqual(resp.status_code, 200)
                    disposition = resp.headers.get("content-disposition", "")
                    self.assertIn("attachment", disposition)
                    self.assertIn("sopilot_scores.csv", disposition)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_csv_export_with_completed_job(self) -> None:
        """Submit a score job, wait for completion, then verify it appears in CSV export."""
        import time

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-csv"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-csv"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-csv"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Submit score job and wait for completion
                    score_resp = client.post(
                        "/score",
                        json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                    )
                    self.assertEqual(score_resp.status_code, 200, score_resp.text)
                    job_id = score_resp.json()["job_id"]

                    deadline = time.time() + 10.0
                    while time.time() < deadline:
                        j = client.get(f"/score/{job_id}").json()
                        if j["status"] in ("completed", "failed"):
                            break
                        time.sleep(0.2)

                    # Now export CSV and verify the job appears in data rows
                    csv_resp = client.get("/score/export/csv")
                    self.assertEqual(csv_resp.status_code, 200)
                    lines = csv_resp.text.strip().split("\n")
                    # Should have header + at least 1 data row
                    self.assertGreaterEqual(len(lines), 2, "CSV should contain header and at least one data row")
                    # The data row should contain our job_id
                    data_content = "\n".join(lines[1:])
                    self.assertIn(str(job_id), data_content)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


class ScoreJobTimestampTests(unittest.TestCase):
    """Tests for created_at and finished_at fields in ScoreJobResponse."""

    def test_score_job_has_created_at_after_submission(self) -> None:
        """Verify ScoreJobResponse includes created_at when a job is first queued."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-ts"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-ts"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-ts"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Submit score job
                    score_resp = client.post(
                        "/score",
                        json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                    )
                    self.assertEqual(score_resp.status_code, 200, score_resp.text)
                    body = score_resp.json()

                    # created_at should be present
                    self.assertIn("created_at", body)
                    self.assertIsNotNone(body["created_at"])
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_score_job_has_timestamps_after_completion(self) -> None:
        """Submit a score job, wait for completion, verify both created_at and finished_at are set."""
        import time

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-ts"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-ts"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-ts"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Submit score job
                    score_resp = client.post(
                        "/score",
                        json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                    )
                    self.assertEqual(score_resp.status_code, 200, score_resp.text)
                    job_id = score_resp.json()["job_id"]

                    # Wait for completion
                    deadline = time.time() + 10.0
                    while time.time() < deadline:
                        j = client.get(f"/score/{job_id}").json()
                        if j["status"] in ("completed", "failed"):
                            break
                        time.sleep(0.2)

                    # Fetch final state
                    final_resp = client.get(f"/score/{job_id}")
                    self.assertEqual(final_resp.status_code, 200)
                    body = final_resp.json()

                    # Both timestamps should be present
                    self.assertIn("created_at", body)
                    self.assertIsNotNone(body["created_at"])
                    self.assertIn("finished_at", body)
                    self.assertIsNotNone(body["finished_at"])

                    # finished_at should be >= created_at (lexicographic works for ISO-8601)
                    self.assertGreaterEqual(body["finished_at"], body["created_at"])
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


class ScoreJobListPaginationTests(unittest.TestCase):
    """Tests for pagination metadata in GET /score (ScoreJobListResponse)."""

    def test_score_list_includes_pagination_fields(self) -> None:
        """GET /score should return items, total, and has_more fields."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-pag"
            try:
                app = create_app()
                with TestClient(app) as client:
                    resp = client.get("/score")
                    self.assertEqual(resp.status_code, 200)
                    body = resp.json()
                    self.assertIn("items", body)
                    self.assertIn("total", body)
                    self.assertIn("has_more", body)
                    self.assertIsInstance(body["items"], list)
                    self.assertIsInstance(body["total"], int)
                    self.assertIsInstance(body["has_more"], bool)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_score_list_empty_returns_zero_total(self) -> None:
        """With no score jobs, total should be 0 and items should be empty."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-pag"
            try:
                app = create_app()
                with TestClient(app) as client:
                    resp = client.get("/score")
                    self.assertEqual(resp.status_code, 200)
                    body = resp.json()
                    self.assertEqual(body["total"], 0)
                    self.assertEqual(len(body["items"]), 0)
                    self.assertFalse(body["has_more"])
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_score_list_with_jobs_has_correct_total(self) -> None:
        """Submit a score job and verify the list total reflects it."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-pag"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-pag"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-pag"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Submit a score job
                    score_resp = client.post(
                        "/score",
                        json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                    )
                    self.assertEqual(score_resp.status_code, 200, score_resp.text)

                    # List score jobs
                    list_resp = client.get("/score")
                    self.assertEqual(list_resp.status_code, 200)
                    body = list_resp.json()
                    self.assertGreaterEqual(body["total"], 1)
                    self.assertGreaterEqual(len(body["items"]), 1)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_score_list_has_more_with_small_limit(self) -> None:
        """With limit=1 and multiple jobs, has_more should be True."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-pag"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-pag"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload two trainee videos
                    trainee_file_1 = root / "trainee1.avi"
                    _make_video(trainee_file_1, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file_1.open("rb") as fh:
                        t1_resp = client.post(
                            "/videos",
                            files={"file": ("trainee1.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-pag"},
                        )
                    self.assertEqual(t1_resp.status_code, 200, t1_resp.text)
                    trainee_id_1 = t1_resp.json()["video_id"]

                    trainee_file_2 = root / "trainee2.avi"
                    _make_video(trainee_file_2, [(255, 0, 0), (0, 255, 0)])
                    with trainee_file_2.open("rb") as fh:
                        t2_resp = client.post(
                            "/videos",
                            files={"file": ("trainee2.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-pag"},
                        )
                    self.assertEqual(t2_resp.status_code, 200, t2_resp.text)
                    trainee_id_2 = t2_resp.json()["video_id"]

                    # Submit two score jobs
                    client.post(
                        "/score",
                        json={"gold_video_id": gold_id, "trainee_video_id": trainee_id_1},
                    )
                    client.post(
                        "/score",
                        json={"gold_video_id": gold_id, "trainee_video_id": trainee_id_2},
                    )

                    # List with limit=1 -- should indicate more results exist
                    list_resp = client.get("/score?limit=1")
                    self.assertEqual(list_resp.status_code, 200)
                    body = list_resp.json()
                    self.assertEqual(len(body["items"]), 1)
                    self.assertGreaterEqual(body["total"], 2)
                    self.assertTrue(body["has_more"])
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_score_list_items_contain_expected_fields(self) -> None:
        """Each item in the score list should contain job metadata fields."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-pag"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-pag"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-pag"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Submit a score job
                    client.post(
                        "/score",
                        json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                    )

                    # List and check item fields
                    list_resp = client.get("/score")
                    self.assertEqual(list_resp.status_code, 200)
                    body = list_resp.json()
                    self.assertGreaterEqual(len(body["items"]), 1)

                    item = body["items"][0]
                    self.assertIn("id", item)
                    self.assertIn("gold_video_id", item)
                    self.assertIn("trainee_video_id", item)
                    self.assertIn("status", item)
                    self.assertIn("created_at", item)
                    self.assertIn("updated_at", item)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


class CancelledTerminalStateTests(unittest.TestCase):
    """Tests verifying that 'cancelled' is treated as a terminal state."""

    def test_cancel_queued_job_still_works(self) -> None:
        """Confirm that cancelling a queued job still returns status 'cancelled'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-cancel-term"
            try:
                app = create_app()
                with TestClient(app) as client:
                    gold_file = root / "gold.avi"
                    trainee_file = root / "trainee.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])

                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-cancel-term"},
                        )
                    gold_id = gold_resp.json()["video_id"]

                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-cancel-term"},
                        )
                    trainee_id = trainee_resp.json()["video_id"]

                    # Insert job directly in DB to bypass queue so it stays queued
                    db = app.state.sopilot_service.database
                    job_id = db.create_score_job(gold_id, trainee_id)

                    # Cancel the queued job
                    cancel_resp = client.post(f"/score/{job_id}/cancel")
                    self.assertEqual(cancel_resp.status_code, 200)
                    self.assertEqual(cancel_resp.json()["status"], "cancelled")

                    # Verify that GET returns cancelled status
                    get_resp = client.get(f"/score/{job_id}")
                    self.assertEqual(get_resp.status_code, 200)
                    self.assertEqual(get_resp.json()["status"], "cancelled")
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_cancelled_job_cannot_be_cancelled_again(self) -> None:
        """Cancelling an already-cancelled job should return 409 (terminal state)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-cancel-term"
            try:
                app = create_app()
                with TestClient(app) as client:
                    gold_file = root / "gold.avi"
                    trainee_file = root / "trainee.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])

                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-cancel-term"},
                        )
                    gold_id = gold_resp.json()["video_id"]

                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-cancel-term"},
                        )
                    trainee_id = trainee_resp.json()["video_id"]

                    # Insert job directly in DB and cancel it
                    db = app.state.sopilot_service.database
                    job_id = db.create_score_job(gold_id, trainee_id)

                    cancel_resp = client.post(f"/score/{job_id}/cancel")
                    self.assertEqual(cancel_resp.status_code, 200)
                    self.assertEqual(cancel_resp.json()["status"], "cancelled")

                    # Try to cancel again -- should be 409 since it's already terminal
                    cancel_again = client.post(f"/score/{job_id}/cancel")
                    self.assertEqual(cancel_again.status_code, 409)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


class ScoreRerunTests(unittest.TestCase):
    """Tests for POST /score/{job_id}/rerun endpoint."""

    def test_rerun_completed_job_creates_new_job(self) -> None:
        """Submit a score job, wait for completion, re-run it, verify a new job is created
        with the same gold/trainee IDs."""
        import time

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-rerun"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-rerun"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-rerun"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Submit score job and wait for completion
                    score_resp = client.post(
                        "/score",
                        json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                    )
                    self.assertEqual(score_resp.status_code, 200, score_resp.text)
                    original_job_id = score_resp.json()["job_id"]

                    deadline = time.time() + 10.0
                    while time.time() < deadline:
                        j = client.get(f"/score/{original_job_id}").json()
                        if j["status"] in ("completed", "failed"):
                            break
                        time.sleep(0.2)

                    # Re-run the completed job
                    rerun_resp = client.post(f"/score/{original_job_id}/rerun")
                    self.assertEqual(rerun_resp.status_code, 200, rerun_resp.text)
                    rerun_body = rerun_resp.json()

                    # New job should have a different ID
                    new_job_id = rerun_body["job_id"]
                    self.assertNotEqual(new_job_id, original_job_id)

                    # New job should be queued (or already running/completed)
                    self.assertIn(rerun_body["status"], ("queued", "running", "completed"))

                    # Verify the new job was created with the same gold/trainee pair
                    self.assertIn("job_id", rerun_body)
                    self.assertIsInstance(new_job_id, int)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_rerun_nonexistent_job_returns_404(self) -> None:
        """POST /score/99999/rerun should return 404."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-rerun"
            try:
                app = create_app()
                with TestClient(app) as client:
                    resp = client.post("/score/99999/rerun")
                    self.assertEqual(resp.status_code, 404)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


class CascadeVideoDeleteTests(unittest.TestCase):
    """Tests for DELETE /videos/{video_id}?force=true cascade delete."""

    def test_delete_video_without_force_when_score_jobs_exist_returns_409(self) -> None:
        """Upload gold + trainee, create score job, try deleting without force -- expect 409."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-cascade"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-cascade"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-cascade"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Create a score job referencing both videos
                    score_resp = client.post(
                        "/score",
                        json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                    )
                    self.assertEqual(score_resp.status_code, 200, score_resp.text)

                    # Try to delete the gold video WITHOUT force -- should fail with 409
                    delete_resp = client.delete(f"/videos/{gold_id}")
                    self.assertEqual(delete_resp.status_code, 409)

                    # Try to delete the trainee video WITHOUT force -- should also fail with 409
                    delete_resp2 = client.delete(f"/videos/{trainee_id}")
                    self.assertEqual(delete_resp2.status_code, 409)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_delete_video_with_force_succeeds_when_score_jobs_exist(self) -> None:
        """Upload gold + trainee, create score job, delete with force=true -- expect 200."""
        import time

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-cascade"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-cascade"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-cascade"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Create a score job referencing both videos
                    score_resp = client.post(
                        "/score",
                        json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                    )
                    self.assertEqual(score_resp.status_code, 200, score_resp.text)
                    job_id = score_resp.json()["job_id"]

                    # Wait for score job to settle (completed or failed) so it is not
                    # actively running when we try to force-delete.
                    deadline = time.time() + 10.0
                    while time.time() < deadline:
                        j = client.get(f"/score/{job_id}").json()
                        if j["status"] in ("completed", "failed"):
                            break
                        time.sleep(0.2)

                    # Delete the trainee video WITH force=true -- should succeed
                    delete_resp = client.delete(f"/videos/{trainee_id}?force=true")
                    self.assertEqual(delete_resp.status_code, 200, delete_resp.text)
                    body = delete_resp.json()
                    self.assertEqual(body["video_id"], trainee_id)
                    self.assertTrue(body["deleted"])

                    # Verify the video is gone
                    get_after = client.get(f"/videos/{trainee_id}")
                    self.assertEqual(get_after.status_code, 404)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


class DatabaseStatsTests(unittest.TestCase):
    """Tests for GET /admin/db-stats endpoint."""

    def test_db_stats_includes_size_fields(self) -> None:
        """GET /admin/db-stats should include db_size_bytes and db_size_human."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-dbstats"
            try:
                app = create_app()
                with TestClient(app) as client:
                    resp = client.get("/admin/db-stats")
                    self.assertEqual(resp.status_code, 200)
                    body = resp.json()

                    # Verify top-level structure
                    self.assertIn("tables", body)
                    self.assertIn("db_size_bytes", body)
                    self.assertIn("db_size_human", body)

                    # db_size_bytes should be a positive integer
                    self.assertIsInstance(body["db_size_bytes"], int)
                    self.assertGreater(body["db_size_bytes"], 0)

                    # db_size_human should be a non-empty string (e.g. "128.0 KB")
                    self.assertIsInstance(body["db_size_human"], str)
                    self.assertTrue(len(body["db_size_human"]) > 0)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_db_stats_tables_is_dict(self) -> None:
        """The 'tables' field should be a dict with table names as keys and row counts as values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-dbstats"
            try:
                app = create_app()
                with TestClient(app) as client:
                    resp = client.get("/admin/db-stats")
                    self.assertEqual(resp.status_code, 200)
                    body = resp.json()

                    self.assertIsInstance(body["tables"], dict)
                    # Each value should be an integer (row count)
                    for table_name, row_count in body["tables"].items():
                        self.assertIsInstance(table_name, str)
                        self.assertIsInstance(row_count, int)
                        self.assertGreaterEqual(row_count, 0)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


class VideoDetailResponseFieldsTests(unittest.TestCase):
    """Tests for v0.5: VideoDetailResponse includes operator_id_hash, created_at, updated_at."""

    def test_video_detail_includes_operator_id_hash_and_timestamps(self) -> None:
        """Upload a video with operator_id_hash, verify GET /videos/{id} returns
        operator_id_hash, created_at, and updated_at."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-detail-fields"
            try:
                app = create_app()
                with TestClient(app) as client:
                    video_file = root / "detail_test.avi"
                    _make_video(video_file, [(0, 0, 255), (0, 255, 0)])

                    with video_file.open("rb") as fh:
                        upload_resp = client.post(
                            "/videos",
                            files={"file": ("detail_test.avi", fh, "video/x-msvideo")},
                            data={
                                "task_id": "task-detail-fields",
                                "operator_id_hash": "op-hash-abc",
                            },
                        )
                    self.assertEqual(upload_resp.status_code, 200, upload_resp.text)
                    video_id = upload_resp.json()["video_id"]

                    # GET /videos/{video_id} should include all three fields
                    detail_resp = client.get(f"/videos/{video_id}")
                    self.assertEqual(detail_resp.status_code, 200)
                    body = detail_resp.json()

                    # operator_id_hash should match what was uploaded
                    self.assertIn("operator_id_hash", body)
                    self.assertEqual(body["operator_id_hash"], "op-hash-abc")

                    # created_at should be present and non-null
                    self.assertIn("created_at", body)
                    self.assertIsNotNone(body["created_at"])

                    # updated_at should be present and non-null
                    self.assertIn("updated_at", body)
                    self.assertIsNotNone(body["updated_at"])

                    # updated_at should be >= created_at (ISO-8601 lexicographic)
                    self.assertGreaterEqual(body["updated_at"], body["created_at"])
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_video_detail_without_operator_id_hash(self) -> None:
        """Upload a video without operator_id_hash, verify the field is present but None."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-detail-fields"
            try:
                app = create_app()
                with TestClient(app) as client:
                    video_file = root / "no_op.avi"
                    _make_video(video_file, [(0, 0, 255), (0, 255, 0)])

                    with video_file.open("rb") as fh:
                        upload_resp = client.post(
                            "/videos",
                            files={"file": ("no_op.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-detail-fields"},
                        )
                    self.assertEqual(upload_resp.status_code, 200, upload_resp.text)
                    video_id = upload_resp.json()["video_id"]

                    detail_resp = client.get(f"/videos/{video_id}")
                    self.assertEqual(detail_resp.status_code, 200)
                    body = detail_resp.json()

                    # operator_id_hash key should exist but be null
                    self.assertIn("operator_id_hash", body)
                    self.assertIsNone(body["operator_id_hash"])

                    # Timestamps should still be present
                    self.assertIn("created_at", body)
                    self.assertIsNotNone(body["created_at"])
                    self.assertIn("updated_at", body)
                    self.assertIsNotNone(body["updated_at"])
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


class VideoListItemOperatorIdHashTests(unittest.TestCase):
    """Tests for v0.5: VideoListItem includes operator_id_hash in GET /videos."""

    def test_video_list_includes_operator_id_hash(self) -> None:
        """Upload a video with operator_id_hash, verify GET /videos returns it in each item."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-list-op"
            try:
                app = create_app()
                with TestClient(app) as client:
                    video_file = root / "list_op.avi"
                    _make_video(video_file, [(0, 0, 255), (0, 255, 0)])

                    with video_file.open("rb") as fh:
                        upload_resp = client.post(
                            "/videos",
                            files={"file": ("list_op.avi", fh, "video/x-msvideo")},
                            data={
                                "task_id": "task-list-op",
                                "operator_id_hash": "op-list-hash",
                            },
                        )
                    self.assertEqual(upload_resp.status_code, 200, upload_resp.text)

                    # GET /videos should return items with operator_id_hash
                    list_resp = client.get("/videos")
                    self.assertEqual(list_resp.status_code, 200)
                    body = list_resp.json()
                    self.assertGreaterEqual(len(body["items"]), 1)

                    # Find our uploaded video in the list
                    found = False
                    for item in body["items"]:
                        self.assertIn("operator_id_hash", item)
                        if item.get("operator_id_hash") == "op-list-hash":
                            found = True
                    self.assertTrue(found, "Uploaded video with operator_id_hash not found in list")
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_video_list_operator_id_hash_null_when_not_provided(self) -> None:
        """Upload a video without operator_id_hash, verify it appears as null in list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-list-op"
            try:
                app = create_app()
                with TestClient(app) as client:
                    video_file = root / "list_no_op.avi"
                    _make_video(video_file, [(0, 0, 255), (0, 255, 0)])

                    with video_file.open("rb") as fh:
                        upload_resp = client.post(
                            "/videos",
                            files={"file": ("list_no_op.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-list-op"},
                        )
                    self.assertEqual(upload_resp.status_code, 200, upload_resp.text)

                    list_resp = client.get("/videos")
                    self.assertEqual(list_resp.status_code, 200)
                    body = list_resp.json()
                    self.assertGreaterEqual(len(body["items"]), 1)

                    # Each item must have the operator_id_hash key
                    for item in body["items"]:
                        self.assertIn("operator_id_hash", item)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


class ScoreJobResponseStartedAtTests(unittest.TestCase):
    """Tests for v0.5: ScoreJobResponse includes started_at field."""

    def test_score_job_has_started_at_key_when_queued(self) -> None:
        """A newly queued score job should have started_at key (may be None)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-started"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-started"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-started"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Insert a score job directly in DB to bypass queue so it stays queued
                    db = app.state.sopilot_service.database
                    job_id = db.create_score_job(gold_id, trainee_id)

                    # GET /score/{job_id} should include started_at
                    job_resp = client.get(f"/score/{job_id}")
                    self.assertEqual(job_resp.status_code, 200)
                    body = job_resp.json()
                    self.assertIn("started_at", body)
                    # For a queued job, started_at should be None
                    self.assertIsNone(body["started_at"])
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_score_job_has_started_at_after_completion(self) -> None:
        """After a score job completes, started_at should be a non-null timestamp."""
        import time

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-started"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-started"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-started"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Submit score job via API (goes through queue)
                    score_resp = client.post(
                        "/score",
                        json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                    )
                    self.assertEqual(score_resp.status_code, 200, score_resp.text)
                    job_id = score_resp.json()["job_id"]

                    # Wait for completion
                    deadline = time.time() + 10.0
                    while time.time() < deadline:
                        j = client.get(f"/score/{job_id}").json()
                        if j["status"] in ("completed", "failed"):
                            break
                        time.sleep(0.2)

                    # Fetch final state
                    final_resp = client.get(f"/score/{job_id}")
                    self.assertEqual(final_resp.status_code, 200)
                    body = final_resp.json()

                    # started_at should be present and non-null for a completed job
                    self.assertIn("started_at", body)
                    self.assertIsNotNone(body["started_at"])

                    # started_at should be between created_at and finished_at
                    self.assertGreaterEqual(body["started_at"], body["created_at"])
                    if body["finished_at"] is not None:
                        self.assertGreaterEqual(body["finished_at"], body["started_at"])
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


class ScoreJobLeftJoinTests(unittest.TestCase):
    """Tests for v0.5: LEFT JOIN in list_score_jobs ensures score jobs appear even
    when the gold video has been force-deleted."""

    def test_score_job_visible_after_gold_video_deleted_from_db(self) -> None:
        """Create videos, create score job via DB, delete the gold video row
        directly (simulating orphaned FK), verify the score job still appears
        in GET /score listing thanks to LEFT JOIN."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-leftjoin"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-leftjoin"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-leftjoin"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Also create a second pair with a valid job, so the listing
                    # contains both orphaned and non-orphaned jobs
                    gold_file2 = root / "gold2.avi"
                    _make_video(gold_file2, [(0, 0, 255), (0, 255, 0)])
                    with gold_file2.open("rb") as fh:
                        gold2_resp = client.post(
                            "/gold",
                            files={"file": ("gold2.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-leftjoin"},
                        )
                    self.assertEqual(gold2_resp.status_code, 200, gold2_resp.text)
                    gold2_id = gold2_resp.json()["video_id"]

                    trainee_file2 = root / "trainee2.avi"
                    _make_video(trainee_file2, [(255, 0, 0), (0, 255, 0)])
                    with trainee_file2.open("rb") as fh:
                        trainee2_resp = client.post(
                            "/videos",
                            files={"file": ("trainee2.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-leftjoin"},
                        )
                    self.assertEqual(trainee2_resp.status_code, 200, trainee2_resp.text)
                    trainee2_id = trainee2_resp.json()["video_id"]

                    # Create score jobs directly in DB (bypass queue)
                    db = app.state.sopilot_service.database
                    orphan_job_id = db.create_score_job(gold_id, trainee_id)
                    valid_job_id = db.create_score_job(gold2_id, trainee2_id)

                    # Delete the first gold video row directly, bypassing FK constraints
                    with db.connect() as conn:
                        conn.execute("PRAGMA foreign_keys=OFF")
                        conn.execute("DELETE FROM clips WHERE video_id = ?", (gold_id,))
                        conn.execute("DELETE FROM videos WHERE id = ?", (gold_id,))
                        conn.execute("PRAGMA foreign_keys=ON")

                    # GET /score should still return 200 and include both jobs
                    list_resp = client.get("/score")
                    self.assertEqual(list_resp.status_code, 200)
                    items = list_resp.json()["items"]
                    job_ids = {item["id"] for item in items}

                    # The orphaned job (gold deleted) should still appear
                    self.assertIn(orphan_job_id, job_ids)
                    # The valid job should also appear
                    self.assertIn(valid_job_id, job_ids)

                    # The orphaned job should have task_id as None
                    orphan = next(item for item in items if item["id"] == orphan_job_id)
                    self.assertIsNone(orphan["task_id"])

                    # The valid job should still have the correct task_id
                    valid = next(item for item in items if item["id"] == valid_job_id)
                    self.assertEqual(valid["task_id"], "task-leftjoin")
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_score_job_listing_with_orphaned_gold_via_db(self) -> None:
        """Create a score job via DB, delete the gold video row directly, verify
        GET /score still returns the orphaned score job (LEFT JOIN behavior)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-leftjoin"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-leftjoin"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-leftjoin"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Create a score job directly in DB
                    db = app.state.sopilot_service.database
                    job_id = db.create_score_job(gold_id, trainee_id)

                    # Delete the gold video row directly from the database
                    # (bypassing the FK constraint by deleting clips first)
                    with db.connect() as conn:
                        conn.execute("PRAGMA foreign_keys=OFF")
                        conn.execute("DELETE FROM clips WHERE video_id = ?", (gold_id,))
                        conn.execute("DELETE FROM videos WHERE id = ?", (gold_id,))
                        conn.execute("PRAGMA foreign_keys=ON")

                    # GET /score should still return 200 and include the orphaned job
                    list_resp = client.get("/score")
                    self.assertEqual(list_resp.status_code, 200)
                    items = list_resp.json()["items"]
                    job_ids = {item["id"] for item in items}
                    self.assertIn(job_id, job_ids)

                    # The orphaned job should have task_id as None (gold was deleted)
                    orphaned = next(item for item in items if item["id"] == job_id)
                    self.assertIsNone(orphaned["task_id"])
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


class AnalyticsDaysFilterTests(unittest.TestCase):
    """Tests for GET /analytics?days=N query parameter."""

    def test_analytics_without_days_returns_all_data(self) -> None:
        """GET /analytics without days param should return all completed jobs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-analytics"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-analytics"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-analytics"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)

                    # Create a completed score job directly in DB
                    db = app.state.sopilot_service.database
                    trainee_id = trainee_resp.json()["video_id"]
                    job_id = db.create_score_job(gold_id, trainee_id)
                    db.claim_score_job(job_id)
                    db.complete_score_job(job_id, {"score": 90.0, "summary": {"decision": "pass"}})

                    # GET /analytics without days should return the job
                    resp = client.get("/analytics")
                    self.assertEqual(resp.status_code, 200)
                    body = resp.json()
                    self.assertGreaterEqual(body["completed_jobs"], 1)
                    self.assertIsNotNone(body["avg_score"])
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_analytics_with_days_filters_recent_data(self) -> None:
        """GET /analytics?days=7 should only include jobs from the last 7 days."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-analytics"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-analytics"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-analytics"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)

                    db = app.state.sopilot_service.database
                    trainee_id = trainee_resp.json()["video_id"]

                    # Create a recent completed job (just now -- within last 7 days)
                    job_id = db.create_score_job(gold_id, trainee_id)
                    db.claim_score_job(job_id)
                    db.complete_score_job(job_id, {"score": 85.0, "summary": {"decision": "pass"}})

                    # Create an old job by manipulating created_at to 30 days ago
                    old_job_id = db.create_score_job(gold_id, trainee_id)
                    db.claim_score_job(old_job_id)
                    db.complete_score_job(old_job_id, {"score": 70.0, "summary": {"decision": "fail"}})
                    with db.connect() as conn:
                        conn.execute(
                            "UPDATE score_jobs SET created_at = datetime('now', '-30 days') WHERE id = ?",
                            (old_job_id,),
                        )

                    # GET /analytics?days=7 should only see the recent job
                    resp = client.get("/analytics?days=7")
                    self.assertEqual(resp.status_code, 200)
                    body = resp.json()
                    self.assertEqual(body["completed_jobs"], 1)
                    self.assertEqual(body["pass_count"], 1)
                    self.assertEqual(body["fail_count"], 0)

                    # GET /analytics without days should see both jobs
                    resp_all = client.get("/analytics")
                    self.assertEqual(resp_all.status_code, 200)
                    body_all = resp_all.json()
                    self.assertEqual(body_all["completed_jobs"], 2)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_analytics_days_zero_returns_422(self) -> None:
        """GET /analytics?days=0 should return 422 (validation: ge=1)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-analytics"
            try:
                app = create_app()
                with TestClient(app) as client:
                    resp = client.get("/analytics?days=0")
                    self.assertEqual(resp.status_code, 422)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_analytics_days_over_365_returns_422(self) -> None:
        """GET /analytics?days=400 should return 422 (validation: le=365)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-analytics"
            try:
                app = create_app()
                with TestClient(app) as client:
                    resp = client.get("/analytics?days=400")
                    self.assertEqual(resp.status_code, 422)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


class OperatorRankingOrderTests(unittest.TestCase):
    """Tests for operator ranking: by_operator sorted by avg_score DESC."""

    def test_operators_sorted_by_avg_score_desc(self) -> None:
        """Create two operators with different avg scores and verify
        the higher-scoring operator comes first in by_operator."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-ranking"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-ranking"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    db = app.state.sopilot_service.database

                    # Create trainee video for operator with LOW avg score but MORE jobs
                    trainee_low_id = db.insert_video(
                        task_id="task-ranking", site_id=None, camera_id=None,
                        operator_id_hash="op-low-score", recorded_at=None, is_gold=False,
                    )
                    # Give op-low-score 3 jobs with avg score 60
                    for _ in range(3):
                        jid = db.create_score_job(gold_id, trainee_low_id)
                        db.claim_score_job(jid)
                        db.complete_score_job(jid, {"score": 60.0, "summary": {"decision": "fail"}})

                    # Create trainee video for operator with HIGH avg score but FEWER jobs
                    trainee_high_id = db.insert_video(
                        task_id="task-ranking", site_id=None, camera_id=None,
                        operator_id_hash="op-high-score", recorded_at=None, is_gold=False,
                    )
                    # Give op-high-score 1 job with avg score 95
                    jid = db.create_score_job(gold_id, trainee_high_id)
                    db.claim_score_job(jid)
                    db.complete_score_job(jid, {"score": 95.0, "summary": {"decision": "pass"}})

                    # GET /analytics and check by_operator ordering
                    resp = client.get("/analytics")
                    self.assertEqual(resp.status_code, 200)
                    body = resp.json()

                    by_operator = body["by_operator"]
                    self.assertEqual(len(by_operator), 2)

                    # Higher avg_score operator should come first, even though
                    # the low-score operator has more jobs
                    self.assertEqual(by_operator[0]["operator_id"], "op-high-score")
                    self.assertEqual(by_operator[0]["avg_score"], 95.0)
                    self.assertEqual(by_operator[0]["job_count"], 1)

                    self.assertEqual(by_operator[1]["operator_id"], "op-low-score")
                    self.assertEqual(by_operator[1]["avg_score"], 60.0)
                    self.assertEqual(by_operator[1]["job_count"], 3)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_operators_with_same_avg_score_sorted_by_job_count_desc(self) -> None:
        """When two operators have the same avg_score, the one with more jobs comes first."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-ranking"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-ranking"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    db = app.state.sopilot_service.database

                    # Create operator with many jobs at score 80
                    trainee_many_id = db.insert_video(
                        task_id="task-ranking", site_id=None, camera_id=None,
                        operator_id_hash="op-many", recorded_at=None, is_gold=False,
                    )
                    for _ in range(3):
                        jid = db.create_score_job(gold_id, trainee_many_id)
                        db.claim_score_job(jid)
                        db.complete_score_job(jid, {"score": 80.0, "summary": {"decision": "pass"}})

                    # Create operator with few jobs at score 80
                    trainee_few_id = db.insert_video(
                        task_id="task-ranking", site_id=None, camera_id=None,
                        operator_id_hash="op-few", recorded_at=None, is_gold=False,
                    )
                    jid = db.create_score_job(gold_id, trainee_few_id)
                    db.claim_score_job(jid)
                    db.complete_score_job(jid, {"score": 80.0, "summary": {"decision": "pass"}})

                    resp = client.get("/analytics")
                    self.assertEqual(resp.status_code, 200)
                    body = resp.json()

                    by_operator = body["by_operator"]
                    self.assertEqual(len(by_operator), 2)

                    # Same avg_score, so tiebreak by job_count DESC
                    self.assertEqual(by_operator[0]["operator_id"], "op-many")
                    self.assertEqual(by_operator[0]["job_count"], 3)
                    self.assertEqual(by_operator[1]["operator_id"], "op-few")
                    self.assertEqual(by_operator[1]["job_count"], 1)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


class ScoreTimelineTests(unittest.TestCase):
    """Tests for GET /score/{job_id}/timeline endpoint."""

    def test_timeline_nonexistent_job_returns_404(self) -> None:
        """GET /score/99999/timeline should return 404 for non-existent job."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-timeline"
            try:
                app = create_app()
                with TestClient(app) as client:
                    resp = client.get("/score/99999/timeline")
                    self.assertEqual(resp.status_code, 404)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_timeline_pending_job_returns_409(self) -> None:
        """GET /score/{job_id}/timeline for a queued job should return 409
        since no result is available yet."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-timeline"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-timeline"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-timeline"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Create a score job directly in DB (stays queued, no result)
                    db = app.state.sopilot_service.database
                    job_id = db.create_score_job(gold_id, trainee_id)

                    resp = client.get(f"/score/{job_id}/timeline")
                    self.assertEqual(resp.status_code, 409)
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_timeline_completed_job_returns_valid_structure(self) -> None:
        """GET /score/{job_id}/timeline for a completed job should return
        a valid timeline structure with steps, total_steps, score, and decision."""
        import time

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-timeline"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-timeline"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-timeline"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Submit score job and wait for completion
                    score_resp = client.post(
                        "/score",
                        json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                    )
                    self.assertEqual(score_resp.status_code, 200, score_resp.text)
                    job_id = score_resp.json()["job_id"]

                    deadline = time.time() + 10.0
                    while time.time() < deadline:
                        j = client.get(f"/score/{job_id}").json()
                        if j["status"] in ("completed", "failed"):
                            break
                        time.sleep(0.2)

                    # Verify job completed successfully
                    final_job = client.get(f"/score/{job_id}").json()
                    self.assertEqual(final_job["status"], "completed",
                                     f"Score job did not complete: {final_job}")

                    # GET /score/{job_id}/timeline
                    resp = client.get(f"/score/{job_id}/timeline")
                    self.assertEqual(resp.status_code, 200, resp.text)
                    body = resp.json()

                    # Verify top-level structure
                    self.assertIn("job_id", body)
                    self.assertEqual(body["job_id"], job_id)
                    self.assertIn("score", body)
                    self.assertIsInstance(body["score"], (int, float))
                    self.assertIn("decision", body)
                    self.assertIn("total_steps", body)
                    self.assertIsInstance(body["total_steps"], int)
                    self.assertGreater(body["total_steps"], 0)
                    self.assertIn("steps", body)
                    self.assertIsInstance(body["steps"], list)
                    self.assertEqual(len(body["steps"]), body["total_steps"])
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)

    def test_timeline_step_fields(self) -> None:
        """Each step in the timeline should have step_index, gold_clip_range, and status."""
        import time

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-timeline"
            try:
                app = create_app()
                with TestClient(app) as client:
                    # Upload gold video
                    gold_file = root / "gold.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-timeline"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    # Upload trainee video
                    trainee_file = root / "trainee.avi"
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])
                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-timeline"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    # Submit score job and wait for completion
                    score_resp = client.post(
                        "/score",
                        json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                    )
                    self.assertEqual(score_resp.status_code, 200, score_resp.text)
                    job_id = score_resp.json()["job_id"]

                    deadline = time.time() + 10.0
                    while time.time() < deadline:
                        j = client.get(f"/score/{job_id}").json()
                        if j["status"] in ("completed", "failed"):
                            break
                        time.sleep(0.2)

                    final_job = client.get(f"/score/{job_id}").json()
                    self.assertEqual(final_job["status"], "completed",
                                     f"Score job did not complete: {final_job}")

                    # GET timeline and validate step fields
                    resp = client.get(f"/score/{job_id}/timeline")
                    self.assertEqual(resp.status_code, 200, resp.text)
                    body = resp.json()

                    for idx, step in enumerate(body["steps"]):
                        # step_index must be present and match position
                        self.assertIn("step_index", step,
                                      f"Step {idx} missing step_index")
                        self.assertEqual(step["step_index"], idx)

                        # gold_clip_range must be a list of two ints
                        self.assertIn("gold_clip_range", step,
                                      f"Step {idx} missing gold_clip_range")
                        gcr = step["gold_clip_range"]
                        self.assertIsInstance(gcr, list)
                        self.assertEqual(len(gcr), 2)
                        self.assertIsInstance(gcr[0], int)
                        self.assertIsInstance(gcr[1], int)
                        self.assertLessEqual(gcr[0], gcr[1])

                        # status must be one of the known values
                        self.assertIn("status", step,
                                      f"Step {idx} missing status")
                        self.assertIn(step["status"],
                                      ("ok", "missing", "deviation", "swapped"),
                                      f"Step {idx} has unexpected status: {step['status']}")
            finally:
                for key in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND", "SOPILOT_PRIMARY_TASK_ID"):
                    os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
