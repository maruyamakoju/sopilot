import os
import tempfile
import time
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


class ApiEndpointsTests(unittest.TestCase):
    def test_end_to_end_http_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            old_task = os.environ.get("SOPILOT_PRIMARY_TASK_ID")
            os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-http"
            os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"  # disable rate limiting in test
            try:
                app = create_app()
                with TestClient(app) as client:
                    bad_task_file = root / "bad_task.avi"
                    _make_video(bad_task_file, [(255, 255, 255)])
                    with bad_task_file.open("rb") as fh:
                        bad_resp = client.post(
                            "/videos",
                            files={"file": ("bad_task.avi", fh, "video/x-msvideo")},
                            data={"task_id": "other-task"},
                        )
                    self.assertEqual(bad_resp.status_code, 409)

                    gold_file = root / "gold.avi"
                    trainee_file = root / "trainee.avi"
                    _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                    _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])

                    with gold_file.open("rb") as fh:
                        gold_resp = client.post(
                            "/gold",
                            files={"file": ("gold.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-http"},
                        )
                    self.assertEqual(gold_resp.status_code, 200, gold_resp.text)
                    gold_id = gold_resp.json()["video_id"]

                    with trainee_file.open("rb") as fh:
                        trainee_resp = client.post(
                            "/videos",
                            files={"file": ("trainee.avi", fh, "video/x-msvideo")},
                            data={"task_id": "task-http"},
                        )
                    self.assertEqual(trainee_resp.status_code, 200, trainee_resp.text)
                    trainee_id = trainee_resp.json()["video_id"]

                    score_resp = client.post(
                        "/score",
                        json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                    )
                    self.assertEqual(score_resp.status_code, 200, score_resp.text)
                    job_id = score_resp.json()["job_id"]

                    deadline = time.time() + 10.0
                    latest = None
                    while time.time() < deadline:
                        latest = client.get(f"/score/{job_id}")
                        if latest.status_code != 200:
                            break
                        status = latest.json()["status"]
                        if status in {"completed", "failed"}:
                            break
                        time.sleep(0.1)

                    self.assertIsNotNone(latest)
                    self.assertEqual(latest.status_code, 200)
                    payload = latest.json()
                    self.assertEqual(payload["status"], "completed")
                    self.assertIn("score", payload["result"])

                    video_detail = client.get(f"/videos/{gold_id}")
                    self.assertEqual(video_detail.status_code, 200)
                    self.assertEqual(video_detail.json()["video_id"], gold_id)

                    stream_resp = client.get(f"/videos/{gold_id}/stream")
                    self.assertEqual(stream_resp.status_code, 200)
                    self.assertGreater(len(stream_resp.content), 0)

                    # /status endpoint
                    status_resp = client.get("/status")
                    self.assertEqual(status_resp.status_code, 200)
                    status_data = status_resp.json()
                    from sopilot import __version__
                    self.assertEqual(status_data["version"], __version__)
                    self.assertIn("primary_task_id", status_data)
                    self.assertIn("embedder", status_data)
                    self.assertIn("queue_depth", status_data)
                    self.assertIn("data_dir", status_data)

                    # original_filename in list response
                    list_resp2 = client.get("/videos?limit=100")
                    self.assertEqual(list_resp2.status_code, 200)
                    items = list_resp2.json()["items"]
                    self.assertTrue(all("original_filename" in item for item in items))
                    # at least one item should carry the actual filename
                    filenames = {item["original_filename"] for item in items}
                    self.assertIn("gold.avi", filenames)

                    ui_resp = client.get("/")
                    self.assertEqual(ui_resp.status_code, 200)
                    self.assertIn("SOP評価コンソール", ui_resp.text)

                    profile_resp = client.get("/task-profile")
                    self.assertEqual(profile_resp.status_code, 200)
                    self.assertEqual(profile_resp.json()["task_id"], "task-http")

                    summary_resp = client.get("/dataset/summary")
                    self.assertEqual(summary_resp.status_code, 200)
                    self.assertGreaterEqual(summary_resp.json()["total_videos"], 2)

                    list_resp = client.get("/videos?limit=100")
                    self.assertEqual(list_resp.status_code, 200)
                    self.assertGreaterEqual(len(list_resp.json()["items"]), 2)

                    review_resp = client.put(
                        f"/score/{job_id}/review",
                        json={"verdict": "pass", "note": "ok"},
                    )
                    self.assertEqual(review_resp.status_code, 200)
                    self.assertEqual(review_resp.json()["verdict"], "pass")

                    export_resp = client.get(f"/score/{job_id}/export")
                    self.assertEqual(export_resp.status_code, 200)
                    self.assertEqual(export_resp.json()["job_id"], job_id)

                    # /analytics endpoint
                    analytics_resp = client.get("/analytics")
                    self.assertEqual(analytics_resp.status_code, 200)
                    analytics_data = analytics_resp.json()
                    self.assertGreaterEqual(analytics_data["completed_jobs"], 1)
                    self.assertIn("score_distribution", analytics_data)
                    self.assertIn("recent_trend", analytics_data)

                    # /score/{job_id}/report HTML
                    report_resp = client.get(f"/score/{job_id}/report")
                    self.assertEqual(report_resp.status_code, 200)
                    self.assertIn("SOPilot", report_resp.text)

                    # /score/{job_id}/report/pdf
                    pdf_resp = client.get(f"/score/{job_id}/report/pdf")
                    self.assertEqual(pdf_resp.status_code, 200)
                    self.assertTrue(pdf_resp.content[:4] == b"%PDF")

                    # DELETE video should fail (referenced by score job)
                    del_gold = client.delete(f"/videos/{gold_id}")
                    self.assertEqual(del_gold.status_code, 409)

                    # /readiness endpoint
                    readiness_resp = client.get("/readiness")
                    self.assertEqual(readiness_resp.status_code, 200)
                    readiness_data = readiness_resp.json()
                    self.assertTrue(readiness_data["ready"])
                    self.assertEqual(readiness_data["details"]["database"], "ok")
                    self.assertEqual(readiness_data["details"]["embedder"], "ok")

                    # PATCH /videos/{video_id} metadata update
                    patch_resp = client.patch(
                        f"/videos/{trainee_id}",
                        json={"site_id": "factory-x", "camera_id": "cam-3"},
                    )
                    self.assertEqual(patch_resp.status_code, 200)
                    patched = patch_resp.json()
                    self.assertEqual(patched["site_id"], "factory-x")
                    self.assertEqual(patched["camera_id"], "cam-3")

                    # Verify the update persisted via GET
                    verify_resp = client.get(f"/videos/{trainee_id}")
                    self.assertEqual(verify_resp.json()["site_id"], "factory-x")

                    # Score status filter validation
                    bad_status = client.get("/score?status=bogus")
                    self.assertEqual(bad_status.status_code, 422)
                    good_status = client.get("/score?status=completed")
                    self.assertEqual(good_status.status_code, 200)

                    # Admin: db-stats
                    dbstats_resp = client.get("/admin/db-stats")
                    self.assertEqual(dbstats_resp.status_code, 200)
                    tables = dbstats_resp.json()["tables"]
                    self.assertGreaterEqual(tables["videos"], 2)
                    self.assertGreaterEqual(tables["clips"], 1)
                    self.assertGreaterEqual(tables["score_jobs"], 1)

                    # Admin: optimize
                    opt_resp = client.post("/admin/optimize")
                    self.assertEqual(opt_resp.status_code, 200)
                    self.assertEqual(opt_resp.json()["status"], "optimized")

                    # Admin: backup
                    backup_resp = client.post("/admin/backup")
                    self.assertEqual(backup_resp.status_code, 200)
                    self.assertIn("backup_path", backup_resp.json())

                    # /api/v1 versioned endpoint
                    v1_status = client.get("/api/v1/status")
                    self.assertEqual(v1_status.status_code, 200)
                    self.assertEqual(v1_status.json()["version"], __version__)

                    # Security headers
                    self.assertEqual(
                        analytics_resp.headers.get("X-Content-Type-Options"), "nosniff"
                    )
                    self.assertIn("X-Request-ID", analytics_resp.headers)
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
                os.environ.pop("SOPILOT_RATE_LIMIT_RPM", None)


if __name__ == "__main__":
    unittest.main()
