from __future__ import annotations

from pathlib import Path
import os
import tempfile
import time
from unittest.mock import patch

import cv2
from fastapi.testclient import TestClient
import numpy as np

from sopilot.api import create_app


def _base_env(data_dir: Path, **overrides) -> dict[str, str]:
    """Return isolated env dict. Keys set to "" simulate absent env vars."""
    env = {
        "SOPILOT_DATA_DIR": str(data_dir),
        "SOPILOT_EMBEDDER_BACKEND": "heuristic",
        "SOPILOT_EMBEDDING_DEVICE": "auto",
        "SOPILOT_VJEPA2_REPO": "facebookresearch/vjepa2",
        "SOPILOT_VJEPA2_VARIANT": "vjepa2_vit_large",
        "SOPILOT_NIGHTLY_ENABLED": "0",
        "SOPILOT_QUEUE_BACKEND": "inline",
        "SOPILOT_SCORE_WORKERS": "1",
        "SOPILOT_TRAIN_WORKERS": "1",
        "SOPILOT_ENABLE_FEATURE_ADAPTER": "1",
        "SOPILOT_AUTH_REQUIRED": "0",
        "SOPILOT_MIN_SCORING_CLIPS": "1",
        "SOPILOT_UPLOAD_MAX_MB": "512",
        # Keys below are "" to simulate absent (config treats "" same as absent)
        "SOPILOT_ADAPT_COMMAND": "",
        "SOPILOT_API_TOKEN": "",
        "SOPILOT_API_TOKEN_ROLE": "admin",
        "SOPILOT_API_ROLE_TOKENS": "",
        "SOPILOT_BASIC_USER": "",
        "SOPILOT_BASIC_PASSWORD": "",
        "SOPILOT_BASIC_ROLE": "admin",
        "SOPILOT_AUTH_DEFAULT_ROLE": "admin",
        "SOPILOT_AUDIT_SIGNING_KEY": "",
        "SOPILOT_AUDIT_SIGNING_KEY_ID": "local",
        "SOPILOT_PRIVACY_MASK_ENABLED": "",
        "SOPILOT_PRIVACY_MASK_MODE": "",
        "SOPILOT_PRIVACY_MASK_RECTS": "",
        "SOPILOT_PRIVACY_FACE_BLUR": "",
    }
    env.update(overrides)
    return env


def _make_video(path: Path, variant: str) -> None:
    width, height = 160, 120
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("video writer failed to open")

    for i in range(72):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if variant == "gold":
            if i < 24:
                cv2.rectangle(frame, (10, 20), (58, 95), (255, 40, 40), -1)
            elif i < 48:
                cv2.circle(frame, (80, 60), 22, (40, 255, 40), -1)
            else:
                cv2.line(frame, (20, 105), (140, 18), (40, 40, 255), 5)
        else:
            if i < 24:
                cv2.rectangle(frame, (14, 24), (62, 92), (240, 50, 50), -1)
            elif i < 48:
                cv2.line(frame, (20, 105), (140, 18), (55, 55, 235), 5)
            else:
                cv2.circle(frame, (84, 62), 20, (50, 240, 50), -1)
        writer.write(frame)

    writer.release()


def _poll_score(client: TestClient, score_job_id: str) -> dict:
    for _ in range(120):
        res = client.get(f"/score/{score_job_id}")
        assert res.status_code == 200, res.text
        payload = res.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("score job did not complete in time")


def _poll_ingest(client: TestClient, ingest_job_id: str, headers: dict | None = None) -> dict:
    for _ in range(120):
        res = client.get(f"/videos/jobs/{ingest_job_id}", headers=headers)
        assert res.status_code == 200, res.text
        payload = res.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("ingest job did not complete in time")


def _poll_training(client: TestClient, training_job_id: str) -> dict:
    for _ in range(120):
        res = client.get(f"/train/jobs/{training_job_id}")
        assert res.status_code == 200, res.text
        payload = res.json()
        if payload["status"] in {"completed", "skipped", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("training job did not complete in time")


def test_async_score_and_training_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with patch.dict(os.environ, _base_env(root / "data")):
            app = create_app()

            gold_path = root / "gold.mp4"
            trainee_path = root / "trainee.mp4"
            _make_video(gold_path, "gold")
            _make_video(trainee_path, "trainee")

            with TestClient(app) as client:
                with gold_path.open("rb") as f:
                    r_gold = client.post(
                        "/gold",
                        data={"task_id": "maintenance_filter_swap"},
                        files={"file": ("gold.mp4", f, "video/mp4")},
                    )
                assert r_gold.status_code == 200, r_gold.text
                gold_ingest = _poll_ingest(client, r_gold.json()["ingest_job_id"])
                assert gold_ingest["status"] == "completed"
                gold_id = gold_ingest["video_id"]
                assert gold_id is not None

                with trainee_path.open("rb") as f:
                    r_tr = client.post(
                        "/videos",
                        data={"task_id": "maintenance_filter_swap", "role": "trainee"},
                        files={"file": ("trainee.mp4", f, "video/mp4")},
                    )
                assert r_tr.status_code == 200, r_tr.text
                trainee_ingest = _poll_ingest(client, r_tr.json()["ingest_job_id"])
                assert trainee_ingest["status"] == "completed"
                trainee_id = trainee_ingest["video_id"]
                assert trainee_id is not None

                r_score = client.post(
                    "/score",
                    json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                )
                assert r_score.status_code == 200, r_score.text
                score_job_id = r_score.json()["score_job_id"]
                assert r_score.json()["status"] == "queued"

                final_score = _poll_score(client, score_job_id)
                assert final_score["status"] == "completed"
                assert final_score["score"] is not None

                ui = client.get("/ui")
                assert ui.status_code == 200, ui.text
                assert "SOPilot Field PoC Console" in ui.text

                video_meta = client.get(f"/videos/{gold_id}")
                assert video_meta.status_code == 200, video_meta.text
                assert video_meta.json()["video_id"] == gold_id

                video_file = client.get(f"/videos/{gold_id}/file")
                assert video_file.status_code == 200, video_file.text
                assert len(video_file.content) > 0

                video_list = client.get("/videos", params={"task_id": "maintenance_filter_swap"})
                assert video_list.status_code == 200, video_list.text
                assert len(video_list.json()["items"]) >= 2

                pdf = client.get(f"/score/{score_job_id}/report.pdf")
                assert pdf.status_code == 200, pdf.text
                assert pdf.headers.get("content-type", "").startswith("application/pdf")

                r_train = client.post("/train/nightly")
                assert r_train.status_code == 200, r_train.text
                training_job_id = r_train.json()["training_job_id"]
                assert r_train.json()["status"] == "queued"

                final_train = _poll_training(client, training_job_id)
                assert final_train["status"] in {"completed", "skipped"}

                nightly = client.get("/train/nightly/status")
                assert nightly.status_code == 200, nightly.text
                assert "enabled" in nightly.json()


def test_auto_embedder_fallback_when_vjepa2_load_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        env = _base_env(
            root / "data",
            SOPILOT_EMBEDDER_BACKEND="auto",
            SOPILOT_EMBEDDER_FALLBACK="1",
            SOPILOT_EMBEDDING_DEVICE="this_device_does_not_exist",
        )
        with patch.dict(os.environ, env):
            app = create_app()

            sample_path = root / "sample.mp4"
            _make_video(sample_path, "gold")

            with TestClient(app) as client:
                with sample_path.open("rb") as f:
                    res = client.post(
                        "/videos",
                        data={"task_id": "fallback_case", "role": "trainee"},
                        files={"file": ("sample.mp4", f, "video/mp4")},
                    )
                assert res.status_code == 200, res.text
                payload = _poll_ingest(client, res.json()["ingest_job_id"])
                assert payload["status"] == "completed"
                assert payload["num_clips"] > 0
                assert "heuristic" in payload["embedding_model"]


def test_token_auth_and_audit_trail() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        env = _base_env(root / "data", SOPILOT_API_TOKEN="test-token")
        with patch.dict(os.environ, env):
            app = create_app()

            sample = root / "sample.mp4"
            _make_video(sample, "gold")

            with TestClient(app) as client:
                public_ui = client.get("/ui")
                assert public_ui.status_code == 200, public_ui.text
                public_ui_slash = client.get("/ui/")
                assert public_ui_slash.status_code == 200, public_ui_slash.text
                public_docs = client.get("/docs")
                assert public_docs.status_code == 200, public_docs.text

                blocked = client.get("/videos")
                assert blocked.status_code == 401, blocked.text

                headers = {"Authorization": "Bearer test-token"}
                allowed = client.get("/videos", headers=headers)
                assert allowed.status_code == 200, allowed.text

                with sample.open("rb") as f:
                    ingest = client.post(
                        "/gold",
                        data={"task_id": "auth_case"},
                        files={"file": ("sample.mp4", f, "video/mp4")},
                        headers=headers,
                    )
                assert ingest.status_code == 200, ingest.text
                final_ingest = _poll_ingest(client, ingest.json()["ingest_job_id"], headers=headers)
                assert final_ingest["requested_by"] == "token:api"

                trail = client.get("/audit/trail", headers=headers)
                assert trail.status_code == 200, trail.text
                items = trail.json()["items"]
                assert len(items) > 0
                assert any(x["job_type"] == "ingest" and x["requested_by"] == "token:api" for x in items)


def test_score_endpoint_returns_500_on_runtime_error() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with patch.dict(os.environ, _base_env(root / "data")):
            app = create_app()

            def _boom(*args, **kwargs):
                raise RuntimeError("queue unavailable")

            app.state.service.enqueue_score = _boom

            with TestClient(app) as client:
                res = client.post(
                    "/score",
                    json={"gold_video_id": 1, "trainee_video_id": 2},
                )
            assert res.status_code == 500, res.text
            assert res.json()["detail"] == "queue unavailable"


def test_video_file_endpoint_returns_500_on_runtime_error() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with patch.dict(os.environ, _base_env(root / "data")):
            app = create_app()

            def _boom(*args, **kwargs):
                raise RuntimeError("disk read failed")

            app.state.service.get_video_file_path = _boom

            with TestClient(app) as client:
                res = client.get("/videos/1/file")
            assert res.status_code == 500, res.text
            assert res.json()["detail"] == "disk read failed"


def test_delete_video_endpoint_removes_video() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with patch.dict(os.environ, _base_env(root / "data")):
            app = create_app()

            sample = root / "sample.mp4"
            _make_video(sample, "gold")

            with TestClient(app) as client:
                with sample.open("rb") as f:
                    ingest = client.post(
                        "/gold",
                        data={"task_id": "delete_case"},
                        files={"file": ("sample.mp4", f, "video/mp4")},
                    )
                assert ingest.status_code == 200, ingest.text
                done = _poll_ingest(client, ingest.json()["ingest_job_id"])
                assert done["status"] == "completed"
                video_id = done["video_id"]
                assert video_id is not None

                deleted = client.delete(f"/videos/{video_id}")
                assert deleted.status_code == 200, deleted.text
                payload = deleted.json()
                assert payload["video_id"] == video_id
                assert payload["task_id"] == "delete_case"

                not_found = client.get(f"/videos/{video_id}")
                assert not_found.status_code == 404, not_found.text


def test_auth_required_without_credentials_returns_503() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        env = _base_env(root / "data", SOPILOT_AUTH_REQUIRED="1")
        with patch.dict(os.environ, env):
            app = create_app()

            with TestClient(app) as client:
                health = client.get("/health")
                assert health.status_code == 200, health.text

                blocked = client.get("/videos")
                assert blocked.status_code == 503, blocked.text
                assert "credentials are not configured" in blocked.json()["detail"]


def test_upload_rejected_when_too_large() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        env = _base_env(root / "data", SOPILOT_UPLOAD_MAX_MB="1")
        with patch.dict(os.environ, env):
            app = create_app()

            oversized = root / "oversized.bin"
            oversized.write_bytes(b"x" * (2 * 1024 * 1024))

            with TestClient(app) as client:
                with oversized.open("rb") as f:
                    res = client.post(
                        "/videos",
                        data={"task_id": "oversize_case", "role": "trainee"},
                        files={"file": ("oversized.bin", f, "application/octet-stream")},
                    )
                assert res.status_code == 400, res.text
                assert "upload too large" in res.json()["detail"]


def test_score_job_fails_when_clip_coverage_is_too_low() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        env = _base_env(root / "data", SOPILOT_MIN_SCORING_CLIPS="10")
        with patch.dict(os.environ, env):
            app = create_app()

            gold_path = root / "gold.mp4"
            trainee_path = root / "trainee.mp4"
            _make_video(gold_path, "gold")
            _make_video(trainee_path, "trainee")

            with TestClient(app) as client:
                with gold_path.open("rb") as f:
                    r_gold = client.post(
                        "/gold",
                        data={"task_id": "clip_gate_case"},
                        files={"file": ("gold.mp4", f, "video/mp4")},
                    )
                assert r_gold.status_code == 200, r_gold.text
                gold_ingest = _poll_ingest(client, r_gold.json()["ingest_job_id"])
                assert gold_ingest["status"] == "completed"
                gold_id = gold_ingest["video_id"]
                assert gold_id is not None

                with trainee_path.open("rb") as f:
                    r_tr = client.post(
                        "/videos",
                        data={"task_id": "clip_gate_case", "role": "trainee"},
                        files={"file": ("trainee.mp4", f, "video/mp4")},
                    )
                assert r_tr.status_code == 200, r_tr.text
                trainee_ingest = _poll_ingest(client, r_tr.json()["ingest_job_id"])
                assert trainee_ingest["status"] == "completed"
                trainee_id = trainee_ingest["video_id"]
                assert trainee_id is not None

                r_score = client.post(
                    "/score",
                    json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                )
                assert r_score.status_code == 200, r_score.text
                score_job_id = r_score.json()["score_job_id"]
                final_score = _poll_score(client, score_job_id)
                assert final_score["status"] == "failed"
                assert "insufficient clip coverage for scoring" in (final_score["error_message"] or "")
