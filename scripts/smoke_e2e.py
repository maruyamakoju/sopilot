#!/usr/bin/env python
"""E2E smoke test for SOPilot pipeline.

Runs the full workflow in-process with inline queue and synthetic videos:
  1. Health check
  2. Gold video ingest
  3. Trainee video ingest
  4. Scoring (alignment + report)
  5. Embedder fallback (auto → heuristic)
  6. Training trigger
  7. Audit trail verification
  8. DB state consistency
  9. NaN / data integrity checks
 10. Performance gate

Usage:
    python scripts/smoke_e2e.py            # run all checks
    python scripts/smoke_e2e.py --verbose  # detailed output
    python -m pytest scripts/smoke_e2e.py  # also works as pytest
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_video(path: Path, variant: str, n_frames: int = 72) -> None:
    """Create a small synthetic MP4 with distinct visual patterns per step."""
    width, height = 160, 120
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {path}")
    for i in range(n_frames):
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


def _env(data_dir: Path, **overrides) -> dict[str, str]:
    """Build an isolated env-var dict for inline queue + heuristic embedder."""
    base = {
        "SOPILOT_DATA_DIR": str(data_dir),
        "SOPILOT_EMBEDDER_BACKEND": "heuristic",
        "SOPILOT_EMBEDDING_DEVICE": "auto",
        "SOPILOT_NIGHTLY_ENABLED": "0",
        "SOPILOT_QUEUE_BACKEND": "inline",
        "SOPILOT_SCORE_WORKERS": "1",
        "SOPILOT_TRAIN_WORKERS": "1",
        "SOPILOT_ENABLE_FEATURE_ADAPTER": "1",
        "SOPILOT_AUTH_REQUIRED": "0",
        "SOPILOT_MIN_SCORING_CLIPS": "1",
        "SOPILOT_UPLOAD_MAX_MB": "512",
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
    base.update(overrides)
    return base


def _poll(client, path: str, done: set[str], max_iter: int = 120) -> dict:
    """Poll a job endpoint until status is in *done*."""
    for _ in range(max_iter):
        res = client.get(path)
        assert res.status_code == 200, f"GET {path} → {res.status_code}: {res.text}"
        payload = res.json()
        if payload.get("status") in done:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"job at {path} did not reach {done} within {max_iter} iterations")


# ---------------------------------------------------------------------------
# Check functions — each returns (pass: bool, message: str)
# ---------------------------------------------------------------------------


class SmokeResult:
    def __init__(self, verbose: bool = False):
        self.results: list[tuple[str, bool, str]] = []
        self.verbose = verbose

    def record(self, name: str, passed: bool, msg: str = "") -> None:
        self.results.append((name, passed, msg))
        status = "PASS" if passed else "FAIL"
        if self.verbose or not passed:
            detail = f" - {msg}" if msg else ""
            print(f"  [{status}] {name}{detail}")
        elif passed:
            print(f"  [{status}] {name}")

    @property
    def all_passed(self) -> bool:
        return all(ok for _, ok, _ in self.results)

    def summary(self) -> str:
        total = len(self.results)
        passed = sum(1 for _, ok, _ in self.results if ok)
        failed = total - passed
        return f"{passed}/{total} passed, {failed} failed"


def run_smoke_test(verbose: bool = False) -> SmokeResult:
    """Execute the full E2E smoke test suite. Returns SmokeResult."""
    # Lazy imports so the module can be loaded without side effects
    from fastapi.testclient import TestClient

    from sopilot.api import create_app

    result = SmokeResult(verbose=verbose)
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_dir = root / "data"

        # Create synthetic videos
        gold_path = root / "gold.mp4"
        trainee_path = root / "trainee.mp4"
        _make_video(gold_path, "gold")
        _make_video(trainee_path, "trainee")

        with patch.dict(os.environ, _env(data_dir)):
            app = create_app()

            with TestClient(app) as client:
                # ── 1. Health check ──────────────────────────────────
                try:
                    r = client.get("/health")
                    health = r.json()
                    result.record(
                        "health_check",
                        r.status_code == 200 and health.get("status") == "ok",
                        f"status={health.get('status')}, db={health.get('db')}",
                    )
                except Exception as exc:
                    result.record("health_check", False, str(exc))

                # ── 2. Gold video ingest ─────────────────────────────
                gold_id = None
                try:
                    with gold_path.open("rb") as f:
                        r = client.post(
                            "/gold",
                            data={"task_id": "smoke_test"},
                            files={"file": ("gold.mp4", f, "video/mp4")},
                        )
                    assert r.status_code == 200, r.text
                    gold_ingest = _poll(client, f"/videos/jobs/{r.json()['ingest_job_id']}", {"completed", "failed"})
                    ok = gold_ingest["status"] == "completed" and gold_ingest.get("video_id") is not None
                    gold_id = gold_ingest.get("video_id")
                    result.record(
                        "gold_ingest",
                        ok,
                        f"video_id={gold_id}, clips={gold_ingest.get('num_clips')}, model={gold_ingest.get('embedding_model')}",
                    )
                except Exception as exc:
                    result.record("gold_ingest", False, str(exc))

                # ── 3. Trainee video ingest ──────────────────────────
                trainee_id = None
                try:
                    with trainee_path.open("rb") as f:
                        r = client.post(
                            "/videos",
                            data={"task_id": "smoke_test", "role": "trainee"},
                            files={"file": ("trainee.mp4", f, "video/mp4")},
                        )
                    assert r.status_code == 200, r.text
                    trainee_ingest = _poll(client, f"/videos/jobs/{r.json()['ingest_job_id']}", {"completed", "failed"})
                    ok = trainee_ingest["status"] == "completed" and trainee_ingest.get("video_id") is not None
                    trainee_id = trainee_ingest.get("video_id")
                    result.record(
                        "trainee_ingest",
                        ok,
                        f"video_id={trainee_id}, clips={trainee_ingest.get('num_clips')}, model={trainee_ingest.get('embedding_model')}",
                    )
                except Exception as exc:
                    result.record("trainee_ingest", False, str(exc))

                # ── 4. Scoring ───────────────────────────────────────
                score_job_id = None
                score_value = None
                score_result = None
                if gold_id is not None and trainee_id is not None:
                    try:
                        t_score_start = time.monotonic()
                        r = client.post(
                            "/score",
                            json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
                        )
                        assert r.status_code == 200, r.text
                        score_job_id = r.json()["score_job_id"]
                        final = _poll(client, f"/score/{score_job_id}", {"completed", "failed"})
                        t_score_dur = time.monotonic() - t_score_start
                        ok = final["status"] == "completed" and final.get("score") is not None
                        score_value = final.get("score")
                        score_result = final.get("result")
                        result.record(
                            "scoring",
                            ok,
                            f"score={score_value}, duration={t_score_dur:.2f}s",
                        )
                    except Exception as exc:
                        result.record("scoring", False, str(exc))
                else:
                    result.record("scoring", False, "skipped — ingest failed")

                # ── 5. NaN / data integrity ──────────────────────────
                if score_result is not None:
                    try:
                        metrics = score_result.get("metrics", {})
                        nan_fields = [
                            k for k, v in metrics.items() if isinstance(v, float) and (math.isnan(v) or math.isinf(v))
                        ]
                        score_nan = score_value is not None and (math.isnan(score_value) or math.isinf(score_value))
                        ok = len(nan_fields) == 0 and not score_nan
                        msg = "all clean" if ok else f"NaN/Inf in: score={score_nan}, metrics={nan_fields}"
                        result.record("nan_check", ok, msg)
                    except Exception as exc:
                        result.record("nan_check", False, str(exc))

                    # Score range
                    try:
                        ok = score_value is not None and 0.0 <= score_value <= 100.0
                        result.record("score_range", ok, f"score={score_value}")
                    except Exception as exc:
                        result.record("score_range", False, str(exc))
                else:
                    result.record("nan_check", False, "skipped — no score result")
                    result.record("score_range", False, "skipped — no score result")

                # ── 6. PDF report ────────────────────────────────────
                if score_job_id is not None:
                    try:
                        r = client.get(f"/score/{score_job_id}/report.pdf")
                        ok = (
                            r.status_code == 200
                            and r.headers.get("content-type", "").startswith("application/pdf")
                            and len(r.content) > 100
                        )
                        result.record("pdf_report", ok, f"size={len(r.content)} bytes")
                    except Exception as exc:
                        result.record("pdf_report", False, str(exc))
                else:
                    result.record("pdf_report", False, "skipped — no score job")

                # ── 7. DB state consistency ──────────────────────────
                try:
                    r = client.get("/videos", params={"task_id": "smoke_test"})
                    assert r.status_code == 200, r.text
                    items = r.json()["items"]
                    roles = {x["role"] for x in items}
                    ok = len(items) >= 2 and "gold" in roles and "trainee" in roles
                    result.record("db_video_list", ok, f"count={len(items)}, roles={roles}")
                except Exception as exc:
                    result.record("db_video_list", False, str(exc))

                if gold_id is not None:
                    try:
                        r = client.get(f"/videos/{gold_id}")
                        assert r.status_code == 200, r.text
                        meta = r.json()
                        ok = (
                            meta["video_id"] == gold_id
                            and meta["task_id"] == "smoke_test"
                            and meta["role"] == "gold"
                            and meta["num_clips"] > 0
                        )
                        result.record(
                            "db_video_meta", ok, f"clips={meta['num_clips']}, model={meta['embedding_model']}"
                        )
                    except Exception as exc:
                        result.record("db_video_meta", False, str(exc))

                # ── 8. Audit trail ───────────────────────────────────
                try:
                    r = client.get("/audit/trail")
                    assert r.status_code == 200, r.text
                    items = r.json()["items"]
                    job_types = {x["job_type"] for x in items}
                    ok = len(items) >= 2 and "ingest" in job_types
                    has_score = "score" in job_types
                    result.record("audit_trail", ok, f"entries={len(items)}, types={job_types}, has_score={has_score}")
                except Exception as exc:
                    result.record("audit_trail", False, str(exc))

                # ── 9. Training trigger ──────────────────────────────
                try:
                    r = client.post("/train/nightly")
                    assert r.status_code == 200, r.text
                    tj = r.json()["training_job_id"]
                    final = _poll(client, f"/train/jobs/{tj}", {"completed", "skipped", "failed"})
                    ok = final["status"] in {"completed", "skipped"}
                    result.record("training_trigger", ok, f"status={final['status']}")
                except Exception as exc:
                    result.record("training_trigger", False, str(exc))

                # ── 10. Embedder fallback ────────────────────────────
        # Run fallback test in separate env
        fallback_video = root / "fallback.mp4"
        _make_video(fallback_video, "gold")
        fb_env = _env(
            root / "data_fb",
            SOPILOT_EMBEDDER_BACKEND="auto",
            SOPILOT_EMBEDDER_FALLBACK="1",
            SOPILOT_EMBEDDING_DEVICE="this_device_does_not_exist",
        )
        with patch.dict(os.environ, fb_env):
            fb_app = create_app()
            with TestClient(fb_app) as fb_client:
                try:
                    with fallback_video.open("rb") as f:
                        r = fb_client.post(
                            "/videos",
                            data={"task_id": "fb_test", "role": "trainee"},
                            files={"file": ("fb.mp4", f, "video/mp4")},
                        )
                    assert r.status_code == 200, r.text
                    fb_done = _poll(fb_client, f"/videos/jobs/{r.json()['ingest_job_id']}", {"completed", "failed"})
                    ok = (
                        fb_done["status"] == "completed"
                        and fb_done.get("num_clips", 0) > 0
                        and "heuristic" in (fb_done.get("embedding_model") or "")
                    )
                    result.record(
                        "embedder_fallback",
                        ok,
                        f"model={fb_done.get('embedding_model')}, clips={fb_done.get('num_clips')}",
                    )
                except Exception as exc:
                    result.record("embedder_fallback", False, str(exc))

    # ── 11. Overall performance gate ─────────────────────────────
    elapsed = time.monotonic() - t0
    ok = elapsed < 60.0  # entire suite must complete under 60s
    result.record("performance_gate", ok, f"total={elapsed:.2f}s (limit=60s)")

    return result


# ---------------------------------------------------------------------------
# CLI + pytest entry points
# ---------------------------------------------------------------------------


def test_smoke_e2e() -> None:
    """Pytest-compatible entry point."""
    res = run_smoke_test(verbose=True)
    assert res.all_passed, f"Smoke test failed: {res.summary()}"


def main() -> int:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    print("SOPilot E2E Smoke Test")
    print("=" * 50)
    res = run_smoke_test(verbose=verbose)
    print("=" * 50)
    print(f"Result: {res.summary()}")
    return 0 if res.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
