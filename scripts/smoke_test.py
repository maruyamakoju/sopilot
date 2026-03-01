#!/usr/bin/env python3
"""SOPilot end-to-end smoke test.

Connects to a running SOPilot server and verifies the full scoring pipeline
works correctly.  Works against both a local uvicorn process and a Docker
container.

Usage:
    # Against a local server (no auth):
    python scripts/smoke_test.py

    # Against a Docker container with API key:
    python scripts/smoke_test.py --url http://localhost:8000 --api-key YOUR_KEY

Exit codes:
    0 -- all checks passed
    1 -- one or more checks failed
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
INFO = "\033[94m·\033[0m"


def _ok(msg: str) -> None:
    print(f"  {PASS} {msg}")


def _fail(msg: str) -> None:
    print(f"  {FAIL} {msg}")


def _info(msg: str) -> None:
    print(f"  {INFO} {msg}")


def _section(title: str) -> None:
    print(f"\n{title}")
    print("─" * len(title))


def _make_video_bytes(colors: list[tuple[int, int, int]]) -> bytes:
    """Create a minimal synthetic AVI video with one frame per color."""
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as f:
        tmp_path = f.name
    try:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(tmp_path, fourcc, 8.0, (96, 96))
        for color in colors:
            frame = np.full((96, 96, 3), color, dtype=np.uint8)
            for _ in range(24):  # 3 seconds at 8 fps
                writer.write(frame)
        writer.release()
        return Path(tmp_path).read_bytes()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Smoke test runner
# ---------------------------------------------------------------------------

class SmokeTest:
    def __init__(self, base_url: str, api_key: str, task_id: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.task_id = task_id
        headers: dict[str, str] = {}
        if api_key:
            headers["X-API-Key"] = api_key
        self.client = httpx.Client(base_url=self.base_url, headers=headers, timeout=120)
        self.failures: list[str] = []
        self._gold_id: int | None = None
        self._trainee_id: int | None = None
        self._job_id: int | None = None

    def _check(self, label: str, condition: bool, detail: str = "") -> bool:
        if condition:
            _ok(label)
            return True
        msg = f"{label}" + (f": {detail}" if detail else "")
        _fail(msg)
        self.failures.append(msg)
        return False

    def _get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.client.get(path, **kwargs)

    def _post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.client.post(path, **kwargs)

    # --- individual checks --------------------------------------------------

    def check_health(self) -> bool:
        _section("1. Health check")
        try:
            r = self._get("/health")
            ok = self._check("HTTP 200", r.status_code == 200, str(r.status_code))
            if ok:
                data = r.json()
                self._check("status=healthy", data.get("status") == "healthy", str(data))
                _info(f"version={data.get('version')}")
            return ok
        except Exception as e:
            msg = f"Connection failed: {e}"
            _fail(msg)
            self.failures.append(msg)
            return False

    def upload_gold(self) -> bool:
        _section("2. Upload gold video")
        video_bytes = _make_video_bytes([(0, 0, 200), (0, 200, 0)])
        _info(f"Generated synthetic video ({len(video_bytes):,} bytes)")
        try:
            r = self._post(
                "/gold",
                data={"task_id": self.task_id},
                files={"file": ("smoke_gold.avi", io.BytesIO(video_bytes), "video/x-msvideo")},
            )
            ok = self._check("HTTP 200", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
            if ok:
                data = r.json()
                self._gold_id = data.get("video_id")
                self._check("video_id present", self._gold_id is not None, str(data))
                self._check("is_gold=true", data.get("is_gold") is True, str(data))
                _info(f"Gold video ID: {self._gold_id}")
            return ok
        except Exception as e:
            self._fail(f"Upload failed: {e}")
            self.failures.append(str(e))
            return False

    def upload_trainee(self) -> bool:
        _section("3. Upload trainee video")
        video_bytes = _make_video_bytes([(0, 0, 200), (200, 0, 0)])
        _info(f"Generated synthetic video ({len(video_bytes):,} bytes)")
        try:
            r = self._post(
                "/videos",
                data={"task_id": self.task_id},
                files={"file": ("smoke_trainee.avi", io.BytesIO(video_bytes), "video/x-msvideo")},
            )
            ok = self._check("HTTP 200", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
            if ok:
                data = r.json()
                self._trainee_id = data.get("video_id")
                self._check("video_id present", self._trainee_id is not None, str(data))
                self._check("is_gold=false", data.get("is_gold") is False, str(data))
                _info(f"Trainee video ID: {self._trainee_id}")
            return ok
        except Exception as e:
            self._fail(f"Upload failed: {e}")
            self.failures.append(str(e))
            return False

    def wait_for_video_ready(self, video_id: int, label: str) -> bool:
        _section(f"4. Wait for {label} processing (video {video_id})")
        for attempt in range(30):
            try:
                r = self._get(f"/videos/{video_id}")
                if r.status_code == 200:
                    status = r.json().get("status", "")
                    if status == "ready":
                        self._check(f"{label} status=ready", True)
                        return True
                    if status == "failed":
                        error = r.json().get("error", "unknown")
                        self._check(f"{label} processing", False, f"failed: {error}")
                        return False
                    _info(f"  Waiting... ({attempt+1}/30) status={status}")
            except Exception as e:
                _info(f"  Poll error: {e}")
            time.sleep(2)
        self._check(f"{label} ready within 60s", False, "timeout")
        return False

    def submit_score_job(self) -> bool:
        _section("5. Submit score job")
        if self._gold_id is None or self._trainee_id is None:
            self._check("Video IDs available", False, "skipped due to earlier failure")
            return False
        try:
            r = self._post(
                "/score",
                json={"gold_video_id": self._gold_id, "trainee_video_id": self._trainee_id},
            )
            ok = self._check("HTTP 200", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
            if ok:
                data = r.json()
                self._job_id = data.get("job_id")
                self._check("job_id present", self._job_id is not None, str(data))
                _info(f"Score job ID: {self._job_id}")
            return ok
        except Exception as e:
            self._fail(f"Score submit failed: {e}")
            self.failures.append(str(e))
            return False

    def wait_for_score_job(self) -> bool:
        _section("6. Wait for score job completion")
        if self._job_id is None:
            self._check("Job ID available", False, "skipped")
            return False
        for attempt in range(60):
            try:
                r = self._get(f"/score/{self._job_id}")
                if r.status_code == 200:
                    data = r.json()
                    status = data.get("status", "")
                    if status == "completed":
                        self._check("Job completed", True)
                        result = data.get("result") or {}
                        score = result.get("score")
                        summary = (result.get("summary") or {})
                        decision = summary.get("decision")
                        score_band = summary.get("score_band")
                        decision_basis = summary.get("decision_basis")
                        self._check("score is numeric", isinstance(score, (int, float)), str(score))
                        self._check("decision present", decision is not None, str(decision))
                        self._check("score_band present", score_band is not None, str(score_band))
                        self._check("decision_basis present", decision_basis is not None, str(decision_basis))
                        _info(f"Score={score:.1f}  decision={decision}  band={score_band}")
                        return True
                    if status == "failed":
                        error = data.get("error", "unknown")
                        self._check("Job succeeded", False, f"failed: {error}")
                        return False
                    _info(f"  Waiting... ({attempt+1}/60) status={status}")
            except Exception as e:
                _info(f"  Poll error: {e}")
            time.sleep(2)
        self._check("Job completed within 120s", False, "timeout")
        return False

    def check_score_list(self) -> bool:
        _section("7. Score job list includes score_band")
        try:
            r = self._get("/score", params={"limit": 5})
            ok = self._check("HTTP 200", r.status_code == 200, str(r.status_code))
            if ok:
                items = r.json().get("items", [])
                self._check("List non-empty", len(items) > 0, f"got {len(items)}")
                if items:
                    item = items[0]
                    self._check(
                        "score_band field in list item",
                        "score_band" in item,
                        str(list(item.keys())),
                    )
            return ok
        except Exception as e:
            self._fail(f"List check failed: {e}")
            self.failures.append(str(e))
            return False

    def check_admin_rescore(self) -> bool:
        _section("8. Admin rescore (dry_run)")
        try:
            r = self._post("/admin/rescore", params={"dry_run": "true"})
            ok = self._check("HTTP 200", r.status_code == 200, f"{r.status_code}: {r.text[:200]}")
            if ok:
                data = r.json()
                self._check("total_jobs_processed present", "total_jobs_processed" in data, str(data))
                self._check("dry_run=true confirmed", data.get("dry_run") is True, str(data))
                _info(f"dry_run result: {data.get('total_jobs_processed')} jobs processed")
            return ok
        except Exception as e:
            self._fail(f"Admin rescore failed: {e}")
            self.failures.append(str(e))
            return False

    def check_task_profile(self) -> bool:
        _section("9. Task profile has updated thresholds")
        try:
            r = self._get("/task-profile")
            ok = self._check("HTTP 200", r.status_code == 200, str(r.status_code))
            if ok:
                data = r.json()
                pass_score = data.get("pass_score")
                retrain_score = data.get("retrain_score")
                self._check(
                    f"pass_score=60.0 (got {pass_score})",
                    pass_score == 60.0,
                    f"expected 60.0, got {pass_score}",
                )
                self._check(
                    f"retrain_score=50.0 (got {retrain_score})",
                    retrain_score == 50.0,
                    f"expected 50.0, got {retrain_score}",
                )
            return ok
        except Exception as e:
            self._fail(f"Task profile check failed: {e}")
            self.failures.append(str(e))
            return False

    # --- main ---------------------------------------------------------------

    def run(self) -> int:
        print("=" * 60)
        print("SOPilot Smoke Test")
        print(f"Target: {self.base_url}")
        print("=" * 60)

        if not self.check_health():
            print("\n\033[91mServer not reachable — aborting.\033[0m")
            return 1

        self.check_task_profile()
        self.upload_gold()

        # Wait for gold video if upload succeeded
        if self._gold_id is not None:
            self.wait_for_video_ready(self._gold_id, "gold")

        self.upload_trainee()

        # Wait for trainee video if upload succeeded
        if self._trainee_id is not None:
            self.wait_for_video_ready(self._trainee_id, "trainee")

        self.submit_score_job()
        self.wait_for_score_job()
        self.check_score_list()
        self.check_admin_rescore()

        # --- Summary --------------------------------------------------------
        _section("Summary")
        total = 9
        failed = len(self.failures)
        passed = total - failed

        if not self.failures:
            print(f"\n  \033[92mAll checks passed ({passed}/{total})\033[0m\n")
            return 0
        else:
            print(f"\n  \033[91m{failed} check(s) failed:\033[0m")
            for f in self.failures:
                print(f"    • {f}")
            print()
            return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="SOPilot end-to-end smoke test")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the SOPilot server")
    parser.add_argument("--api-key", default=os.environ.get("SOPILOT_API_KEY", ""), help="API key (or set SOPILOT_API_KEY env var)")
    parser.add_argument("--task-id", default="pilot_task", help="Task ID to use for test videos")
    args = parser.parse_args()

    runner = SmokeTest(base_url=args.url, api_key=args.api_key, task_id=args.task_id)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
