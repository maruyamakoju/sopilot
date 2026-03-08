#!/usr/bin/env python3
"""SOPilot キラーデモ: ライブブラウザデモ.

FastAPI サーバーをバックグラウンド起動し、MP4 フレームをストリーミング送信。
ブラウザで MJPEG プレビューを自動表示し、ターミナルに違反ログをリアルタイム出力する。

Usage::

    python scripts/demo_live.py --input factory.mp4 --preset warehouse_safety
    python scripts/demo_live.py --input factory.mp4 --preset construction_safety --loop --fps 5
    python scripts/demo_live.py --input factory.mp4 --port 9000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from threading import Event, Thread

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


def _start_server(port: int) -> subprocess.Popen:
    """Spawn uvicorn in a background process and return the Popen handle."""
    env = os.environ.copy()
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "sopilot.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        cwd=str(_ROOT),
    )
    logger.info("Server PID %d starting on http://localhost:%d …", proc.pid, port)
    return proc


def _wait_for_server(port: int, timeout: float = 30.0) -> bool:
    """Poll health endpoint until ready or timeout (seconds)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(
                f"http://localhost:{port}/health", timeout=2
            )
            return True
        except Exception:
            time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _http_json(url: str, method: str = "GET", body: bytes | None = None, content_type: str = "application/json") -> dict:
    """Simple synchronous HTTP call; returns parsed JSON dict."""
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": content_type} if body else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _create_session(port: int, name: str, rules: list[str]) -> str:
    """POST /vigil/stream/sessions; return session_id."""
    body = json.dumps({"name": name, "rules": rules}).encode()
    data = _http_json(f"http://localhost:{port}/vigil/stream/sessions", "POST", body)
    return data["session_id"]


def _delete_session(port: int, sid: str) -> None:
    """DELETE /vigil/stream/sessions/{sid} (best-effort)."""
    try:
        _http_json(f"http://localhost:{port}/vigil/stream/sessions/{sid}", "DELETE")
    except Exception:
        pass


def _post_frame(port: int, sid: str, frame_bytes: bytes) -> dict | None:
    """POST raw JPEG to session; return result dict or None on error."""
    try:
        req = urllib.request.Request(
            f"http://localhost:{port}/vigil/stream/sessions/{sid}/frame",
            data=frame_bytes,
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        logger.debug("Frame POST error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Streaming thread
# ---------------------------------------------------------------------------


def _stream_loop(
    port: int,
    sid: str,
    input_path: Path,
    fps: float,
    loop: bool,
    stop_event: Event,
) -> None:
    """Read frames from *input_path* and POST them to the session endpoint."""
    from sopilot.vigil.extractor import iter_frames

    interval = 1.0 / max(fps, 0.1)
    round_num = 0

    while not stop_event.is_set():
        round_num += 1
        if round_num > 1:
            logger.info("Loop %d …", round_num)

        with tempfile.TemporaryDirectory(prefix="sopilot_live_") as tmpdir:
            tmp_path = Path(tmpdir)
            try:
                for frame_num, ts_sec, jpeg_path in iter_frames(
                    input_path,
                    sample_fps=fps,
                    output_dir=tmp_path,
                ):
                    if stop_event.is_set():
                        return

                    t0 = time.perf_counter()
                    frame_bytes = jpeg_path.read_bytes()
                    result = _post_frame(port, sid, frame_bytes)

                    if result is not None:
                        violations = result.get("violations", [])
                        det_count = len(result.get("detections", []))
                        proc_ms = result.get("frame_ms", 0.0)

                        if violations:
                            logger.warning(
                                "[%5d | %.2fs]  VIOLATION %s  (det=%d, %.1fms)",
                                frame_num,
                                ts_sec,
                                violations,
                                det_count,
                                proc_ms,
                            )
                        else:
                            logger.info(
                                "[%5d | %.2fs]  OK  det=%d  %.1fms",
                                frame_num,
                                ts_sec,
                                det_count,
                                proc_ms,
                            )

                    # Pace to target fps
                    elapsed = time.perf_counter() - t0
                    sleep_for = interval - elapsed
                    if sleep_for > 0:
                        stop_event.wait(timeout=sleep_for)

            except Exception:
                logger.exception("Stream error in round %d", round_num)

        if not loop:
            break

    logger.info("Stream loop finished.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from sopilot.vigil.factory_presets import get_preset, list_presets

    p = argparse.ArgumentParser(
        description="SOPilot ライブブラウザデモ — FastAPI + MJPEG プレビュー",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"利用可能なプリセット: {list_presets()}",
    )
    p.add_argument("--input", required=True, help="入力動画ファイル")
    p.add_argument(
        "--preset",
        default="warehouse_safety",
        help="プリセット名 (デフォルト: warehouse_safety)",
    )
    p.add_argument("--port", type=int, default=8000, help="サーバーポート (デフォルト: 8000)")
    p.add_argument("--fps", type=float, default=3.0, help="送信 FPS (デフォルト: 3)")
    p.add_argument("--loop", action="store_true", help="動画を繰り返し再生")
    args = p.parse_args()

    # ── Validate input ──────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    preset = get_preset(args.preset)
    rules: list[str] = preset["rules"]

    logger.info("Preset  : %s (%s)", args.preset, preset["name"])
    logger.info("Rules   : %s", rules)
    logger.info("Input   : %s", input_path)
    logger.info("FPS     : %s", args.fps)
    logger.info("Loop    : %s", args.loop)

    # ── Start server ────────────────────────────────────────────────────────
    server_proc = _start_server(args.port)

    try:
        if not _wait_for_server(args.port, timeout=30.0):
            logger.error("Server did not become ready within 30 s. Check logs.")
            server_proc.terminate()
            sys.exit(1)

        logger.info("Server ready at http://localhost:%d", args.port)

        # ── Create session ──────────────────────────────────────────────────
        sid = _create_session(args.port, preset["name"], rules)
        preview_url = f"http://localhost:{args.port}/vigil/stream/sessions/{sid}/preview"
        logger.info("Session : %s", sid)
        logger.info("Preview : %s", preview_url)

        # ── Open browser ────────────────────────────────────────────────────
        time.sleep(0.5)
        webbrowser.open(preview_url)
        logger.info("Browser opened. Press Ctrl+C to stop.")

        # ── Streaming thread ────────────────────────────────────────────────
        stop_event = Event()
        stream_thread = Thread(
            target=_stream_loop,
            args=(args.port, sid, input_path, args.fps, args.loop, stop_event),
            daemon=True,
            name="demo-stream",
        )
        stream_thread.start()

        try:
            stream_thread.join()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping …")
            stop_event.set()
            stream_thread.join(timeout=5.0)

    finally:
        # ── Cleanup ─────────────────────────────────────────────────────────
        try:
            _delete_session(args.port, sid)  # type: ignore[possibly-undefined]
        except Exception:
            pass
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5.0)
        except Exception:
            server_proc.kill()
        logger.info("Done.")


if __name__ == "__main__":
    main()
