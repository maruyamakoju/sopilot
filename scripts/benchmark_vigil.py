#!/usr/bin/env python3
"""
VigilPilot Performance Benchmark
Measures VLM latency, frame throughput, and event storage performance.

Usage:
    python scripts/benchmark_vigil.py [--host http://localhost:8000] [--frames 20]

Output:
    - Per-frame VLM analysis latency (mean, p50, p95, p99)
    - End-to-end throughput (frames/sec)
    - Event storage latency
    - Summary table suitable for technical documentation
"""

import argparse
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx")
    sys.exit(1)

BASE_URL = "http://localhost:8000"
API_KEY = os.environ.get("SOPILOT_API_KEY", "")

# ── ANSI colours ────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
RESET = "\033[0m"

def _hdr(text: str) -> str:
    return f"{BOLD}{CYAN}{text}{RESET}"

def _ok(text: str) -> str:
    return f"{GREEN}✓{RESET} {text}"

def _warn(text: str) -> str:
    return f"{YELLOW}⚠{RESET} {text}"

def _err(text: str) -> str:
    return f"{RED}✗{RESET} {text}"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _headers() -> dict:
    h = {"Accept": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def _make_jpeg(width: int = 320, height: int = 240) -> bytes:
    """Create a synthetic JPEG frame."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    # Add some structure to make it look like a real scene
    cv2.rectangle(img, (50, 50), (150, 150), (200, 100, 50), -1)
    cv2.circle(img, (250, 100), 40, (50, 200, 100), -1)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _make_video(path: Path, seconds: int = 3, fps: float = 8.0) -> None:
    """Write a minimal synthetic AVI."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    total = int(seconds * fps)
    for i in range(total):
        val = (i * 20) % 255
        frame = np.full((240, 320, 3), (val, 120, 80), dtype=np.uint8)
        writer.write(frame)
    writer.release()


# ── Benchmarks ──────────────────────────────────────────────────────────────

def bench_health(client: httpx.Client) -> bool:
    """Check server is reachable."""
    try:
        r = client.get(f"{BASE_URL}/health", headers=_headers(), timeout=5)
        r.raise_for_status()
        data = r.json()
        print(_ok(f"Server healthy — version {data.get('version', '?')}"))
        return True
    except Exception as e:
        print(_err(f"Health check failed: {e}"))
        return False


def bench_webcam_frames(
    client: httpx.Client,
    session_id: int,
    n_frames: int,
) -> dict:
    """
    Benchmark POST /vigil/sessions/{id}/webcam-frame
    Returns latency statistics in milliseconds.
    """
    jpeg = _make_jpeg()
    latencies = []

    print(f"\n  Sending {n_frames} frames to session {session_id}…")
    for i in range(n_frames):
        t0 = time.perf_counter()
        r = client.post(
            f"{BASE_URL}/vigil/sessions/{session_id}/webcam-frame",
            headers=_headers(),
            files={"file": ("frame.jpg", io.BytesIO(jpeg), "image/jpeg")},
            params={"store": "false"},   # don't store — pure VLM latency
            timeout=30,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        status = "✓" if r.status_code == 200 else f"HTTP {r.status_code}"
        bar = "█" * min(int(elapsed_ms / 100), 30)
        print(f"    [{i+1:>3}/{n_frames}] {elapsed_ms:6.0f} ms  {bar}  {status}")

    if not latencies:
        return {}

    latencies_sorted = sorted(latencies)
    return {
        "n": len(latencies),
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": latencies_sorted[int(len(latencies) * 0.95)],
        "p99_ms": latencies_sorted[int(len(latencies) * 0.99)],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "throughput_fps": 1000 / statistics.mean(latencies),
    }


def bench_session_creation(client: httpx.Client, n: int = 20) -> dict:
    """Benchmark POST /vigil/sessions creation latency."""
    latencies = []
    session_ids = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.post(
            f"{BASE_URL}/vigil/sessions",
            headers={**_headers(), "Content-Type": "application/json"},
            content=json.dumps({
                "name": "bench",
                "rules": ["テストルール"],
                "sample_fps": 1.0,
                "severity_threshold": "warning",
            }),
            timeout=10,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            latencies.append(elapsed_ms)
            session_ids.append(r.json()["session_id"])

    # Clean up
    for sid in session_ids:
        try:
            client.delete(f"{BASE_URL}/vigil/sessions/{sid}", headers=_headers(), timeout=5)
        except Exception:
            pass

    if not latencies:
        return {}
    return {
        "n": len(latencies),
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }


def bench_event_list(client: httpx.Client, session_id: int) -> dict:
    """Benchmark GET /vigil/sessions/{id}/events after analysis completes."""
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        client.get(
            f"{BASE_URL}/vigil/sessions/{session_id}/events",
            headers=_headers(),
            timeout=10,
        )
        latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": statistics.mean(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VigilPilot performance benchmark")
    parser.add_argument("--host", default=BASE_URL, help="Server base URL")
    parser.add_argument("--frames", type=int, default=10,
                        help="Number of webcam frames to send for VLM latency test")
    parser.add_argument("--api-key", default=API_KEY, help="API key")
    args = parser.parse_args()

    global BASE_URL, API_KEY
    BASE_URL = args.host.rstrip("/")
    API_KEY = args.api_key

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         VigilPilot Performance Benchmark                 ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    with httpx.Client() as client:
        # Health check
        print(_hdr("1. Health Check"))
        if not bench_health(client):
            sys.exit(1)

        # Session creation latency
        print()
        print(_hdr("2. Session Creation Latency (n=20)"))
        sc = bench_session_creation(client, n=20)
        if sc:
            print(f"  mean={sc['mean_ms']:.1f}ms  "
                  f"median={sc['median_ms']:.1f}ms  "
                  f"min={sc['min_ms']:.1f}ms  max={sc['max_ms']:.1f}ms")

        # Create a session for VLM benchmark
        r = client.post(
            f"{BASE_URL}/vigil/sessions",
            headers={**_headers(), "Content-Type": "application/json"},
            content=json.dumps({
                "name": "benchmark-session",
                "rules": [
                    "ヘルメット未着用の作業者を検出",
                    "安全ベルトなしで高所作業している人を検出",
                ],
                "sample_fps": 1.0,
                "severity_threshold": "info",
            }),
            timeout=10,
        )
        if r.status_code != 200:
            print(_err(f"Could not create benchmark session: {r.status_code}"))
            sys.exit(1)
        session_id = r.json()["session_id"]

        # VLM latency benchmark
        print()
        print(_hdr(f"3. VLM Frame Analysis Latency (n={args.frames})"))
        print(f"  Endpoint: POST /vigil/sessions/{session_id}/webcam-frame")
        print(f"  Frame size: 320×240 JPEG (synthetic)")
        vlm = bench_webcam_frames(client, session_id, args.frames)

        # Event list latency
        print()
        print(_hdr("4. Event List Query Latency (n=10)"))
        el = bench_event_list(client, session_id)
        if el:
            print(f"  mean={el['mean_ms']:.1f}ms  "
                  f"min={el['min_ms']:.1f}ms  max={el['max_ms']:.1f}ms")

        # Clean up
        client.delete(f"{BASE_URL}/vigil/sessions/{session_id}",
                      headers=_headers(), timeout=5)

    # ── Summary table ────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                     Results Summary                      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    if vlm:
        print(f"  {'Metric':<30} {'Value':>12}")
        print(f"  {'─'*30} {'─'*12}")
        print(f"  {'VLM latency — mean':<30} {vlm['mean_ms']:>11.0f}ms")
        print(f"  {'VLM latency — median (p50)':<30} {vlm['median_ms']:>11.0f}ms")
        print(f"  {'VLM latency — p95':<30} {vlm['p95_ms']:>11.0f}ms")
        print(f"  {'VLM latency — p99':<30} {vlm['p99_ms']:>11.0f}ms")
        print(f"  {'VLM latency — min':<30} {vlm['min_ms']:>11.0f}ms")
        print(f"  {'VLM latency — max':<30} {vlm['max_ms']:>11.0f}ms")
        print(f"  {'Effective throughput':<30} {vlm['throughput_fps']:>10.2f}fps")
        if sc:
            print(f"  {'Session creation — mean':<30} {sc['mean_ms']:>11.0f}ms")
        if el:
            print(f"  {'Event list query — mean':<30} {el['mean_ms']:>11.0f}ms")
        print()

        # Interpretation
        mean = vlm["mean_ms"]
        if mean < 2000:
            tier = f"{GREEN}Excellent{RESET} (< 2s/frame)"
        elif mean < 5000:
            tier = f"{YELLOW}Good{RESET} (2-5s/frame, suitable for 0.2–0.5 fps sampling)"
        else:
            tier = f"{RED}Slow{RESET} (> 5s/frame — consider lower sample_fps or local Qwen3)"

        print(f"  Performance tier: {tier}")
        print()
        print("  Note: latency is dominated by the VLM API (Claude Sonnet 4.6).")
        print("  For offline/GPU deployment, switch to VIGIL_VLM_BACKEND=qwen3")
        print("  to achieve < 500ms/frame on a modern GPU.")

    print()


if __name__ == "__main__":
    main()
