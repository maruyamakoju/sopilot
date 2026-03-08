#!/usr/bin/env python3
"""SOPilot キラーデモ: 合成ファクトリー動画ジェネレーター.

OpenCV のみで動作する自己完結型の合成ビデオ生成スクリプト。
demo_video.py / demo_live.py 用のテスト映像を作成する。

Usage::

    python scripts/generate_demo_video.py
    python scripts/generate_demo_video.py --output my_factory.mp4 --duration 40 --fps 25
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root on path (for factory_presets import)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------

W, H = 640, 480
FPS = 25

# Colours (BGR)
C_BG_FLOOR = (55, 55, 55)
C_BG_STRIPE = (45, 45, 45)
C_WALL = (70, 65, 60)
C_ZONE_NORMAL = (40, 100, 40)       # safe zone border
C_ZONE_RESTRICTED = (30, 30, 180)   # restricted zone border
C_ZONE_FILL = (0, 0, 60, 60)        # restricted zone fill (semi-transparent)
C_PERSON_BODY = (200, 120, 60)      # person body colour
C_PERSON_HELMET = (30, 200, 30)     # helmet indicator (green = has helmet)
C_PERSON_NOHELMET = (30, 30, 220)   # no helmet (red)
C_FORKLIFT_BODY = (30, 180, 220)    # forklift body
C_FORKLIFT_FORK = (20, 130, 160)    # forklift forks
C_VIOLATION_BANNER = (0, 0, 160)
C_WHITE = (255, 255, 255)
C_YELLOW = (0, 220, 220)
C_RED = (0, 0, 240)
C_GREEN = (30, 200, 30)


# ---------------------------------------------------------------------------
# Entity definitions
# ---------------------------------------------------------------------------


@dataclass
class Person:
    x: float          # centre x (pixels)
    y: float          # centre y (pixels)
    w: int = 28
    h: int = 52
    has_helmet: bool = True
    vx: float = 0.0   # velocity x (px/frame)
    vy: float = 0.0   # velocity y (px/frame)
    label: str = "worker"

    @property
    def rect(self) -> tuple[int, int, int, int]:
        hx, hy = self.w // 2, self.h // 2
        return (int(self.x - hx), int(self.y - hy), int(self.x + hx), int(self.y + hy))


@dataclass
class Forklift:
    x: float
    y: float
    w: int = 60
    h: int = 45
    vx: float = 0.0
    vy: float = 0.0

    @property
    def rect(self) -> tuple[int, int, int, int]:
        hx, hy = self.w // 2, self.h // 2
        return (int(self.x - hx), int(self.y - hy), int(self.x + hx), int(self.y + hy))


@dataclass
class Zone:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    restricted: bool = False


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _draw_background(img: Any) -> None:
    """Draw factory floor with stripes."""
    import cv2  # type: ignore[import-untyped]

    img[:] = C_BG_FLOOR
    # Wall at top
    img[0:60, :] = C_WALL
    # Floor stripes
    for i in range(0, W, 80):
        cv2.line(img, (i, 60), (i, H), C_BG_STRIPE, 1)
    for j in range(60, H, 60):
        cv2.line(img, (0, j), (W, j), C_BG_STRIPE, 1)


def _draw_zone(img: Any, zone: Zone, alpha: float = 0.25) -> None:
    import cv2  # type: ignore[import-untyped]
    import numpy as np

    color = C_ZONE_RESTRICTED if zone.restricted else C_ZONE_NORMAL
    # Semi-transparent fill
    overlay = img.copy()
    fill = (0, 0, 80) if zone.restricted else (0, 60, 0)
    cv2.rectangle(overlay, (zone.x1, zone.y1), (zone.x2, zone.y2), fill, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    # Border
    thickness = 2
    if zone.restricted:
        # Dashed border
        pts = [(zone.x1, zone.y1), (zone.x2, zone.y1),
               (zone.x2, zone.y2), (zone.x1, zone.y2), (zone.x1, zone.y1)]
        for k in range(len(pts) - 1):
            x1, y1 = pts[k]
            x2, y2 = pts[k + 1]
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            dash, gap = 12, 6
            total = dash + gap
            n_segs = int(length / total)
            for s in range(n_segs):
                t0 = s * total / length
                t1 = (s * total + dash) / length
                p0 = (int(x1 + dx * t0), int(y1 + dy * t0))
                p1 = (int(x1 + dx * t1), int(y1 + dy * t1))
                cv2.line(img, p0, p1, color, thickness)
    else:
        cv2.rectangle(img, (zone.x1, zone.y1), (zone.x2, zone.y2), color, thickness)
    # Label
    cv2.putText(img, zone.label, (zone.x1 + 4, zone.y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _draw_person(img: Any, p: Person) -> None:
    import cv2  # type: ignore[import-untyped]

    x1, y1, x2, y2 = p.rect
    # Body
    cv2.rectangle(img, (x1, y1 + 14), (x2, y2), C_PERSON_BODY, -1)
    cv2.rectangle(img, (x1, y1 + 14), (x2, y2), (255, 255, 255), 1)
    # Head circle
    cx = (x1 + x2) // 2
    head_y = y1 + 10
    cv2.circle(img, (cx, head_y), 9, C_PERSON_BODY, -1)
    cv2.circle(img, (cx, head_y), 9, (255, 255, 255), 1)
    # Helmet
    hcol = C_PERSON_HELMET if p.has_helmet else C_PERSON_NOHELMET
    cv2.ellipse(img, (cx, head_y - 3), (11, 7), 0, 180, 360, hcol, -1)
    cv2.ellipse(img, (cx, head_y - 3), (11, 7), 0, 180, 360, C_WHITE, 1)
    # Label
    cv2.putText(img, p.label, (x1, y2 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_WHITE, 1, cv2.LINE_AA)


def _draw_forklift(img: Any, fk: Forklift) -> None:
    import cv2  # type: ignore[import-untyped]

    x1, y1, x2, y2 = fk.rect
    # Body
    cv2.rectangle(img, (x1, y1), (x2, y2), C_FORKLIFT_BODY, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), C_WHITE, 1)
    # Cab
    cab_x = x1 + (x2 - x1) * 3 // 4
    cv2.rectangle(img, (cab_x, y1 + 5), (x2 - 2, y2 - 2), (100, 200, 230), -1)
    # Forks
    fork_y = y2
    cv2.rectangle(img, (x1 + 4, fork_y), (x1 + 10, fork_y + 20), C_FORKLIFT_FORK, -1)
    cv2.rectangle(img, (x1 + 16, fork_y), (x1 + 22, fork_y + 20), C_FORKLIFT_FORK, -1)
    # Label
    cv2.putText(img, "forklift", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_YELLOW, 1, cv2.LINE_AA)


def _draw_hud(img: Any, frame_num: int, t_sec: float, scene: str, violations: list[str]) -> None:
    import cv2  # type: ignore[import-untyped]

    # Top bar
    cv2.rectangle(img, (0, 0), (W, 22), (20, 20, 20), -1)
    cv2.putText(img, f"SOPilot Demo | Scene: {scene}",
                (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_YELLOW, 1, cv2.LINE_AA)
    ts_str = f"t={t_sec:.1f}s  f={frame_num}"
    (tw, _), _ = cv2.getTextSize(ts_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(img, ts_str, (W - tw - 6, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # Violation banner
    if violations:
        banner_h = 20 + 18 * len(violations)
        cv2.rectangle(img, (0, 22), (W, 22 + banner_h), C_VIOLATION_BANNER, -1)
        for i, v in enumerate(violations):
            cv2.putText(img, f"! {v}", (6, 38 + 18 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_WHITE, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------


def _scene_normal(t: float) -> tuple[list[Person], list[Forklift], list[str]]:
    """0–8 s: 2 workers with helmets, walking normally."""
    p1 = Person(x=120 + t * 12, y=220, has_helmet=True, label="worker A")
    p2 = Person(x=400 - t * 8, y=340, has_helmet=True, label="worker B")
    return [p1, p2], [], []


def _scene_zone_intrusion(t: float) -> tuple[list[Person], list[Forklift], list[str]]:
    """8–16 s: worker B enters restricted zone."""
    p1 = Person(x=200 + t * 5, y=220, has_helmet=True, label="worker A")
    # worker B walks into restricted zone (x=380–520, y=200–380)
    enter_x = 330 + t * 16
    p2 = Person(x=enter_x, y=280, has_helmet=True, label="worker B")
    violations = ["立入禁止エリア侵入"] if enter_x > 390 else []
    return [p1, p2], [], violations


def _scene_ppe_violation(t: float) -> tuple[list[Person], list[Forklift], list[str]]:
    """16–24 s: new worker appears without helmet."""
    p1 = Person(x=180 + t * 6, y=240, has_helmet=True, label="worker A")
    p2 = Person(x=340, y=180 + t * 10, has_helmet=False, label="worker C")
    violations: list[str] = ["ヘルメット未着用"]
    return [p1, p2], [], violations


def _scene_combined(t: float) -> tuple[list[Person], list[Forklift], list[str]]:
    """24–32 s: forklift + proximity zone + no-helmet worker."""
    p1 = Person(x=300 + t * 8, y=280, has_helmet=True, label="worker A")
    p2 = Person(x=260, y=300, has_helmet=False, label="worker C")
    fk = Forklift(x=280 + t * 10, y=310)
    violations: list[str] = ["フォークリフト接近禁止エリア侵入", "ヘルメット未着用"]
    return [p1, p2], [fk], violations


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate(output: Path, duration: float, fps: int, quiet: bool) -> None:
    import cv2  # type: ignore[import-untyped]
    import numpy as np

    total_frames = int(duration * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (W, H))
    if not writer.isOpened():
        print(f"ERROR: cannot open VideoWriter for {output}", file=sys.stderr)
        sys.exit(1)

    # Zones
    zones = [
        Zone(30, 80, 350, 430, "作業エリア A", restricted=False),
        Zone(370, 190, 570, 390, "立入禁止", restricted=True),
    ]

    scenes = [
        (0.0, 8.0, "正常運行", _scene_normal),
        (8.0, 16.0, "エリア侵入", _scene_zone_intrusion),
        (16.0, 24.0, "PPE 違反", _scene_ppe_violation),
        (24.0, duration, "複合違反", _scene_combined),
    ]

    for frame_num in range(total_frames):
        t_sec = frame_num / fps

        # Select scene
        scene_label = "正常運行"
        scene_fn = _scene_normal
        scene_t = t_sec
        for s_start, s_end, s_label, s_fn in scenes:
            if s_start <= t_sec < s_end:
                scene_label = s_label
                scene_fn = s_fn
                scene_t = t_sec - s_start
                break

        persons, forklifts, violations = scene_fn(scene_t)

        # Render frame
        img = np.zeros((H, W, 3), dtype=np.uint8)
        _draw_background(img)
        for z in zones:
            _draw_zone(img, z)
        for fk in forklifts:
            _draw_forklift(img, fk)
        for p in persons:
            _draw_person(img, p)
        _draw_hud(img, frame_num, t_sec, scene_label, violations)

        writer.write(img)

        if not quiet and frame_num % fps == 0:
            print(f"  {t_sec:5.1f}s / {duration:.0f}s  [{scene_label}]  viol={violations}")

    writer.release()


def main() -> None:
    p = argparse.ArgumentParser(
        description="SOPilot: 合成ファクトリー動画ジェネレーター",
    )
    p.add_argument("--output", default="factory_demo.mp4", help="出力ファイル (デフォルト: factory_demo.mp4)")
    p.add_argument("--duration", type=float, default=32.0, help="動画の長さ (秒, デフォルト: 32)")
    p.add_argument("--fps", type=int, default=FPS, help=f"FPS (デフォルト: {FPS})")
    p.add_argument("--quiet", action="store_true", help="進捗表示を省略")
    args = p.parse_args()

    out = Path(args.output)
    print(f"Generating {out}  ({args.duration}s @ {args.fps}fps)…")
    generate(out, args.duration, args.fps, args.quiet)
    print(f"Done → {out}")
    print()
    print("次のコマンドでデモを実行:")
    print(f"  python scripts/demo_video.py --input {out} --preset warehouse_safety --show")
    print(f"  python scripts/demo_live.py  --input {out} --preset warehouse_safety --loop")


if __name__ == "__main__":
    main()
