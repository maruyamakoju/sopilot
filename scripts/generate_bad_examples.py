from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def list_videos(input_dir: Path, recursive: bool) -> list[Path]:
    if not input_dir.exists():
        raise SystemExit(f"input-dir not found: {input_dir}")
    globber = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted([path for path in globber if path.is_file() and path.suffix.lower() in VIDEO_EXTS])


def _read_all_frames(path: Path, max_frames: int | None) -> tuple[list, float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 8.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        width, height = 640, 360

    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    capture.release()
    return frames, fps, (width, height)


def _write_mp4(path: Path, frames: list, fps: float, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, fps),
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {path}")
    try:
        for frame in frames:
            if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
                frame = cv2.resize(frame, size)
            writer.write(frame)
    finally:
        writer.release()


def make_bad_variants(
    src: Path,
    dst_dir: Path,
    keep_head_ratio: float,
    keep_tail_ratio: float,
    freeze_seconds: float,
    max_frames: int | None,
) -> list[Path]:
    frames, fps, size = _read_all_frames(src, max_frames=max_frames)
    if len(frames) < 8:
        return []

    outputs: list[Path] = []
    stem = src.stem

    head_n = max(4, int(len(frames) * keep_head_ratio))
    head_frames = frames[:head_n]
    out_head = dst_dir / f"{stem}_bad_cut_tail.mp4"
    _write_mp4(out_head, head_frames, fps, size)
    outputs.append(out_head)

    tail_start = max(0, int(len(frames) * (1.0 - keep_tail_ratio)))
    tail_frames = frames[tail_start:]
    if len(tail_frames) >= 4:
        out_tail = dst_dir / f"{stem}_bad_skip_start.mp4"
        _write_mp4(out_tail, tail_frames, fps, size)
        outputs.append(out_tail)

    freeze_count = max(4, int(max(0.5, freeze_seconds) * fps))
    freeze_frame = frames[min(len(frames) - 1, max(0, int(len(frames) * 0.25)))]
    freeze_frames = [freeze_frame for _ in range(freeze_count)]
    out_freeze = dst_dir / f"{stem}_bad_freeze.mp4"
    _write_mp4(out_freeze, freeze_frames, fps, size)
    outputs.append(out_freeze)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate intentionally bad trainee examples from existing videos.")
    parser.add_argument("--input-dir", required=True, help="Source trainee folder")
    parser.add_argument("--output-dir", required=True, help="Destination folder for bad trainee videos")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-source", type=int, default=6, help="Max source videos to transform")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic source selection")
    parser.add_argument("--selection-output", default=None, help="Optional JSON path for selected input manifest")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap while reading source video")
    parser.add_argument("--keep-head-ratio", type=float, default=0.35)
    parser.add_argument("--keep-tail-ratio", type=float, default=0.35)
    parser.add_argument("--freeze-seconds", type=float, default=4.0)
    args = parser.parse_args()

    src_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    all_videos = list_videos(src_dir, recursive=args.recursive)
    if args.max_source < 1:
        raise SystemExit("--max-source must be >= 1")
    if len(all_videos) > int(args.max_source):
        rng = random.Random(int(args.seed))
        selected = rng.sample(all_videos, int(args.max_source))
        videos = sorted(selected)
    else:
        videos = all_videos
    if not videos:
        raise SystemExit("no source videos found")

    selection_payload: dict[str, Any] = {
        "seed": int(args.seed),
        "source_dir": str(src_dir),
        "output_dir": str(out_dir),
        "max_source": int(args.max_source),
        "recursive": bool(args.recursive),
        "candidate_count": len(all_videos),
        "selected_count": len(videos),
        "selected": [str(path.resolve()) for path in videos],
    }
    if args.selection_output:
        selection_path = Path(args.selection_output).resolve()
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection_path.write_text(json.dumps(selection_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"selection_manifest={selection_path}")

    created: list[Path] = []
    for idx, src in enumerate(videos, start=1):
        print(f"[{idx}/{len(videos)}] source={src.name}")
        try:
            made = make_bad_variants(
                src=src,
                dst_dir=out_dir,
                keep_head_ratio=max(0.1, min(0.9, args.keep_head_ratio)),
                keep_tail_ratio=max(0.1, min(0.9, args.keep_tail_ratio)),
                freeze_seconds=max(0.5, args.freeze_seconds),
                max_frames=args.max_frames,
            )
            created.extend(made)
            for item in made:
                print(f"  -> {item.name}")
        except Exception as exc:
            print(f"  -> failed {exc}")

    print(f"created={len(created)} output_dir={out_dir}")


if __name__ == "__main__":
    main()
