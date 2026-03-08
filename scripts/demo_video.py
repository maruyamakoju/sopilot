#!/usr/bin/env python3
"""SOPilot キラーデモ: オフライン動画処理スクリプト.

MP4 動画を読み込み、違反検出 + アノテーション付き MP4 と JSON サマリーを出力する。

Usage::

    python scripts/demo_video.py --input factory.mp4 --preset warehouse_safety
    python scripts/demo_video.py --input factory.mp4 --preset construction_safety --show
    python scripts/demo_video.py --input 0 --preset retail_security --fps 10 --no-output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

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
# Detector priority chain
# ---------------------------------------------------------------------------


class _DetectorAdapter:
    """Adapt any detector to the ``detect(frame) -> list`` signature.

    ``FrameInferrer._detect()`` calls ``detector.detect(frame)`` with a single
    positional argument.  Some detectors (e.g. MockDetector) also require a
    *prompts* argument.  This adapter catches the TypeError and retries with an
    empty prompts list.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._needs_prompts: bool | None = None  # discovered on first call

    def detect(self, frame: np.ndarray) -> list:
        if self._needs_prompts is False:
            return self._inner.detect(frame)
        if self._needs_prompts is True:
            return self._inner.detect(frame, [])
        # Auto-detect on first call
        try:
            result = self._inner.detect(frame)
            self._needs_prompts = False
            return result
        except TypeError:
            result = self._inner.detect(frame, [])
            self._needs_prompts = True
            return result


class _YOLODetector:
    """Wrap ``ultralytics.YOLO`` to the ``detect(frame) -> list[Detection]`` interface."""

    def __init__(self, model: Any) -> None:
        self._model = model
        from sopilot.perception.types import BBox, Detection

        self._Detection = Detection
        self._BBox = BBox

    def detect(self, frame: np.ndarray) -> list:
        results = self._model(frame, verbose=False)[0]
        dets: list = []
        h, w = frame.shape[:2]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            dets.append(
                self._Detection(
                    label=label,
                    confidence=conf,
                    bbox=self._BBox(x1 / w, y1 / h, x2 / w, y2 / h),
                )
            )
        return dets


def _build_detector() -> Any:
    """Return best available detector: YOLO → ONNX → MockDetector."""
    # 1. Try ultralytics YOLOv8n (CPU)
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]

        model = YOLO("yolov8n.pt")
        logger.info("Detector: YOLOv8n (ultralytics)")
        return _YOLODetector(model)
    except Exception as exc:
        logger.info("YOLOv8 unavailable (%s), trying ONNX…", exc)

    # 2. Try SOPilot ONNX detector
    try:
        from sopilot.perception.onnx_detector import OnnxDetector  # type: ignore[import]

        det = OnnxDetector()
        logger.info("Detector: ONNX")
        return _DetectorAdapter(det)
    except Exception as exc:
        logger.info("ONNX detector unavailable (%s), using MockDetector…", exc)

    # 3. MockDetector fallback
    from sopilot.perception.detector import MockDetector

    logger.info("Detector: MockDetector (use_heuristics=True)")
    return _DetectorAdapter(MockDetector(use_heuristics=True))


# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------


def _add_hud(
    frame: np.ndarray,
    frame_num: int,
    proc_ms: float,
    det_count: int,
    viol_count: int,
) -> np.ndarray:
    """Add frame-info HUD to an already-annotated frame (in-place).

    Adds:
    * Top-right  : frame number + processing time
    * Bottom-left: detection count + violation count
    """
    try:
        import cv2  # type: ignore[import-untyped]

        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        ltype = cv2.LINE_AA
        color = (220, 220, 80)  # yellow

        # Top-right: frame # / proc ms
        info = f"Frame:{frame_num:5d}  {proc_ms:.1f}ms"
        (tw, th), _ = cv2.getTextSize(info, font, scale, thickness)
        cv2.putText(frame, info, (w - tw - 8, th + 4), font, scale, color, thickness, ltype)

        # Bottom-left: det / viol counts
        counts = f"Det:{det_count}  Viol:{viol_count}"
        cv2.putText(frame, counts, (6, h - 8), font, scale, (200, 200, 200), thickness, ltype)
    except Exception:
        logger.debug("HUD overlay failed", exc_info=True)
    return frame


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_video(args: argparse.Namespace) -> None:
    import cv2  # type: ignore[import-untyped]

    from sopilot.vigil.extractor import iter_frames
    from sopilot.vigil.factory_presets import get_preset
    from sopilot.vigil.frame_inferrer import FrameInferrer

    preset = get_preset(args.preset)
    rules: list[str] = preset["rules"]

    logger.info("Preset  : %s (%s)", args.preset, preset["name"])
    logger.info("Rules   : %s", rules)

    detector = _build_detector()
    inferrer = FrameInferrer(detector=detector)

    # ── Resolve input ───────────────────────────────────────────────────────
    # Support webcam index (integer string) or file path
    try:
        cam_index = int(args.input)
        input_path: Path | None = None
        is_camera = True
    except ValueError:
        input_path = Path(args.input)
        is_camera = False
        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            sys.exit(1)

    # ── Probe video dimensions for writer ──────────────────────────────────
    source = cam_index if is_camera else str(input_path)
    cap_probe = cv2.VideoCapture(source)
    if not cap_probe.isOpened():
        logger.error("Cannot open: %s", args.input)
        sys.exit(1)
    frame_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    cap_probe.release()

    # ── Output path ────────────────────────────────────────────────────────
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        out_path = Path(args.output)
    elif input_path is not None:
        out_path = input_path.parent / f"{input_path.stem}_demo_{ts_str}.mp4"
    else:
        out_path = Path(f"demo_output_{ts_str}.mp4")

    writer: Any = None
    if not args.no_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(args.fps), (frame_w, frame_h))
        if not writer.isOpened():
            logger.warning("VideoWriter failed; disabling MP4 output")
            writer = None

    # ── Stats ──────────────────────────────────────────────────────────────
    total_frames_processed = 0
    total_det = 0
    total_viol = 0
    violations_log: list[dict] = []

    # ── Frame loop ─────────────────────────────────────────────────────────
    logger.info("Processing… (Ctrl+C or 'q' to stop)")

    with tempfile.TemporaryDirectory(prefix="sopilot_demo_") as tmpdir:
        tmp_path = Path(tmpdir)

        if is_camera:
            # Webcam: read frames directly without extractor
            cap = cv2.VideoCapture(cam_index)
            interval = 1.0 / args.fps
            frame_num = 0
            while cap.isOpened():
                ret, bgr = cap.read()
                if not ret:
                    break
                _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = bytes(buf)
                frame_num += 1
                ts_sec = frame_num / args.fps
                _process_frame(
                    frame_bytes, frame_num, ts_sec, inferrer, rules, writer,
                    args.show, total_det, total_viol, violations_log,
                )
                total_frames_processed = frame_num
                time.sleep(max(0.0, interval - 0.001))
                if args.show and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
        else:
            # MP4 file: use iter_frames extractor
            for frame_num, ts_sec, jpeg_path in iter_frames(
                input_path,  # type: ignore[arg-type]
                sample_fps=float(args.fps),
                output_dir=tmp_path,
            ):
                frame_bytes = jpeg_path.read_bytes()
                result = _infer_and_annotate(
                    frame_bytes, frame_num, inferrer, rules, writer, args.show,
                )
                total_frames_processed = frame_num
                total_det += len(result["detections"])
                total_viol += len(result["violations"])
                if result["violations"]:
                    violations_log.append(
                        {
                            "frame": frame_num,
                            "ts": round(ts_sec, 2),
                            "violations": result["violations"],
                            "detections": len(result["detections"]),
                        }
                    )
                    logger.warning(
                        "[%5d] VIOLATION: %s", frame_num, result["violations"]
                    )
                else:
                    logger.debug(
                        "[%5d] OK  det=%d  %.1fms",
                        frame_num,
                        len(result["detections"]),
                        result["frame_ms"],
                    )

                if args.show and cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    # ── Cleanup ────────────────────────────────────────────────────────────
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # ── JSON summary ───────────────────────────────────────────────────────
    summary: dict = {
        "input": str(args.input),
        "preset": args.preset,
        "preset_name": preset["name"],
        "rules": rules,
        "fps_processed": args.fps,
        "total_frames": total_frames_processed,
        "total_detections": total_det,
        "total_violations": total_viol,
        "violation_frames": len(violations_log),
        "output": str(out_path) if (not args.no_output and writer is not None) else None,
        "violations_log": violations_log,
    }

    if not args.no_output:
        summary_path = out_path.with_suffix(".json")
    else:
        summary_path = Path(f"demo_summary_{ts_str}.json")

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "Done.  frames=%d  detections=%d  violations=%d",
        total_frames_processed,
        total_det,
        total_viol,
    )
    if not args.no_output and writer is not None:
        logger.info("Output : %s", out_path)
    logger.info("Summary: %s", summary_path)


def _infer_and_annotate(
    frame_bytes: bytes,
    frame_num: int,
    inferrer: Any,
    rules: list[str],
    writer: Any,
    show: bool,
) -> dict:
    """Run inferrer + HUD overlay, write to video writer and/or display."""
    import cv2  # type: ignore[import-untyped]

    result = inferrer.infer(frame_bytes, rules)

    # Decode the already-annotated JPEG to add our HUD on top
    arr = np.frombuffer(result["annotated_jpeg"], dtype=np.uint8)
    annotated_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if annotated_frame is not None:
        annotated_frame = _add_hud(
            annotated_frame,
            frame_num,
            result["frame_ms"],
            len(result["detections"]),
            len(result["violations"]),
        )

        if writer is not None:
            writer.write(annotated_frame)

        if show:
            cv2.imshow("SOPilot Demo", annotated_frame)

    return result


def _process_frame(
    frame_bytes: bytes,
    frame_num: int,
    ts_sec: float,
    inferrer: Any,
    rules: list[str],
    writer: Any,
    show: bool,
    total_det: int,
    total_viol: int,
    violations_log: list,
) -> None:
    """Helper for webcam loop (mutates totals in-place via list append)."""
    result = _infer_and_annotate(frame_bytes, frame_num, inferrer, rules, writer, show)
    total_det += len(result["detections"])
    total_viol += len(result["violations"])
    if result["violations"]:
        violations_log.append(
            {
                "frame": frame_num,
                "ts": round(ts_sec, 2),
                "violations": result["violations"],
                "detections": len(result["detections"]),
            }
        )
        logger.warning("[%5d] VIOLATION: %s", frame_num, result["violations"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    from sopilot.vigil.factory_presets import list_presets

    p = argparse.ArgumentParser(
        description="SOPilot キラーデモ: MP4 → アノテーション付き MP4 + JSON サマリー",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"利用可能なプリセット: {list_presets()}",
    )
    p.add_argument("--input", required=True, help="入力動画ファイルまたは Webcam インデックス (0, 1, …)")
    p.add_argument("--output", help="出力 MP4 ファイル名 (デフォルト: <input>_demo_YYYYMMDD.mp4)")
    p.add_argument(
        "--preset",
        default="warehouse_safety",
        help="プリセット名 (デフォルト: warehouse_safety)",
    )
    p.add_argument("--fps", type=float, default=5.0, help="処理 FPS — サンプリングレート (デフォルト: 5)")
    p.add_argument("--show", action="store_true", help="cv2.imshow でリアルタイムプレビューを表示")
    p.add_argument("--no-output", action="store_true", help="MP4 ファイルを出力しない (プレビュー専用)")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
