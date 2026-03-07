#!/usr/bin/env python3
"""End-to-end Perception Orchestration Engine demo.

Phase 31-36 stack demonstration:
  - ModelOrchestrator (VRAM management)
  - DINOv2Backbone (768-dim visual features)
  - CascadeDetector (YOLO->DINO->VLM escalation)
  - SAMSegmenter (precision segmentation)
  - TemporalFusionEngine (multi-frame reasoning)
  - OnlineLearner (few-shot deployment adaptation)

Usage:
    python scripts/demo_perception_orchestration.py
    python scripts/demo_perception_orchestration.py --image path/to/image.jpg
    python scripts/demo_perception_orchestration.py --frames 30 --verbose
"""
import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Rich console — graceful fallback if not installed
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint

    console = Console()

    def print_header(title: str) -> None:
        console.print(Panel(f"[bold cyan]{title}[/bold cyan]", expand=False))

    def print_ok(msg: str) -> None:
        console.print(f"  [green]OK[/green]  {msg}")

    def print_warn(msg: str) -> None:
        console.print(f"  [yellow]WARN[/yellow] {msg}")

    def print_info(msg: str) -> None:
        console.print(f"  [blue]INFO[/blue] {msg}")

    def print_section(title: str) -> None:
        console.rule(f"[bold]{title}[/bold]")

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

    def print_header(title: str) -> None:  # type: ignore[misc]
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    def print_ok(msg: str) -> None:  # type: ignore[misc]
        print(f"  [OK]   {msg}")

    def print_warn(msg: str) -> None:  # type: ignore[misc]
        print(f"  [WARN] {msg}")

    def print_info(msg: str) -> None:  # type: ignore[misc]
        print(f"  [INFO] {msg}")

    def print_section(title: str) -> None:  # type: ignore[misc]
        print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SOPilot Phase 31-36 Perception Orchestration Engine demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image file to use as the test frame (JPEG/PNG). "
             "If omitted, synthetic frames are generated.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Number of frames to process in the inference loop (default: 10).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-frame statistics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device (default: auto — CUDA if available, else CPU).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Synthetic frame generation
# ---------------------------------------------------------------------------

def _generate_synthetic_frame(width: int = 640, height: int = 480):
    """Return a single synthetic BGR frame as a numpy array."""
    import numpy as np

    rng = np.random.default_rng()
    frame = rng.integers(30, 80, (height, width, 3), dtype=np.uint8)

    # Draw 1-3 random rectangles to simulate people / objects
    n_rects = rng.integers(1, 4)
    for _ in range(n_rects):
        x1 = int(rng.integers(0, width - 60))
        y1 = int(rng.integers(0, height - 100))
        x2 = x1 + int(rng.integers(40, 80))
        y2 = y1 + int(rng.integers(80, 130))
        x2 = min(x2, width - 1)
        y2 = min(y2, height - 1)
        color = (
            int(rng.integers(100, 255)),
            int(rng.integers(100, 255)),
            int(rng.integers(100, 255)),
        )
        frame[y1:y2, x1:x2] = color

    return frame


def _load_image_as_frame(path: str):
    """Load an image from disk as a BGR numpy array."""
    try:
        import cv2
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f"cv2.imread returned None for '{path}'")
        return frame
    except ImportError:
        # Fallback: use PIL + numpy
        from PIL import Image
        import numpy as np

        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        # RGB -> BGR
        return arr[:, :, ::-1].copy()


# ---------------------------------------------------------------------------
# Component initialisation helpers
# ---------------------------------------------------------------------------

def _init_orchestrator(device: str):
    try:
        from sopilot.perception.orchestrator import ModelOrchestrator
        orch = ModelOrchestrator(device=device)
        print_ok(f"ModelOrchestrator ready  (device={device})")
        return orch
    except Exception as exc:
        print_warn(f"ModelOrchestrator unavailable: {exc}")
        return None


def _init_dinov2(orchestrator, device: str):
    try:
        from sopilot.perception.dinov2_backbone import DINOv2Backbone
        backbone = DINOv2Backbone(orchestrator=orchestrator, device=device)
        print_ok("DINOv2Backbone ready     (ViT-B/14, 768-dim CLS)")
        return backbone
    except Exception as exc:
        print_warn(f"DINOv2Backbone unavailable: {exc}")
        return None


def _init_cascade(orchestrator, device: str):
    try:
        from sopilot.perception.cascade_detector import CascadeDetector
        detector = CascadeDetector(orchestrator=orchestrator, device=device)
        print_ok("CascadeDetector ready    (YOLO->DINO->VLM)")
        return detector
    except Exception as exc:
        print_warn(f"CascadeDetector unavailable: {exc}")
        return None


def _init_sam(orchestrator, device: str):
    try:
        from sopilot.perception.sam_segmenter import SAMSegmenter  # Phase 34
        seg = SAMSegmenter(orchestrator=orchestrator, device=device)
        print_ok("SAMSegmenter ready       (Segment-Anything 2)")
        return seg
    except Exception as exc:
        print_warn(f"SAMSegmenter unavailable (Phase 34 not yet deployed): {exc}")
        return None


def _init_temporal(orchestrator, device: str):
    try:
        from sopilot.perception.temporal_fusion import TemporalFusionEngine  # Phase 35
        engine = TemporalFusionEngine(orchestrator=orchestrator, device=device)
        print_ok("TemporalFusionEngine ready (multi-frame reasoning)")
        return engine
    except Exception as exc:
        print_warn(f"TemporalFusionEngine unavailable (Phase 35 not yet deployed): {exc}")
        return None


def _init_online_learner(orchestrator, device: str):
    try:
        from sopilot.perception.online_learner import OnlineLearner  # Phase 36
        learner = OnlineLearner(orchestrator=orchestrator, device=device)
        print_ok("OnlineLearner ready      (few-shot adaptation)")
        return learner
    except Exception as exc:
        print_warn(f"OnlineLearner unavailable (Phase 36 not yet deployed): {exc}")
        return None


# ---------------------------------------------------------------------------
# Per-frame inference
# ---------------------------------------------------------------------------

def _run_frame(
    frame_idx: int,
    frame,
    cascade,
    backbone,
    temporal,
    learner,
    sam,
    verbose: bool,
) -> dict:
    """Run all available components on a single frame. Returns stats dict."""
    stats: dict = {
        "frame": frame_idx,
        "detections": 0,
        "cascade_stage": None,
        "feature_dim": None,
        "action": None,
        "learner_class": None,
        "segmented": False,
        "errors": [],
    }

    # --- Stage 1: Cascade detection ---
    detections = []
    if cascade is not None:
        try:
            result = cascade.detect(frame)
            # Normalise: result may be list[dict] or object with .boxes
            if isinstance(result, (list, tuple)):
                detections = list(result)
            elif hasattr(result, "boxes"):
                detections = result.boxes if result.boxes is not None else []
            elif hasattr(result, "detections"):
                detections = result.detections or []
            else:
                detections = []
            stats["detections"] = len(detections)
            # Try to retrieve escalation stage info
            if hasattr(result, "stage"):
                stats["cascade_stage"] = result.stage
            elif isinstance(result, dict):
                stats["cascade_stage"] = result.get("stage", "unknown")
        except Exception as exc:
            stats["errors"].append(f"cascade: {exc}")

    # --- Stage 2: DINOv2 feature extraction ---
    if backbone is not None:
        try:
            import numpy as np
            # extract_cls accepts a BGR numpy array or PIL image
            feat = backbone.extract_cls(frame)
            if feat is not None:
                # feat may be a torch.Tensor or numpy array
                if hasattr(feat, "shape"):
                    dim = feat.shape[-1] if len(feat.shape) > 0 else 0
                elif hasattr(feat, "__len__"):
                    dim = len(feat)
                else:
                    dim = "?"
                stats["feature_dim"] = dim
        except Exception as exc:
            stats["errors"].append(f"dinov2: {exc}")

    # --- Stage 3: Temporal fusion ---
    if temporal is not None:
        try:
            feat_vec = None
            if backbone is not None and stats["feature_dim"] is not None:
                # Reuse already-extracted feature if available
                try:
                    feat_vec = backbone.extract_cls(frame)
                except Exception:
                    pass
            temporal.observe(frame, feature=feat_vec)
            action = temporal.classify_action()
            stats["action"] = str(action) if action is not None else "unknown"
        except Exception as exc:
            stats["errors"].append(f"temporal: {exc}")

    # --- Stage 4: Online learner classification ---
    if learner is not None:
        try:
            # Only classify if the learner has been seeded with at least one exemplar
            has_exemplars = (
                hasattr(learner, "exemplar_count") and learner.exemplar_count > 0
            ) or (
                hasattr(learner, "classes") and len(getattr(learner, "classes", [])) > 0
            )
            if has_exemplars:
                pred = learner.classify(frame)
                stats["learner_class"] = str(pred)
            else:
                stats["learner_class"] = "(no exemplars yet)"
        except Exception as exc:
            stats["errors"].append(f"learner: {exc}")

    # --- Stage 5: SAM segmentation (only if detections found) ---
    if sam is not None and stats["detections"] > 0:
        try:
            masks = sam.segment_frame(frame, detections=detections)
            stats["segmented"] = masks is not None and len(masks) > 0
        except Exception as exc:
            stats["errors"].append(f"sam: {exc}")

    return stats


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

def _print_frame_stats(stats: dict, verbose: bool) -> None:
    det = stats["detections"]
    stage = stats["cascade_stage"] or "-"
    dim = stats["feature_dim"] or "-"
    action = stats["action"] or "-"
    lc = stats["learner_class"] or "-"
    seg = "yes" if stats["segmented"] else "no"
    errors = stats["errors"]

    if verbose:
        print_info(
            f"frame={stats['frame']:03d} | detections={det} | cascade_stage={stage} | "
            f"feat_dim={dim} | action={action} | learner={lc} | segmented={seg}"
        )
        for err in errors:
            print_warn(f"  -> {err}")
    else:
        # Compact single-line output
        print(
            f"  frame {stats['frame']:03d}: det={det}  stage={stage}  "
            f"feat={dim}  action={action}  seg={seg}"
        )


def _print_summary(all_stats: list[dict], elapsed: float) -> None:
    print_section("Summary")
    total_frames = len(all_stats)
    total_det = sum(s["detections"] for s in all_stats)
    total_seg = sum(1 for s in all_stats if s["segmented"])
    total_errors = sum(len(s["errors"]) for s in all_stats)
    fps = total_frames / elapsed if elapsed > 0 else 0.0

    if HAS_RICH:
        table = Table(title="Inference Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right")
        table.add_row("Frames processed", str(total_frames))
        table.add_row("Total detections", str(total_det))
        table.add_row("Frames with segmentation", str(total_seg))
        table.add_row("Component warnings/errors", str(total_errors))
        table.add_row("Elapsed time (s)", f"{elapsed:.2f}")
        table.add_row("Throughput (fps)", f"{fps:.1f}")
        console.print(table)
    else:
        print(f"  Frames processed          : {total_frames}")
        print(f"  Total detections          : {total_det}")
        print(f"  Frames with segmentation  : {total_seg}")
        print(f"  Component warnings/errors : {total_errors}")
        print(f"  Elapsed time (s)          : {elapsed:.2f}")
        print(f"  Throughput (fps)          : {fps:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    print_header("SOPilot Perception Orchestration Engine — Phase 31-36 Demo")

    # Resolve device
    device = args.device
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"
    print_info(f"Inference device: {device}")

    # Load test frame(s) if --image provided
    seed_frame = None
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"ERROR: image file not found: {img_path}", file=sys.stderr)
            return 1
        try:
            seed_frame = _load_image_as_frame(str(img_path))
            print_ok(f"Loaded image: {img_path.name}  shape={seed_frame.shape}")
        except Exception as exc:
            print_warn(f"Failed to load image ({exc}); falling back to synthetic frames")
            seed_frame = None

    # ---------------------------------------------------------------------------
    print_section("Component Initialisation")
    # ---------------------------------------------------------------------------
    orchestrator = _init_orchestrator(device)
    backbone = _init_dinov2(orchestrator, device)
    cascade = _init_cascade(orchestrator, device)
    sam = _init_sam(orchestrator, device)
    temporal = _init_temporal(orchestrator, device)
    learner = _init_online_learner(orchestrator, device)

    # Count available components
    available = sum(
        c is not None for c in [orchestrator, backbone, cascade, sam, temporal, learner]
    )
    print_info(f"{available}/6 components available")

    # ---------------------------------------------------------------------------
    print_section(f"Inference Loop  ({args.frames} frames)")
    # ---------------------------------------------------------------------------
    all_stats: list[dict] = []
    t_start = time.perf_counter()

    for i in range(args.frames):
        # Build this frame
        if seed_frame is not None:
            frame = seed_frame.copy()
        else:
            try:
                frame = _generate_synthetic_frame()
            except ImportError:
                # numpy not available — create a minimal placeholder
                frame = None  # type: ignore[assignment]

        stats = _run_frame(
            frame_idx=i,
            frame=frame,
            cascade=cascade,
            backbone=backbone,
            temporal=temporal,
            learner=learner,
            sam=sam,
            verbose=args.verbose,
        )
        all_stats.append(stats)
        _print_frame_stats(stats, verbose=args.verbose)

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    # ---------------------------------------------------------------------------
    _print_summary(all_stats, elapsed)
    # ---------------------------------------------------------------------------

    print_section("Done")
    print_ok("Phase 31-36 Perception Orchestration Engine demo complete.")
    print_info(
        "To enable full GPU inference, set SOPILOT_NEURAL_REID=true, "
        "SOPILOT_ORCHESTRATOR_ENABLED=true, SOPILOT_DINOV2_ENABLED=true in your environment "
        "and run with --device cuda."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
