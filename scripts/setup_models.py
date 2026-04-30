"""Phase 58: First-time model download utility for RTX 4090 inference stack.

Downloads and verifies all model weights required for the real inference stack:
    - YOLO v8 World (ultralytics)
    - DINOv2 ViT-B/14 (torch.hub)
    - Grounding DINO 1.5-edge (HuggingFace)
    - LLaVA-NeXT 7B (HuggingFace)
    - SAM 2 small (HuggingFace)

Usage::

    # Check CUDA and show what would be downloaded
    python scripts/setup_models.py --dry-run

    # Download all models to HuggingFace cache
    python scripts/setup_models.py --device cuda

    # Download only specific models
    python scripts/setup_models.py --models yolo dinov2

VRAM requirements (RTX 4090, concurrent):
    YOLO v8 World           ~1.5 GB
    DINOv2 ViT-B/14         ~0.35 GB
    Grounding DINO 1.5-edge ~1.5 GB
    LLaVA-NeXT 7B (4-bit)   ~4.5 GB
    SAM 2 small             ~2.5 GB
    ──────────────────────────────
    Total (all loaded)      ~10.35 GB  ✓ fits in 22 GB usable
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("setup_models")

# ── Model registry ─────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "yolo": {
        "name": "YOLO v8 World",
        "vram_gb": 1.5,
        "size_gb": 0.15,
        "description": "ultralytics yolov8s-worldv2.pt (open-vocabulary detection)",
    },
    "dinov2": {
        "name": "DINOv2 ViT-B/14",
        "vram_gb": 0.35,
        "size_gb": 0.35,
        "hf_repo": None,  # torch.hub
        "description": "facebookresearch/dinov2 — universal visual features (768-dim CLS)",
    },
    "grounding_dino": {
        "name": "Grounding DINO 1.5-edge",
        "vram_gb": 1.5,
        "size_gb": 0.9,
        "hf_repo": "IDEA-Research/grounding-dino-1.5-edge",
        "description": "Open-set detector, cascade Stage 2",
    },
    "llava": {
        "name": "LLaVA-NeXT 7B (4-bit quantized)",
        "vram_gb": 4.5,
        "size_gb": 14.0,
        "hf_repo": "llava-hf/llava-v1.6-mistral-7b-hf",
        "description": "Vision-language model, cascade Stage 3, scene description",
    },
    "sam2": {
        "name": "SAM 2 small",
        "vram_gb": 2.5,
        "size_gb": 0.18,
        "hf_repo": "facebook/sam2-hiera-small",
        "description": "Segment Anything Model 2, precision segmentation",
    },
}

TOTAL_VRAM_GB = sum(m["vram_gb"] for m in MODELS.values())
TOTAL_SIZE_GB = sum(m["size_gb"] for m in MODELS.values())


# ── CUDA check ─────────────────────────────────────────────────────────────


def check_cuda() -> tuple[bool, str]:
    """Check CUDA availability. Returns (available, device_name)."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return True, f"{name} ({vram:.1f} GB VRAM)"
        return False, "CUDA not available"
    except ImportError:
        return False, "torch not installed"


# ── Individual downloaders ──────────────────────────────────────────────────


def download_yolo(dry_run: bool = False) -> bool:
    """Download YOLO v8 World weights via ultralytics."""
    logger.info("[YOLO] Checking yolov8s-worldv2.pt ...")
    if dry_run:
        logger.info("[YOLO] dry-run: would download yolov8s-worldv2.pt (~150 MB)")
        return True
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8s-worldv2.pt")  # auto-downloads if not cached
        logger.info("[YOLO] ✓ Ready: %s", model.model_name if hasattr(model, "model_name") else "yolov8s-worldv2")
        return True
    except ImportError:
        logger.warning("[YOLO] ultralytics not installed — pip install ultralytics")
        return False
    except Exception as e:
        logger.error("[YOLO] Download failed: %s", e)
        return False


def download_dinov2(dry_run: bool = False) -> bool:
    """Trigger torch.hub cache for DINOv2 ViT-B/14."""
    logger.info("[DINOv2] Checking facebookresearch/dinov2 ...")
    if dry_run:
        logger.info("[DINOv2] dry-run: would cache DINOv2 ViT-B/14 (~350 MB)")
        return True
    try:
        import torch

        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
        model.eval()
        logger.info("[DINOv2] ✓ Ready: dinov2_vitb14 (768-dim CLS)")
        return True
    except ImportError:
        logger.warning("[DINOv2] torch not installed — pip install -e '.[gpu]'")
        return False
    except Exception as e:
        logger.error("[DINOv2] Download failed: %s", e)
        return False


def download_grounding_dino(dry_run: bool = False) -> bool:
    """Download Grounding DINO 1.5-edge from HuggingFace."""
    repo_id = "IDEA-Research/grounding-dino-1.5-edge"
    logger.info("[GroundingDINO] Checking %s ...", repo_id)
    if dry_run:
        logger.info("[GroundingDINO] dry-run: would download %s (~900 MB)", repo_id)
        return True
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=repo_id, ignore_patterns=["*.msgpack", "flax*"])
        logger.info("[GroundingDINO] ✓ Cached at: %s", path)
        return True
    except ImportError:
        logger.warning("[GroundingDINO] huggingface_hub not installed — pip install huggingface-hub")
        return False
    except Exception as e:
        logger.error("[GroundingDINO] Download failed: %s", e)
        return False


def download_llava(dry_run: bool = False) -> bool:
    """Download LLaVA-NeXT 7B from HuggingFace."""
    repo_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    logger.info("[LLaVA] Checking %s (~14 GB) ...", repo_id)
    if dry_run:
        logger.info("[LLaVA] dry-run: would download %s (~14 GB) — longest download", repo_id)
        return True
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=repo_id, ignore_patterns=["*.msgpack", "flax*", "*.ot"])
        logger.info("[LLaVA] ✓ Cached at: %s", path)
        return True
    except ImportError:
        logger.warning("[LLaVA] huggingface_hub not installed — pip install huggingface-hub")
        return False
    except Exception as e:
        logger.error("[LLaVA] Download failed: %s", e)
        return False


def download_sam2(dry_run: bool = False) -> bool:
    """Download SAM 2 small weights from HuggingFace."""
    repo_id = "facebook/sam2-hiera-small"
    logger.info("[SAM2] Checking %s ...", repo_id)
    if dry_run:
        logger.info("[SAM2] dry-run: would download %s (~180 MB)", repo_id)
        return True
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=repo_id)
        logger.info("[SAM2] ✓ Cached at: %s", path)
        return True
    except ImportError:
        logger.warning("[SAM2] huggingface_hub not installed — pip install huggingface-hub")
        return False
    except Exception as e:
        logger.error("[SAM2] Download failed: %s", e)
        return False


_DOWNLOADERS = {
    "yolo": download_yolo,
    "dinov2": download_dinov2,
    "grounding_dino": download_grounding_dino,
    "llava": download_llava,
    "sam2": download_sam2,
}


# ── Main ───────────────────────────────────────────────────────────────────


def print_summary(selected: list[str]) -> None:
    """Print VRAM and download size summary."""
    vram = sum(MODELS[k]["vram_gb"] for k in selected)
    size = sum(MODELS[k]["size_gb"] for k in selected)
    print("\n┌─ Model Download Plan ─────────────────────────────────────────┐")
    for key in selected:
        m = MODELS[key]
        print(f"│  {m['name']:<35} {m['vram_gb']:>4.1f} GB VRAM  {m['size_gb']:>5.1f} GB download")
    print(f"├───────────────────────────────────────────────────────────────┤")
    print(f"│  Total                                    {vram:>4.1f} GB VRAM  {size:>5.1f} GB download")
    print(f"└───────────────────────────────────────────────────────────────┘")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 58: Download models for SOPilot RTX 4090 inference stack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target device (default: auto)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(_DOWNLOADERS.keys()),
        default=None,
        help="Models to download (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    args = parser.parse_args()

    # CUDA check
    cuda_available, cuda_info = check_cuda()
    print(f"\nSOPilot Phase 58 — Model Setup")
    print(f"GPU:  {cuda_info}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'DOWNLOAD'}")

    if not cuda_available and args.device == "cuda":
        logger.error("CUDA requested but not available. Use --device cpu for CPU-only mode.")
        return 1

    # Select models
    selected = args.models if args.models else list(_DOWNLOADERS.keys())
    print_summary(selected)

    if args.dry_run:
        print("\n[dry-run] No files downloaded. Run without --dry-run to proceed.\n")
        return 0

    # Download
    results: dict[str, bool] = {}
    for key in selected:
        results[key] = _DOWNLOADERS[key](dry_run=False)

    # Report
    print("\n── Results ──────────────────────────────────────────────────────")
    all_ok = True
    for key in selected:
        status = "✓ OK" if results[key] else "✗ FAILED"
        print(f"  {MODELS[key]['name']:<40} {status}")
        if not results[key]:
            all_ok = False

    if all_ok:
        print("\n✓ All models ready. Run: python scripts/demo_phase58.py\n")
        return 0
    else:
        print("\n✗ Some models failed. Check logs above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
