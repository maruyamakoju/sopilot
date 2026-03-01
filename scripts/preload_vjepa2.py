from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Preload V-JEPA2 model cache for offline/on-prem use.")
    parser.add_argument("--variant", default="vjepa2_vit_large", help="torch.hub callable")
    parser.add_argument("--pretrained", default="true", help="true/false")
    parser.add_argument("--crop-size", type=int, default=256, help="Preprocessor crop size")
    args = parser.parse_args()

    pretrained = args.pretrained.strip().lower() in {"1", "true", "yes", "y", "on"}
    model = torch.hub.load("facebookresearch/vjepa2", args.variant, pretrained=pretrained)
    _ = model
    processor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor", crop_size=args.crop_size)
    _ = processor
    print(f"cached variant={args.variant} pretrained={pretrained} crop_size={args.crop_size}")


if __name__ == "__main__":
    main()

