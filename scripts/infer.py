#!/usr/bin/env python
"""
Entry point for test-set inference.

Usage:
    python scripts/infer.py \\
        --checkpoint models/best_model.pth \\
        --test_dir   data/test/ \\
        --output_dir predictions/
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import cfg
from src.inferencer import Inferencer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BraTS 2023 — Inference")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pth checkpoint file",
    )
    p.add_argument(
        "--test_dir",
        required=True,
        help="Directory containing test subjects",
    )
    p.add_argument(
        "--output_dir",
        default="./predictions",
        help="Where to save output NIfTI files",
    )
    p.add_argument(
        "--model_name",
        default=cfg.model_name,
        choices=["segresnet", "unet"],
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg.model_name = args.model_name

    runner = Inferencer(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
    )
    runner.run(test_dir=args.test_dir)


if __name__ == "__main__":
    main()
