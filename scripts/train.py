#!/usr/bin/env python
"""
Entry point for training.

Usage:
    python scripts/train.py
    python scripts/train.py --data_dir /path/to/data --max_epochs 300
"""

import argparse
import logging
import sys
from pathlib import Path

# Make sure `src` is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import cfg
from src.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BraTS 2023 — Training")
    p.add_argument("--data_dir",    default=cfg.data_dir)
    p.add_argument("--output_dir",  default=cfg.output_dir)
    p.add_argument("--model_dir",   default=cfg.model_dir)
    p.add_argument("--model_name",  default=cfg.model_name,
                   choices=["segresnet", "unet"])
    p.add_argument("--max_epochs",  type=int, default=cfg.max_epochs)
    p.add_argument("--batch_size",  type=int, default=cfg.batch_size)
    p.add_argument("--lr",          type=float, default=cfg.learning_rate)
    p.add_argument("--cache_rate",  type=float, default=cfg.cache_rate)
    p.add_argument("--val_every",   type=int, default=cfg.val_every)
    p.add_argument("--seed",        type=int, default=cfg.seed)
    return p.parse_args()


def main():
    args = parse_args()

    # Patch config with CLI overrides
    cfg.data_dir = args.data_dir
    cfg.output_dir = args.output_dir
    cfg.model_dir = args.model_dir
    cfg.model_name = args.model_name
    cfg.max_epochs = args.max_epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.cache_rate = args.cache_rate
    cfg.val_every = args.val_every
    cfg.seed = args.seed

    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()
