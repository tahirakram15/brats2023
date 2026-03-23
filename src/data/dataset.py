"""
Dataset utilities for BraTS 2023.

Responsibilities:
  - Crawl the BraTS directory tree and build file-list dicts
  - Train / validation split
  - Build MONAI Dataset / CacheDataset instances
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from monai.data import CacheDataset, DataLoader, Dataset

from src.config import cfg
from src.data.transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# File-list builder
# ─────────────────────────────────────────────────────────────────────────────

def build_file_list(data_dir: str) -> List[Dict[str, str]]:
    """
    Crawl a BraTS2023 training directory and return a list of dicts:
        {"image": [t1n_path, t1c_path, t2w_path, t2f_path], "label": seg_path}

    Handles two common filename conventions:
        <SID>-<mod>.nii.gz   (dash separator)
        <SID><mod>.nii.gz    (no separator)
    """
    data_dir = Path(data_dir)
    data_list: List[Dict] = []

    for subject_dir in sorted(data_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        sid = subject_dir.name

        images = []
        for mod in cfg.modalities:
            p = subject_dir / f"{sid}-{mod}.nii.gz"
            if not p.exists():
                p = subject_dir / f"{sid}{mod}.nii.gz"
            images.append(str(p))

        seg = subject_dir / f"{sid}-seg.nii.gz"
        if not seg.exists():
             seg = subject_dir / f"{sid}_seg.nii.gz"

        if all(Path(i).exists() for i in images) and seg.exists():
            data_list.append({"image": images, "label": str(seg)})
        else:
            logger.warning("Skipping incomplete subject: %s", sid)

    logger.info("Found %d subjects in %s", len(data_list), data_dir)
    return data_list


def build_test_file_list(test_dir: str) -> List[Dict[str, str]]:
    """Build a file list for the test set (no label)."""
    test_dir = Path(test_dir)
    data_list: List[Dict] = []

    for subject_dir in sorted(test_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        sid = subject_dir.name
        images = []
        for mod in cfg.modalities:
            
            p = subject_dir / f"{sid}-{mod}.nii.gz"
            if not p.exists():
                p = subject_dir / f"{sid}_{mod}.nii.gz"
            images.append(str(p))

        if all(Path(i).exists() for i in images):
            data_list.append({"image": images, "subject_id": sid})
        else:
            logger.warning("Skipping incomplete test subject: %s", sid)

    return data_list


# ─────────────────────────────────────────────────────────────────────────────
# Train / val split
# ─────────────────────────────────────────────────────────────────────────────

def train_val_split(
    data_list: List[Dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data_list))
    n_val = max(1, int(len(data_list) * val_ratio))
    return (
        [data_list[i] for i in idx[n_val:]],
        [data_list[i] for i in idx[:n_val]],
    )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factories
# ─────────────────────────────────────────────────────────────────────────────

def get_train_loader() -> DataLoader:
    data_list = build_file_list(cfg.data_dir)
    train_list, _ = train_val_split(data_list, cfg.val_ratio, cfg.seed)
    logger.info("Train set size: %d", len(train_list))

    ds = CacheDataset(
        data=train_list,
        transform=get_train_transforms(),
        cache_rate=cfg.cache_rate,
        num_workers=cfg.num_workers,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )


def get_val_loader() -> DataLoader:
    data_list = build_file_list(cfg.data_dir)
    _, val_list = train_val_split(data_list, cfg.val_ratio, cfg.seed)
    logger.info("Val set size: %d", len(val_list))

    ds = Dataset(data=val_list, transform=get_val_transforms())
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
