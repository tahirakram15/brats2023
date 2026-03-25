"""
Dataset utilities for BraTS 2023.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from monai.data import CacheDataset, DataLoader, Dataset

from src.config import cfg
from src.data.transforms import (
    get_inference_transforms,
    get_train_transforms,
    get_val_transforms,
)

logger = logging.getLogger(__name__)


def _first_existing(paths: List[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _resolve_modality_path(subject_dir: Path, sid: str, modality: str) -> Path | None:
    candidates = [
        subject_dir / f"{sid}-{modality}.nii.gz",
        subject_dir / f"{sid}_{modality}.nii.gz",
        subject_dir / f"{sid}{modality}.nii.gz",
    ]
    return _first_existing(candidates)


def _resolve_label_path(subject_dir: Path, sid: str) -> Path | None:
    candidates = [
        subject_dir / f"{sid}-seg.nii.gz",
        subject_dir / f"{sid}_seg.nii.gz",
        subject_dir / f"{sid}seg.nii.gz",
    ]
    return _first_existing(candidates)


def build_file_list(data_dir: str) -> List[Dict[str, str]]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Training data directory does not exist: {data_dir}")

    data_list: List[Dict[str, str]] = []
    for subject_dir in sorted(data_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        sid = subject_dir.name
        images: List[str] = []
        missing = False
        for modality in cfg.modalities:
            path = _resolve_modality_path(subject_dir, sid, modality)
            if path is None:
                missing = True
                logger.warning("Missing modality '%s' for subject %s", modality, sid)
                break
            images.append(str(path))

        seg = _resolve_label_path(subject_dir, sid)
        if missing or seg is None:
            logger.warning("Skipping incomplete subject: %s", sid)
            continue

        data_list.append({"image": images, "label": str(seg)})

    if not data_list:
        raise RuntimeError(
            f"No valid BraTS subjects were found in {data_dir}. "
            "Check the directory structure and file naming."
        )

    logger.info("Found %d subjects in %s", len(data_list), data_dir)
    return data_list


def build_test_file_list(test_dir: str) -> List[Dict[str, str]]:
    test_dir = Path(test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test data directory does not exist: {test_dir}")

    data_list: List[Dict[str, str]] = []
    for subject_dir in sorted(test_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        sid = subject_dir.name
        images: List[str] = []
        missing = False
        for modality in cfg.modalities:
            path = _resolve_modality_path(subject_dir, sid, modality)
            if path is None:
                missing = True
                logger.warning("Missing modality '%s' for test subject %s", modality, sid)
                break
            images.append(str(path))

        if missing:
            logger.warning("Skipping incomplete test subject: %s", sid)
            continue

        data_list.append({"image": images, "subject_id": sid})

    if not data_list:
        raise RuntimeError(
            f"No valid test subjects were found in {test_dir}. "
            "Check the directory structure and file naming."
        )

    logger.info("Found %d test subjects in %s", len(data_list), test_dir)
    return data_list


def train_val_split(
    data_list: List[Dict[str, str]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if len(data_list) < 2:
        raise RuntimeError(
            "At least 2 valid subjects are required for a train/validation split."
        )

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data_list))
    n_val = int(round(len(data_list) * val_ratio))
    n_val = min(max(1, n_val), len(data_list) - 1)

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return [data_list[i] for i in train_idx], [data_list[i] for i in val_idx]


def _dataloader_kwargs(shuffle: bool) -> Dict:
    kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": True,
    }
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = True
    if shuffle:
        kwargs["drop_last"] = False
    return kwargs


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
        **_dataloader_kwargs(shuffle=True),
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
        **_dataloader_kwargs(shuffle=False),
    )


def get_test_loader(test_dir: str) -> DataLoader:
    data_list = build_test_file_list(test_dir)
    ds = Dataset(data=data_list, transform=get_inference_transforms())
    return DataLoader(
        ds,
        batch_size=cfg.infer_batch_size,
        shuffle=False,
        **_dataloader_kwargs(shuffle=False),
    )
