from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
)

from src.config import cfg


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

MODALITIES: Sequence[str] = ("t1c", "t1n", "t2f", "t2w")


def _get_cfg_value(name: str, default):
    return getattr(cfg, name, default)


def _get_roi_size() -> Tuple[int, int, int]:
    roi = _get_cfg_value("roi_size", (128, 128, 128))
    if isinstance(roi, (list, tuple)) and len(roi) == 3:
        return tuple(int(x) for x in roi)
    return (128, 128, 128)


def _get_num_workers() -> int:
    return int(_get_cfg_value("num_workers", 2))


def _get_batch_size() -> int:
    return int(_get_cfg_value("batch_size", 1))


def _get_val_ratio() -> float:
    return float(_get_cfg_value("val_ratio", 0.2))


def _get_divisible_k() -> int:
    # 16 is safer than 8 for common 3D UNet / SegResNet depth settings.
    return int(_get_cfg_value("divisible_k", 16))


def _find_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def _modality_candidates(subject_dir: Path, sid: str, mod: str) -> List[Path]:
    return [
        subject_dir / f"{sid}-{mod}.nii.gz",
        subject_dir / f"{sid}_{mod}.nii.gz",
        subject_dir / f"{mod}.nii.gz",
    ]


def _seg_candidates(subject_dir: Path, sid: str) -> List[Path]:
    return [
        subject_dir / f"{sid}-seg.nii.gz",
        subject_dir / f"{sid}_seg.nii.gz",
        subject_dir / "seg.nii.gz",
    ]


# ---------------------------------------------------------------------
# Label conversion
# ---------------------------------------------------------------------

def convert_brats_labels(seg: torch.Tensor) -> torch.Tensor:
    """
    Convert BraTS segmentation labels:
      0 = background
      1 = NCR/NET
      2 = ED
      3 = ET

    into 3 multilabel channels:
      TC = [1, 3]
      WT = [1, 2, 3]
      ET = [3]

    Input shape after loading:
      [1, D, H, W]

    Output shape:
      [3, D, H, W]
    """
    if seg.ndim == 4 and seg.shape[0] == 1:
        seg = seg[0]

    tc = ((seg == 1) | (seg == 3)).float()
    wt = ((seg == 1) | (seg == 2) | (seg == 3)).float()
    et = (seg == 3).float()

    out = torch.stack([tc, wt, et], dim=0)
    return out


# ---------------------------------------------------------------------
# File list builders
# ---------------------------------------------------------------------

def build_file_list(data_dir: str, require_label: bool = True) -> List[Dict[str, object]]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    items: List[Dict[str, object]] = []

    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue

        sid = subject_dir.name

        image_paths: List[str] = []
        missing_modality = False

        for mod in MODALITIES:
            chosen = _find_existing(_modality_candidates(subject_dir, sid, mod))
            if chosen is None:
                logger.warning("Skipping %s: missing modality %s", sid, mod)
                missing_modality = True
                break
            image_paths.append(str(chosen))

        if missing_modality:
            continue

        seg_path = _find_existing(_seg_candidates(subject_dir, sid))

        if require_label and seg_path is None:
            logger.warning("Skipping %s: missing segmentation", sid)
            continue

        item: Dict[str, object] = {
            "image": image_paths,
            "case_id": sid,
        }

        if seg_path is not None:
            item["label"] = str(seg_path)

        items.append(item)

    logger.info("Found %d subjects in %s", len(items), root)
    return items


def build_test_file_list(data_dir: str) -> List[Dict[str, object]]:
    """
    Compatibility wrapper expected by the repo.
    Allows unlabeled test/inference folders.
    """
    return build_file_list(data_dir, require_label=False)


def train_val_split(
    data_list: List[Dict[str, object]],
    val_ratio: Optional[float] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if val_ratio is None:
        val_ratio = _get_val_ratio()

    if len(data_list) == 0:
        return [], []

    split_idx = int(len(data_list) * (1.0 - float(val_ratio)))
    split_idx = max(1, min(split_idx, len(data_list) - 1)) if len(data_list) > 1 else len(data_list)

    train_files = data_list[:split_idx]
    val_files = data_list[split_idx:]

    return train_files, val_files


# ---------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------

def get_train_transforms():
    roi_size = _get_roi_size()
    divisible_k = _get_divisible_k()

    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Lambdad(keys="label", func=convert_brats_labels),

            # Make spatial sizes safe for encoder/decoder skip connections.
            DivisiblePadd(keys=["image", "label"], k=divisible_k),

            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=roi_size,
                random_size=False,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_val_transforms():
    divisible_k = _get_divisible_k()

    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Lambdad(keys="label", func=convert_brats_labels),

            # Critical fix for validation/inference shape mismatch.
            DivisiblePadd(keys=["image", "label"], k=divisible_k),

            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_test_transforms():
    divisible_k = _get_divisible_k()

    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            DivisiblePadd(keys=["image"], k=divisible_k),
            EnsureTyped(keys=["image"]),
        ]
    )


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------

def get_train_loader():
    data_list = build_file_list(cfg.data_dir, require_label=True)
    train_files, _ = train_val_split(data_list)

    logger.info("Train set size: %d", len(train_files))

    ds = Dataset(data=train_files, transform=get_train_transforms())
    loader = DataLoader(
        ds,
        batch_size=_get_batch_size(),
        shuffle=True,
        num_workers=_get_num_workers(),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return loader


def get_val_loader():
    data_list = build_file_list(cfg.data_dir, require_label=True)
    _, val_files = train_val_split(data_list)

    logger.info("Val set size: %d", len(val_files))

    ds = Dataset(data=val_files, transform=get_val_transforms())
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=_get_num_workers(),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader


def get_test_loader(data_dir: Optional[str] = None):
    if data_dir is None:
        data_dir = cfg.data_dir

    test_files = build_test_file_list(data_dir)

    logger.info("Test set size: %d", len(test_files))

    ds = Dataset(data=test_files, transform=get_test_transforms())
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=_get_num_workers(),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader