from pathlib import Path
from typing import List, Dict

from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Lambdad,
    DivisiblePadd,
)

from src.config import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)


def convert_brats_labels(seg):
    """
    Convert BraTS labels:
      0 = background
      1 = NCR/NET
      2 = ED
      3 = ET

    into 3 multilabel channels:
      TC = labels 1 or 3
      WT = labels 1 or 2 or 3
      ET = label 3
    """
    tc = ((seg == 1) | (seg == 3)).float()
    wt = ((seg == 1) | (seg == 2) | (seg == 3)).float()
    et = (seg == 3).float()
    return [tc, wt, et]


def build_file_list(data_dir: str) -> List[Dict[str, str]]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    items = []

    for subject_dir in sorted(data_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        sid = subject_dir.name

        # BraTS 2023 modalities
        modality_paths = []
        for mod in ["t1c", "t1n", "t2f", "t2w"]:
            candidates = [
                subject_dir / f"{sid}-{mod}.nii.gz",
                subject_dir / f"{sid}_{mod}.nii.gz",
                subject_dir / f"{mod}.nii.gz",
            ]

            chosen = None
            for p in candidates:
                if p.exists():
                    chosen = p
                    break

            if chosen is None:
                logger.warning(f"Skipping {sid}: missing modality {mod}")
                modality_paths = []
                break

            modality_paths.append(str(chosen))

        if not modality_paths:
            continue

        seg_candidates = [
            subject_dir / f"{sid}-seg.nii.gz",
            subject_dir / f"{sid}_seg.nii.gz",
            subject_dir / "seg.nii.gz",
        ]

        seg_path = None
        for p in seg_candidates:
            if p.exists():
                seg_path = p
                break

        if seg_path is None:
            logger.warning(f"Skipping {sid}: missing segmentation")
            continue

        items.append(
            {
                "image": modality_paths,
                "label": str(seg_path),
            }
        )

    logger.info(f"Found {len(items)} subjects in {data_dir}")
    return items


def get_train_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

            # Convert segmentation to TC / WT / ET channels
            Lambdad(keys="label", func=convert_brats_labels),

            # Ensure spatial size is compatible before crop/forward
            DivisiblePadd(keys=["image", "label"], k=8),

            # Patch-based training
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=cfg.roi_size,
                random_size=False,
            ),

            # Simple augmentation
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),

            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_val_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

            # Convert segmentation to TC / WT / ET channels
            Lambdad(keys="label", func=convert_brats_labels),

            # Critical fix for SegResNet validation shape mismatch
            DivisiblePadd(keys=["image", "label"], k=8),

            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_train_loader():
    data_list = build_file_list(cfg.data_dir)
    split_idx = int(len(data_list) * 0.8)
    train_files = data_list[:split_idx]

    logger.info(f"Train set size: {len(train_files)}")

    ds = Dataset(data=train_files, transform=get_train_transforms())
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=min(2, getattr(cfg, "num_workers", 2)),
        pin_memory=True,
        drop_last=True,
    )
    return loader


def get_val_loader():
    data_list = build_file_list(cfg.data_dir)
    split_idx = int(len(data_list) * 0.8)
    val_files = data_list[split_idx:]

    logger.info(f"Val set size: {len(val_files)}")

    ds = Dataset(data=val_files, transform=get_val_transforms())
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=min(2, getattr(cfg, "num_workers", 2)),
        pin_memory=True,
    )
    return loader

def build_test_file_list(data_dir: str):
    return build_file_list(data_dir)


def train_val_split(data_list, val_ratio=0.2):
    split_idx = int(len(data_list) * (1 - val_ratio))
    train_files = data_list[:split_idx]
    val_files = data_list[split_idx:]
    return train_files, val_files