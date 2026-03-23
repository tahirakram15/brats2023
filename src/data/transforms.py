"""
MONAI transform pipelines for BraTS 2023.

Three pipelines:
  get_train_transforms()   — training (augmentation + label conversion)
  get_val_transforms()     — validation (no augmentation)
  get_inference_transforms() — test set (no label)
"""

from typing import Dict

import numpy as np
import torch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
)

from src.config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Label conversion
# ─────────────────────────────────────────────────────────────────────────────

class ConvertBratsLabels(MapTransform):
    """
    Convert BraTS scalar label (0/1/2/3) to 3-channel binary tensor:
        ch0 → TC  (tumor core)   = labels {1, 3}
        ch1 → WT  (whole tumor)  = labels {1, 2, 3}
        ch2 → ET  (enhancing)    = label  {3}
    """

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.keys:
            lbl = d[key]
            result = [
                torch.logical_or(lbl == 1, lbl == 3),
                torch.logical_or(
                    torch.logical_or(lbl == 1, lbl == 2), lbl == 3
                ),
                lbl == 3,
            ]
            d[key] = torch.stack(result, dim=0).float()
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms() -> Compose:
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertBratsLabels(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=cfg.voxel_spacing,
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            k_divisible=list(cfg.roi_size),
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=cfg.roi_size,
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(np.pi / 18,) * 3,
            scale_range=(0.1,) * 3,
            mode=("bilinear", "nearest"),
        ),
        RandGaussianNoised(keys="image", prob=0.2, mean=0, std=0.1),
        RandGaussianSmoothd(
            keys="image",
            prob=0.2,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
        ),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms() -> Compose:
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertBratsLabels(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=cfg.voxel_spacing,
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_inference_transforms() -> Compose:
    """Test-set pipeline — no label key."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=cfg.voxel_spacing, mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ])
