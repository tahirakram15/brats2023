"""
MONAI transform pipelines for BraTS 2023.
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


_ORIENTATION_LABELS = (("L", "R"), ("P", "A"), ("I", "S"))


class ConvertBratsLabels(MapTransform):
    """
    Convert BraTS scalar label map {0,1,2,3} to three binary channels:
    ch0 = TC = {1, 3}
    ch1 = WT = {1, 2, 3}
    ch2 = ET = {3}
    """

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.keys:
            lbl = d[key]
            if not torch.is_tensor(lbl):
                lbl = torch.as_tensor(lbl)

            # If a singleton channel exists, remove it before conversion.
            if lbl.ndim == 4 and lbl.shape[0] == 1:
                lbl = lbl[0]

            tc = torch.logical_or(lbl == 1, lbl == 3)
            wt = torch.logical_or(torch.logical_or(lbl == 1, lbl == 2), lbl == 3)
            et = lbl == 3
            d[key] = torch.stack([tc, wt, et], dim=0).float()
        return d


def get_train_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            ConvertBratsLabels(keys=["label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=_ORIENTATION_LABELS),
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
                padding_mode="border",
            ),
            RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.1),
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
        ]
    )


def get_val_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            ConvertBratsLabels(keys=["label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=_ORIENTATION_LABELS),
            Spacingd(
                keys=["image", "label"],
                pixdim=cfg.voxel_spacing,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_inference_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS", labels=_ORIENTATION_LABELS),
            Spacingd(keys=["image"], pixdim=cfg.voxel_spacing, mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )
