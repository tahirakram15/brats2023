"""
3D UNet wrapper for BraTS 2023.

Standard MONAI UNet with residual units, suitable as an alternative to
SegResNet for lower-VRAM scenarios.
"""

import torch.nn as nn
from monai.networks.nets import UNet

from src.config import cfg


def build_unet() -> nn.Module:
    """
    Build a 3D UNet for BraTS segmentation.

    Channel progression: 32 → 64 → 128 → 256 → 512 with stride-2 downsampling.
    num_res_units=2 adds two residual units per level.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=len(cfg.modalities),
        out_channels=3,        # TC, WT, ET
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2,
    )
    return model
