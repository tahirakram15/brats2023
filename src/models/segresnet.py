"""
3D SegResNet wrapper for BraTS 2023.

SegResNet is a residual encoder-decoder network with variational auto-encoder
regularisation, originally proposed for BraTS2018 and re-used in BraTS2021.
Reference: https://arxiv.org/abs/1810.11654
"""

import torch.nn as nn
from monai.networks.nets import SegResNet

from src.config import cfg


def build_segresnet() -> nn.Module:
    """
    Build a SegResNet tuned for BraTS 4-channel input → 3-channel output
    (TC / WT / ET).

    Architecture highlights:
      - blocks_down / blocks_up control encoder/decoder depth
      - init_filters=16 → doubles at each scale (16, 32, 64, 128)
      - dropout_prob=0.2 for regularisation
    """
    model = SegResNet(
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        init_filters=16,
        in_channels=len(cfg.modalities),
        out_channels=3,        # TC, WT, ET
        dropout_prob=0.2,
    )
    return model
