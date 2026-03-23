"""
Loss functions for BraTS 2023 segmentation.

BraTSLoss: weighted per-region Dice + cross-entropy.
  Higher weight for ET (smallest / hardest sub-region).
"""

from typing import List

import torch
import torch.nn as nn
from monai.losses import DiceCELoss

from src.config import cfg


class BraTSLoss(nn.Module):
    """
    Weighted composite loss computed independently for each output channel
    (TC, WT, ET) with per-region scalar weights.

    Each region uses sigmoid-activated DiceCELoss so the three channels
    are treated as independent binary segmentation tasks.

    Args:
        weights: list of three floats [w_tc, w_wt, w_et].
                 Defaults to cfg values.
    """

    def __init__(self, weights: List[float] = None):
        super().__init__()
        self.weights = weights or [
            cfg.loss_weight_tc,
            cfg.loss_weight_wt,
            cfg.loss_weight_et,
        ]
        self._dice_ce = DiceCELoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=True,
            reduction="mean",
        )

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            preds:   (B, 3, H, W, D) — raw logits
            targets: (B, 3, H, W, D) — binary ground-truth channels

        Returns:
            Scalar weighted loss.
        """
        total = preds.new_zeros(1).squeeze()
        for i, w in enumerate(self.weights):
            total = total + w * self._dice_ce(
                preds[:, i : i + 1],
                targets[:, i : i + 1],
            )
        return total / sum(self.weights)
