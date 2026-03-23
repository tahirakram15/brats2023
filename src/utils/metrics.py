"""
Evaluation metrics for BraTS 2023.

Two modes:
  1. MONAI-based (tensor, used during training validation loop)
  2. NIfTI-file-based (used for offline evaluation / submission scoring)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Activations, AsDiscrete, Compose

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MONAI metric helpers (used in training loop)
# ─────────────────────────────────────────────────────────────────────────────

def build_metrics() -> Tuple[DiceMetric, HausdorffDistanceMetric, Compose]:
    """
    Returns:
        dice_metric   — MONAI DiceMetric (reset between epochs)
        hd95_metric   — MONAI HausdorffDistanceMetric at 95th percentile
        post_trans    — sigmoid → threshold(0.5) post-processing transform
    """
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(
        include_background=True,
        distance_metric="euclidean",
        percentile=95,
        reduction="mean",
    )
    post_trans = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5),
    ])
    return dice_metric, hd95_metric, post_trans


# ─────────────────────────────────────────────────────────────────────────────
# File-based metrics (offline evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0


def _hd95(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial.distance import directed_hausdorff

    a_pts = np.argwhere(a)
    b_pts = np.argwhere(b)
    if len(a_pts) == 0 or len(b_pts) == 0:
        return 0.0
    d1 = directed_hausdorff(a_pts, b_pts)[0]
    d2 = directed_hausdorff(b_pts, a_pts)[0]
    return float(max(d1, d2))


def compute_metrics_from_arrays(
    pred: np.ndarray, gt: np.ndarray
) -> Dict[str, float]:
    """
    Compute Dice and HD95 from integer label maps.

    Args:
        pred: (H, W, D) uint8 array with values in {0, 1, 2, 3}
        gt:   (H, W, D) uint8 array with values in {0, 1, 2, 3}

    Returns:
        Dict with keys dice_tc, dice_wt, dice_et, hd95_tc, hd95_wt,
        hd95_et, mean_dice.
    """
    regions = {
        "tc": (np.isin(pred, [1, 3]), np.isin(gt, [1, 3])),
        "wt": (np.isin(pred, [1, 2, 3]), np.isin(gt, [1, 2, 3])),
        "et": (pred == 3, gt == 3),
    }
    results: Dict[str, float] = {}
    for name, (p, g) in regions.items():
        results[f"dice_{name}"] = _dice(p, g)
        results[f"hd95_{name}"] = _hd95(p, g)

    results["mean_dice"] = float(
        np.mean([results["dice_tc"], results["dice_wt"], results["dice_et"]])
    )
    return results


def compute_metrics_from_files(
    pred_path: str, gt_path: str
) -> Dict[str, float]:
    """Load NIfTI files and compute metrics."""
    import nibabel as nib

    pred = nib.load(pred_path).get_fdata().astype(np.uint8)
    gt = nib.load(gt_path).get_fdata().astype(np.uint8)
    return compute_metrics_from_arrays(pred, gt)


def aggregate_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
    """Average per-subject metrics across a cohort."""
    keys = [k for k in results[0] if k != "subject"]
    return {k: float(np.mean([r[k] for r in results])) for k in keys}
