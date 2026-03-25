"""
Evaluation metrics for BraTS 2023.
"""

import logging
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Activations, AsDiscrete, Compose

logger = logging.getLogger(__name__)


def build_metrics() -> Tuple[DiceMetric, DiceMetric, HausdorffDistanceMetric, Compose]:
    """
    Returns:
        dice_mean_metric: MONAI DiceMetric with scalar mean reduction
        dice_batch_metric: MONAI DiceMetric with per-channel mean_batch reduction
        hd95_batch_metric: MONAI HD95 with per-channel mean_batch reduction
        post_trans: sigmoid -> threshold(0.5)
    """
    dice_mean_metric = DiceMetric(include_background=False, reduction="mean")
    dice_batch_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd95_batch_metric = HausdorffDistanceMetric(
        include_background=False,
        distance_metric="euclidean",
        percentile=95,
        reduction="mean_batch",
    )
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    return dice_mean_metric, dice_batch_metric, hd95_batch_metric, post_trans


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0


def _hd95(a: np.ndarray, b: np.ndarray) -> float:
    """
    Approximate HD95 from foreground point clouds.
    """
    from scipy.spatial.distance import cdist

    a_pts = np.argwhere(a)
    b_pts = np.argwhere(b)
    if len(a_pts) == 0 and len(b_pts) == 0:
        return 0.0
    if len(a_pts) == 0 or len(b_pts) == 0:
        return float("inf")

    dist = cdist(a_pts, b_pts)
    d_ab = dist.min(axis=1)
    d_ba = dist.min(axis=0)
    return float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))


def compute_metrics_from_arrays(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
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
    results["hd95_mean"] = float(
        np.mean([results["hd95_tc"], results["hd95_wt"], results["hd95_et"]])
    )
    return results


def compute_metrics_from_files(pred_path: str, gt_path: str) -> Dict[str, float]:
    pred = nib.load(pred_path).get_fdata().astype(np.uint8)
    gt = nib.load(gt_path).get_fdata().astype(np.uint8)
    return compute_metrics_from_arrays(pred, gt)


def aggregate_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
    keys = [k for k in results[0] if k != "subject"]
    return {k: float(np.mean([r[k] for r in results])) for k in keys}
