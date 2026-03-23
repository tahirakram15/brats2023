"""
Post-processing for BraTS 2023 predictions.

Steps applied after sigmoid + threshold:
  1. Remove connected components smaller than `min_size` voxels.
  2. Enforce BraTS anatomical hierarchy: ET ⊆ TC ⊆ WT.
  3. Convert 3-channel binary (TC, WT, ET) to BraTS integer label map.
"""

import numpy as np
from scipy import ndimage

from src.config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Connected-component filtering
# ─────────────────────────────────────────────────────────────────────────────

def remove_small_components(
    mask: np.ndarray, min_size: int = None
) -> np.ndarray:
    """
    Remove connected components with fewer than `min_size` voxels.

    Args:
        mask:     Binary 3-D array.
        min_size: Minimum number of voxels to keep. Defaults to cfg.min_cc_size.

    Returns:
        Cleaned binary mask (same shape as input).
    """
    min_size = min_size if min_size is not None else cfg.min_cc_size
    mask = mask.copy()
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    for i, size in enumerate(sizes, start=1):
        if size < min_size:
            mask[labeled == i] = 0
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchy enforcement + full post-processing
# ─────────────────────────────────────────────────────────────────────────────

def postprocess_channels(
    pred: np.ndarray, min_size: int = None
) -> np.ndarray:
    """
    Clean a 3-channel binary prediction (TC, WT, ET).

    Args:
        pred:     (3, H, W, D) binary float/bool array.
        min_size: Passed to remove_small_components.

    Returns:
        (3, H, W, D) cleaned binary array.
    """
    result = np.zeros_like(pred, dtype=np.uint8)

    # Per-channel component filtering
    for c in range(pred.shape[0]):
        result[c] = remove_small_components(
            pred[c].astype(np.uint8), min_size
        )

    # Enforce anatomical hierarchy: ET ⊆ TC ⊆ WT
    result[0] = np.logical_and(result[0], result[1]).astype(np.uint8)   # TC ⊆ WT
    result[2] = np.logical_and(result[2], result[0]).astype(np.uint8)   # ET ⊆ TC

    return result


def channels_to_label_map(pred: np.ndarray) -> np.ndarray:
    """
    Convert 3-channel (TC, WT, ET) binary prediction to the BraTS integer
    label map used for submission.

    Label convention:
        0 = background
        1 = NCR  (necrotic core, part of TC but not ET)
        2 = ED   (edema, part of WT but not TC)
        3 = ET   (enhancing tumor)

    Priority applied bottom-up: ET overwrites NCR which overwrites ED.

    Args:
        pred: (3, H, W, D) binary array with channels [TC, WT, ET].

    Returns:
        (H, W, D) uint8 label map.
    """
    label = np.zeros(pred.shape[1:], dtype=np.uint8)
    label[pred[1] == 1] = 2   # ED  = WT region
    label[pred[0] == 1] = 1   # NCR = TC region (overwrites ED where TC ⊆ WT)
    label[pred[2] == 1] = 3   # ET  = ET region (overwrites NCR where ET ⊆ TC)
    return label
