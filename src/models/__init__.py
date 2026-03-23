"""Model factory for BraTS 2023."""

import logging

import torch.nn as nn

from src.config import cfg
from src.models.segresnet import build_segresnet
from src.models.unet import build_unet

logger = logging.getLogger(__name__)

_REGISTRY = {
    "segresnet": build_segresnet,
    "unet": build_unet,
}


def build_model(name: str = None) -> nn.Module:
    """
    Build a segmentation model by name.

    Args:
        name: one of "segresnet" or "unet". Defaults to cfg.model_name.

    Returns:
        Instantiated (un-trained) nn.Module.
    """
    name = name or cfg.model_name
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    model = _REGISTRY[name]()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Built model: %s | Parameters: %.2fM", name, n_params)
    return model


__all__ = ["build_model", "build_segresnet", "build_unet"]
