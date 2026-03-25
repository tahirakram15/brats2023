"""
Central configuration dataclass for BraTS 2023.
All scripts import from here, so a single edit updates the whole pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # Paths
    data_dir: str = "./data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    output_dir: str = "./outputs"
    model_dir: str = "./models"

    # Modalities (must match NIfTI suffixes)
    modalities: List[str] = field(default_factory=lambda: ["t1n", "t1c", "t2w", "t2f"])

    # Preprocessing / patching
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    roi_size: Tuple[int, int, int] = (128, 128, 128)

    # Training
    seed: int = 42
    num_workers: int = 2
    batch_size: int = 2
    cache_rate: float = 0.0
    max_epochs: int = 300
    val_every: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    val_ratio: float = 0.2
    grad_clip_norm: float = 1.0
    amp: bool = True

    # Model
    model_name: str = "segresnet"  # "segresnet" | "unet"
    num_output_channels: int = 3   # TC, WT, ET

    # Loss weights (TC / WT / ET)
    loss_weight_tc: float = 1.0
    loss_weight_wt: float = 1.0
    loss_weight_et: float = 2.0

    # Inference / validation
    sw_batch_size: int = 1
    overlap: float = 0.5
    infer_batch_size: int = 1

    # Post-processing
    min_cc_size: int = 100

    @property
    def use_amp(self) -> bool:
        return self.amp


cfg = Config()
