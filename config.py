"""
Central configuration dataclass for BraTS 2023.
All scripts import from here — edit once, applies everywhere.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: str = "./data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    output_dir: str = "./outputs"
    model_dir: str = "./models"

    # ── Modalities (channel order must match NIfTI file suffixes) ──────────
    modalities: List[str] = field(
        default_factory=lambda: ["t1n", "t1c", "t2w", "t2f"]
    )

    # ── Preprocessing ──────────────────────────────────────────────────────
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    roi_size: Tuple[int, int, int] = (128, 128, 128)

    # ── Training ───────────────────────────────────────────────────────────
    seed: int = 42
    num_workers: int = 4
    batch_size: int = 1           # 3-D patches — keep at 1 unless large GPU
    cache_rate: float = 0.0       # increase if RAM allows (e.g. 0.5)
    max_epochs: int = 300
    val_every: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    val_ratio: float = 0.2

    # ── Model ──────────────────────────────────────────────────────────────
    model_name: str = "segresnet"   # "segresnet" | "unet"
    num_classes: int = 4            # 0=BG, 1=NCR, 2=ED, 3=ET (raw label map)

    # ── Loss weights (TC / WT / ET) ────────────────────────────────────────
    loss_weight_tc: float = 1.0
    loss_weight_wt: float = 1.0
    loss_weight_et: float = 2.0     # ET is smallest — upweight it

    # ── Inference ──────────────────────────────────────────────────────────
    sw_batch_size: int = 4
    overlap: float = 0.5

    # ── Post-processing ────────────────────────────────────────────────────
    min_cc_size: int = 100          # voxels; smaller components removed


# Singleton used across the codebase
cfg = Config()
