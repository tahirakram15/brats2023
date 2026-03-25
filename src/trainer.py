"""
Trainer class for BraTS 2023.

Handles the full training + validation loop:
  - Mixed-precision (AMP) forward + backward
  - Gradient clipping
  - Cosine LR schedule
  - Sliding-window validation
  - Best-model checkpointing
  - JSON history logging
"""

import json
import logging
import os
from typing import Dict
import numpy as np

import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import cfg
from src.data.dataset import get_train_loader, get_val_loader
from src.losses.losses import BraTSLoss
from src.models import build_model
from src.utils import build_metrics, metrics

logger = logging.getLogger(__name__)


class Trainer:
    """End-to-end training orchestrator."""

    def __init__(self):
        set_determinism(seed=cfg.seed)
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.model_dir, exist_ok=True)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Device: %s", self.device)

        # Data
        self.train_loader = get_train_loader()
        self.val_loader = get_val_loader()

        # Model + optimiser + loss
        self.model = build_model(cfg.model_name).to(self.device)
        self.loss_fn = BraTSLoss().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.max_epochs, eta_min=1e-6
        )
        self.scaler = GradScaler()

        # Metrics
        self.dice_metric, self.hd95_metric, self.post_trans = build_metrics()

        self.best_mean_dice = 0.0
        self.history: Dict = {"train_loss": [], "val_dice": []}

    # ─────────────────────────────────────────────────────────────────────
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss, step = 0.0, 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            step += 1

        return epoch_loss / max(step, 1)

    def _validate(self):
   
        self.model.eval()
        self.dice_metric.reset()

        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                # forward
                outputs = self.model(images)

                # loss
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()
                num_batches += 1

                # BraTS here is multilabel region prediction, so use sigmoid
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

            # update MONAI dice metric
                self.dice_metric(y_pred=preds, y=labels)

    # mean validation loss
        mean_val_loss = val_loss / max(1, num_batches)

            # aggregate mean dice
        dice_mean = self.dice_metric.aggregate()
        if isinstance(dice_mean, torch.Tensor):
            dice_mean = float(dice_mean.detach().cpu().item())
        else:
            dice_mean = float(dice_mean)

        # aggregate per-region dice safely
        per_region = self.dice_metric.aggregate(reduction=None)

        if isinstance(per_region, torch.Tensor):
            per_region = per_region.detach().cpu().numpy()
        else:
            per_region = np.asarray(per_region)

        # Robust handling for different MONAI output shapes
        # Possible cases:
        # scalar
        # (3,)
        # (N,3)
        # (1,3)
        # weird flattened forms
        if per_region.ndim == 0:
            per_region = np.array([float(per_region)] * 3, dtype=np.float32)

        elif per_region.ndim == 1:
            if per_region.shape[0] == 1:
                per_region = np.array([float(per_region[0])] * 3, dtype=np.float32)
            elif per_region.shape[0] < 3:
                padded = np.zeros(3, dtype=np.float32)
                padded[:per_region.shape[0]] = per_region
                per_region = padded
            else:
                per_region = per_region[:3].astype(np.float32)

        else:
            # e.g. (num_cases, 3) or (1, 3)
            per_region = np.asarray(per_region, dtype=np.float32)
            per_region = per_region.reshape(-1, per_region.shape[-1]).mean(axis=0)

            if per_region.shape[0] < 3:
                padded = np.zeros(3, dtype=np.float32)
                padded[:per_region.shape[0]] = per_region
                per_region = padded
            else:
                per_region = per_region[:3].astype(np.float32)

        metrics = {
            "val_loss": float(mean_val_loss),
            "dice": float(dice_mean),
            "dice_tc": float(per_region[0]),
            "dice_wt": float(per_region[1]),
            "dice_et": float(per_region[2]),
        }

        self.dice_metric.reset()
        self.model.train()

        return metrics

    # ─────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, metrics: Dict):
        path = os.path.join(cfg.model_dir, "best_model.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_mean_dice": self.best_mean_dice,
                "metrics": metrics,
                "config": cfg.__dict__,
            },
            path,
        )
        logger.info("Checkpoint saved → %s", path)

    # ─────────────────────────────────────────────────────────────────────
    def train(self):
        logger.info("Training for %d epochs …", cfg.max_epochs)

        for epoch in range(1, cfg.max_epochs + 1):
            loss = self._train_epoch(epoch)
            self.scheduler.step()
            self.history["train_loss"].append(loss)

            if epoch % cfg.val_every == 0:
                m = self._validate()
                self.history["val_dice"].append(m["mean_dice"])
                logger.info(
                    "Epoch %4d/%d | Loss %.4f | "
                    "Dice TC/WT/ET %.4f/%.4f/%.4f | HD95 %.2f",
                    epoch,
                    cfg.max_epochs,
                    loss,
                    m["dice_tc"],
                    m["dice_wt"],
                    m["dice_et"],
                    m["hd95"],
                )
                if m["mean_dice"] > self.best_mean_dice:
                    self.best_mean_dice = m["mean_dice"]
                    self._save_checkpoint(epoch, m)
                    logger.info(
                        "  ↳ New best mean Dice: %.4f", self.best_mean_dice
                    )
            else:
                logger.info(
                    "Epoch %4d/%d | Loss %.4f", epoch, cfg.max_epochs, loss
                )

        # Save final weights and training history
        torch.save(
            self.model.state_dict(),
            os.path.join(cfg.model_dir, "final_model.pth"),
        )
        with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(
            "Training complete. Best mean Dice: %.4f", self.best_mean_dice
        )
