"""
Trainer class for BraTS 2023.
"""

import json
import logging
import os
from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import cfg
from src.data.dataset import get_train_loader, get_val_loader
from src.losses.losses import BraTSLoss
from src.models import build_model
from src.utils import build_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """End-to-end training orchestrator."""

    def __init__(self):
        set_determinism(seed=cfg.seed)
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.model_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(cfg.use_amp and self.device.type == "cuda")
        logger.info("Device: %s", self.device)
        logger.info("AMP enabled: %s", self.use_amp)

        self.train_loader = get_train_loader()
        self.val_loader = get_val_loader()

        self.model = build_model(cfg.model_name).to(self.device)
        self.loss_fn = BraTSLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.max_epochs,
            eta_min=1e-6,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        (
            self.dice_mean_metric,
            self.dice_batch_metric,
            self.hd95_batch_metric,
            self.post_trans,
        ) = build_metrics()

        self.best_mean_dice = -float("inf")
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_mean_dice": [],
            "val_dice_tc": [],
            "val_dice_wt": [],
            "val_dice_et": [],
            "val_hd95_tc": [],
            "val_hd95_wt": [],
            "val_hd95_et": [],
            "lr": [],
        }

    def _amp_autocast(self):
        if self.use_amp:
            return torch.cuda.amp.autocast()
        return nullcontext()

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        num_steps = 0

        for step, batch in enumerate(self.train_loader, start=1):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with self._amp_autocast():
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += float(loss.item())
            num_steps = step

        return epoch_loss / max(num_steps, 1)

    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        self.dice_mean_metric.reset()
        self.dice_batch_metric.reset()
        self.hd95_batch_metric.reset()

        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                with self._amp_autocast():
                    logits = sliding_window_inference(
                        inputs=images,
                        roi_size=cfg.roi_size,
                        sw_batch_size=cfg.sw_batch_size,
                        predictor=self.model,
                        overlap=cfg.overlap,
                    )
                    loss = self.loss_fn(logits, labels)

                preds = self.post_trans(logits)
                labels_for_metric = labels.float()

                self.dice_mean_metric(y_pred=preds, y=labels_for_metric)
                self.dice_batch_metric(y_pred=preds, y=labels_for_metric)
                self.hd95_batch_metric(y_pred=preds, y=labels_for_metric)

                val_loss += float(loss.item())
                num_batches += 1

        mean_val_loss = val_loss / max(1, num_batches)

        mean_dice = self.dice_mean_metric.aggregate()
        if isinstance(mean_dice, torch.Tensor):
            mean_dice = float(mean_dice.detach().cpu().item())
        else:
            mean_dice = float(mean_dice)

        per_region_dice = self.dice_batch_metric.aggregate()
        per_region_hd95 = self.hd95_batch_metric.aggregate()

        if isinstance(per_region_dice, torch.Tensor):
            per_region_dice = per_region_dice.detach().cpu().numpy()
        else:
            per_region_dice = np.asarray(per_region_dice)

        if isinstance(per_region_hd95, torch.Tensor):
            per_region_hd95 = per_region_hd95.detach().cpu().numpy()
        else:
            per_region_hd95 = np.asarray(per_region_hd95)

        per_region_dice = np.nan_to_num(per_region_dice.astype(np.float32), nan=0.0)
        per_region_hd95 = np.nan_to_num(per_region_hd95.astype(np.float32), nan=0.0, posinf=0.0)

        metrics = {
            "val_loss": float(mean_val_loss),
            "mean_dice": float(mean_dice),
            "dice_tc": float(per_region_dice[0]),
            "dice_wt": float(per_region_dice[1]),
            "dice_et": float(per_region_dice[2]),
            "hd95_tc": float(per_region_hd95[0]),
            "hd95_wt": float(per_region_hd95[1]),
            "hd95_et": float(per_region_hd95[2]),
            "hd95_mean": float(np.mean(per_region_hd95)),
        }

        self.dice_mean_metric.reset()
        self.dice_batch_metric.reset()
        self.hd95_batch_metric.reset()
        self.model.train()
        return metrics

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        path = os.path.join(cfg.model_dir, "best_model.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_mean_dice": self.best_mean_dice,
                "metrics": metrics,
                "config": cfg.__dict__,
            },
            path,
        )
        logger.info("Checkpoint saved -> %s", path)

    def train(self) -> None:
        logger.info("Training for %d epochs ...", cfg.max_epochs)

        for epoch in range(1, cfg.max_epochs + 1):
            train_loss = self._train_epoch(epoch)
            self.scheduler.step()
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            self.history["train_loss"].append(train_loss)
            self.history["lr"].append(current_lr)

            if epoch % cfg.val_every == 0:
                m = self._validate()
                self.history["val_loss"].append(m["val_loss"])
                self.history["val_mean_dice"].append(m["mean_dice"])
                self.history["val_dice_tc"].append(m["dice_tc"])
                self.history["val_dice_wt"].append(m["dice_wt"])
                self.history["val_dice_et"].append(m["dice_et"])
                self.history["val_hd95_tc"].append(m["hd95_tc"])
                self.history["val_hd95_wt"].append(m["hd95_wt"])
                self.history["val_hd95_et"].append(m["hd95_et"])

                logger.info(
                    (
                        "Epoch %4d/%d | train_loss %.4f | val_loss %.4f | "
                        "Dice mean/TC/WT/ET %.4f/%.4f/%.4f/%.4f | "
                        "HD95 mean/TC/WT/ET %.4f/%.4f/%.4f/%.4f | lr %.2e"
                    ),
                    epoch,
                    cfg.max_epochs,
                    train_loss,
                    m["val_loss"],
                    m["mean_dice"],
                    m["dice_tc"],
                    m["dice_wt"],
                    m["dice_et"],
                    m["hd95_mean"],
                    m["hd95_tc"],
                    m["hd95_wt"],
                    m["hd95_et"],
                    current_lr,
                )

                if m["mean_dice"] > self.best_mean_dice:
                    self.best_mean_dice = m["mean_dice"]
                    self._save_checkpoint(epoch, m)
                    logger.info("New best mean Dice: %.4f", self.best_mean_dice)
            else:
                logger.info(
                    "Epoch %4d/%d | train_loss %.4f | lr %.2e",
                    epoch,
                    cfg.max_epochs,
                    train_loss,
                    current_lr,
                )

        torch.save(
            self.model.state_dict(),
            os.path.join(cfg.model_dir, "final_model.pth"),
        )
        with open(os.path.join(cfg.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        logger.info("Training complete. Best mean Dice: %.4f", self.best_mean_dice)
