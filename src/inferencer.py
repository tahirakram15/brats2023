"""
Inference engine for BraTS 2023.
"""

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose

from src.config import cfg
from src.data import build_test_file_list, get_inference_transforms
from src.models import build_model
from src.utils import channels_to_label_map, postprocess_channels

logger = logging.getLogger(__name__)


class Inferencer:
    """Run inference on a test directory and save NIfTI segmentation masks."""

    def __init__(self, checkpoint: str, output_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(cfg.use_amp and self.device.type == "cuda")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = build_model(cfg.model_name).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        logger.info("Loaded checkpoint: %s", checkpoint)
        logger.info("Device: %s | AMP enabled: %s", self.device, self.use_amp)

        self.transforms = get_inference_transforms()
        self.post_trans = Compose(
            [
                Activations(sigmoid=True),
                AsDiscrete(threshold=0.5),
            ]
        )

    def _amp_autocast(self):
        if self.use_amp:
            return torch.cuda.amp.autocast()
        return nullcontext()

    def _predict(self, image_paths: List[str]) -> np.ndarray:
        data = self.transforms({"image": image_paths})
        image = data["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            with self._amp_autocast():
                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=cfg.roi_size,
                    sw_batch_size=cfg.sw_batch_size,
                    predictor=self.model,
                    overlap=cfg.overlap,
                )
            pred = self.post_trans(logits[0]).cpu().numpy().astype(np.uint8)

        return postprocess_channels(pred)

    def run(self, test_dir: str) -> None:
        data_list = build_test_file_list(test_dir)
        logger.info("Running inference on %d subjects ...", len(data_list))

        for item in data_list:
            sid = item["subject_id"]
            image_paths = item["image"]
            pred_channels = self._predict(image_paths)
            label_map = channels_to_label_map(pred_channels)

            ref = nib.load(image_paths[0])
            nib_out = nib.Nifti1Image(label_map.astype(np.uint8), affine=ref.affine, header=ref.header)
            out_path = self.output_dir / f"{sid}.nii.gz"
            nib.save(nib_out, str(out_path))
            logger.info("Saved: %s", out_path)

        logger.info("Inference complete.")
