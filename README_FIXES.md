# BraTS 2023 repo fixes

This patch bundle focuses on the code paths that currently break training or inference.

## Main fixes included

1. `src/trainer.py`
   - fixed validation key mismatch (`mean_dice` vs `dice`)
   - fixed missing HD95 fields in validation output
   - switched validation to sliding-window inference to avoid full-volume OOM
   - made AMP conditional on CUDA
   - saved scheduler state in the best checkpoint
   - made training history keys consistent

2. `src/data/__init__.py`
   - removed broken exports (`get_test_transforms` from the wrong module)
   - added `get_inference_transforms`

3. `src/data/dataset.py`
   - added explicit errors for missing or empty dataset directories
   - added stronger path resolution for `-`, `_`, and no-separator naming
   - prevented silent empty-train / empty-val loader creation
   - added `get_test_loader`

4. `src/data/transforms.py`
   - added explicit orientation labels to suppress MONAI orientation warning
   - kept train / val / inference pipelines aligned

5. `src/inferencer.py`
   - fixed package import path by relying on corrected `src.data`
   - made AMP conditional on CUDA
   - kept sliding-window inference path stable

6. `src/utils/metrics.py`
   - added a real HD95 approximation for offline scoring
   - returned both mean Dice and per-region Dice cleanly

7. `requirements.txt`
   - normalized filename casing and kept install command consistent

## Recommended repo cleanup

- Delete the duplicate root-level `config.py` and keep only `src/config.py`.
- Rename `Requirements.txt` to `requirements.txt` in the repo itself.
- Add a real `README.md` with:
  - expected BraTS folder structure
  - exact setup commands
  - train and inference commands
  - GPU memory notes for `segresnet` and `unet`
- Add a small `scripts/check_data.py` sanity script that verifies subject counts and missing modalities before training starts.
- Add checkpoint resume support to `scripts/train.py`.

## Suggested training command

```bash
python3 scripts/train.py       --data_dir "/root/workspace/data"       --model_name unet       --max_epochs 50       --batch_size 1       --output_dir "/root/workspace/outputs"       --model_dir "/root/workspace/models"
```

If GPU memory is tight, start with:

- `--batch_size 1`
- `cfg.sw_batch_size = 1`
- `cfg.roi_size = (128, 128, 128)` or smaller if needed
