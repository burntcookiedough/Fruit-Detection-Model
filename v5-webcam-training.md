# V5 Webcam-Augmented Training Dataset

## Goal
Add synthetic webcam-degraded training images to V5 so the model explicitly
trains on bad-camera conditions — directly fixing apple (13%) and orange (11%)
webcam collapse from V4.

## Why
YOLO's HSV/scale/flip augmentation does NOT simulate JPEG compression, BGR
channel color casts, or low-res downscale artifacts. The existing
`tools/generate_synthetic_webcam_holdout.py` already does all of this.
We just need to point it at TRAIN images (not val/test) and merge the output
into the V5 training split.

## Tasks

- [ ] **Task 1**: Write `pipeline/09_generate_webcam_train.py`
  - Copy from `tools/generate_synthetic_webcam_holdout.py`
  - Change `--splits` default to `["train"]` only (never val/test — avoids leakage)
  - Change `--output` default to `dataset_v4_webcam_train/`
  - Change sampling: instead of balanced-per-class, process ALL train images
    (or pass `--fraction 0.6` to take 60% of each class, ~10,700 images)
  - Write a flat `train/images/` + `train/labels/` YOLO structure for output
  - Verify: run script, confirm output has `train/images/` with ~10K `.jpg` files,
    no images from `valid/` or `test/` sources, labels match 1:1

- [ ] **Task 2**: Create `data_v5_webcam.yaml`
  - YOLO supports multiple train paths as a list
  - `train:` lists both `dataset_v4_balanced/train/images` AND `dataset_v4_webcam_train/train/images`
  - `val:` stays as `dataset_v4_balanced/valid/images` (unchanged — clean eval)
  - `test:` stays as `dataset_v4_balanced/test/images` (unchanged — clean eval)
  - Verify: `python -c "from ultralytics import YOLO; m=YOLO('yolov8s.pt'); m.val(data='data_v5_webcam.yaml', split='val')"` loads without error and shows correct image count (~3,815 val)

- [ ] **Task 3**: Update `config.py`
  - Set `DATA_YAML = ROOT / "data_v5_webcam.yaml"`
  - Set `RUN_NAME = "fruit_v5_quality"`
  - Keep V5 augmentation profile (stronger HSV — belt + suspenders)
  - Verify: `python -c "import config; print(config.DATA_YAML, config.RUN_NAME)"` prints correct paths

- [ ] **Task 4**: Smoke-test train for 2 epochs
  - Run: `python train.py --epochs 2 --name fruit_v5_smoke --batch 8 --workers 4`
  - Confirm: both dataset paths load, no CUDA OOM, both clean + degraded images appear in mosaic
  - Confirm train image count is ~28K (17,876 + ~10,700 webcam), val is still 3,815
  - Kill after 2 epochs — this is only a sanity check

- [ ] **Task 5**: Launch V5 full training run
  - `python train.py --name fruit_v5_quality --epochs 120 --batch 8 --workers 4`
  - Verify launch: `results.csv` row 1 appears in `runs/fruit_v5_quality/`
  - Confirm in terminal: train images count shows ~28K, no errors in first epoch

## Done When
- [ ] `runs/fruit_v5_quality/weights/best.pt` exists after training
- [ ] `evaluate.py --data synthetic_webcam_holdout/data_synthetic_holdout.yaml`
      shows apple webcam mAP50 > 40% and orange webcam mAP50 > 40%
      (up from 13% and 11% in V4)
- [ ] Overall webcam mAP50 > 55% (up from 41.7% in V4)

## Notes

### Leakage safety rule (CRITICAL)
The webcam generator MUST only source from `dataset_v4_balanced/train/`.
Never touch `valid/` or `test/` — those are the clean evaluation splits.
The 155-image `synthetic_webcam_holdout/` was built from val/test images
and stays evaluation-only. Do NOT regenerate or modify it.

### Fraction choice
~60% of training images (~10,700) is recommended:
- Enough to give the model strong webcam signal
- Avoids making degraded images > 40% of total training set
  (we still want the model to detect clean fruit too)
- Keeps total dataset ~28K images — reasonable epoch time on RTX 3060

### Expected training time
~28K images vs original 17,876 = ~1.57× more images per epoch
V4 was ~7.6 min/epoch → V5 expect ~10-12 min/epoch → 120 epochs ≈ 20-24 hours

### Transforms in the generator (all already implemented)
- Low-res downscale/upscale (webcam optics)
- JPEG compression (streaming artifact)
- BGR channel color cast (yellow indoor light)
- Brightness/contrast shift (dim rooms)
- Gaussian noise + motion blur
- Vignette
- Random occlusion rectangles
