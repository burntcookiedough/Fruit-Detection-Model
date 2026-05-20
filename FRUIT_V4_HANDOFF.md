# Fruit Detection V4 Handoff

Last updated: 2026-05-20

Workspace: `F:\Fruit Detection Model`

## Objective

Build a strong local fruit detector for bad webcam conditions on a laptop with:

- GPU: RTX 3060, 6 GB VRAM
- RAM: 16 GB
- Target classes: `apple`, `banana`, `orange`, `mango`, `pineapple`, `watermelon`, `grapes`, `pomegranate`
- Model family: YOLOv8

Important scope decision: V4 is a fruit-type detector only. It does not detect fruit damage, rot, ripeness, quality condition, or material damage. Any future damage/quality model needs a separate labeled dataset and class schema.

## Executive Status At Handoff

The V4 dataset pipeline is built and has passed the intended pre-training gates. The final training dataset is:

```text
F:\Fruit Detection Model\data_v4_balanced.yaml
F:\Fruit Detection Model\dataset_v4_balanced
```

The main quality model is currently:

```text
runs\fruit_v4_quality
```

The run was interrupted once with `Ctrl+C`, then correctly resumed with:

```powershell
.\venv\Scripts\python.exe train.py --resume --name fruit_v4_quality
```

The terminal confirmed:

```text
Resuming training ... runs\fruit_v4_quality\weights\last.pt from epoch 95 to 120 total epochs
```

As of the last file read, `results.csv` had recorded through epoch 94. The active resumed process may already be beyond that, but the file only updates after completed epochs.

Latest recorded epoch:

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: | ---: |
| 94 | 0.80755 | 0.67806 | 0.75425 | 0.57808 |

This is already materially better than the frozen Phase A baseline mAP50 of 0.584.

Current key files:

```text
runs\fruit_v4_quality\weights\best.pt
runs\fruit_v4_quality\weights\last.pt
runs\fruit_v4_quality\results.csv
```

Do not delete or overwrite `runs\fruit_v4_quality`.

## Chronological History

1. Started from frozen Phase A baseline: `runs/fruit_v4_s_local`.
2. Decided V4 must be a strict 8-class fruit-type detector.
3. Decided dataset quality is a hard blocker: no training until labels, splits, leakage, and class coverage are acceptable.
4. Audited raw datasets under `raw_datasets/`.
5. Built `build_v4_raw.py` to merge raw YOLO datasets into canonical 8-class labels.
6. Excluded synthetic/clean-background Fruit-360-style sources from the training chain.
7. Mapped non-standard source classes into the canonical 8 fruit classes.
8. Built `dataset_v4_raw` and `dataset_v4_raw_manifest.csv`.
9. Ran `prepare_v4_quality.py` to remove malformed labels, empty labels, loose boxes, synthetic sources, and bad banana/finger-style samples.
10. Ran leakage checks and found cross-split near-duplicates.
11. Wrote and ran `clean_split_leakage.py`, quarantining near-duplicates.
12. Rechecked leakage until MD5 and pHash gates passed.
13. Ran a tiny-box cleanup with an 8 px minimum box threshold.
14. Removed empty image/label pairs created by tiny-box cleanup.
15. Balanced the training split while leaving validation/test natural.
16. Created `dataset_v4_balanced` and `data_v4_balanced.yaml`.
17. Created a synthetic webcam stress holdout because real fruits/webcam were not available.
18. Evaluated the frozen Phase A model on V4 val/test and synthetic webcam stress data.
19. Confirmed Phase A performs poorly on the new V4/webcam-like domain.
20. Configured full V4 augmentation and disk caching for laptop training.
21. Ran worker/cache benchmarks; workers 0 was too slow, workers 2 was faster, and the user chose workers 4 because previous cached runs were stable.
22. Started the main YOLOv8s quality training run: `fruit_v4_quality`.
23. Training reached epoch 94 with mAP50 0.75425 and mAP50-95 0.57808.
24. Training was manually interrupted.
25. An accidental fresh run folder `fruit_v4_quality_resume` was created using the wrong resume style.
26. Verified original `fruit_v4_quality` weights were not overwritten.
27. Correctly resumed original `fruit_v4_quality` using `--resume --name fruit_v4_quality`.

## Frozen Baseline

Baseline run:

```text
runs/fruit_v4_s_local/
```

Baseline model:

```text
runs/fruit_v4_s_local/weights/best.pt
```

Known baseline metrics from the original Phase A run:

| Metric | Best epoch around 61 | Final epoch 80 |
| --- | ---: | ---: |
| mAP50 | 0.584 | 0.580 |
| mAP50-95 | 0.460 | 0.460 |
| Precision | 0.677 | 0.723 |
| Recall | 0.538 | 0.523 |

Pre-training stress evaluation of this frozen baseline on the new V4-style data showed it was not strong enough for the new dataset and webcam-like conditions:

| Dataset | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| V4 validation | 0.4215 | 0.3144 |
| V4 test | 0.4530 | 0.3376 |
| Synthetic webcam holdout | 0.1959 | 0.1300 |

Interpretation:

- The Phase A model was not reliable on the broader V4 dataset.
- Synthetic webcam degradation hurt the baseline heavily.
- This justified rebuilding the dataset and training V4 instead of only tuning thresholds.

## Data Pipeline Built

The dataset was rebuilt before training. Training was intentionally blocked until the quality gates passed.

Main pipeline:

```text
raw_datasets/
  -> source_audit_v4.py
  -> build_v4_raw.py
  -> dataset_v4_raw/
  -> prepare_v4_quality.py
  -> dataset_v4_quality/
  -> check_split_leakage.py
  -> clean_split_leakage.py
  -> dataset_v4_quality_min8/
  -> drop_empty_yolo_images.py
  -> balance_v4_train.py
  -> dataset_v4_balanced/
  -> data_v4_balanced.yaml
```

Final training dataset:

```text
data_v4_balanced.yaml
dataset_v4_balanced/
```

Do not train from `data_v4_quality.yaml` unless intentionally debugging. The current V4 training run uses `data_v4_balanced.yaml`.

## Scripts Added

| Script | Purpose |
| --- | --- |
| `source_audit_v4.py` | Audit raw sources and image counts |
| `build_v4_raw.py` | Merge raw YOLO datasets, map class IDs to the canonical 8 classes, dedupe by MD5 |
| `check_split_leakage.py` | Check exact MD5 duplicates and pHash near-duplicates across train/valid/test |
| `clean_split_leakage.py` | Quarantine cross-split near-duplicates |
| `balance_v4_train.py` | Balance the training split without touching validation/test |
| `create_pseudo_label_queue.py` | Queue unlabeled images for pseudo-label/manual review instead of training on them directly |
| `generate_synthetic_webcam_holdout.py` | Generate temporary webcam-like stress-test images from labeled valid/test images |
| `predict_stress_samples.py` | Save annotated predictions for manual stress-set review |
| `drop_empty_yolo_images.py` | Remove image/label pairs left empty after tiny-box filtering |
| `run_v4_training.py` | Gate-aware benchmark/smoke/final training helper |

Modified files:

| File | Change |
| --- | --- |
| `prepare_v4_quality.py` | Defaults to `dataset_v4_raw`, enforces validation/test class gates, no stale clean-profile recommendation |
| `config.py` | Uses full V4 augmentation and `cache="disk"` |
| `dashboard.html` | Defaults to `fruit_v4_quality`, 120 epochs, batch 8, workers 4, YOLOv8s metadata |

## Raw Dataset Build

`build_v4_raw.py` output:

```text
dataset_v4_raw/
dataset_v4_raw_manifest.csv
```

Final raw kept images:

```text
36,527
```

Raw box counts:

| Class | Raw boxes |
| --- | ---: |
| apple | 27,475 |
| banana | 61,054 |
| orange | 31,794 |
| mango | 18,105 |
| pineapple | 3,681 |
| watermelon | 3,630 |
| grapes | 20,161 |
| pomegranate | 6,589 |

Special mapping decisions:

- `fruits_360`, `fruits_262`, and `fruits_360_yolo` were skipped to avoid synthetic/clean-background bias.
- `mango_ripeness_yolo` ripeness classes were mapped to `mango`, except `Not_Mango`.
- `watermelon-peel` was skipped.
- Pomegranate buds/flowers were skipped.

## Quality Dataset

`prepare_v4_quality.py --source dataset_v4_raw` produced:

```text
dataset_v4_quality/
data_v4_quality.yaml
quality_report_v4.txt
```

Quality filtering removed:

- Finger-style banana images: 1,237
- Loose boxes covering more than 85 percent image area: 3,744
- Malformed or empty labels
- Synthetic/Fruit-360-style sources

Initial quality dataset:

```text
31,546 images
```

Then leakage checking found cross-split pHash near-duplicates. `clean_split_leakage.py` quarantined:

```text
6,411 near-duplicate images
```

Quarantine folder:

```text
dataset_v4_quality_leakage_quarantine/
```

After that, `check_split_leakage.py` passed with:

- Zero MD5 duplicates across train/valid/test
- Zero unresolved pHash near-duplicates across train/valid/test

## Tiny Box Cleanup

A conservative 8 px minimum box filter was applied:

```powershell
.\venv\Scripts\python.exe filter_dataset.py --data data_v4_quality.yaml --out dataset_v4_quality_min8 --min_px 8
```

Approximate removed boxes:

- Train: 2.7 percent
- Validation: 1.9 percent
- Test: 3.0 percent

Then empty label pairs were removed:

```powershell
.\venv\Scripts\python.exe drop_empty_yolo_images.py --dataset dataset_v4_quality_min8
```

Empty pairs moved to:

```text
dataset_v4_quality_min8_empty_quarantine/
```

## Final Balanced Dataset

Balancing command used:

```powershell
.\venv\Scripts\python.exe balance_v4_train.py --source dataset_v4_quality_min8 --target-min 4000 --soft-cap 12000 --hard-cap 18000
```

Final dataset:

```text
dataset_v4_balanced/
data_v4_balanced.yaml
```

Final split sizes:

| Split | Images | Labels |
| --- | ---: | ---: |
| train | 17,876 | 17,876 |
| valid | 3,815 | 3,815 |
| test | 4,728 | 4,728 |

Final training box counts:

| Class | Train boxes |
| --- | ---: |
| apple | 13,445 |
| banana | 7,949 |
| orange | 12,514 |
| mango | 11,019 |
| pineapple | 4,021 |
| watermelon | 4,004 |
| grapes | 9,872 |
| pomegranate | 4,573 |

Validation box counts:

| Class | Validation boxes |
| --- | ---: |
| apple | 2,395 |
| banana | 1,699 |
| orange | 3,364 |
| mango | 2,526 |
| pineapple | 414 |
| watermelon | 452 |
| grapes | 2,249 |
| pomegranate | 953 |

Test box counts:

| Class | Test boxes |
| --- | ---: |
| apple | 2,691 |
| banana | 2,870 |
| orange | 4,666 |
| mango | 2,716 |
| pineapple | 448 |
| watermelon | 521 |
| grapes | 2,361 |
| pomegranate | 1,007 |

Final leakage gate:

```text
dataset_v4_balanced passed:
- zero MD5 duplicates across splits
- zero pHash near-duplicates across splits
```

Important: validation and test were not balanced by augmentation. They remain natural evaluation splits.

## Why Class Counts Are Not Exactly Equal

The classes are intentionally not forced to exactly equal image counts. Equal image counts are not the same thing as equal learning signal for object detection because:

- One image may contain multiple boxes.
- Some source datasets have many duplicate-like images.
- Some classes have more noisy labels than others.
- Over-augmenting rare classes can create leakage-like repetition and make validation performance look better than real performance.
- Validation and test should represent natural data, not artificially equalized data.

The balancing strategy used here is conservative:

- Training split was balanced by box count, not just image count.
- Rare classes were raised to at least about 4,000 training boxes.
- Overrepresented classes were softly capped instead of aggressively deleting useful diversity.
- Validation and test were left natural, but with enough boxes per class to be meaningful.
- Augmented training images were rejected if they created leakage risk.

This is why pineapple, watermelon, and pomegranate are much lower than apple/orange/mango but still above the minimum useful threshold. That is preferable to making thousands of low-value copies and teaching the model repeated artifacts.

## Quality Gates That Passed

The final training dataset was accepted only after these checks:

| Gate | Result |
| --- | --- |
| All included images have labels | Passed |
| All labels mapped to class IDs `0..7` | Passed |
| No empty labels after cleanup | Passed |
| No malformed boxes | Passed |
| No huge loose boxes over the configured threshold | Passed |
| No Fruit-360/synthetic clean-background sources in V4 chain | Passed |
| At least 200 raw boxes per class before filtering | Passed |
| At least 100 boxes per class in validation/test | Passed |
| Zero exact MD5 duplicates across splits | Passed |
| Zero unresolved pHash near-duplicates across splits | Passed |
| Final train split has rare classes lifted to about 4,000 boxes | Passed |

## What Was Not Done Yet

The real webcam holdout is not done because the user did not have fruits available and the webcam is unreliable.

Still required for final real-world claims:

```text
webcam_holdout/images/
webcam_holdout/labels/
webcam_holdout/data_holdout.yaml
```

The current synthetic webcam set is useful for stress testing only. It should not be used to claim deployment quality.

## Unlabeled Images Policy

Unlabeled images are not used directly for training.

Allowed path for image-only sources:

```text
unlabeled images
  -> pseudo-label with current best model
  -> manual review/fix labels
  -> quality checks
  -> only then merge into a future dataset version
```

Script for this workflow:

```text
create_pseudo_label_queue.py
```

Do not add unlabeled images directly into `dataset_v4_balanced`.

## Synthetic Webcam Holdout

Because real fruits/webcam access was unavailable, a temporary synthetic webcam stress set was created.

Folder:

```text
synthetic_webcam_holdout/
synthetic_webcam_holdout/data_synthetic_holdout.yaml
```

Latest generated set:

```text
155 images/labels
```

Class coverage:

| Class | Images containing class |
| --- | ---: |
| apple | 22 |
| banana | 22 |
| orange | 21 |
| mango | 21 |
| pineapple | 20 |
| watermelon | 20 |
| grapes | 20 |
| pomegranate | 20 |

Synthetic transforms include:

- Low resolution downscale/upscale
- JPEG compression
- Gaussian blur
- Light motion blur
- Brightness/contrast shifts
- Yellow indoor color cast
- Noise/grain
- Mild shadow/vignette
- Partial occlusion rectangles where possible

This set is only for fast stress testing. It is not a replacement for a real webcam holdout.

Real holdout still needed later:

```text
webcam_holdout/
  images/
  labels/
  data_holdout.yaml
```

Target real holdout:

- Minimum: 25 manually labeled images per class
- Preferred: 50 per class
- Include low light, blur, clutter, hand occlusion, close/far fruit, and multiple fruits

## Training Configuration

Current quality training run:

```text
runs/fruit_v4_quality/
```

Current model:

```text
YOLOv8s
```

Dataset:

```text
data_v4_balanced.yaml
```

Settings:

| Setting | Value |
| --- | --- |
| model | `yolov8s.pt` |
| epochs | 120 |
| image size | 640 |
| batch | 8 |
| workers | 4 |
| patience | 25 |
| cache | disk |
| augmentation | full V4 webcam-oriented profile |
| close_mosaic | 10 |

Disk cache is enabled in `config.py`. The cache is large, roughly 40+ GB, but it speeds training. Do not delete `.npy` cache files while training is active.

## Laptop Performance Decisions

The laptop constraints were:

- RTX 3060 with 6 GB VRAM
- 16 GB system RAM
- Windows environment
- IDE overhead should be avoided during final training

Resulting decisions:

- Use `imgsz=640`; larger image sizes would risk VRAM pressure and slower epochs.
- Use `batch=8` for YOLOv8s; this is the safe quality setting on 6 GB VRAM.
- Use `workers=4` because the user requested it and cache-backed training had run stably with it.
- Use `cache=disk`, not RAM cache, because RAM is only 16 GB.
- Move temporary directories and Ultralytics config to `F:` to avoid pressure on `C:`.
- Run final training from PowerShell instead of inside the IDE.

Observed benchmark notes:

- A YOLOv8n workers-0 benchmark completed but was very slow.
- Workers 2 was clearly faster than workers 0.
- Workers 4 was selected for the main run after cached training appeared stable.
- If Windows dataloader problems appear, fallback is workers 2.
- If CUDA out-of-memory appears, fallback is batch 6 or batch 4 for YOLOv8s.

Current disk/cache state from the last check:

| Item | Value |
| --- | ---: |
| F: free space | 62.56 GB |
| F: used space | 138.03 GB |
| `dataset_v4_balanced` `.npy` cache files | 21,691 |
| `dataset_v4_balanced` cache size | 39.89 GB |

Keep this cache during training. It is large but intentional.

## Current Training State

Original run:

```text
runs/fruit_v4_quality/
```

Current checkpoint files exist:

```text
runs/fruit_v4_quality/weights/best.pt
runs/fruit_v4_quality/weights/last.pt
```

As of the last file check, `results.csv` had completed through epoch 94:

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: | ---: |
| 94 | 0.80755 | 0.67806 | 0.75425 | 0.57808 |

The terminal screenshot confirmed the correct resume behavior:

```text
Resuming training ... runs\fruit_v4_quality\weights\last.pt from epoch 95 to 120 total epochs
Logging results to ... runs\fruit_v4_quality
Using 4 dataloader workers
```

This means the active resume did not restart from scratch.

The `results.csv` header is:

```text
epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2,lr/pg3,lr/pg4,lr/pg5,lr/pg6,lr/pg7
```

Epoch 94 row:

```text
94,42996.4,0.96353,0.76134,1.2035,0.80755,0.67806,0.75425,0.57808,0.89303,0.69549,1.1743,...
```

Approximate epoch time near the end of the run was around 7.6 minutes per epoch based on the cumulative time difference in `results.csv`. From epoch 95 to 120, expect roughly 3 to 3.5 additional hours if the same speed holds.

Accidental fresh run:

```text
runs/fruit_v4_quality_resume/
```

This was created when `--model last.pt` was used with a new `--name`. It did not overwrite the original run's `best.pt` or `last.pt`. It can be ignored or deleted after confirming no process is using it.

## Correct Resume Command

Use this if training is stopped and needs to continue:

```powershell
Set-Location "F:\Fruit Detection Model"

$env:YOLO_CONFIG_DIR = "F:\Fruit Detection Model\Ultralytics"
$env:TMP = "F:\Fruit Detection Model\tmp"
$env:TEMP = "F:\Fruit Detection Model\tmp"

.\venv\Scripts\python.exe train.py --resume --name fruit_v4_quality
```

Do not pass `--model`, `--data`, or a new `--name` when resuming. `train.py --resume --name fruit_v4_quality` loads:

```text
runs/fruit_v4_quality/weights/last.pt
```

and restores optimizer/LR/epoch state.

Wrong resume command that should not be used:

```powershell
.\venv\Scripts\python.exe train.py `
  --model runs\fruit_v4_quality\weights\last.pt `
  --data data_v4_balanced.yaml `
  --name fruit_v4_quality_resume `
  --epochs 120 `
  --batch 8 `
  --patience 25 `
  --workers 4
```

That starts a new run because it treats `last.pt` as initial weights, not as a full training resume. This is what created:

```text
runs\fruit_v4_quality_resume
```

## High Priority Resume Command

Use this if launching from a clean PowerShell and you want high process priority:

```powershell
Set-Location "F:\Fruit Detection Model"

$env:YOLO_CONFIG_DIR = "F:\Fruit Detection Model\Ultralytics"
$env:TMP = "F:\Fruit Detection Model\tmp"
$env:TEMP = "F:\Fruit Detection Model\tmp"

New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null

$p = Start-Process `
  -FilePath ".\venv\Scripts\python.exe" `
  -ArgumentList @(
    "train.py",
    "--resume",
    "--name",
    "fruit_v4_quality"
  ) `
  -WorkingDirectory "F:\Fruit Detection Model" `
  -NoNewWindow `
  -PassThru

$p.PriorityClass = "High"
$p.WaitForExit()
$p.ExitCode
```

PowerShell does not support `Start-Process -Priority` on this system. Start the process first, then set `$p.PriorityClass = "High"`.

## Dashboard

Dashboard file:

```text
dashboard.html
```

It is configured for:

```text
fruit_v4_quality
```

Start a lightweight dashboard server from the workspace:

```powershell
Set-Location "F:\Fruit Detection Model"
python -m http.server 8000 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8000/dashboard.html?run=fruit_v4_quality
```

The dashboard reads:

```text
runs/fruit_v4_quality/results.csv
```

It updates after each completed epoch.

## Evaluation Commands After Training

Evaluate the final V4 model on the balanced test split:

```powershell
Set-Location "F:\Fruit Detection Model"

.\venv\Scripts\python.exe evaluate.py `
  --model runs\fruit_v4_quality\weights\best.pt `
  --data data_v4_balanced.yaml `
  --split test
```

Evaluate on the temporary synthetic webcam stress set:

```powershell
.\venv\Scripts\python.exe evaluate.py `
  --model runs\fruit_v4_quality\weights\best.pt `
  --data synthetic_webcam_holdout\data_synthetic_holdout.yaml `
  --split test
```

Once real webcam holdout images are labeled, evaluate:

```powershell
.\venv\Scripts\python.exe evaluate.py `
  --model runs\fruit_v4_quality\weights\best.pt `
  --data webcam_holdout\data_holdout.yaml `
  --split test
```

Save visual prediction samples after training:

```powershell
.\venv\Scripts\python.exe predict_stress_samples.py `
  --model runs\fruit_v4_quality\weights\best.pt `
  --data synthetic_webcam_holdout\data_synthetic_holdout.yaml `
  --out runs\fruit_v4_quality_synthetic_predictions `
  --limit 80
```

If the script options differ, run:

```powershell
.\venv\Scripts\python.exe predict_stress_samples.py --help
```

The visual inspection should focus on:

- Pineapple
- Watermelon
- Pomegranate
- Mango
- Low light
- Blur
- Clutter
- Hand/partial occlusion
- Multiple fruits in one frame

## Full Command Reference

Audit raw sources:

```powershell
.\venv\Scripts\python.exe source_audit_v4.py
```

Build raw merged V4 dataset:

```powershell
.\venv\Scripts\python.exe build_v4_raw.py
```

Prepare quality dataset:

```powershell
.\venv\Scripts\python.exe prepare_v4_quality.py --source dataset_v4_raw
```

Check split leakage:

```powershell
.\venv\Scripts\python.exe check_split_leakage.py --dataset dataset_v4_quality
```

Clean split leakage:

```powershell
.\venv\Scripts\python.exe clean_split_leakage.py --dataset dataset_v4_quality
```

Tiny-box filter:

```powershell
.\venv\Scripts\python.exe filter_dataset.py --data data_v4_quality.yaml --out dataset_v4_quality_min8 --min_px 8
```

Drop images left with empty labels:

```powershell
.\venv\Scripts\python.exe drop_empty_yolo_images.py --dataset dataset_v4_quality_min8
```

Balance train split:

```powershell
.\venv\Scripts\python.exe balance_v4_train.py --source dataset_v4_quality_min8 --target-min 4000 --soft-cap 12000 --hard-cap 18000
```

Final leakage check:

```powershell
.\venv\Scripts\python.exe check_split_leakage.py --dataset dataset_v4_balanced
```

Generate synthetic webcam holdout:

```powershell
.\venv\Scripts\python.exe generate_synthetic_webcam_holdout.py
```

Evaluate frozen baseline on balanced test:

```powershell
.\venv\Scripts\python.exe evaluate.py `
  --model runs\fruit_v4_s_local\weights\best.pt `
  --data data_v4_balanced.yaml `
  --split test
```

Evaluate frozen baseline on synthetic webcam holdout:

```powershell
.\venv\Scripts\python.exe evaluate.py `
  --model runs\fruit_v4_s_local\weights\best.pt `
  --data synthetic_webcam_holdout\data_synthetic_holdout.yaml `
  --split test
```

Fresh main training command originally used:

```powershell
Set-Location "F:\Fruit Detection Model"

$env:YOLO_CONFIG_DIR = "F:\Fruit Detection Model\Ultralytics"
$env:TMP = "F:\Fruit Detection Model\tmp"
$env:TEMP = "F:\Fruit Detection Model\tmp"

.\venv\Scripts\python.exe train.py `
  --model yolov8s.pt `
  --data data_v4_balanced.yaml `
  --name fruit_v4_quality `
  --epochs 120 `
  --batch 8 `
  --patience 25 `
  --workers 4
```

Only use the fresh command if starting a brand new run with a new name. For the current run, use the resume command.

## Next Steps

1. Let `fruit_v4_quality` finish to epoch 120 unless early stopping triggers.
2. Evaluate `runs/fruit_v4_quality/weights/best.pt` on `data_v4_balanced.yaml --split test`.
3. Evaluate the same model on `synthetic_webcam_holdout/data_synthetic_holdout.yaml`.
4. Save/inspect prediction samples, especially for pineapple, watermelon, pomegranate, and mango.
5. If YOLOv8s webcam FPS is acceptable, keep YOLOv8s as the quality model.
6. If YOLOv8s is too slow for webcam use, train YOLOv8n using the same dataset/settings as a fast fallback.
7. When fruits/webcam are available, build the real `webcam_holdout/` and evaluate before making final deployment claims.

## YOLOv8n Fallback Plan

Only run this if YOLOv8s is too slow or unstable for live webcam inference:

```powershell
Set-Location "F:\Fruit Detection Model"

.\venv\Scripts\python.exe train.py `
  --model yolov8n.pt `
  --data data_v4_balanced.yaml `
  --name fruit_v4_nano `
  --epochs 120 `
  --batch 16 `
  --patience 25 `
  --workers 4
```

If batch 16 causes CUDA out-of-memory, rerun with:

```powershell
--batch 8
```

## Failure Handling

If CUDA out-of-memory occurs:

```text
YOLOv8s: reduce batch from 8 to 6 or 4
YOLOv8n: reduce batch from 16 to 8
```

If Windows dataloader issues occur:

```text
reduce workers from 4 to 2
```

If disk space becomes tight:

- Do not delete the active run.
- Do not delete `runs/fruit_v4_quality/weights`.
- Do not delete `.npy` cache while training is active.
- Safe cleanup candidates after training: old benchmark runs, accidental `fruit_v4_quality_resume`, old prediction samples, old quarantines if no longer needed.

## How To Verify The Run Is Healthy

Check the latest training row:

```powershell
Get-Content "F:\Fruit Detection Model\runs\fruit_v4_quality\results.csv" -Tail 5
```

Check checkpoints:

```powershell
Get-ChildItem "F:\Fruit Detection Model\runs\fruit_v4_quality\weights" |
  Select-Object Name,Length,LastWriteTime
```

Check whether the resume checkpoint exists:

```powershell
Test-Path "F:\Fruit Detection Model\runs\fruit_v4_quality\weights\last.pt"
```

Check the dashboard:

```text
http://127.0.0.1:8000/dashboard.html?run=fruit_v4_quality
```

Healthy signs:

- Epoch number continues from 95 or later, not 0.
- Logging path is `runs\fruit_v4_quality`.
- `last.pt` timestamp updates after epochs complete.
- `results.csv` gains one row per completed epoch.
- mAP does not collapse suddenly.

Concerning signs:

- New folder such as `fruit_v4_quality2` or `fruit_v4_quality_resume` appears and receives new rows.
- Epoch starts from 0 when intending to resume.
- CUDA out-of-memory appears.
- Dataloader worker errors appear repeatedly.
- Disk space drops close to zero.

## Handoff To Another Assistant

If another assistant or engineer takes over, tell them:

```text
We are training a YOLOv8s 8-class fruit detector on Windows at F:\Fruit Detection Model.
The final dataset is data_v4_balanced.yaml and dataset_v4_balanced.
The active run is runs\fruit_v4_quality.
Resume only with: .\venv\Scripts\python.exe train.py --resume --name fruit_v4_quality
Do not use --model last.pt with a new name.
Do not delete disk cache or the active run.
After training, evaluate best.pt on data_v4_balanced.yaml test and synthetic_webcam_holdout.
Real webcam holdout is still required later.
```

## Current Key Paths

| Item | Path |
| --- | --- |
| Final dataset yaml | `F:\Fruit Detection Model\data_v4_balanced.yaml` |
| Final dataset folder | `F:\Fruit Detection Model\dataset_v4_balanced` |
| Current training run | `F:\Fruit Detection Model\runs\fruit_v4_quality` |
| Current best weights | `F:\Fruit Detection Model\runs\fruit_v4_quality\weights\best.pt` |
| Current resume weights | `F:\Fruit Detection Model\runs\fruit_v4_quality\weights\last.pt` |
| Synthetic holdout yaml | `F:\Fruit Detection Model\synthetic_webcam_holdout\data_synthetic_holdout.yaml` |
| Real holdout yaml | `F:\Fruit Detection Model\webcam_holdout\data_holdout.yaml` |
| Dashboard | `F:\Fruit Detection Model\dashboard.html` |

## Important Do-Not-Do List

- Do not train from unlabeled images.
- Do not mix synthetic webcam holdout into train/valid/test.
- Do not mix real webcam holdout into train/valid/test.
- Do not resume with `--model last.pt --name some_new_name`; that starts a fresh run.
- Do not overwrite or delete `runs/fruit_v4_quality` while training is active.
- Do not claim real webcam deployment quality until `webcam_holdout/` has manually labeled real images.
