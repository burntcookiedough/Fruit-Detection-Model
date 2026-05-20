# Dataset Build Pipeline

Scripts are numbered and must be run **in order**. Each script's output is
the input to the next. Skip steps 01–08 if `dataset_v4_balanced/` already
exists — it is the stable, pre-built V4 dataset.

## When to re-run the pipeline

- Adding new source images to `raw_datasets/`
- Changing quality filter thresholds
- Changing class balance targets
- Building V6+ dataset from scratch

## Scripts

### 01_source_audit.py
Audits `raw_datasets/` and produces a source manifest CSV.
Counts images per class per source, flags duplicates, reports format issues.

```bash
python pipeline/01_source_audit.py
```

---

### 02_build_raw.py
Merges all sources from `raw_datasets/` into `dataset_v4_raw/`.
Normalises to YOLO label format. Remaps class IDs from heterogeneous sources
to the canonical 8-class schema.

```bash
python pipeline/02_build_raw.py
```

---

### 03_prepare_quality.py
Filters `dataset_v4_raw/` for quality:
- Minimum image resolution (drops tiny thumbnails)
- Minimum bounding box area (drops pixel-sized annotations)
- Minimum boxes per image threshold
Outputs `dataset_v4_quality/`.

```bash
python pipeline/03_prepare_quality.py
```

---

### 04_check_leakage.py
Detects exact-duplicate images that appear across train/val/test splits
using perceptual hashing. Prints a leakage report.

```bash
python pipeline/04_check_leakage.py
```

---

### 05_clean_leakage.py
Removes cross-split duplicates found by step 04.
Duplicates in val/test are moved to quarantine (not deleted) for audit.
Outputs clean dataset in-place.

```bash
python pipeline/05_clean_leakage.py
```

---

### 06_filter_tiny_boxes.py
Drops bounding boxes below the minimum size threshold (8 px on longest side
at training resolution). Images where all boxes are removed become background
images or are moved to quarantine.

```bash
python pipeline/06_filter_tiny_boxes.py
```

---

### 07_drop_empty_pairs.py
Removes image+label pairs where the label file is empty or missing after
previous filtering steps. Prevents YOLO from training on background-only
images unintentionally.

```bash
python pipeline/07_drop_empty_pairs.py
```

---

### 08_balance_train.py
Balances the **training split only** (val/test are left at natural
distribution). Uses soft-cap (12K) and hard-cap (18K) per class to prevent
dominant classes from overwhelming minority classes.

Outputs the final `dataset_v4_balanced/` used for training.

```bash
python pipeline/08_balance_train.py
```

---

### 09_generate_webcam_train.py ← V5 addition

Generates synthetic webcam-degraded copies of training images.
Applies: JPEG compression, BGR colour casts (warm + cool), low-res
downscale/upscale, brightness/contrast variation, Gaussian noise, vignette,
motion blur, random occlusion rectangles.

Output goes to `dataset_v4_webcam_train/` which is referenced by
`data_v5_webcam.yaml` alongside the original `dataset_v4_balanced/train/`.

**Why**: YOLO's built-in HSV augmentation does not simulate JPEG artefacts,
BGR channel colour casts, or low-res downscale. V4 evaluation showed apple
(13%) and orange (11%) catastrophically fail under webcam conditions because
the model learned colour-only shortcuts. Training on degraded images forces
shape/texture learning.

```bash
python pipeline/09_generate_webcam_train.py              # all images, dual-pass
python pipeline/09_generate_webcam_train.py --fraction 0.6   # lighter version
python pipeline/09_generate_webcam_train.py --no-dual-pass   # single cast pass
```

---

## Dataset summary after full pipeline

| Dataset | Location | Images | Purpose |
|---|---|---:|---|
| Raw merged | `dataset_v4_raw/` | ~36K | Intermediate — deleteable |
| Quality filtered | `dataset_v4_quality/` | ~31K | Intermediate — deleteable |
| Final balanced | `dataset_v4_balanced/` | 26,419 | **Training — keep** |
| Webcam augmented | `dataset_v4_webcam_train/` | ~35K | V5 training addition |
| Webcam stress test | `synthetic_webcam_holdout/` | 155 | **Evaluation — keep** |

> Intermediate directories (`dataset_v4_raw/`, `dataset_v4_quality/`, quarantine dirs)
> can be deleted once `dataset_v4_balanced/` is confirmed. Re-run pipeline from
> `raw_datasets/` to rebuild them.
