# 🍎 Fruit Detection Model

A real-world fruit detection system built with **YOLOv8s** and trained on a
quality-filtered, leakage-free dataset of **26,000+ images** across 8 fruit classes.
Designed for iterative improvement toward embedded hardware deployment.

---

## 📊 Model Performance (V4 — Current Champion)

**Model**: `models/best.pt` — YOLOv8s, 11M params, 21.5 MB (inference-ready)
**Training**: 120 epochs on RTX 3060 6 GB, `dataset_v4_balanced`

| Split | Images | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| **Validation** | 3,815 | **75.4%** | 57.8% | 80.5% | 68.0% |
| **Test** (holdout) | 4,728 | **66.2%** | 51.6% | 80.4% | 67.6% |
| **Webcam stress** | 155 | **41.7%** | 28.7% | 70.8% | 46.5% |

### Per-Class Test Results

| Class | mAP@50 | Recall | Webcam mAP@50 | Status |
|---|---:|---:|---:|---|
| 🥭 Mango | 97.1% | 96.9% | 79.3% | 🟢 Excellent |
| 🍎 Pomegranate | 91.6% | 91.8% | 61.1% | 🟢 Good |
| 🍊 Orange | 64.4% | 66.1% | 10.7% | 🔴 Webcam critical |
| 🍇 Grapes | 54.5% | 57.5% | 39.5% | 🟠 Moderate |
| 🍌 Banana | 56.8% | 58.5% | 30.2% | 🟠 Moderate |
| 🍍 Pineapple | 53.4% | 54.9% | 40.2% | 🟠 Moderate |
| 🍉 Watermelon | 51.5% | 52.2% | 59.0% | 🟡 Decent |
| 🍎 Apple | 60.3% | 63.1% | 13.3% | 🔴 Webcam critical |

> Apple and orange collapse under webcam conditions because the model learned
> colour-only shortcuts. V5 training addresses this with synthetic webcam-degraded
> training images.

---

## 🚀 Quick Start

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Run inference (internet image)
```bash
python demo.py
```

### Run webcam inference
```bash
python demo.py --mode webcam
```

### Evaluate on test set
```bash
python evaluate.py --model models/best.pt --data data_v4_balanced.yaml --split test
```

### Evaluate on webcam stress set
```bash
python evaluate.py --model models/best.pt --data synthetic_webcam_holdout/data_synthetic_holdout.yaml
```

---

## 📁 Project Structure

```
Fruit Detection Model/
│
├── train.py                    # Training entry point
├── evaluate.py                 # Evaluation (test/val/webcam splits)
├── demo.py                     # Live inference: webcam or internet image
├── config.py                   # All hyperparameters — single source of truth
├── telemetry.py                # Training metrics logger
├── dashboard.html              # Live training dashboard
│
├── pipeline/                   # V4 dataset build pipeline (numbered, ordered)
│   ├── 01_source_audit.py          # Audit raw source datasets
│   ├── 02_build_raw.py             # Merge raw sources into one dataset
│   ├── 03_prepare_quality.py       # Quality filter (resolution, box size)
│   ├── 04_check_leakage.py         # Detect train/val/test leakage
│   ├── 05_clean_leakage.py         # Remove duplicate images across splits
│   ├── 06_filter_tiny_boxes.py     # Drop boxes below minimum size
│   ├── 07_drop_empty_pairs.py      # Remove images with no valid labels
│   ├── 08_balance_train.py         # Balance training set per-class
│   └── 09_generate_webcam_train.py # Generate webcam-degraded training images (V5)
│
├── tools/                      # Utility scripts (recurring use)
│   ├── generate_synthetic_webcam_holdout.py  # Regenerate stress-test set
│   ├── predict_samples.py
│   └── predict_stress_samples.py
│
├── export/                     # Model export
│   ├── export_onnx.py              # ONNX (Raspberry Pi, Jetson, laptop)
│   └── export_tflite.py            # TFLite (Android, Coral)
│
├── inference/                  # Standalone inference scripts
│   ├── webcam.py
│   └── image.py
│
├── archive/                    # V3-era scripts (superseded, kept for reference)
│
├── models/
│   └── best.pt                 # Current champion weights (YOLOv8s, V4)
│
├── dataset_v4_balanced/        # Training dataset (42 GB, 26K images, 8 classes)
├── raw_datasets/               # Original downloaded sources (14 GB, irreplaceable)
├── synthetic_webcam_holdout/   # 155-image webcam stress-test set
│
├── data_v4_balanced.yaml       # V4 dataset config (clean)
├── data_v5_webcam.yaml         # V5 dataset config (clean + webcam-degraded)
│
└── runs/
    ├── fruit_v4_quality/       # V4 champion run (best mAP50: 75.4% @ epoch 93)
    └── fruit_v4_s_local/       # Phase A baseline (frozen for comparison)
```

---

## 🔄 Improvement Cycle

This project uses an iterative improvement loop until the architecture ceiling is reached:

```
Train → Evaluate (test + webcam) → Identify weak classes → Improve data → Retrain
```

### Current Cycle: V4 → V5
**Target**: Fix apple and orange webcam collapse (13%, 11% → 50%+)

**Approach**: Add `dataset_v4_webcam_train/` to training — synthetic webcam-degraded
copies of all V4 training images (JPEG compression, colour casts, blur, low-res,
vignette). This directly trains the model on bad-camera conditions.

**Build V5 dataset**:
```bash
python pipeline/09_generate_webcam_train.py
```

**Train V5**:
```bash
python train.py --name fruit_v5_quality
```

**Evaluate V5**:
```bash
python evaluate.py --model runs/fruit_v5_quality/weights/best.pt --data data_v5_webcam.yaml --split test
python evaluate.py --model runs/fruit_v5_quality/weights/best.pt --data synthetic_webcam_holdout/data_synthetic_holdout.yaml
```

### Acceptance Criteria for "Good Enough"

| Metric | V4 (current) | V5 Target |
|---|---:|---:|
| Test mAP@50 | 66.2% | ≥ 68% |
| Webcam mAP@50 | 41.7% | ≥ 58% |
| Apple webcam mAP@50 | 13.3% | ≥ 45% |
| Orange webcam mAP@50 | 10.7% | ≥ 45% |

---

## 🔧 Retrain from Scratch

### 1. Download source datasets
```bash
kaggle datasets download -d henningheyen/lvis-fruits-and-vegetables-dataset -p raw_datasets/lvis_fruits
kaggle datasets download -d lakshaytyagi01/fruit-detection -p raw_datasets/fruit_detection_kaggle
# + other sources documented in pipeline/01_source_audit.py
```

### 2. Run build pipeline (in order)
```bash
python pipeline/02_build_raw.py
python pipeline/03_prepare_quality.py
python pipeline/04_check_leakage.py
python pipeline/05_clean_leakage.py
python pipeline/06_filter_tiny_boxes.py
python pipeline/07_drop_empty_pairs.py
python pipeline/08_balance_train.py
python pipeline/09_generate_webcam_train.py   # V5 only
```

### 3. Train
```bash
python train.py --name fruit_v5_quality
```

---

## 📦 Export for Deployment

```bash
# ONNX — hardware-agnostic (Raspberry Pi 4, Jetson Nano, laptop CPU)
python export/export_onnx.py --model models/best.pt

# TFLite — Android, Coral TPU
python export/export_tflite.py --model models/best.pt
```

**Target hardware guide**:
| Hardware | Format | Expected FPS |
|---|---|---|
| RTX 3060 (GPU) | PyTorch | 212 FPS |
| Laptop CPU (no GPU) | ONNX Runtime | ~20–40 FPS |
| Raspberry Pi 4 | ONNX Runtime | ~3–6 FPS |
| Jetson Nano | TensorRT | ~25–35 FPS |
| Android phone | TFLite | ~15–25 FPS |

---

## ⚙️ CLI Reference

| Script | Key Args |
|---|---|
| `train.py` | `--name`, `--epochs`, `--batch`, `--data`, `--resume` |
| `evaluate.py` | `--model`, `--data`, `--split [test\|val]` |
| `demo.py` | `--mode [internet\|webcam]`, `--url`, `--conf`, `--model` |
| `pipeline/09_generate_webcam_train.py` | `--fraction`, `--seed`, `--no-dual-pass` |

---

## 📋 Requirements

- Python 3.9+
- CUDA GPU recommended (RTX 3060 6 GB used for development)
- See `requirements.txt` for full dependency list
