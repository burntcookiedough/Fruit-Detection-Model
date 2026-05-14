# 🍎 Fruit Detection Model

A lightweight, real-world fruit detection system built with **YOLOv8n** (Nano) and trained on a merged dataset of 8,630 real-world images. Designed for fast inference and embedded hardware deployment.

---

## 📦 Classes Detected

| # | Fruit | mAP@50 |
|---|-------|--------|
| 0 | 🍎 Apple | 68.6% |
| 1 | 🍌 Banana | 62.2% |
| 2 | 🍊 Orange | 71.4% |
| 3 | 🥭 Mango | 82.1% |
| 4 | 🍍 Pineapple | 92.1% |
| 5 | 🍉 Watermelon | 82.1% |
| 6 | 🍇 Grapes | 63.7% |
| 7 | 🍎 Pomegranate | 85.9% |

---

## 🏆 Benchmark Results (Held-out Test Set — 863 real-world images)

| Metric | Score |
|--------|-------|
| **mAP@50** | **76.0%** |
| **mAP@50-95** | **63.4%** |
| **Precision** | **85.1%** |
| **Recall** | **70.6%** |
| **Inference Speed** | **4.7ms/image (GPU)** |

> Model: YOLOv8n — 6MB, 212 FPS on GPU  
> Hardware: RTX 3060 6GB  
> Training: 80 epochs, ~112 minutes  

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

> **Note:** The base pretrained `yolov8n.pt` is not included. It is auto-downloaded by ultralytics on first run.

### 2. Run inference on a random internet image

```bash
python demo.py
```

### 3. Run inference on a specific image URL

```bash
python demo.py --url "https://example.com/fruit.jpg"
```

### 4. Run live webcam inference

```bash
python demo.py --mode webcam
```

### 5. Evaluate on the test set

```bash
python evaluate.py --model models/best.pt --data data_v2.yaml
```

---

## 📁 Project Structure

```
Fruit Detection Model/
├── demo.py                  # Live demo: internet images & webcam
├── train.py                 # Training script (YOLOv8n fine-tuning)
├── evaluate.py              # Evaluate model on test split
├── prepare_dataset.py       # Dataset prep v1 (original synthetic)
├── prepare_dataset_v2.py    # Dataset prep v2 (real-world Kaggle + LVIS)
├── data.yaml                # Dataset config v1
├── data_v2.yaml             # Dataset config v2 (used for training)
├── requirements.txt         # Python dependencies
├── models/
│   ├── best.pt              # Current best model (v2, real-world trained)
│   ├── best_v1.pt           # Old model (v1, synthetic trained)
│   └── best.onnx            # ONNX export for edge deployment
├── export/
│   ├── export_onnx.py       # Export to ONNX format
│   └── export_tflite.py     # Export to TFLite format
└── inference/
    ├── image.py             # Batch inference on local images
    └── webcam.py            # Standalone webcam inference
```

---

## 🔁 Retrain from Scratch

### 1. Download datasets

```bash
kaggle datasets download -d henningheyen/lvis-fruits-and-vegetables-dataset -p raw_datasets/lvis_fruits
kaggle datasets download -d lakshaytyagi01/fruit-detection -p raw_datasets/fruit_detection_kaggle
```

### 2. Prepare the merged dataset

```bash
python prepare_dataset_v2.py
```

### 3. Train

```bash
python train.py --epochs 80 --batch 16
```

---

## 📤 Export for Deployment

```bash
# ONNX (hardware-agnostic, Raspberry Pi, Jetson, etc.)
python export/export_onnx.py

# TFLite (Android, microcontrollers)
python export/export_tflite.py
```

---

## 🛠️ CLI Reference

| Script | Key Args |
|--------|----------|
| `demo.py` | `--mode [internet\|webcam]`, `--url`, `--conf`, `--imgsz`, `--model` |
| `train.py` | `--epochs`, `--batch`, `--data`, `--device`, `--name` |
| `evaluate.py` | `--model`, `--data`, `--split` |

---

## 📋 Requirements

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU
- See `requirements.txt` for full dependency list

---

## 📝 Notes

- The model was fine-tuned from `yolov8n.pt` (COCO pretrained) using transfer learning.
- Training data: **8,630 images** merged from LVIS, Kaggle Fruit Detection, and Roboflow datasets.
- The v1 model (`best_v1.pt`) was trained on ~2,200 synthetic/photoshopped images and is kept for comparison only.
