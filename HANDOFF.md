# 🤝 Agent Handoff Document — Fruit Detection Model
**Date:** 2026-05-14  
**Project Root:** `f:\Fruit Detection Model\`  
**Git Repo:** `burntcookiedough/Fruit-Detection-Model` (local, on main branch)  
**Status:** ✅ Trained, benchmarked, committed. Ready for improvement or deployment.

---

## 1. What This Project Is

A **real-world fruit detection system** using YOLOv8n (Nano), capable of detecting 8 fruits in photos, webcam feeds, or internet images. Built with:
- **Framework:** Ultralytics YOLOv8
- **Model:** YOLOv8n (Nano) — fine-tuned via transfer learning from COCO weights
- **Training Hardware:** NVIDIA RTX 3060 6GB
- **Target Deployment:** Edge hardware (Raspberry Pi, Jetson Nano) via ONNX/TFLite

### 8 Target Classes (in order, 0-indexed)
```
0: apple | 1: banana | 2: orange | 3: mango
4: pineapple | 5: watermelon | 6: grapes | 7: pomegranate
```

---

## 2. What Has Been Done (Full History)

### Phase 1 — Synthetic Baseline (DONE, superseded)
- Trained YOLOv8n on ~2,200 synthetic/photoshopped Roboflow images
- Achieved mAP@50 = 0.80 on synthetic test set
- **Problem discovered:** Model completely failed on real-world photos due to domain shift — it had learned photoshop artifact patterns, not real fruit features
- Old model saved as: `models/best_v1.pt`

### Phase 2 — Real-World Dataset & Retraining (DONE ✅)
**Dataset sources merged:**
| Source | Images | Notes |
|--------|--------|-------|
| Kaggle: `lakshaytyagi01/fruit-detection` | 6,392 (of 8,479) | 6 classes, real photos, 640×640 |
| LVIS Fruits & Vegetables (`henningheyen`) | 53 | Real COCO images, 63 classes |
| Roboflow synthetic (3 datasets) | 2,185 | Kept for class diversity (mango/pomegranate only appear here) |
| **Total** | **8,630** | Filtered, deduped, split 70/20/10 |

**Training run:** 80 epochs, ~112 minutes, RTX 3060 6GB  
**Best checkpoint saved to:** `models/best.pt`

### Phase 3 — Benchmark (DONE ✅)
Evaluated on 863 completely unseen real-world test images:

| Metric | Score |
|--------|-------|
| mAP@50 | **76.0%** |
| mAP@50-95 | **63.4%** |
| Precision | **85.1%** |
| Recall | **70.6%** |
| Inference Speed | **4.7ms/image on GPU (212 FPS)** |

Per-class mAP@50:
```
pineapple:   92.1%   (best)
pomegranate: 85.9%
mango:       82.1%
watermelon:  82.1%
orange:      71.4%
apple:       68.6%
grapes:      63.7%
banana:      62.2%   (worst — look-alike with lighting)
```

### Phase 4 — Infrastructure (DONE ✅)
- `.gitignore` created (excludes datasets, venv, runs/)
- `README.md` written with full docs
- Initial git commit made with 18 files including all model weights

---

## 3. Current State of Each File

```
f:\Fruit Detection Model\
├── demo.py                  ✅ Works. Internet + webcam modes. PyTorch 2.6 fix applied.
├── train.py                 ✅ Works. Defaults to data_v2.yaml, 80 epochs, fruit_v2 run.
├── evaluate.py              ✅ Works. PyTorch 2.6 fix applied.
├── prepare_dataset.py       ✅ Works. (v1, for the old synthetic datasets)
├── prepare_dataset_v2.py    ✅ Works. Downloads, unzips, merges, deduplicates, splits.
├── data.yaml                ✅ For v1 synthetic dataset (dataset/)
├── data_v2.yaml             ✅ For v2 real-world dataset (dataset_v2/) ← USE THIS
├── requirements.txt         ✅ All deps listed
├── .gitignore               ✅ Correct — excludes heavy files
├── README.md                ✅ Full project documentation
├── models/
│   ├── best.pt              ✅ The CURRENT production model (v2, real-world trained)
│   ├── best_v1.pt           📦 Old synthetic model — keep for comparison
│   ├── best.onnx            ✅ ONNX export of v2 model (11.7MB, for deployment)
│   └── last.pt              📦 Last epoch checkpoint from v2 training run
├── export/
│   ├── export_onnx.py       ✅ Works
│   └── export_tflite.py     ✅ Works (requires tensorflow installed separately)
├── inference/
│   ├── image.py             ✅ Batch inference on local image files
│   └── webcam.py            ✅ Standalone webcam inference script
├── runs/fruit_v2/           🚫 Git-ignored. Contains training plots, confusion matrix.
├── dataset_v2/              🚫 Git-ignored. Can regenerate with prepare_dataset_v2.py
├── raw_datasets/            🚫 Git-ignored. Contains Kaggle ZIPs + extracted folders.
└── venv/                    🚫 Git-ignored. Python 3.x virtual environment.
```

---

## 4. Critical Technical Details (Must Know)

### PyTorch 2.6 Compatibility Fix
PyTorch 2.6 changed `torch.load()` to default `weights_only=True`, which breaks Ultralytics model loading. **All scripts already have this patch applied at the top:**
```python
import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
```
If adding new scripts that load the YOLO model, **always include this block before importing YOLO**.

### Windows-Specific Settings
- All scripts use **ASCII-only output** (no Unicode em dashes, checkmarks, etc.) — Windows terminal breaks on them.
- `train.py` uses `workers=2` (not default 8) to prevent Windows multiprocessing deadlocks.
- All paths use `Path(__file__).resolve()` for absolute paths — YOLO needs absolute `data.yaml` paths.

### Dataset v2 Structure
```
dataset_v2/
├── train/images/ + train/labels/    (6,041 images, 70%)
├── valid/images/ + valid/labels/    (1,726 images, 20%)
└── test/images + test/labels/       (863 images, 10%)
```
Label format: YOLO normalized `class_id cx cy w h` (values 0–1).

### Class Index Mapping
This is critical. If you add new data, labels MUST use these exact indices:
```
0=apple, 1=banana, 2=orange, 3=mango, 4=pineapple, 5=watermelon, 6=grapes, 7=pomegranate
```

---

## 5. Environment Setup (for new machine)

```powershell
cd "f:\Fruit Detection Model"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**To re-download Kaggle datasets (if dataset_v2/ is missing):**
```powershell
# Requires kaggle CLI configured with API key (~/.kaggle/kaggle.json)
python prepare_dataset_v2.py
# This auto-downloads, unzips, filters, merges, and splits everything
```

**Base model weights** (`yolov8n.pt`) are auto-downloaded by ultralytics on first `train.py` run.

---

## 6. How to Run Everything

```powershell
# Activate venv first
venv\Scripts\activate

# Test on random internet image
python demo.py

# Test on specific URL
python demo.py --url "https://example.com/apple.jpg"

# Live webcam (lower conf for better detection)
python demo.py --mode webcam --conf 0.10 --imgsz 320

# Evaluate on test set
python evaluate.py --model models/best.pt --data data_v2.yaml

# Retrain from scratch (uses data_v2.yaml by default)
python train.py

# Export to ONNX
python export/export_onnx.py
```

---

## 7. Known Issues & Limitations

### 🔴 Critical
| Issue | Details | Fix |
|-------|---------|-----|
| **Webcam performance is poor** | Training data = studio/catalog photos. Webcam images = noisy, dark, cluttered. The model hasn't seen "webcam-style" frames. | See roadmap item #1 below |
| **Mango & Pomegranate underrepresented** | Only ~260 bounding boxes each vs 5,000+ for apple/grapes. These come ONLY from Roboflow synthetic data. | See roadmap item #2 |
| **Grapes & Banana lowest accuracy** | Grapes cluster tightly (hard to box individually), banana look changes dramatically with lighting | More training data needed |

### 🟡 Minor
| Issue | Details |
|-------|---------|
| One Wikimedia URL is 404 | `Fruit_bowl.jpg` in `SAMPLE_URLS` list in `demo.py` returns 404. Remove or replace it. |
| `last.pt` is in git | Could be removed (it's just the final checkpoint, `best.pt` is more useful). |

---

## 8. Roadmap — What To Do Next (Prioritized)

### Priority 1 — Fix Webcam Performance 🎯
**Problem:** Training images are clean studio shots. Webcam is noisy, dark, in-scene.  
**Solution options:**
- Add **Roboflow "fruits in hand" / "fruits on table"** datasets — images that look like webcam frames
- Apply heavy **augmentation** during training: brightness jitter, blur, noise, perspective
- In `train.py`, add augmentation args: `hsv_h=0.015, hsv_s=0.7, blur=0.3, degrees=10`

```python
# In train.py model.train() call, add:
hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
degrees=10, translate=0.1, scale=0.5,
blur=0.3, noise=0.01
```

### Priority 2 — More Mango & Pomegranate Data
These are the most underrepresented classes.  
**Sources to try:**
- `kaggle datasets download -d sshikamaru/fruit-recognition` (has mango)
- HuggingFace: search "mango detection dataset"
- `prepare_dataset_v2.py` is already set up to merge new sources — just add them to the `dataset_dirs` list

### Priority 3 — Export & Deploy
The model is already ONNX-exported (`models/best.onnx`, 11.7MB).  
Next: test on target hardware.
```powershell
# For Raspberry Pi / Jetson Nano
python export/export_tflite.py    # Requires tensorflow, produces .tflite

# Test ONNX inference speed
python -c "from ultralytics import YOLO; m=YOLO('models/best.onnx'); m.info()"
```

### Priority 4 — Push to GitHub/HuggingFace
Currently only local git. To push:
```powershell
git remote add origin https://github.com/YOUR_USERNAME/fruit-detection-model.git
git push -u origin main
```
For model weights on HuggingFace Hub:
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(path_or_fileobj="models/best.pt", path_in_repo="best.pt", repo_id="YOUR_USERNAME/fruit-detection")
```

### Priority 5 — Per-Class Confidence Thresholds
Currently a global `--conf 0.25` threshold. Mango/pomegranate (low-data classes) might benefit from a lower threshold. Consider implementing per-class thresholds in `demo.py`.

---

## 9. Quick Comparison: v1 vs v2

| | v1 (Synthetic) | v2 (Real-World) |
|--|--|--|
| Training images | 2,200 synthetic | 8,630 real photos |
| mAP@50 (test set) | 0.80 (synthetic test) | **0.76** (real-world test) |
| Real-world performance | ❌ Fails completely | ✅ Works |
| Inference speed | 4.7ms | 4.7ms (same) |
| Model file | models/best_v1.pt | models/best.pt |

The v1 number looks higher, but it was measured on a synthetic test set — an unfair comparison. The v2 model is the only deployable one.

---

## 10. Things NOT to Break

1. **Do not change class indices.** Everything (labels, data.yaml, demo.py color map) is tied to `0=apple...7=pomegranate`.
2. **Always include the PyTorch 2.6 patch** in any new script that loads the model.
3. **Use `workers=2`** in any future `model.train()` calls on Windows.
4. **Keep ASCII-only output** — no Unicode in print statements.
5. **Use absolute paths** for `data.yaml` when calling `model.train(data=...)`.
