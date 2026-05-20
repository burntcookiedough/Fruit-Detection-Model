"""
config.py — Fruit Detection Model: Single source of truth for all hyperparameters.

Every script (train.py, evaluate.py, demo.py) imports constants from here.
Change a value here once; it propagates everywhere.

Versions
--------
V4  (fruit_v4_quality)  — YOLOv8s, 120 epochs, full V4 augmentation
                          Best val mAP50: 75.43% @ epoch 93
                          Test mAP50: 66.19%  |  Webcam stress mAP50: 41.66%

V5  (fruit_v5_quality)  — YOLOv8s, stronger HSV augmentation targeting
                          webcam colour-cast robustness.  Next run.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths  (all relative to this file → portable across machines)
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent
RUNS_DIR   = ROOT / "runs"
MODELS_DIR = ROOT / "models"

# Canonical dataset (V4 balanced, leakage-free, 8-class)
DATA_YAML  = ROOT / "data_v5_webcam.yaml"    # V5: clean + webcam-degraded train

# Champion weights — always the best model produced so far
CHAMPION_MODEL = MODELS_DIR / "best.pt"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# YOLOv8s: 11M params — best balance of accuracy and webcam deployability.
# On local RTX 3060: batch=8 keeps VRAM under 5 GB.
BASE_MODEL = "yolov8s.pt"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
EPOCHS      = 120
IMGSZ       = 640
BATCH       = 8        # 8 keeps VRAM safe on local RTX 3060 6 GB
PATIENCE    = 30       # V5: increased from 25 — plateau was very flat in V4
SAVE_PERIOD = 10
RUN_NAME    = "fruit_v5_quality"

# ---------------------------------------------------------------------------
# V5 Augmentation profile  (webcam-optimised, colour-cast hardened)
#
# Key changes vs V4:
#   hsv_h  0.020 → 0.040  wider hue jitter → survives yellow/cool casts
#   hsv_s  0.80  → 0.90   more saturation variation → greyed-out webcam
#   hsv_v  0.50  → 0.65   more brightness variation → dim rooms
#   erasing 0.40 → 0.50   more occlusion → forces shape/texture learning
#   close_mosaic 10 → 15  more stable final fine-tuning phase
#
# Rationale:
#   V4 evaluation showed apple (13%) and orange (11%) catastrophically fail
#   under webcam colour degradation — model learned colour shortcuts.
#   Forcing stronger colour randomisation makes it learn shape instead.
# ---------------------------------------------------------------------------
AUGMENT_KWARGS = {
    "hsv_h":       0.040,   # V5: 0.020→0.040 — wider hue: survives colour casts
    "hsv_s":       0.90,    # V5: 0.80→0.90  — greyed/oversaturated webcam
    "hsv_v":       0.65,    # V5: 0.50→0.65  — dark rooms, backlit windows
    "degrees":     12,      # rotation: tilted fruit / tilted camera
    "translate":   0.12,    # translation: partial in-frame fruit
    "scale":       0.60,    # zoom: different distances from camera
    "shear":       4.0,     # slight perspective distortion
    "perspective": 0.0003,
    "flipud":      0.0,     # fruit is never upside-down
    "fliplr":      0.5,     # horizontal flip: fine for fruit
    "mosaic":      1.0,     # always on: multi-object scenes
    "mixup":       0.15,    # helps minority classes
    "erasing":     0.50,    # V5: 0.40→0.50 — more occlusion → shape learning
}

# ---------------------------------------------------------------------------
# Quality training settings (always applied, not augmentation-specific)
# ---------------------------------------------------------------------------
TRAIN_QUALITY_KWARGS = {
    "cos_lr":       True,   # cosine LR decay: smoother convergence
    "close_mosaic": 15,     # V5: 10→15 — disable mosaic earlier for stability
    "val":          True,   # always validate each epoch
    "cache":        "disk", # fastest local training; needs ~45 GB free
}

# ---------------------------------------------------------------------------
# V4 augmentation profile — kept for reference / ablation comparison
# ---------------------------------------------------------------------------
AUGMENT_KWARGS_V4 = {
    "hsv_h":       0.020,
    "hsv_s":       0.80,
    "hsv_v":       0.50,
    "degrees":     12,
    "translate":   0.12,
    "scale":       0.60,
    "shear":       4.0,
    "perspective": 0.0003,
    "flipud":      0.0,
    "fliplr":      0.5,
    "mosaic":      1.0,
    "mixup":       0.15,
    "erasing":     0.40,
}
