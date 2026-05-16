"""
config.py — Fruit Detection Model: Single source of truth for all hyperparameters.

Every script (train.py, evaluate.py, demo.py) imports constants from here.
Change a value here once; it propagates everywhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths  (all relative to this file → portable across machines)
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent
DATA_YAML   = ROOT / "data_v3.yaml"
DATASET_DIR = ROOT / "dataset_v3"
RUNS_DIR    = ROOT / "runs"
MODELS_DIR  = ROOT / "models"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# yolov8s: 11M params — chosen over nano (too small, failed on webcam) and
# medium (safe for 6 GB VRAM but trains slower with no meaningful gain at
# this dataset scale).
BASE_MODEL = "yolov8s.pt"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
EPOCHS       = 120
IMGSZ        = 640
BATCH        = 16        # 16 uses ~2.5GB VRAM, much faster than AutoBatch's conservative estimate
PATIENCE     = 25        # early-stop if mAP50 doesn't improve for 25 epochs
SAVE_PERIOD  = 10        # save a checkpoint every N epochs
RUN_NAME     = "fruit_v3"

# ---------------------------------------------------------------------------
# Augmentation profile  (webcam-optimised)
#
# Goal: make the model robust to the conditions it will actually face:
#   • dark rooms / yellow-tinted lighting  → hsv jitter
#   • fruit held at angles                 → rotation
#   • varying distance from camera         → scale
#   • partial hand occlusion               → random erasing
#   • multiple fruits in frame             → mosaic
#   • minority class generalisation        → mixup
#
# Fruit is never upside-down in practice → flipud=0.0
# ---------------------------------------------------------------------------
AUGMENT_KWARGS = {
    "hsv_h":       0.020,   # hue jitter: handles colour casts
    "hsv_s":       0.80,    # saturation: over/under-saturated webcam
    "hsv_v":       0.50,    # brightness: dark rooms, backlit windows
    "degrees":     12,      # rotation: tilted fruit / tilted camera
    "translate":   0.12,    # translation: partial in-frame fruit
    "scale":       0.60,    # zoom: different distances from camera
    "shear":       4.0,     # slight perspective distortion
    "perspective": 0.0003,
    "flipud":      0.0,     # fruit is never upside-down
    "fliplr":      0.5,     # horizontal flip: fine for fruit
    "mosaic":      1.0,     # always on: multi-object scenes (bowl of fruit)
    "mixup":       0.15,    # helps minority classes
    "erasing":     0.40,    # simulates hand occlusion
}

# ---------------------------------------------------------------------------
# Quality training settings (always applied, not augmentation-specific)
# ---------------------------------------------------------------------------
TRAIN_QUALITY_KWARGS = {
    "cos_lr":       True,  # cosine LR decay: smoother convergence than step decay
    "close_mosaic": 10,   # disable mosaic for final 10 epochs (fine-tuning stability)
    "val":          True,  # always validate each epoch
    "cache":        "disk", # disk cache on E: (216 GB free) — speeds up epoch 2+
}
