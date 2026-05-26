"""All hyperparameters in one place."""
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset_v4_balanced")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATA_DIR, "train", "labels")
VAL_IMG_DIR = os.path.join(DATA_DIR, "valid", "images")
VAL_LBL_DIR = os.path.join(DATA_DIR, "valid", "labels")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "images")
TEST_LBL_DIR = os.path.join(DATA_DIR, "test", "labels")

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_custom", "fruit_v1")
WEIGHTS_DIR = os.path.join(RUNS_DIR, "weights")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_320")
TARGET_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "target_cache_320")

IMG_SIZE = 320
NUM_CLASSES = 8
BATCH_SIZE = 16
NUM_EPOCHS = 50
LR = 1e-3
WARMUP_EPOCHS = 3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
NEG_POS_RATIO = 100
POS_IOU = 0.5
NEG_IOU = 0.4
CONF_THRESH = 0.05
NMS_IOU = 0.45
PRE_NMS_TOPK = 1000
MAX_DETECTIONS = 100
MATCHER_TYPE = "grid"  # "grid" is fast enough to train; "iou" is exact but slow.
ANCHOR_SCALES = [48, 96, 192]
ANCHOR_RATIOS = [0.5, 1.0, 2.0]
NUM_ANCHOR_RATIOS = len(ANCHOR_RATIOS)
FM_SIZES = [40, 20, 10]
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True

CLASS_NAMES = [
    "apple", "banana", "orange", "mango",
    "pineapple", "watermelon", "grapes", "pomegranate",
]
