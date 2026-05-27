"""Quick profiling script to find exactly where training time is being spent."""
import time
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from custom_config import *
from custom_detector.dataset import FruitDataset, collate_fn
from custom_detector.model import FruitDetector
from custom_detector.loss import DetectionLoss, compute_class_weights
from torch.utils.data import DataLoader

def cache_subdir(split):
    return os.path.join(CACHE_DIR, split) if CACHE_DIR else None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Build dataset
train_ds = FruitDataset(
    TRAIN_IMG_DIR, TRAIN_LBL_DIR, IMG_SIZE, augment=True,
    cache_dir=cache_subdir("train"), cache_images=True,
)
loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                    collate_fn=collate_fn, pin_memory=(device.type == 'cuda'))

model = FruitDetector(NUM_CLASSES, IMG_SIZE, ANCHOR_SCALES, ANCHOR_RATIOS, FM_SIZES).to(device)
if device.type == 'cuda':
    model = model.to(memory_format=torch.channels_last)

class_weights = compute_class_weights(TRAIN_LBL_DIR, NUM_CLASSES)
criterion = DetectionLoss(
    NUM_CLASSES, POS_IOU, NEG_IOU, FOCAL_GAMMA, FOCAL_ALPHA,
    matcher_type=MATCHER_TYPE, img_size=IMG_SIZE,
    fm_sizes=FM_SIZES, ratios=ANCHOR_RATIOS,
    neg_pos_ratio=NEG_POS_RATIO,
    class_weights=class_weights
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(f"\nAnchors: {model.anchors.shape[0]}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Dataset size: {len(train_ds)}")
print(f"Batches per epoch: {len(train_ds) // BATCH_SIZE}")
print()

# Profile 5 batches
model.train()
for batch_idx, batch in enumerate(loader):
    if batch_idx >= 5:
        break

    images, boxes_list, labels_list = batch[:3]
    target_keys = batch[3] if len(batch) == 4 else None

    # Time: Data to GPU
    t0 = time.perf_counter()
    images = images.to(device, non_blocking=True)
    if device.type == 'cuda':
        images = images.contiguous(memory_format=torch.channels_last)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_data = time.perf_counter() - t0

    # Time: Forward pass
    t0 = time.perf_counter()
    cls_pred, box_pred, anchors = model(images)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_forward = time.perf_counter() - t0

    # Time: Loss computation
    t0 = time.perf_counter()
    loss_dict = criterion(cls_pred, box_pred, anchors, boxes_list, labels_list, target_keys)
    loss = loss_dict['total']
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_loss = time.perf_counter() - t0

    # Time: Backward
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_backward = time.perf_counter() - t0

    # Time: Optimizer step
    t0 = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_optim = time.perf_counter() - t0

    total = t_data + t_forward + t_loss + t_backward + t_optim
    print(f"Batch {batch_idx}: total={total*1000:.0f}ms  "
          f"data_to_gpu={t_data*1000:.0f}ms  "
          f"forward={t_forward*1000:.0f}ms  "
          f"loss={t_loss*1000:.0f}ms  "
          f"backward={t_backward*1000:.0f}ms  "
          f"optim={t_optim*1000:.0f}ms")

print("\n--- Now timing DataLoader iteration (10 batches, workers=0) ---")
t0 = time.perf_counter()
for i, batch in enumerate(loader):
    if i >= 10:
        break
t_load = time.perf_counter() - t0
print(f"10 batches loaded in {t_load:.2f}s ({t_load/10*1000:.0f}ms per batch)")

print("\n--- Now timing DataLoader iteration (10 batches, workers=4) ---")
loader4 = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                     collate_fn=collate_fn, pin_memory=(device.type == 'cuda'),
                     persistent_workers=True, prefetch_factor=2)
# warm up workers
for i, batch in enumerate(loader4):
    if i >= 2:
        break
t0 = time.perf_counter()
for i, batch in enumerate(loader4):
    if i >= 10:
        break
t_load4 = time.perf_counter() - t0
print(f"10 batches loaded in {t_load4:.2f}s ({t_load4/10*1000:.0f}ms per batch)")
