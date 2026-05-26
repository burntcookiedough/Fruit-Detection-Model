"""Training loop."""
import argparse
import csv
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from custom_config import (
    TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR, TEST_IMG_DIR, TEST_LBL_DIR,
    IMG_SIZE, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, LR,
    WARMUP_EPOCHS, WEIGHT_DECAY, GRAD_CLIP,
    RUNS_DIR, WEIGHTS_DIR, CACHE_DIR, TARGET_CACHE_DIR,
    NUM_WORKERS, PREFETCH_FACTOR, PERSISTENT_WORKERS,
    ANCHOR_SCALES, ANCHOR_RATIOS, FM_SIZES, FOCAL_GAMMA, FOCAL_ALPHA,
    POS_IOU, NEG_IOU, NEG_POS_RATIO, MATCHER_TYPE, PRE_NMS_TOPK, MAX_DETECTIONS,
    CLASS_NAMES
)
from custom_detector.dataset import FruitDataset, collate_fn
from custom_detector.model import FruitDetector
from custom_detector.loss import DetectionLoss
from custom_detector.utils import cxcywh_to_xyxy
import torchvision.ops as ops


def decode_predictions(cls_pred, box_pred, anchors, conf_thresh=0.05, nms_iou=0.45,
                       pre_nms_topk=1000, max_detections=100):
    """Decode one image's predictions."""
    device = cls_pred.device
    scores = torch.sigmoid(cls_pred)
    max_scores, labels = scores.max(dim=1)
    keep = max_scores > conf_thresh
    if keep.sum() == 0:
        return torch.zeros((0, 4), device=device), torch.zeros((0,), device=device, dtype=torch.long), torch.zeros((0,), device=device)

    box_pred = box_pred[keep]
    labels = labels[keep]
    max_scores = max_scores[keep]
    anchors_k = anchors[keep]

    if max_scores.numel() > pre_nms_topk:
        top_scores, top_idx = max_scores.topk(pre_nms_topk)
        box_pred = box_pred[top_idx]
        labels = labels[top_idx]
        anchors_k = anchors_k[top_idx]
        max_scores = top_scores

    tx, ty, tw, th = box_pred[:, 0], box_pred[:, 1], box_pred[:, 2], box_pred[:, 3]
    tw = tw.clamp(min=-8.0, max=8.0)
    th = th.clamp(min=-8.0, max=8.0)
    cx = tx * anchors_k[:, 2] + anchors_k[:, 0]
    cy = ty * anchors_k[:, 3] + anchors_k[:, 1]
    w = torch.exp(tw) * anchors_k[:, 2]
    h = torch.exp(th) * anchors_k[:, 3]
    boxes = torch.stack([cx, cy, w, h], dim=1)
    boxes_xyxy = cxcywh_to_xyxy(boxes).clamp(min=0, max=IMG_SIZE)

    nms_keep = ops.batched_nms(boxes_xyxy, max_scores, labels, nms_iou)
    nms_keep = nms_keep[:max_detections]
    return boxes_xyxy[nms_keep], labels[nms_keep], max_scores[nms_keep]


def unpack_batch(batch):
    if len(batch) == 4:
        images, boxes_list, labels_list, target_keys = batch
    else:
        images, boxes_list, labels_list = batch
        target_keys = None
    return images, boxes_list, labels_list, target_keys


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip, max_batches=None):
    model.train()
    total_loss = 0
    total_cls = 0
    total_box = 0
    num_batches = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, boxes_list, labels_list, target_keys = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        if device.type == 'cuda':
            images = images.contiguous(memory_format=torch.channels_last)
        cls_pred, box_pred, anchors = model(images)
        loss_dict = criterion(cls_pred, box_pred, anchors, boxes_list, labels_list, target_keys)
        loss = loss_dict['total']

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_cls += loss_dict['cls'].item()
        total_box += loss_dict['box'].item()
        num_batches += 1
    return total_loss / num_batches, total_cls / num_batches, total_box / num_batches


@torch.no_grad()
def validate(model, loader, device, anchors, max_batches=None):
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError as exc:
        raise RuntimeError("Install torchmetrics before validation: pip install torchmetrics") from exc

    model.eval()
    metric = MeanAveragePrecision(iou_type='bbox', class_metrics=False)
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, boxes_list, labels_list, _ = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        if device.type == 'cuda':
            images = images.contiguous(memory_format=torch.channels_last)
        cls_pred, box_pred, _ = model(images)
        for b in range(images.shape[0]):
            pred_boxes, pred_labels, pred_scores = decode_predictions(
                cls_pred[b], box_pred[b], anchors,
                pre_nms_topk=PRE_NMS_TOPK, max_detections=MAX_DETECTIONS
            )
            preds = [{
                'boxes': pred_boxes.cpu(),
                'scores': pred_scores.cpu(),
                'labels': pred_labels.cpu(),
            }]
            targets = [{
                'boxes': cxcywh_to_xyxy(boxes_list[b]).cpu(),
                'labels': labels_list[b].cpu(),
            }]
            metric.update(preds, targets)
    return metric.compute()


def cache_subdir(split_name):
    return os.path.join(CACHE_DIR, split_name)


def target_cache_subdir(split_name):
    fm = "-".join(str(x) for x in FM_SIZES)
    scales = "-".join(str(x) for x in ANCHOR_SCALES)
    ratios = "-".join(str(x).replace(".", "p") for x in ANCHOR_RATIOS)
    signature = f"img{IMG_SIZE}_{MATCHER_TYPE}_fm{fm}_s{scales}_r{ratios}"
    return os.path.join(TARGET_CACHE_DIR, signature, split_name)


def build_image_caches(overwrite=False):
    specs = [
        ("train", TRAIN_IMG_DIR, TRAIN_LBL_DIR),
        ("valid", VAL_IMG_DIR, VAL_LBL_DIR),
        ("test", TEST_IMG_DIR, TEST_LBL_DIR),
    ]
    for split_name, img_dir, lbl_dir in specs:
        print(f"Building {split_name} cache...")
        ds = FruitDataset(
            img_dir, lbl_dir, IMG_SIZE, augment=False,
            cache_dir=cache_subdir(split_name), cache_images=False
        )
        stats = ds.build_cache(overwrite=overwrite)
        print(f"  {split_name}: written={stats['written']} skipped={stats['skipped']} total={stats['total']}")


def make_loader(dataset, batch_size, shuffle, workers, pin_memory, prefetch_factor, persistent_workers):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(dataset, **kwargs)


def resolve_workers(dataset, requested_workers, pin_memory, prefetch_factor, persistent_workers):
    if requested_workers <= 0:
        return 0
    candidates = [requested_workers]
    if requested_workers > 2:
        candidates.append(2)
    candidates.append(0)
    for workers in candidates:
        if workers == 0:
            return 0
        try:
            probe = make_loader(
                dataset, batch_size=1, shuffle=False, workers=workers,
                pin_memory=pin_memory, prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers
            )
            next(iter(probe))
            return workers
        except Exception as exc:
            print(f"DataLoader workers={workers} failed. Reason: {exc}")
    return 0


def save_history_row(path, row):
    exists = os.path.exists(path)
    fieldnames = [
        "epoch", "lr", "loss", "cls_loss", "box_loss",
        "map50", "map", "epoch_seconds"
    ]
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def config_snapshot(args):
    arg_values = vars(args) if args is not None else {}
    return {
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "batch_size": BATCH_SIZE,
        "num_epochs": arg_values.get("epochs", NUM_EPOCHS),
        "lr": LR,
        "warmup_epochs": WARMUP_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "anchor_scales": ANCHOR_SCALES,
        "anchor_ratios": ANCHOR_RATIOS,
        "fm_sizes": FM_SIZES,
        "matcher_type": MATCHER_TYPE,
        "neg_pos_ratio": NEG_POS_RATIO,
        "cache_images": arg_values.get("cache_images", False),
        "cache_targets": arg_values.get("cache_targets", False),
        "workers": arg_values.get("workers", NUM_WORKERS),
        "prefetch_factor": arg_values.get("prefetch_factor", PREFETCH_FACTOR),
        "persistent_workers": arg_values.get("persistent_workers", PERSISTENT_WORKERS),
        "val_every": arg_values.get("val_every", 1),
        "command_args": arg_values,
    }


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_map, best_loss, args):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_map50': best_map,
        'best_loss': best_loss,
        'config': config_snapshot(args),
    }, path)


def main(num_epochs=NUM_EPOCHS, resume='', limit_train_batches=None, limit_val_batches=None,
         val_every=1, skip_val=False, cache_images=False, workers=NUM_WORKERS,
         prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS,
         build_cache=False, overwrite_cache=False, cache_targets=False, args=None):
    if limit_train_batches is not None and limit_train_batches < 1:
        raise ValueError("--limit-train-batches must be >= 1")
    if limit_val_batches is not None and limit_val_batches < 1:
        raise ValueError("--limit-val-batches must be >= 1")
    if val_every < 1:
        raise ValueError("--val-every must be >= 1")

    if build_cache:
        build_image_caches(overwrite=overwrite_cache)
        return

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    train_ds = FruitDataset(
        TRAIN_IMG_DIR, TRAIN_LBL_DIR, IMG_SIZE, augment=True,
        cache_dir=cache_subdir("train"), cache_images=cache_images,
        return_meta=cache_targets
    )
    val_ds = FruitDataset(
        VAL_IMG_DIR, VAL_LBL_DIR, IMG_SIZE, augment=False,
        cache_dir=cache_subdir("valid"), cache_images=cache_images
    )

    pin_memory = device.type == 'cuda'
    workers = resolve_workers(train_ds, workers, pin_memory, prefetch_factor, persistent_workers)
    print(f"DataLoader workers: {workers}")
    train_loader = make_loader(train_ds, BATCH_SIZE, True, workers, pin_memory, prefetch_factor, persistent_workers)
    val_loader = make_loader(val_ds, BATCH_SIZE, False, workers, pin_memory, prefetch_factor, persistent_workers)

    model = FruitDetector(NUM_CLASSES, IMG_SIZE, ANCHOR_SCALES, ANCHOR_RATIOS, FM_SIZES).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = DetectionLoss(
        NUM_CLASSES, POS_IOU, NEG_IOU, FOCAL_GAMMA, FOCAL_ALPHA,
        matcher_type=MATCHER_TYPE, img_size=IMG_SIZE,
        fm_sizes=FM_SIZES, ratios=ANCHOR_RATIOS,
        neg_pos_ratio=NEG_POS_RATIO,
        cache_targets=cache_targets,
        target_cache_dir=target_cache_subdir("train")
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup_epochs = min(WARMUP_EPOCHS, max(num_epochs - 1, 0))
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(num_epochs - warmup_epochs, 1), eta_min=LR * 0.01)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(num_epochs, 1), eta_min=LR * 0.01)

    best_map = 0.0
    best_loss = float('inf')
    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except (KeyError, ValueError):
            print("Warning: scheduler state did not match this run configuration; scheduler was reset.")
        start_epoch = ckpt['epoch'] + 1
        best_map = ckpt.get('best_map50', 0.0)
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f"Resumed {resume} from epoch {start_epoch}")

    anchors = model.anchors.to(device)
    history_path = os.path.join(RUNS_DIR, "history.csv")
    if args is not None:
        with open(os.path.join(RUNS_DIR, "config_snapshot.json"), "w") as f:
            json.dump(config_snapshot(args), f, indent=2)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.perf_counter()
        loss, cls_loss, box_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, GRAD_CLIP,
            max_batches=limit_train_batches
        )
        scheduler.step()

        should_validate = (not skip_val) and ((epoch + 1) % val_every == 0 or epoch == num_epochs - 1)
        if should_validate:
            try:
                result = validate(model, val_loader, device, anchors, max_batches=limit_val_batches)
                map50 = result['map_50'].item()
                map_val = result['map'].item()
            except RuntimeError as exc:
                if 'torchmetrics' not in str(exc):
                    raise
                print("Validation skipped: install torchmetrics for mAP (`pip install torchmetrics`).")
                skip_val = True
                map50 = best_map
                map_val = float('nan')
        else:
            map50 = best_map
            map_val = float('nan')

        epoch_seconds = time.perf_counter() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | loss={loss:.4f} cls={cls_loss:.4f} box={box_loss:.4f} | mAP50={map50:.4f} mAP={map_val:.4f} | time={epoch_seconds:.1f}s")
        save_history_row(history_path, {
            "epoch": epoch + 1,
            "lr": lr,
            "loss": loss,
            "cls_loss": cls_loss,
            "box_loss": box_loss,
            "map50": map50,
            "map": map_val,
            "epoch_seconds": epoch_seconds,
        })

        if loss < best_loss:
            best_loss = loss
            save_checkpoint(os.path.join(WEIGHTS_DIR, 'best_loss.pt'), epoch, model, optimizer, scheduler, best_map, best_loss, args)

        best_map_path = os.path.join(WEIGHTS_DIR, 'best_map50.pt')
        if should_validate and (map50 > best_map or not os.path.exists(best_map_path)):
            best_map = map50
            save_checkpoint(best_map_path, epoch, model, optimizer, scheduler, best_map, best_loss, args)
            print(f"  -> New best mAP50: {best_map:.4f}")

        last_path = os.path.join(WEIGHTS_DIR, 'last.pt')
        save_checkpoint(last_path, epoch, model, optimizer, scheduler, best_map, best_loss, args)

    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--limit-train-batches', type=int, default=None)
    parser.add_argument('--limit-val-batches', type=int, default=None)
    parser.add_argument('--val-every', type=int, default=1)
    parser.add_argument('--skip-val', action='store_true')
    parser.add_argument('--cache-images', action='store_true')
    parser.add_argument('--cache-targets', action='store_true')
    parser.add_argument('--build-cache', action='store_true')
    parser.add_argument('--overwrite-cache', action='store_true')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--prefetch-factor', type=int, default=PREFETCH_FACTOR)
    parser.add_argument('--persistent-workers', action=argparse.BooleanOptionalAction, default=PERSISTENT_WORKERS)
    args = parser.parse_args()
    main(args.epochs, args.resume, args.limit_train_batches, args.limit_val_batches,
         args.val_every, args.skip_val, args.cache_images, args.workers,
         args.prefetch_factor, args.persistent_workers, args.build_cache,
         args.overwrite_cache, args.cache_targets, args)
