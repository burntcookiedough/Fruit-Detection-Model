"""
generate_synthetic_webcam_holdout.py
====================================
Create a temporary synthetic webcam-style holdout from existing labeled V4
validation/test images.

This dataset is only for stress testing and early training confidence. It is
not a replacement for real webcam_holdout/ images.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["apple", "banana", "orange", "mango", "pineapple", "watermelon", "grapes", "pomegranate"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def remove_tree(path: Path) -> None:
    def onexc(func, p, exc):
        try:
            os.chmod(p, 0o700)
            func(p)
        except OSError:
            pass

    for _ in range(5):
        try:
            shutil.rmtree(path, onexc=onexc)
        except TypeError:
            shutil.rmtree(path, onerror=lambda func, p, exc: onexc(func, p, exc))
        except OSError:
            time.sleep(0.5)
            continue
        if not path.exists():
            return
    if path.exists():
        raise OSError(f"Could not remove output directory: {path}")


def read_boxes(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes
    with label_path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:])
            except ValueError:
                continue
            if 0 <= cls_id < len(CLASSES) and 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1:
                boxes.append((cls_id, cx, cy, w, h))
    return boxes


def write_boxes(label_path: Path, boxes) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for cls_id, cx, cy, w, h in boxes:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def collect_records(source: Path, splits: list[str]):
    records = []
    for split in splits:
        img_dir = source / split / "images"
        lbl_dir = source / split / "labels"
        if not img_dir.exists():
            continue
        for image in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            label = lbl_dir / f"{image.stem}.txt"
            boxes = read_boxes(label)
            if not boxes:
                continue
            classes = {b[0] for b in boxes}
            records.append((image, label, boxes, classes, split))
    return records


def sample_balanced(records, per_class: int, seed: int):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for rec in records:
        for cls_id in rec[3]:
            by_class[cls_id].append(rec)

    selected = []
    selected_paths = set()
    class_counts = Counter()
    for cls_id in range(len(CLASSES)):
        candidates = list(by_class.get(cls_id, []))
        rng.shuffle(candidates)
        for rec in candidates:
            if class_counts[cls_id] >= per_class:
                break
            image = rec[0]
            if image in selected_paths:
                continue
            selected.append(rec)
            selected_paths.add(image)
            for present_cls in rec[3]:
                class_counts[present_cls] += 1

    # Fill any short class with reused source images, but produce a different
    # synthetic view. This is acceptable for a stress set, not final metrics.
    for cls_id in range(len(CLASSES)):
        candidates = list(by_class.get(cls_id, []))
        if not candidates:
            continue
        while class_counts[cls_id] < per_class:
            rec = rng.choice(candidates)
            selected.append(rec)
            for present_cls in rec[3]:
                class_counts[present_cls] += 1
    return selected


def apply_low_resolution(img, rng: random.Random):
    h, w = img.shape[:2]
    scale = rng.uniform(0.32, 0.62)
    low_w = max(64, int(w * scale))
    low_h = max(64, int(h * scale))
    small = cv2.resize(img, (low_w, low_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_jpeg_compression(img, rng: random.Random):
    quality = rng.randint(28, 62)
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else img


def apply_color_cast(img, rng: random.Random):
    out = img.astype(np.float32)
    # BGR: indoor webcam yellow/green cast, with reduced blue.
    out[:, :, 0] *= rng.uniform(0.72, 0.95)
    out[:, :, 1] *= rng.uniform(0.95, 1.12)
    out[:, :, 2] *= rng.uniform(1.05, 1.28)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_brightness_contrast(img, rng: random.Random):
    alpha = rng.uniform(0.58, 1.28)
    beta = rng.uniform(-48, 28)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def apply_noise(img, rng: random.Random):
    sigma = rng.uniform(6, 24)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_motion_blur(img, rng: random.Random):
    k = rng.choice([3, 5, 7])
    kernel = np.zeros((k, k), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= k
    return cv2.filter2D(img, -1, kernel)


def apply_vignette(img, rng: random.Random):
    h, w = img.shape[:2]
    strength = rng.uniform(0.35, 0.65)
    y, x = np.ogrid[:h, :w]
    cx = w * rng.uniform(0.42, 0.58)
    cy = h * rng.uniform(0.42, 0.58)
    radius = math.sqrt(cx * cx + cy * cy)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = 1.0 - strength * np.clip(dist / max(radius, 1), 0, 1)
    out = img.astype(np.float32) * mask[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def box_center_pixels(boxes, w: int, h: int):
    centers = []
    for _, cx, cy, bw, bh in boxes:
        centers.append((int(cx * w), int(cy * h), int(bw * w), int(bh * h)))
    return centers


def apply_occlusion(img, boxes, rng: random.Random):
    h, w = img.shape[:2]
    out = img.copy()
    centers = box_center_pixels(boxes, w, h)
    for _ in range(12):
        ow = rng.randint(max(18, w // 12), max(24, w // 5))
        oh = rng.randint(max(18, h // 12), max(24, h // 5))
        x1 = rng.randint(0, max(0, w - ow))
        y1 = rng.randint(0, max(0, h - oh))
        x2, y2 = x1 + ow, y1 + oh
        overlaps_center = False
        for cx, cy, bw, bh in centers:
            center_inside = x1 <= cx <= x2 and y1 <= cy <= y2
            too_large_on_box = ow * oh > max(1, bw * bh) * 0.6
            if center_inside and too_large_on_box:
                overlaps_center = True
                break
        if overlaps_center:
            continue
        color = rng.choice([(35, 35, 35), (70, 55, 45), (115, 95, 75), (25, 25, 30)])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=-1)
        return out
    return out


def webcam_degrade(img, boxes, rng: random.Random):
    out = img.copy()
    ops = [
        apply_low_resolution,
        apply_color_cast,
        apply_brightness_contrast,
        apply_noise,
        apply_vignette,
        apply_jpeg_compression,
    ]
    if rng.random() < 0.60:
        ops.append(lambda frame, r: cv2.GaussianBlur(frame, (3, 3), 0))
    if rng.random() < 0.35:
        ops.append(apply_motion_blur)
    rng.shuffle(ops)
    for op in ops:
        out = op(out, rng)
    if rng.random() < 0.35:
        out = apply_occlusion(out, boxes, rng)
    return out


def write_yaml(output: Path) -> Path:
    yaml_path = output / "data_synthetic_holdout.yaml"
    content = f"""# Temporary synthetic webcam stress-test holdout
path: {output.resolve()}
train: images
val: images
test: images

nc: {len(CLASSES)}

names:
"""
    for idx, name in enumerate(CLASSES):
        content += f"  {idx}: {name}\n"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic webcam-style holdout")
    parser.add_argument("--source", default="dataset_v4_balanced")
    parser.add_argument("--output", default="synthetic_webcam_holdout")
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if output.exists():
        remove_tree(output)
    img_out = output / "images"
    lbl_out = output / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    records = collect_records(source, args.splits)
    selected = sample_balanced(records, args.per_class, args.seed)

    class_counts = Counter()
    for idx, (image, _label, boxes, classes, split) in enumerate(selected):
        frame = cv2.imread(str(image))
        if frame is None:
            continue
        degraded = webcam_degrade(frame, boxes, rng)
        stem = f"synthetic_{idx:05d}_{split}_{image.stem}"
        cv2.imwrite(str(img_out / f"{stem}.jpg"), degraded, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        write_boxes(lbl_out / f"{stem}.txt", boxes)
        for cls_id in classes:
            class_counts[cls_id] += 1

    yaml_path = write_yaml(output)
    failures = [f"{CLASSES[i]}={class_counts[i]}" for i in range(len(CLASSES)) if class_counts[i] < args.per_class]

    print("=" * 72)
    print("  SYNTHETIC WEBCAM HOLDOUT REPORT")
    print("=" * 72)
    print(f"Images: {len(list(img_out.iterdir())):,}")
    print(f"Labels: {len(list(lbl_out.iterdir())):,}")
    for idx, name in enumerate(CLASSES):
        print(f"  {name:<14} {class_counts[idx]:>5} images containing class")
    print(f"YAML: {yaml_path.resolve()}")
    if failures:
        raise SystemExit("[FAIL] Synthetic class minimum not met: " + ", ".join(failures))
    print("[PASS] Synthetic holdout generated.")


if __name__ == "__main__":
    main()
