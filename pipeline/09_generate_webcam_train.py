"""
pipeline/09_generate_webcam_train.py
=====================================
Generate synthetic webcam-degraded copies of training images for V5.

Applies JPEG compression, BGR colour casts, low-res downscale, brightness
variation, Gaussian noise, vignette, motion blur, and random occlusion.
These transforms cannot be replicated by YOLO's built-in HSV augmentation —
they directly target the failure modes revealed in V4 evaluation:
  apple  webcam mAP50: 13%  (CRITICAL — colour-cast destroys hue cues)
  orange webcam mAP50: 11%  (CRITICAL — same root cause)

DESIGN CHOICE — flat 100% fraction
-----------------------------------
V1 of this script attempted class-stratified selection (reading every label
to pick images by class composition). On Windows NTFS with 17,876 small
files, that pre-scan takes >20 minutes due to file-open syscall overhead.

100% flat fraction achieves the same quality outcome because:
  - apple and orange images are distributed across the full training set
  - giving mango/watermelon webcam copies does not harm them (they maintain
    their high performance and also learn degradation invariance)
  - dual-pass produces 2 different degradation styles per image, giving
    maximally diverse colour-cast exposure

Labels are read ONLY at generation time (for each image being processed),
not in a pre-scan pass.

LEAKAGE SAFETY
--------------
Source: dataset_v4_balanced/train/ ONLY — never val/test.
The synthetic_webcam_holdout/ was built from val/test; do not regenerate it.

Usage
-----
  python pipeline/09_generate_webcam_train.py
  python pipeline/09_generate_webcam_train.py --fraction 0.60  # lighter run
  python pipeline/09_generate_webcam_train.py --no-dual-pass   # single pass
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
CLASSES = [
    "apple", "banana", "orange", "mango",
    "pineapple", "watermelon", "grapes", "pomegranate",
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT            = Path(__file__).resolve().parent.parent
SOURCE_DATASET  = ROOT / "dataset_v4_balanced"
DEFAULT_OUTPUT  = ROOT / "dataset_v4_webcam_train"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

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
            shutil.rmtree(path, onerror=lambda fn, p, e: onexc(fn, p, e))
        except OSError:
            time.sleep(0.5)
            continue
        if not path.exists():
            return


def read_boxes(label_path: Path) -> list[tuple]:
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
            if (0 <= cls_id < len(CLASSES)
                    and 0 <= cx <= 1 and 0 <= cy <= 1
                    and 0 < w <= 1 and 0 < h <= 1):
                boxes.append((cls_id, cx, cy, w, h))
    return boxes


def write_boxes(label_path: Path, boxes: list) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for cls_id, cx, cy, w, h in boxes:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ---------------------------------------------------------------------------
# Degradation transforms
# ---------------------------------------------------------------------------

def apply_low_resolution(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    scale = rng.uniform(0.30, 0.65)
    small = cv2.resize(img, (max(64, int(w * scale)), max(64, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_jpeg_compression(img: np.ndarray, rng: random.Random) -> np.ndarray:
    quality = rng.randint(22, 65)
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else img


def apply_warm_cast(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Yellow/warm indoor webcam cast. Primary failure mode for apple and orange."""
    out = img.astype(np.float32)
    out[:, :, 0] *= rng.uniform(0.65, 0.92)   # reduce blue
    out[:, :, 1] *= rng.uniform(0.90, 1.12)   # mild green
    out[:, :, 2] *= rng.uniform(1.05, 1.35)   # boost red/yellow
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_cool_cast(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Cool fluorescent / overcast cast for degradation diversity."""
    out = img.astype(np.float32)
    out[:, :, 0] *= rng.uniform(1.05, 1.30)   # boost blue
    out[:, :, 1] *= rng.uniform(0.90, 1.05)
    out[:, :, 2] *= rng.uniform(0.70, 0.95)   # reduce red
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_brightness_contrast(img: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = rng.uniform(0.50, 1.35)
    beta  = rng.uniform(-55, 35)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def apply_noise(img: np.ndarray, rng: random.Random) -> np.ndarray:
    sigma = rng.uniform(5, 28)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_motion_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    k = rng.choice([3, 5, 7, 9])
    kernel = np.zeros((k, k), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= k
    return cv2.filter2D(img, -1, kernel)


def apply_vignette(img: np.ndarray, rng: random.Random) -> np.ndarray:
    import math
    h, w = img.shape[:2]
    strength = rng.uniform(0.30, 0.70)
    cx = w * rng.uniform(0.38, 0.62)
    cy = h * rng.uniform(0.38, 0.62)
    radius = math.sqrt(cx * cx + cy * cy)
    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    mask = 1.0 - strength * np.clip(dist / max(radius, 1), 0, 1)
    return np.clip(img.astype(np.float32) * mask[:, :, None], 0, 255).astype(np.uint8)


def apply_occlusion(img: np.ndarray, boxes: list, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    centers = [(int(cx * w), int(cy * h), int(bw * w), int(bh * h))
               for _, cx, cy, bw, bh in boxes]
    for _ in range(15):
        ow = rng.randint(max(18, w // 12), max(24, w // 5))
        oh = rng.randint(max(18, h // 12), max(24, h // 5))
        x1 = rng.randint(0, max(0, w - ow))
        y1 = rng.randint(0, max(0, h - oh))
        x2, y2 = x1 + ow, y1 + oh
        bad = any(
            x1 <= bx <= x2 and y1 <= by <= y2 and ow * oh > max(1, bw * bh) * 0.6
            for bx, by, bw, bh in centers
        )
        if not bad:
            color = rng.choice([(35, 35, 35), (70, 55, 45), (115, 95, 75), (25, 25, 30)])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=-1)
            return out
    return out


def webcam_degrade(
    img: np.ndarray,
    boxes: list,
    rng: random.Random,
    warm: bool = True,
) -> np.ndarray:
    """
    Apply a randomised degradation stack.
    warm=True  -> yellow/warm cast  (primary failure mode)
    warm=False -> cool/fluorescent cast  (diversity pass)
    """
    ops = [
        apply_low_resolution,
        warm and apply_warm_cast or apply_cool_cast,
        apply_brightness_contrast,
        apply_noise,
        apply_vignette,
        apply_jpeg_compression,
    ]
    if rng.random() < 0.60:
        ops.append(lambda f, r: cv2.GaussianBlur(f, (3, 3), 0))
    if rng.random() < 0.40:
        ops.append(apply_motion_blur)
    rng.shuffle(ops)
    out = img.copy()
    for op in ops:
        out = op(out, rng)
    if rng.random() < 0.35:
        out = apply_occlusion(out, boxes, rng)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic webcam-degraded training images for V5",
    )
    p.add_argument("--source",   default=str(SOURCE_DATASET),
                   help="Balanced dataset root (default: dataset_v4_balanced)")
    p.add_argument("--output",   default=str(DEFAULT_OUTPUT),
                   help="Output directory (default: dataset_v4_webcam_train)")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Fraction of training images to degrade (default: 1.0 = all)")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--no-dual-pass", action="store_true",
                   help="Generate only one degradation per image instead of two")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    rng    = random.Random(args.seed)
    np.random.seed(args.seed)

    source        = Path(args.source)
    output        = Path(args.output)
    train_img_src = source / "train" / "images"
    train_lbl_src = source / "train" / "labels"
    dual_pass     = not args.no_dual_pass

    if not train_img_src.exists():
        raise FileNotFoundError(
            f"Train images not found: {train_img_src}\n"
            "Ensure --source points to dataset_v4_balanced"
        )

    # Collect image list — NO label scanning at this stage
    all_images = sorted(p for p in train_img_src.iterdir()
                        if p.suffix.lower() in IMG_EXTS)
    rng.shuffle(all_images)
    n_select = max(1, int(len(all_images) * args.fraction))
    selected = all_images[:n_select]
    n_passes = 2 if dual_pass else 1
    total    = n_select * n_passes

    if output.exists():
        print(f"Removing existing: {output}")
        remove_tree(output)
    img_out = output / "train" / "images"
    lbl_out = output / "train" / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  WEBCAM TRAIN GENERATOR")
    print("=" * 60)
    print(f"  Source    : {source}")
    print(f"  Train set : {len(all_images):,} images")
    print(f"  Selected  : {n_select:,}  ({args.fraction:.0%})")
    print(f"  Passes    : {n_passes}  ({'warm+cool casts' if dual_pass else 'warm cast only'})")
    print(f"  Total out : {total:,} images")
    print(f"  Seed      : {args.seed}")
    print("=" * 60)

    generated     = 0
    skipped       = 0
    class_counts: Counter = Counter()

    for pass_idx in range(n_passes):
        warm = (pass_idx == 0)  # pass 0 = warm cast, pass 1 = cool cast
        cast_label = "warm" if warm else "cool"
        print(f"\n  Pass {pass_idx + 1}/{n_passes} ({cast_label} colour cast)")

        for local_idx, image_path in enumerate(selected):
            # Read label ONLY at generation time — avoids 17K pre-scan
            label_path = train_lbl_src / f"{image_path.stem}.txt"
            boxes      = read_boxes(label_path)
            if not boxes:
                skipped += 1
                continue

            frame = cv2.imread(str(image_path))
            if frame is None:
                skipped += 1
                continue

            degraded   = webcam_degrade(frame, boxes, rng, warm=warm)
            global_idx = pass_idx * n_select + local_idx
            stem       = f"wcam_p{pass_idx}_{global_idx:06d}_{image_path.stem}"

            cv2.imwrite(str(img_out / f"{stem}.jpg"), degraded,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            write_boxes(lbl_out / f"{stem}.txt", boxes)

            for cls_id, *_ in boxes:
                class_counts[cls_id] += 1
            generated += 1

            if (global_idx + 1) % 1000 == 0:
                pct = (global_idx + 1) / total * 100
                print(f"    [{global_idx+1:,}/{total:,}]  {pct:.0f}%  "
                      f"generated={generated:,}  skipped={skipped}")

    print("\n" + "=" * 60)
    print("  COMPLETE")
    print(f"  Generated : {generated:,} webcam images")
    print(f"  Skipped   : {skipped} (no label or unreadable)")
    print()
    print("  Boxes per class in webcam set:")
    for i, name in enumerate(CLASSES):
        bar = "#" * min(20, class_counts[i] // 500)
        print(f"    {name:<14} {class_counts[i]:>7,}  {bar}")
    print("=" * 60)
    print()
    print("  NEXT: python train.py --epochs 2 --name fruit_v5_smoke")


if __name__ == "__main__":
    main()
