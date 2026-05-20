"""
Fruit Detection Model - Dataset Balancer (v2 -> v3)

Reads the existing dataset_v2 and produces a new balanced dataset_v3 by:
  1. CAPPING over-represented classes (orange, apple, grapes, banana)
     - Randomly selects a subset of their images until class count is under TARGET_MAX
  2. AUGMENTING under-represented classes (mango, pomegranate, pineapple)
     - Applies realistic transforms to existing images to reach TARGET_MIN

Augmentations simulate webcam-style conditions:
  - Brightness / contrast jitter
  - Horizontal flip (with mirrored bounding boxes)
  - Gaussian noise
  - Slight blur
  - Small rotation (+/- 10 degrees)

Usage:
    python balance_dataset.py
    python balance_dataset.py --max_boxes 2000 --min_boxes 500
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

FINAL_CLASSES = [
    "apple", "banana", "orange", "mango",
    "pineapple", "watermelon", "grapes", "pomegranate",
]

SOURCE_DIR  = Path("dataset_v2")
OUTPUT_DIR  = Path("dataset_v3")
DATA_YAML   = Path("data_v3.yaml")

TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO  = 0.10
RANDOM_SEED = 42

# Default balance targets (overrideable via CLI)
# These are calibrated for v3 (larger dataset)
DEFAULT_TARGET_MAX = 5000   # cap majority classes to this many BOXES in train
DEFAULT_TARGET_MIN = 2500   # augment minority classes to at least this many BOXES

# ============================================================
# AUGMENTATION HELPERS  (pure OpenCV / numpy, no extra deps)
# ============================================================

def aug_brightness_contrast(img, alpha_range=(0.5, 1.5), beta_range=(-40, 40)):
    """Random brightness and contrast."""
    alpha = random.uniform(*alpha_range)
    beta  = random.uniform(*beta_range)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def aug_gaussian_noise(img, sigma_range=(5, 25)):
    """Add Gaussian noise."""
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def aug_blur(img, ksize_choices=(3, 5)):
    """Gaussian blur."""
    k = random.choice(ksize_choices)
    return cv2.GaussianBlur(img, (k, k), 0)


def aug_hflip(img, boxes):
    """Horizontal flip with YOLO box adjustment."""
    flipped = cv2.flip(img, 1)
    new_boxes = []
    for b in boxes:
        cls_id, cx, cy, w, h = b
        new_cx = 1.0 - cx
        new_boxes.append((cls_id, new_cx, cy, w, h))
    return flipped, new_boxes


def aug_rotate(img, boxes, angle_range=(-10, 10)):
    """Small rotation (crops/pads to keep original size). Adjusts boxes approximately."""
    angle = random.uniform(*angle_range)
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)

    # Rotate bounding box corners and recompute YOLO box
    new_boxes = []
    for b in boxes:
        cls_id, cx, cy, bw, bh = b
        # Convert to pixel corners
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        x2 = (cx + bw / 2) * W
        y2 = (cy + bh / 2) * H
        corners = np.array([[x1, y1, 1], [x2, y1, 1],
                             [x1, y2, 1], [x2, y2, 1]], dtype=np.float32)
        rotated_corners = (M @ corners.T).T
        rx1 = np.clip(rotated_corners[:, 0].min(), 0, W)
        rx2 = np.clip(rotated_corners[:, 0].max(), 0, W)
        ry1 = np.clip(rotated_corners[:, 1].min(), 0, H)
        ry2 = np.clip(rotated_corners[:, 1].max(), 0, H)
        new_cx = ((rx1 + rx2) / 2) / W
        new_cy = ((ry1 + ry2) / 2) / H
        new_bw  = (rx2 - rx1) / W
        new_bh  = (ry2 - ry1) / H
        if 0 < new_bw <= 1 and 0 < new_bh <= 1:
            new_boxes.append((cls_id, new_cx, new_cy, new_bw, new_bh))
    return rotated, new_boxes


# ============================================================
# I/O HELPERS
# ============================================================

def read_boxes(label_path):
    """Returns list of (cls_id, cx, cy, w, h) tuples."""
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    boxes.append((int(parts[0]),
                                  float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4])))
                except ValueError:
                    pass
    return boxes


def write_boxes(label_path, boxes):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for b in boxes:
            f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")


def collect_split(split_name):
    """Returns list of (img_path, lbl_path, classes_present) for a split."""
    img_dir = SOURCE_DIR / split_name / "images"
    lbl_dir = SOURCE_DIR / split_name / "labels"
    if not img_dir.exists():
        return []

    records = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        boxes = read_boxes(lbl_path)
        if not boxes:
            continue
        classes = set(b[0] for b in boxes)
        records.append((img_path, lbl_path, boxes, classes))
    return records


# ============================================================
# MAIN BALANCE LOGIC
# ============================================================

def count_boxes(records):
    counts = defaultdict(int)
    for _, _, boxes, _ in records:
        for b in boxes:
            counts[b[0]] += 1
    return counts


def greedy_select_with_cap(records, target_max):
    """
    Greedy selection: include all images, but after each class hits target_max
    boxes, stop including images whose ONLY contribution is that capped class.
    This is a heuristic — real ILP is overkill here.
    """
    random.shuffle(records)
    selected = []
    class_counts = defaultdict(int)

    # First pass: always include images with at least one under-represented class
    # Classes 3=mango, 4=pineapple, 5=watermelon, 7=pomegranate get priority
    minority_classes = {3, 4, 5, 7}
    remainder = []
    for rec in records:
        img_path, lbl_path, boxes, classes = rec
        if classes & minority_classes:
            selected.append(rec)
            for b in boxes:
                class_counts[b[0]] += 1
        else:
            remainder.append(rec)

    # Second pass: add remaining images while respecting cap
    for rec in remainder:
        img_path, lbl_path, boxes, classes = rec
        # Check if this image adds any class that is below cap
        useful = any(class_counts[b[0]] < target_max for b in boxes)
        if useful:
            selected.append(rec)
            for b in boxes:
                class_counts[b[0]] += 1

    return selected, class_counts


def augment_to_target(records, class_counts, target_min, out_img_dir, out_lbl_dir, prefix,
                      max_multiplier=3):
    """
    For classes below target_min, augment images containing those classes.
    max_multiplier: never augment a class beyond (max_multiplier * original_count).
    This prevents one class from flooding the dataset with synthetic images.
    Returns list of (new_img_path, new_lbl_path) tuples added.
    """
    aug_idx = 0
    # Original count before augmentation (hard ceiling per class)
    orig_counts = dict(class_counts)  # snapshot
    aug_ceiling = {cls_id: max(orig_counts.get(cls_id, 1) * max_multiplier, target_min)
                   for cls_id in range(len(FINAL_CLASSES))}

    # Identify which classes need augmentation
    needs_aug = {cls_id for cls_id in range(len(FINAL_CLASSES))
                 if class_counts[cls_id] < target_min}

    if not needs_aug:
        print("  No classes need augmentation.")
        return []

    print(f"  Classes needing augmentation: {[FINAL_CLASSES[i] for i in sorted(needs_aug)]}")

    # Build candidate pool (images containing at least one needy class)
    candidates = [r for r in records if r[3] & needs_aug]
    if not candidates:
        print("  [WARN] No candidate images found for augmentation!")
        return []

    added = []
    max_iterations = 50000  # safety valve (larger dataset needs more iterations)

    for iteration in range(max_iterations):
        # Check: any class still below target AND below ceiling?
        still_needed = {
            cls_id for cls_id in needs_aug
            if class_counts[cls_id] < target_min
            and class_counts[cls_id] < aug_ceiling[cls_id]
        }
        if not still_needed:
            break

        rec = random.choice(candidates)
        img_path, lbl_path, boxes, classes = rec

        # Only augment if this image contributes to a still-needed class
        if not (classes & still_needed):
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Pick 2-4 random augmentations to chain
        ops = random.sample(["brightness", "noise", "blur", "rotate", "hflip"], k=random.randint(2, 4))
        aug_img = img.copy()
        aug_boxes = list(boxes)

        for op in ops:
            if op == "brightness":
                aug_img = aug_brightness_contrast(aug_img)
            elif op == "noise":
                aug_img = aug_gaussian_noise(aug_img)
            elif op == "blur":
                aug_img = aug_blur(aug_img)
            elif op == "hflip":
                aug_img, aug_boxes = aug_hflip(aug_img, aug_boxes)
            elif op == "rotate":
                aug_img, aug_boxes = aug_rotate(aug_img, aug_boxes)

        if not aug_boxes:
            continue

        new_name = f"{prefix}_aug_{aug_idx:05d}{img_path.suffix.lower()}"
        new_img_path = out_img_dir / new_name
        new_lbl_path = out_lbl_dir / f"{prefix}_aug_{aug_idx:05d}.txt"

        cv2.imwrite(str(new_img_path), aug_img)
        write_boxes(new_lbl_path, aug_boxes)

        for b in aug_boxes:
            class_counts[b[0]] += 1
        aug_idx += 1
        added.append((new_img_path, new_lbl_path))

    return added


def copy_records_to_split(records, split_name):
    """Copies original image/label pairs to output split directory."""
    img_dir = OUTPUT_DIR / split_name / "images"
    lbl_dir = OUTPUT_DIR / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i, (img_path, lbl_path, boxes, _) in enumerate(records):
        new_img = img_dir / img_path.name
        new_lbl = lbl_dir / (img_path.stem + ".txt")
        shutil.copy2(img_path, new_img)
        write_boxes(new_lbl, boxes)

    return len(records)


def write_data_yaml(use_absolute=True):
    # Use absolute path so the yaml works from ANY working directory
    # (critical for Colab / Kaggle where cwd differs)
    if use_absolute:
        ds_path = str(OUTPUT_DIR.resolve())
    else:
        ds_path = f"./{OUTPUT_DIR}"

    content = f"""# Fruit Detection Dataset v3 (Balanced)
# Generated by balance_dataset.py
# Absolute path: works on local, Colab, and Kaggle

path: {ds_path}

train: train/images
val:   valid/images
test:  test/images

nc: {len(FINAL_CLASSES)}

names:
"""
    for idx, name in enumerate(FINAL_CLASSES):
        content += f"  {idx}: {name}\n"

    with open(DATA_YAML, "w") as f:
        f.write(content)
    print(f"  Wrote {DATA_YAML} (path: {ds_path})")


def print_distribution(split_name, counts, total_images):
    max_count = max(counts.values()) if counts else 1
    print(f"\n  [{split_name.upper()}] {total_images} images:")
    for idx, name in enumerate(FINAL_CLASSES):
        c = counts.get(idx, 0)
        bar_len = int(c / max(max_count, 1) * 30)
        bar = "#" * bar_len
        print(f"    {idx}: {name:15s} {c:5d}  {bar}")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Balance the fruit detection dataset")
    parser.add_argument("--source", default=None,
                        help="Source dataset dir (default: dataset_v3_raw if exists, else dataset_v2)")
    parser.add_argument("--out", default=None,
                        help="Output dataset dir (default: dataset_v3)")
    parser.add_argument("--max_boxes", type=int, default=DEFAULT_TARGET_MAX,
                        help=f"Cap majority classes to this many boxes in train (default: {DEFAULT_TARGET_MAX})")
    parser.add_argument("--min_boxes", type=int, default=DEFAULT_TARGET_MIN,
                        help=f"Augment minority classes to at least this many boxes in train (default: {DEFAULT_TARGET_MIN})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Resolve source and output dirs
    global SOURCE_DIR, OUTPUT_DIR, DATA_YAML
    if args.source:
        SOURCE_DIR = Path(args.source)
    elif Path("dataset_v3_raw").exists():
        SOURCE_DIR = Path("dataset_v3_raw")
    # else keep default (dataset_v2)

    if args.out:
        OUTPUT_DIR = Path(args.out)
        DATA_YAML = Path(f"{args.out.replace('/', '_')}.yaml").with_suffix(".yaml")
        # Use cleaner yaml name
        DATA_YAML = Path("data_v3.yaml") if "v3" in args.out else Path(f"data_{args.out}.yaml")

    print("=" * 60)
    print("  FRUIT DETECTION - DATASET BALANCER")
    print(f"  Source   : {SOURCE_DIR}")
    print(f"  Output   : {OUTPUT_DIR}")
    print(f"  Max boxes per class (train): {args.max_boxes}")
    print(f"  Min boxes per class (train): {args.min_boxes}")
    print("=" * 60)

    if not SOURCE_DIR.exists():
        print(f"\n[ERROR] Source dataset not found: {SOURCE_DIR}")
        print("Run prepare_dataset_v3.py (or prepare_dataset_v2.py) first.")
        return

    # ----------------------------------------------------------
    # Clear output
    # ----------------------------------------------------------
    if OUTPUT_DIR.exists():
        print(f"\nRemoving old {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)

    # ----------------------------------------------------------
    # Process TRAIN split (balancing applied here)
    # ----------------------------------------------------------
    print("\n[1/3] Processing TRAIN split...")
    train_records = collect_split("train")
    print(f"  Source: {len(train_records)} images")
    raw_counts = count_boxes(train_records)
    print_distribution("train (original)", raw_counts, len(train_records))

    # Select subset with cap on majority classes
    print("\n  Applying majority-class cap...")
    selected_train, selected_counts = greedy_select_with_cap(train_records, args.max_boxes)
    print(f"  After cap: {len(selected_train)} images retained")
    print_distribution("train (after cap)", selected_counts, len(selected_train))

    # Copy selected to output
    train_img_dir = OUTPUT_DIR / "train" / "images"
    train_lbl_dir = OUTPUT_DIR / "train" / "labels"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    for img_path, lbl_path, boxes, _ in selected_train:
        shutil.copy2(img_path, train_img_dir / img_path.name)
        write_boxes(train_lbl_dir / (img_path.stem + ".txt"), boxes)

    # Augment minority classes
    print("\n  Augmenting minority classes...")
    augment_to_target(
        selected_train, selected_counts, args.min_boxes,
        train_img_dir, train_lbl_dir, "tr"
    )
    final_train_counts = count_boxes(collect_split_from_dir(OUTPUT_DIR / "train"))
    final_train_images = len(list((OUTPUT_DIR / "train" / "images").iterdir()))
    print_distribution("train (final)", final_train_counts, final_train_images)

    # ----------------------------------------------------------
    # Process VALID and TEST splits (no balancing, just copy)
    # ----------------------------------------------------------
    for split in ["valid", "test"]:
        print(f"\n[{'2' if split == 'valid' else '3'}/3] Processing {split.upper()} split (copy as-is)...")
        records = collect_split(split)
        out_img = OUTPUT_DIR / split / "images"
        out_lbl = OUTPUT_DIR / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path, boxes, _ in records:
            shutil.copy2(img_path, out_img / img_path.name)
            write_boxes(out_lbl / (img_path.stem + ".txt"), boxes)
        counts = count_boxes(records)
        print_distribution(split, counts, len(records))

    # ----------------------------------------------------------
    # Write data_v3.yaml
    # ----------------------------------------------------------
    print("\n[4/4] Writing data_v3.yaml...")
    write_data_yaml(use_absolute=True)

    print("\n" + "=" * 60)
    print("  BALANCING COMPLETE")
    print(f"  Balanced dataset saved to: {OUTPUT_DIR}/")
    print(f"  Config file: {DATA_YAML}")
    print("\n  Next steps:")
    print("  1. python train.py --data data_v3.yaml --name fruit_v3 --epochs 100")
    print("=" * 60)


def collect_split_from_dir(split_dir):
    """Like collect_split but takes a full path."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    records = []
    if not img_dir.exists():
        return records
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        boxes = read_boxes(lbl_path)
        if not boxes:
            continue
        classes = set(b[0] for b in boxes)
        records.append((img_path, lbl_path, boxes, classes))
    return records


if __name__ == "__main__":
    main()
