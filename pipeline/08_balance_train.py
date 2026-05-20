"""
balance_v4_train.py
===================
Create a train-balanced V4 dataset from dataset_v4_quality/.

Principles:
  - Balance training only.
  - Copy validation and test unchanged so metrics stay honest.
  - Cap overrepresented classes by selecting fewer train images.
  - Augment underrepresented train classes only if real reviewed images are
    still below the minimum training signal target.

Output:
  dataset_v4_balanced/
  data_v4_balanced.yaml
  balance_report_v4.txt
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

from check_split_leakage import BKTree, phash

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
                boxes.append((int(parts[0]), *[float(v) for v in parts[1:]]))
            except ValueError:
                continue
    return boxes


def write_boxes(label_path: Path, boxes) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for cls_id, cx, cy, w, h in boxes:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def collect_records(dataset: Path, split: str):
    img_dir = dataset / split / "images"
    lbl_dir = dataset / split / "labels"
    records = []
    for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        boxes = read_boxes(lbl_path)
        if not boxes:
            continue
        classes = {b[0] for b in boxes}
        records.append((img_path, lbl_path, boxes, classes))
    return records


def count_boxes(records) -> Counter:
    counts = Counter()
    for _, _, boxes, _ in records:
        for cls_id, *_ in boxes:
            counts[cls_id] += 1
    return counts


def select_train_records(records, target_min: int, soft_cap: int, hard_cap: int, seed: int):
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    raw_counts = count_boxes(shuffled)
    scarce_classes = {cls_id for cls_id in range(len(CLASSES)) if raw_counts[cls_id] < soft_cap}

    selected = []
    selected_set = set()
    counts = Counter()

    # Preserve real minority-class coverage first.
    for rec in shuffled:
        img_path, _, boxes, classes = rec
        if classes & scarce_classes:
            selected.append(rec)
            selected_set.add(img_path)
            for cls_id, *_ in boxes:
                counts[cls_id] += 1

    # Add majority-only images while they improve balance or stay below caps.
    for rec in shuffled:
        img_path, _, boxes, classes = rec
        if img_path in selected_set:
            continue
        contributes_to_min = any(counts[cls_id] < target_min for cls_id in classes)
        below_soft = any(counts[cls_id] < soft_cap for cls_id in classes)
        would_break_hard_only = all(counts[cls_id] >= hard_cap for cls_id in classes)
        if contributes_to_min or (below_soft and not would_break_hard_only):
            selected.append(rec)
            selected_set.add(img_path)
            for cls_id, *_ in boxes:
                counts[cls_id] += 1

    return selected, counts, raw_counts


def copy_records(records, out_root: Path, split: str) -> None:
    img_dir = out_root / split / "images"
    lbl_dir = out_root / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for img_path, _, boxes, _ in records:
        shutil.copy2(img_path, img_dir / img_path.name)
        write_boxes(lbl_dir / f"{img_path.stem}.txt", boxes)


def aug_brightness_contrast(img):
    alpha = random.uniform(0.65, 1.35)
    beta = random.uniform(-35, 35)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def aug_noise(img):
    sigma = random.uniform(4, 18)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def aug_blur(img):
    return cv2.GaussianBlur(img, (3, 3), 0)


def aug_hflip(img, boxes):
    flipped = cv2.flip(img, 1)
    new_boxes = [(cls_id, 1.0 - cx, cy, w, h) for cls_id, cx, cy, w, h in boxes]
    return flipped, new_boxes


def phash_image(img, size: int = 8, highfreq_factor: int = 4) -> int:
    img_size = size * highfreq_factor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low_freq = dct[:size, :size].copy()
    coeffs = low_freq.flatten()[1:]
    median = float(np.median(coeffs))
    bits = low_freq.flatten() > median
    bits[0] = False
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def build_forbidden_tree(records):
    tree = BKTree()
    for img_path, *_ in records:
        h = phash(img_path)
        if h is not None:
            tree.add(h, img_path)
    return tree


def augment_to_minimum(
    selected,
    counts: Counter,
    out_root: Path,
    target_min: int,
    max_aug_images: int,
    seed: int,
    forbidden_tree: BKTree,
    phash_threshold: int,
):
    random.seed(seed)
    np.random.seed(seed)

    img_dir = out_root / "train" / "images"
    lbl_dir = out_root / "train" / "labels"
    added = 0
    candidate_by_class = {
        cls_id: [rec for rec in selected if cls_id in rec[3]]
        for cls_id in range(len(CLASSES))
    }

    rejected_leakage = 0
    attempts = 0
    max_attempts = max_aug_images * 20
    while added < max_aug_images and attempts < max_attempts:
        attempts += 1
        needy = [cls_id for cls_id in range(len(CLASSES)) if counts[cls_id] < target_min]
        if not needy:
            break
        cls_id = min(needy, key=lambda c: counts[c])
        candidates = candidate_by_class.get(cls_id, [])
        if not candidates:
            break

        img_path, _, boxes, _ = random.choice(candidates)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        aug_img = img.copy()
        aug_boxes = list(boxes)
        ops = random.sample(["brightness", "noise", "blur", "hflip"], k=random.randint(1, 3))
        for op in ops:
            if op == "brightness":
                aug_img = aug_brightness_contrast(aug_img)
            elif op == "noise":
                aug_img = aug_noise(aug_img)
            elif op == "blur":
                aug_img = aug_blur(aug_img)
            elif op == "hflip":
                aug_img, aug_boxes = aug_hflip(aug_img, aug_boxes)

        aug_hash = phash_image(aug_img)
        if forbidden_tree.query(aug_hash, phash_threshold):
            rejected_leakage += 1
            continue

        stem = f"v4_bal_aug_{added:06d}"
        cv2.imwrite(str(img_dir / f"{stem}{img_path.suffix.lower()}"), aug_img)
        write_boxes(lbl_dir / f"{stem}.txt", aug_boxes)
        for box_cls, *_ in aug_boxes:
            counts[box_cls] += 1
        added += 1

    return added, counts, rejected_leakage


def write_yaml(out_root: Path, yaml_path: Path) -> None:
    content = f"""# Fruit Detection Dataset V4 - Train Balanced
path: {out_root.resolve()}

train: train/images
val:   valid/images
test:  test/images

nc: {len(CLASSES)}

names:
"""
    for idx, name in enumerate(CLASSES):
        content += f"  {idx}: {name}\n"
    yaml_path.write_text(content, encoding="utf-8")


def format_counts(title: str, counts: Counter, image_count: int):
    lines = [title, f"  images: {image_count:,}"]
    max_count = max([counts.get(i, 0) for i in range(len(CLASSES))] or [1])
    for idx, name in enumerate(CLASSES):
        count = counts.get(idx, 0)
        bar = "#" * int((count / max(max_count, 1)) * 32)
        lines.append(f"  {name:<14} {count:>7}  {bar}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance V4 train split only")
    parser.add_argument("--source", default="dataset_v4_quality")
    parser.add_argument("--output", default="dataset_v4_balanced")
    parser.add_argument("--yaml", default="data_v4_balanced.yaml")
    parser.add_argument("--report", default="balance_report_v4.txt")
    parser.add_argument("--target-min", type=int, default=4000)
    parser.add_argument("--soft-cap", type=int, default=12000)
    parser.add_argument("--hard-cap", type=int, default=18000)
    parser.add_argument("--max-aug-images", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if output.exists():
        remove_tree(output)

    train_records = collect_records(source, "train")
    valid_records = collect_records(source, "valid")
    test_records = collect_records(source, "test")

    selected, selected_counts, raw_counts = select_train_records(
        train_records,
        target_min=args.target_min,
        soft_cap=args.soft_cap,
        hard_cap=args.hard_cap,
        seed=args.seed,
    )

    copy_records(selected, output, "train")
    forbidden_tree = build_forbidden_tree(valid_records + test_records)
    added_aug, final_counts, rejected_leakage = augment_to_minimum(
        selected,
        selected_counts,
        output,
        target_min=args.target_min,
        max_aug_images=args.max_aug_images,
        seed=args.seed,
        forbidden_tree=forbidden_tree,
        phash_threshold=8,
    )
    copy_records(valid_records, output, "valid")
    copy_records(test_records, output, "test")
    write_yaml(output, Path(args.yaml))

    final_train_records = collect_records(output, "train")
    final_counts = count_boxes(final_train_records)
    failed = [(CLASSES[i], final_counts[i]) for i in range(len(CLASSES)) if final_counts[i] < args.target_min]

    lines = []
    lines.append("=" * 72)
    lines.append("  DATASET V4 TRAIN BALANCE REPORT")
    lines.append("=" * 72)
    lines.extend(format_counts("Raw train counts:", raw_counts, len(train_records)))
    lines.append("")
    lines.extend(format_counts("Balanced train counts:", final_counts, len(final_train_records)))
    lines.append("")
    lines.extend(format_counts("Validation counts unchanged:", count_boxes(valid_records), len(valid_records)))
    lines.append("")
    lines.extend(format_counts("Test counts unchanged:", count_boxes(test_records), len(test_records)))
    lines.append("")
    lines.append(f"Augmented train images added: {added_aug:,}")
    lines.append(f"Augmented images rejected for leakage risk: {rejected_leakage:,}")
    lines.append(f"Target min: {args.target_min:,}; soft cap: {args.soft_cap:,}; hard cap: {args.hard_cap:,}")
    if failed:
        lines.append("[FAIL] Training balance gate failed: " + ", ".join(f"{name}={count}" for name, count in failed))
    else:
        lines.append("[PASS] Training balance gate passed.")
    lines.append(f"Dataset: {output.resolve()}")
    lines.append(f"YAML: {Path(args.yaml).resolve()}")

    report = "\n".join(lines)
    Path(args.report).write_text(report, encoding="utf-8")
    print(report)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
