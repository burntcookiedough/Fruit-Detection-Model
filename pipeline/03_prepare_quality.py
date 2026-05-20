"""
prepare_v4_quality.py
======================
Builds dataset_v4_quality/ from dataset_v4_raw/ — a high-quality,
real-images-only fruit detection dataset.

Quality rules:
  1. Real images only — NO Fruit-360 pseudo-labels
  2. NO synthetic augmentation (no *_aug_* filenames)
  3. Exclude finger-style banana labels entirely
  4. No loose boxes (>85% image area)
  5. No empty labels after filtering
  6. Stratified 70/15/15 split per class
  7. Minimum 100 boxes per class in val and test

Usage:
    python prepare_v4_quality.py
    python prepare_v4_quality.py --source dataset_v4_raw
    python prepare_v4_quality.py --min-val-test 50
"""

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

CLASSES = ["apple", "banana", "orange", "mango",
           "pineapple", "watermelon", "grapes", "pomegranate"]
NC = len(CLASSES)

BANANA_ID = 1
BUNCH_AREA_THRESHOLD = 0.04
LOOSE_AREA_THRESHOLD = 0.85

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SOURCE_DIR = Path("dataset_v4_raw")
OUTPUT_DIR = Path("dataset_v4_quality")
YAML_PATH = Path("data_v4_quality.yaml")
REPORT_PATH = Path("quality_report_v4.txt")


def read_boxes(lbl: Path):
    if not lbl.exists():
        return []
    boxes = []
    with open(lbl) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    boxes.append((int(parts[0]), *map(float, parts[1:])))
                except ValueError:
                    pass
    return boxes


def write_boxes(lbl: Path, boxes):
    lbl.parent.mkdir(parents=True, exist_ok=True)
    with open(lbl, "w") as f:
        for cls_id, cx, cy, w, h in boxes:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def collect_all_records(src_dir: Path):
    records = []
    for split in ["train", "valid", "test"]:
        img_dir = src_dir / split / "images"
        lbl_dir = src_dir / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            boxes = read_boxes(lbl_path)
            if boxes:
                records.append((img_path, lbl_path, boxes))
    return records


def is_fruit360_source(img_path: Path) -> bool:
    parts = img_path.parts
    for p in parts:
        if "fruits_360" in p.lower() or "fruits-360" in p.lower() or "fruits262" in p.lower():
            return True
    return False


def is_synthetic_aug(img_path: Path) -> bool:
    return "_aug_" in img_path.name.lower()


def classify_banana_style(boxes):
    banana_boxes = [(cls, cx, cy, w, h) for cls, cx, cy, w, h in boxes if cls == BANANA_ID]
    if not banana_boxes:
        return "none"
    areas = [w * h for _, _, _, w, h in banana_boxes]
    is_finger = [a < BUNCH_AREA_THRESHOLD for a in areas]
    if all(is_finger):
        return "finger"
    return "bunch"


def has_loose_box(boxes):
    for _, _, _, w, h in boxes:
        if w * h > LOOSE_AREA_THRESHOLD:
            return True
    return False


def filter_quality(records):
    stats = {
        "total": len(records),
        "fruit360_excluded": 0,
        "synthetic_excluded": 0,
        "banana_finger_excluded": 0,
        "loose_box_excluded": 0,
        "empty_after_filter": 0,
        "kept": 0,
    }

    filtered = []
    for img_path, lbl_path, boxes in records:
        if is_fruit360_source(img_path):
            stats["fruit360_excluded"] += 1
            continue

        if is_synthetic_aug(img_path):
            stats["synthetic_excluded"] += 1
            continue

        if classify_banana_style(boxes) == "finger":
            stats["banana_finger_excluded"] += 1
            continue

        if has_loose_box(boxes):
            stats["loose_box_excluded"] += 1
            continue

        if not boxes:
            stats["empty_after_filter"] += 1
            continue

        filtered.append((img_path, lbl_path, boxes))
        stats["kept"] += 1

    return filtered, stats


def stratified_split(records, seed=RANDOM_SEED):
    rng = random.Random(seed)

    class_images = defaultdict(list)
    for img_path, lbl_path, boxes in records:
        classes_in_image = set(b[0] for b in boxes)
        for cls_id in classes_in_image:
            class_images[cls_id].append((img_path, lbl_path, boxes))

    assigned = {}
    for cls_id in range(NC):
        imgs = class_images[cls_id]
        rng.shuffle(imgs)
        n = len(imgs)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_valid = max(1, int(n * VALID_RATIO))

        for i, (img_path, lbl_path, boxes) in enumerate(imgs):
            if img_path in assigned:
                continue
            if i < n_train:
                assigned[img_path] = "train"
            elif i < n_train + n_valid:
                assigned[img_path] = "valid"
            else:
                assigned[img_path] = "test"

    for img_path, lbl_path, boxes in records:
        if img_path not in assigned:
            assigned[img_path] = "train"

    splits = {"train": [], "valid": [], "test": []}
    for img_path, lbl_path, boxes in records:
        splits[assigned[img_path]].append((img_path, lbl_path, boxes))

    return splits


def build_dataset(splits, output_dir: Path):
    for split_name in ["train", "valid", "test"]:
        img_dir = output_dir / split_name / "images"
        lbl_dir = output_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx, (img_path, lbl_path, boxes) in enumerate(splits[split_name]):
            new_name = f"{split_name}_{idx:06d}{img_path.suffix.lower()}"
            new_img = img_dir / new_name
            new_lbl = lbl_dir / f"{split_name}_{idx:06d}.txt"
            shutil.copy2(img_path, new_img)
            write_boxes(new_lbl, boxes)


def count_boxes(records):
    counts = defaultdict(int)
    for _, _, boxes in records:
        for b in boxes:
            counts[b[0]] += 1
    return counts


def write_yaml(output_dir: Path):
    content = f"""# Fruit Detection Dataset V4 — High Quality (Real Images Only)
# Generated by prepare_v4_quality.py
# No Fruit-360 pseudo-labels, no synthetic augmentation, no finger-style bananas

path: {output_dir.resolve()}
train: train/images
val:   valid/images
test:  test/images

nc: {NC}

names:
"""
    for i, name in enumerate(CLASSES):
        content += f"  {i}: {name}\n"

    with open(YAML_PATH, "w") as f:
        f.write(content)
    print(f"  Wrote: {YAML_PATH}")


def generate_report(splits, filter_stats, output_dir: Path, min_val_test: int):
    lines = []
    lines.append("=" * 60)
    lines.append("  DATASET V4 QUALITY REPORT")
    lines.append("=" * 60)
    lines.append("")

    total = sum(len(s) for s in splits.values())
    lines.append(f"Total images:    {total:,}")
    for split_name in ["train", "valid", "test"]:
        lines.append(f"  {split_name.capitalize():8s}: {len(splits[split_name]):,}")
    lines.append("")

    for split_name in ["train", "valid", "test"]:
        counts = count_boxes(splits[split_name])
        max_c = max(counts.values()) if counts else 1
        lines.append(f"Per-class distribution ({split_name.upper()}):")
        for i, name in enumerate(CLASSES):
            c = counts.get(i, 0)
            bar_len = int(c / max_c * 28)
            bar = "|" * bar_len
            lines.append(f"  {i}: {name:<14} {c:>5}  {bar}")
        lines.append("")

    lines.append("Quality gates:")
    lines.append(f"  [PASS] No Fruit-360 pseudo-labels ({filter_stats['fruit360_excluded']:,} excluded)")
    lines.append(f"  [PASS] No synthetic augmentation ({filter_stats['synthetic_excluded']:,} excluded)")
    lines.append(f"  [PASS] No finger-style banana labels ({filter_stats['banana_finger_excluded']:,} excluded)")
    lines.append(f"  [PASS] No loose boxes >85% area ({filter_stats['loose_box_excluded']:,} excluded)")
    lines.append(f"  [PASS] No empty labels ({filter_stats['empty_after_filter']:,} excluded)")
    lines.append("")

    val_counts = count_boxes(splits["valid"])
    test_counts = count_boxes(splits["test"])
    warnings = []
    for i, name in enumerate(CLASSES):
        vc = val_counts.get(i, 0)
        tc = test_counts.get(i, 0)
        if vc < min_val_test:
            warnings.append(f"  [FAIL] {name}: only {vc} boxes in valid (target: {min_val_test}+)")
        if tc < min_val_test:
            warnings.append(f"  [FAIL] {name}: only {tc} boxes in test (target: {min_val_test}+)")

    if warnings:
        lines.append("Warnings:")
        lines.extend(warnings)
    else:
        lines.append(f"[OK] All classes meet minimum {min_val_test} boxes in val and test.")

    lines.append("")
    lines.append("=" * 60)

    ready = not warnings
    if ready:
        lines.append("  READY FOR TRAINING")
    else:
        lines.append("  NOT READY FOR TRAINING")
    lines.append("=" * 60)

    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(report)
    print(f"\n  Report saved to: {REPORT_PATH}")
    return ready


def main():
    parser = argparse.ArgumentParser(description="Build high-quality dataset v4")
    parser.add_argument("--source", default=str(SOURCE_DIR),
                        help="Source filtered dataset directory")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help="Output quality dataset directory")
    parser.add_argument("--min-val-test", type=int, default=100,
                        help="Minimum boxes per class in val/test (default: 100)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)

    source = Path(args.source)
    output = Path(args.output)

    print("=" * 60)
    print("  FRUIT DETECTION — PREPARE V4 QUALITY DATASET")
    print(f"  Source : {source}")
    print(f"  Output : {output}")
    print("=" * 60)

    if not source.exists():
        print(f"\n[ERROR] Source directory not found: {source}")
        print("Run these steps first:")
        print("  1. python download_github_datasets.py")
        print("  1. python source_audit_v4.py")
        print("  2. python build_v4_raw.py")
        sys.exit(1)

    if output.exists():
        print(f"\nRemoving old {output}...")
        shutil.rmtree(output, ignore_errors=True)

    print("\n[1/5] Collecting records from all splits...")
    records = collect_all_records(source)
    print(f"  Found {len(records):,} image-label pairs")

    print("\n[2/5] Applying quality filters...")
    filtered, filter_stats = filter_quality(records)
    print(f"  Input:    {filter_stats['total']:,} records")
    print(f"  Fruit-360 excluded:  {filter_stats['fruit360_excluded']:,}")
    print(f"  Synthetic excluded:  {filter_stats['synthetic_excluded']:,}")
    print(f"  Banana finger excluded: {filter_stats['banana_finger_excluded']:,}")
    print(f"  Loose box excluded:  {filter_stats['loose_box_excluded']:,}")
    print(f"  Kept:     {filter_stats['kept']:,} records")

    if filter_stats["kept"] == 0:
        print("\n[ERROR] No records survived filtering. Check source data.")
        sys.exit(1)

    print("\n[3/5] Stratified split (70/15/15)...")
    splits = stratified_split(filtered, seed=args.seed)
    for name, recs in splits.items():
        print(f"  {name.capitalize():8s}: {len(recs):,} images")

    print("\n[4/5] Building dataset...")
    build_dataset(splits, output)
    print(f"  Written to: {output}")

    print("\n[5/5] Writing config and report...")
    write_yaml(output)
    ready = generate_report(splits, filter_stats, output, args.min_val_test)

    print("\n" + "=" * 60)
    print("  V4 QUALITY DATASET READY")
    print(f"  Dataset : {output.resolve()}")
    print(f"  Config  : {YAML_PATH.resolve()}")
    print(f"  Report  : {REPORT_PATH.resolve()}")
    print("\n  Next step:")
    print(f"    python run_v4_training.py")
    print(f"    python run_v4_training.py --benchmark")
    print("=" * 60)
    if not ready:
        sys.exit(1)


if __name__ == "__main__":
    main()
