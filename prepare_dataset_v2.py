"""
Fruit Detection Model - Dataset Preparation v2
Downloads real-world fruit datasets from Kaggle, merges them with existing data,
normalizes labels to 8 canonical classes, and splits into train/valid/test.

Usage:
    python prepare_dataset_v2.py
"""

import argparse
import hashlib
import random
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

FINAL_CLASSES = [
    "apple",
    "banana",
    "orange",
    "mango",
    "pineapple",
    "watermelon",
    "grapes",
    "pomegranate",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(FINAL_CLASSES)}

# Comprehensive label aliases covering LVIS, Kaggle, and Roboflow naming conventions
LABEL_ALIASES = {
    # apple (LVIS index 1)
    "apple": "apple", "Apple": "apple", "apples": "apple", "APPLE": "apple",
    "apple_(fruit)": "apple", "red_apple": "apple", "green_apple": "apple",
    # banana (LVIS index 6)
    "banana": "banana", "Banana": "banana", "bananas": "banana", "BANANA": "banana",
    "banana_ripe": "banana",
    # orange (LVIS index 44)
    "orange": "orange", "Orange": "orange", "oranges": "orange", "ORANGE": "orange",
    "orange_(fruit)": "orange", "orange/orange fruit": "orange",
    "mandarin orange": "orange",
    # mango (not in LVIS -- only from Roboflow synthetic)
    "mango": "mango", "Mango": "mango", "mangos": "mango", "mangoes": "mango",
    "MANGO": "mango",
    # pineapple (LVIS index 51)
    "pineapple": "pineapple", "Pineapple": "pineapple", "pineapples": "pineapple",
    "PINEAPPLE": "pineapple",
    # watermelon (LVIS index 61)
    "watermelon": "watermelon", "Watermelon": "watermelon", "watermelons": "watermelon",
    "WATERMELON": "watermelon", "WaterMelon": "watermelon", "water_melon": "watermelon",
    # grapes (LVIS index 32 = "grape")
    "grapes": "grapes", "Grapes": "grapes", "grape": "grapes", "Grape": "grapes",
    "GRAPES": "grapes", "GRAPE": "grapes",
    # pomegranate (not in LVIS -- only from Roboflow synthetic)
    "pomegranate": "pomegranate", "Pomegranate": "pomegranate", "pomegranates": "pomegranate",
    "POMEGRANATE": "pomegranate",
}

TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO = 0.10
RANDOM_SEED = 42

RAW_DIR = Path("raw_datasets")
OUTPUT_DIR = Path("dataset_v2")
DATA_YAML_PATH = Path("data_v2.yaml")


# ============================================================
# HELPERS
# ============================================================

def read_class_names_from_yaml(yaml_path):
    """Parse a YOLO data.yaml to get class name list."""
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return names


def find_yaml_in_dataset(dataset_dir):
    """Find data.yaml in a dataset folder."""
    dataset_dir = Path(dataset_dir)
    for name in ["data.yaml", "dataset.yaml"]:
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate
    yamls = list(dataset_dir.rglob("data.yaml")) + list(dataset_dir.rglob("dataset.yaml"))
    if yamls:
        return yamls[0]
    return None


def find_image_label_pairs(dataset_dir):
    """Find all (image_path, label_path) pairs in YOLO-format dataset.
    
    Handles two common layouts:
      Layout A (standard):  dataset/train/images/, dataset/train/labels/
      Layout B (LVIS-style): dataset/images/train/, dataset/labels/train/
    """
    dataset_dir = Path(dataset_dir)
    pairs = []
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Layout A: split/images/ + split/labels/
    for split_name in ["train", "valid", "val", "validation", "test"]:
        img_dir = dataset_dir / split_name / "images"
        lbl_dir = dataset_dir / split_name / "labels"
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in img_extensions:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))

    # Layout B: images/split/ + labels/split/ (e.g., LVIS dataset)
    for split_name in ["train", "valid", "val", "validation", "test"]:
        img_dir = dataset_dir / "images" / split_name
        lbl_dir = dataset_dir / "labels" / split_name
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in img_extensions:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))

    # Also check root-level images/ and labels/ (flat layout)
    root_img = dataset_dir / "images"
    root_lbl = dataset_dir / "labels"
    if root_img.exists() and root_lbl.exists():
        # Only use root-level if no split subdirs were found inside
        has_split_subdirs = any((root_img / s).is_dir() for s in ["train", "val", "test", "valid"])
        if not has_split_subdirs:
            for img_path in root_img.iterdir():
                if img_path.suffix.lower() not in img_extensions:
                    continue
                lbl_path = root_lbl / (img_path.stem + ".txt")
                if lbl_path.exists():
                    pairs.append((img_path, lbl_path))

    return pairs


def normalize_label_file(label_path, source_class_names, output_path):
    """Remap class indices to our canonical 8 classes. Drops unknown classes."""
    output_lines = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            src_idx = int(parts[0])
            if src_idx >= len(source_class_names):
                continue
            src_name = source_class_names[src_idx]
            canonical = LABEL_ALIASES.get(src_name)
            if canonical is None:
                # Try case-insensitive lookup
                canonical = LABEL_ALIASES.get(src_name.lower())
            if canonical is None:
                continue
            new_idx = CLASS_TO_IDX[canonical]
            # Validate bounding box values
            try:
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    continue
            except ValueError:
                continue
            output_lines.append(f"{new_idx} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

    if output_lines:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines) + "\n")
        return True
    return False


def file_hash(filepath):
    """Quick hash of first 8KB for dedup."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        h.update(f.read(8192))
    return h.hexdigest()


def unzip_dataset(zip_path, extract_dir):
    """Unzip a dataset if not already extracted."""
    extract_dir = Path(extract_dir)
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"  Already extracted: {extract_dir}")
        return
    print(f"  Unzipping {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    print(f"  Extracted to {extract_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("  FRUIT DETECTION - DATASET PREPARATION v2")
    print("  Building real-world fruit detection dataset")
    print("=" * 60)

    # ----------------------------------------------------------
    # Step 1: Unzip downloaded Kaggle datasets
    # ----------------------------------------------------------
    print("\n[1/5] Unzipping Kaggle datasets...")

    kaggle_datasets = {
        "lvis_fruits": RAW_DIR / "lvis_fruits",
        "fruit_detection_kaggle": RAW_DIR / "fruit_detection_kaggle",
    }

    for name, base_dir in kaggle_datasets.items():
        zips = list(base_dir.glob("*.zip")) if base_dir.exists() else []
        for zp in zips:
            unzip_dataset(zp, base_dir / "extracted")

    # ----------------------------------------------------------
    # Step 2: Discover all datasets
    # ----------------------------------------------------------
    print("\n[2/5] Discovering datasets...")

    # Collect all dataset directories to process
    dataset_dirs = []

    # LVIS
    lvis_base = RAW_DIR / "lvis_fruits"
    if lvis_base.exists():
        yaml_file = find_yaml_in_dataset(lvis_base)
        if yaml_file:
            dataset_dirs.append(("LVIS Fruits", yaml_file.parent, yaml_file))

    # Kaggle Fruit Detection
    fd_base = RAW_DIR / "fruit_detection_kaggle"
    if fd_base.exists():
        yaml_file = find_yaml_in_dataset(fd_base)
        if yaml_file:
            dataset_dirs.append(("Fruit Detection (Kaggle)", yaml_file.parent, yaml_file))

    # Existing Roboflow datasets (synthetic, but adds diversity)
    for existing in ["fruit-lpwjt", "watermelon-muqdf", "fruit-detection-dnwrs"]:
        edir = RAW_DIR / existing
        if edir.exists():
            yaml_file = find_yaml_in_dataset(edir)
            if yaml_file:
                dataset_dirs.append((f"Roboflow ({existing})", yaml_file.parent, yaml_file))

    print(f"  Found {len(dataset_dirs)} dataset(s):")
    for name, path, yaml_file in dataset_dirs:
        print(f"    - {name}: {yaml_file}")

    if not dataset_dirs:
        print("\n[ERROR] No datasets found. Please download datasets first.")
        print("  kaggle datasets download -d henningheyen/lvis-fruits-and-vegetables-dataset -p raw_datasets/lvis_fruits")
        print("  kaggle datasets download -d lakshaytyagi01/fruit-detection -p raw_datasets/fruit_detection_kaggle")
        sys.exit(1)

    # ----------------------------------------------------------
    # Step 3: Normalize labels and collect all pairs
    # ----------------------------------------------------------
    print("\n[3/5] Normalizing labels and collecting image-label pairs...")

    staging_dir = Path("_staging_v2")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_img = staging_dir / "images"
    staging_lbl = staging_dir / "labels"
    staging_img.mkdir(parents=True)
    staging_lbl.mkdir(parents=True)

    total_pairs = 0
    total_kept = 0
    stats_per_source = {}
    seen_hashes = set()
    idx_counter = 0

    for ds_name, ds_path, yaml_path in dataset_dirs:
        print(f"\n  Processing: {ds_name}")
        source_classes = read_class_names_from_yaml(yaml_path)
        print(f"    Source classes ({len(source_classes)}): {source_classes[:10]}{'...' if len(source_classes) > 10 else ''}")

        pairs = find_image_label_pairs(ds_path)
        print(f"    Found {len(pairs)} image-label pairs")

        kept = 0
        for img_path, lbl_path in pairs:
            total_pairs += 1

            # Dedup by image content hash
            try:
                h = file_hash(img_path)
            except Exception:
                continue
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # Generate unique filename
            new_name = f"img_{idx_counter:06d}{img_path.suffix.lower()}"
            new_img = staging_img / new_name
            new_lbl = staging_lbl / f"img_{idx_counter:06d}.txt"

            # Normalize label file
            if normalize_label_file(lbl_path, source_classes, new_lbl):
                shutil.copy2(img_path, new_img)
                kept += 1
                idx_counter += 1

        stats_per_source[ds_name] = kept
        total_kept += kept
        print(f"    Kept {kept} images (after filtering to our 8 classes + dedup)")

    print(f"\n  Total images collected: {total_kept}")
    for src, count in stats_per_source.items():
        print(f"    {src}: {count}")

    if total_kept == 0:
        print("[ERROR] No valid images found after filtering. Check class name mappings.")
        sys.exit(1)

    # ----------------------------------------------------------
    # Step 4: Count per-class distribution
    # ----------------------------------------------------------
    class_counts = defaultdict(int)
    for lbl_file in staging_lbl.iterdir():
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_idx = int(parts[0])
                    class_counts[cls_idx] += 1

    print("\n  Per-class bounding box counts:")
    for idx, name in enumerate(FINAL_CLASSES):
        count = class_counts.get(idx, 0)
        bar = "#" * min(count // 20, 40)
        print(f"    {idx}: {name:15s} {count:5d}  {bar}")

    # ----------------------------------------------------------
    # Step 5: Split into train/valid/test
    # ----------------------------------------------------------
    print(f"\n[4/5] Splitting {total_kept} images into train/valid/test...")

    all_images = sorted(staging_img.iterdir())
    random.shuffle(all_images)

    n_train = int(len(all_images) * TRAIN_RATIO)
    n_valid = int(len(all_images) * VALID_RATIO)

    splits = {
        "train": all_images[:n_train],
        "valid": all_images[n_train:n_train + n_valid],
        "test": all_images[n_train + n_valid:],
    }

    # Clear and create output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split_name, split_images in splits.items():
        img_dir = OUTPUT_DIR / split_name / "images"
        lbl_dir = OUTPUT_DIR / split_name / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for img_path in split_images:
            stem = img_path.stem
            lbl_path = staging_lbl / f"{stem}.txt"
            shutil.copy2(img_path, img_dir / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dir / lbl_path.name)

        print(f"    {split_name}: {len(split_images)} images")

    # ----------------------------------------------------------
    # Step 6: Write data.yaml
    # ----------------------------------------------------------
    print(f"\n[5/5] Writing {DATA_YAML_PATH}...")

    yaml_content = f"""# Fruit Detection Dataset v2 (Real-World)
# Generated by prepare_dataset_v2.py

path: ./{OUTPUT_DIR}

train: train/images
val: valid/images
test: test/images

nc: {len(FINAL_CLASSES)}

names:
"""
    for idx, name in enumerate(FINAL_CLASSES):
        yaml_content += f"  {idx}: {name}\n"

    with open(DATA_YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"    Saved to {DATA_YAML_PATH}")

    # Cleanup staging
    shutil.rmtree(staging_dir)

    print("\n" + "=" * 60)
    print("  DATASET PREPARATION COMPLETE")
    print(f"  Total images: {total_kept}")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  Config: {DATA_YAML_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
