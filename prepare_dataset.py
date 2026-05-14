"""
Fruit Detection Model - Dataset Preparation Script
Downloads datasets from Roboflow, normalizes labels, merges, and splits.

Usage:
    # Mode 1: Download from Roboflow (requires API key)
    python prepare_dataset.py --api-key YOUR_ROBOFLOW_API_KEY

    # Mode 2: Use locally downloaded raw datasets
    python prepare_dataset.py --local

    For local mode, place YOLO-format dataset folders in raw_datasets/:
        raw_datasets/
        |-- dataset1/
        |   |-- train/
        |   |   |-- images/
        |   |   |-- labels/
        |   |-- valid/
        |       |-- images/
        |       |-- labels/
        |-- dataset2/
            |-- ...
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# Our canonical 8 fruit classes (index order matches data.yaml)
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

# Label aliases: maps variant names -> canonical name
# Add more aliases as you discover them in downloaded datasets
LABEL_ALIASES = {
    # apple
    "apple": "apple",
    "red_apple": "apple",
    "green_apple": "apple",
    "apples": "apple",
    "Apple": "apple",
    # banana
    "banana": "banana",
    "banana_ripe": "banana",
    "bananas": "banana",
    "Banana": "banana",
    # orange
    "orange": "orange",
    "orange_fruit": "orange",
    "oranges": "orange",
    "Orange": "orange",
    # mango
    "mango": "mango",
    "mangoes": "mango",
    "mangos": "mango",
    "Mango": "mango",
    # pineapple
    "pineapple": "pineapple",
    "pineapples": "pineapple",
    "Pineapple": "pineapple",
    # watermelon
    "watermelon": "watermelon",
    "watermelons": "watermelon",
    "water_melon": "watermelon",
    "Watermelon": "watermelon",
    # grapes
    "grapes": "grapes",
    "grape": "grapes",
    "Grapes": "grapes",
    "Grape": "grapes",
    # pomegranate
    "pomegranate": "pomegranate",
    "pomegranates": "pomegranate",
    "Pomegranate": "pomegranate",
}

# Map canonical class names to their final index
CLASS_TO_IDX = {name: idx for idx, name in enumerate(FINAL_CLASSES)}

# Dataset split ratios
TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42

# Roboflow datasets to download (workspace/project/version)
ROBOFLOW_DATASETS = [
    {
        "workspace": "first-btnnz",
        "project": "fruit-lpwjt",
        "version": 1,
        "description": "Comprehensive fruit detection (63 classes, ~6000 images)",
    },
    {
        "workspace": "pluto2002-kqi66",
        "project": "watermelon-v1ctm",
        "version": 1,
        "description": "Watermelon-specific detection (~309 images)",
    },
    {
        "workspace": "yun-3zrjn",
        "project": "fruit-detection-dnwrs",
        "version": 4,
        "description": "Multi-fruit detection (10 classes, ~106 images)",
    },
]


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
        # {0: 'apple', 1: 'banana', ...}
        names = [names[k] for k in sorted(names.keys())]
    return names


def find_yaml_in_dataset(dataset_dir):
    """Find the data.yaml (or similar) in a downloaded dataset folder."""
    dataset_dir = Path(dataset_dir)
    for name in ["data.yaml", "dataset.yaml", "_darknet.labels"]:
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate

    # Search recursively
    yamls = list(dataset_dir.rglob("data.yaml")) + list(dataset_dir.rglob("dataset.yaml"))
    if yamls:
        return yamls[0]

    return None


def find_image_label_pairs(dataset_dir):
    """
    Find all (image_path, label_path) pairs in a YOLO-format dataset.
    Searches train/, valid/, test/ subdirectories.
    """
    dataset_dir = Path(dataset_dir)
    pairs = []
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split_name in ["train", "valid", "val", "test"]:
        img_dir = dataset_dir / split_name / "images"
        lbl_dir = dataset_dir / split_name / "labels"

        if not img_dir.exists():
            continue

        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in img_extensions:
                continue

            # Find matching label file
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))

    # Also check root-level images/ and labels/
    root_img = dataset_dir / "images"
    root_lbl = dataset_dir / "labels"
    if root_img.exists():
        for img_path in root_img.iterdir():
            if img_path.suffix.lower() not in img_extensions:
                continue
            lbl_path = root_lbl / (img_path.stem + ".txt")
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))

    return pairs


def normalize_label_file(label_path, source_class_names, output_path):
    """
    Read a YOLO label file, remap class indices to our canonical classes.
    Drops bounding boxes for classes not in our final list.

    Returns the number of valid boxes written (0 means image has no relevant fruits).
    """
    valid_lines = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            old_class_idx = int(parts[0])

            # Get old class name
            if old_class_idx >= len(source_class_names):
                continue
            old_name = source_class_names[old_class_idx]

            # Map to canonical name
            canonical = LABEL_ALIASES.get(old_name)
            if canonical is None:
                # Try lowercase
                canonical = LABEL_ALIASES.get(old_name.lower())
            if canonical is None:
                # Not a fruit we care about -- skip this box
                continue

            # Get new class index
            new_idx = CLASS_TO_IDX[canonical]

            # Rebuild the line with new class index
            new_line = f"{new_idx} " + " ".join(parts[1:])
            valid_lines.append(new_line)

    if not valid_lines:
        return 0

    # Write normalized label
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(valid_lines) + "\n")

    return len(valid_lines)


# ============================================================
# DOWNLOAD FROM ROBOFLOW
# ============================================================

def download_roboflow_datasets(api_key, output_dir):
    """Download datasets from Roboflow Universe."""
    from roboflow import Roboflow

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)

    downloaded = []
    for ds_info in ROBOFLOW_DATASETS:
        print(f"\nDownloading: {ds_info['description']}")
        print(f"  Workspace: {ds_info['workspace']}")
        print(f"  Project  : {ds_info['project']}")
        print(f"  Version  : {ds_info['version']}")

        try:
            project = rf.workspace(ds_info["workspace"]).project(ds_info["project"])
            version = project.version(ds_info["version"])
            ds = version.download("yolov8", location=str(output_dir / ds_info["project"]))
            downloaded.append(output_dir / ds_info["project"])
            print(f"  [OK] Downloaded to {output_dir / ds_info['project']}")
        except Exception as e:
            print(f"  [FAIL] Failed: {e}")
            print(f"    You can manually download this dataset from Roboflow Universe")
            print(f"    and place it in: {output_dir / ds_info['project']}")

    return downloaded


# ============================================================
# MERGE AND SPLIT
# ============================================================

def collect_and_normalize(raw_dirs, staging_dir):
    """
    Collect all image-label pairs from raw dataset directories,
    normalize labels to our canonical classes,
    and copy to a flat staging directory.
    """
    staging_dir = Path(staging_dir)
    staging_img = staging_dir / "images"
    staging_lbl = staging_dir / "labels"
    staging_img.mkdir(parents=True, exist_ok=True)
    staging_lbl.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    total_pairs = 0
    skipped = 0
    file_counter = 0

    for raw_dir in raw_dirs:
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            print(f"  [WARN] Skipping {raw_dir} (not found)")
            continue

        print(f"\n  Processing: {raw_dir.name}")

        # Find the class names from the dataset's yaml
        yaml_path = find_yaml_in_dataset(raw_dir)
        if yaml_path is None:
            print(f"    [WARN] No data.yaml found in {raw_dir}, skipping")
            continue

        source_classes = read_class_names_from_yaml(yaml_path)
        print(f"    Source classes: {source_classes}")

        # Find all image-label pairs
        pairs = find_image_label_pairs(raw_dir)
        print(f"    Found {len(pairs)} image-label pairs")

        for img_path, lbl_path in pairs:
            file_counter += 1
            # Use counter prefix to avoid name collisions between datasets
            new_stem = f"{file_counter:06d}"
            new_img = staging_img / f"{new_stem}{img_path.suffix.lower()}"
            new_lbl = staging_lbl / f"{new_stem}.txt"

            # Normalize the label file
            num_boxes = normalize_label_file(lbl_path, source_classes, new_lbl)

            if num_boxes == 0:
                skipped += 1
                # Clean up label file if it was created empty
                if new_lbl.exists():
                    new_lbl.unlink()
                continue

            # Copy image
            shutil.copy2(img_path, new_img)
            total_pairs += 1

            # Count per-class stats
            with open(new_lbl, "r") as f:
                for line in f:
                    cls_idx = int(line.strip().split()[0])
                    stats[FINAL_CLASSES[cls_idx]] += 1

    return total_pairs, skipped, dict(stats)


def split_dataset(staging_dir, output_dir, seed=RANDOM_SEED):
    """
    Split staging directory into train/valid/test with the configured ratios.
    """
    staging_dir = Path(staging_dir)
    output_dir = Path(output_dir)

    # Get all image files
    img_dir = staging_dir / "images"
    lbl_dir = staging_dir / "labels"

    images = sorted(list(img_dir.iterdir()))
    random.seed(seed)
    random.shuffle(images)

    total = len(images)
    n_train = int(total * TRAIN_RATIO)
    n_valid = int(total * VALID_RATIO)
    # Rest goes to test

    splits = {
        "train": images[:n_train],
        "valid": images[n_train:n_train + n_valid],
        "test": images[n_train + n_valid:],
    }

    for split_name, split_images in splits.items():
        split_img_dir = output_dir / split_name / "images"
        split_lbl_dir = output_dir / split_name / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            lbl_path = lbl_dir / (img_path.stem + ".txt")

            shutil.copy2(img_path, split_img_dir / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, split_lbl_dir / lbl_path.name)

        print(f"  {split_name:>5s}: {len(split_images)} images")

    return {name: len(imgs) for name, imgs in splits.items()}


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare fruit detection dataset")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Roboflow API key (for automatic download)")
    parser.add_argument("--local", action="store_true",
                        help="Use locally downloaded datasets in raw_datasets/")
    parser.add_argument("--raw-dir", type=str, default="raw_datasets",
                        help="Directory containing raw datasets (default: raw_datasets)")
    parser.add_argument("--output-dir", type=str, default="dataset",
                        help="Output dataset directory (default: dataset)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed for splitting (default: {RANDOM_SEED})")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    staging_dir = Path("_staging")

    print("=" * 60)
    print("  FRUIT DETECTION - DATASET PREPARATION")
    print("=" * 60)
    print(f"  Classes: {FINAL_CLASSES}")
    print(f"  Split  : {TRAIN_RATIO:.0%} train / {VALID_RATIO:.0%} valid / {TEST_RATIO:.0%} test")
    print("=" * 60)

    # Step 1: Get raw datasets
    if args.api_key:
        print("\n[1/4] Downloading datasets from Roboflow...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        download_roboflow_datasets(args.api_key, raw_dir)
    elif args.local:
        print(f"\n[1/4] Using local datasets from {raw_dir}/")
        if not raw_dir.exists():
            print(f"  [FAIL] Directory {raw_dir} not found!")
            print(f"    Create it and place YOLO-format datasets inside.")
            return
    else:
        print("\n[FAIL] Specify --api-key YOUR_KEY or --local")
        print("  --api-key : Download from Roboflow automatically")
        print("  --local   : Use datasets already in raw_datasets/")
        return

    # Find all dataset subdirectories
    raw_datasets = [d for d in raw_dir.iterdir() if d.is_dir()]
    if not raw_datasets:
        print(f"\n  [FAIL] No dataset directories found in {raw_dir}/")
        return

    print(f"\n  Found {len(raw_datasets)} dataset(s): {[d.name for d in raw_datasets]}")

    # Step 2: Normalize and collect
    print("\n[2/4] Normalizing labels and collecting images...")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    total, skipped, class_stats = collect_and_normalize(raw_datasets, staging_dir)
    print(f"\n  Total valid pairs: {total}")
    print(f"  Skipped (no matching classes): {skipped}")

    if total == 0:
        print("\n  [FAIL] No valid images found! Check your datasets and label aliases.")
        return

    # Step 3: Print class distribution
    print("\n[3/4] Class distribution (bounding box counts):")
    print("  " + "-" * 30)
    for cls_name in FINAL_CLASSES:
        count = class_stats.get(cls_name, 0)
        bar = "#" * (count // 10) if count > 0 else "-"
        print(f"    {cls_name:<15s}: {count:>5d}  {bar}")
    print("  " + "-" * 30)

    # Step 4: Split into train/valid/test
    print(f"\n[4/4] Splitting into train/valid/test...")
    if output_dir.exists():
        print(f"  Clearing existing {output_dir}/")
        shutil.rmtree(output_dir)

    split_counts = split_dataset(staging_dir, output_dir, seed=args.seed)

    # Cleanup staging
    shutil.rmtree(staging_dir)

    print("\n" + "=" * 60)
    print("  [OK] DATASET READY!")
    print("=" * 60)
    print(f"  Output   : {output_dir}/")
    print(f"  Train    : {split_counts['train']} images")
    print(f"  Valid    : {split_counts['valid']} images")
    print(f"  Test     : {split_counts['test']} images")
    print(f"\n  Next step: python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
