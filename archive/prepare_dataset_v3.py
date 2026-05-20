"""
Fruit Detection - Dataset Preparation v3
Merges ALL dataset sources (Kaggle + Roboflow + existing v2) into one
large, diverse training set.

Key additions over v2:
  - Handles Fruit-360 classification images (converts to detection via pseudo-labels)
  - Handles Roboflow YOLOv8 format datasets
  - Handles COCO JSON format datasets
  - Handles Pascal VOC XML format datasets
  - Aggressive deduplication via perceptual hash

Usage:
    python prepare_dataset_v3.py
    python prepare_dataset_v3.py --output dataset_v3_raw
"""

import io
import sys
# Force UTF-8 stdout so non-Latin characters in dataset class names don't crash on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

FINAL_CLASSES = [
    "apple", "banana", "orange", "mango",
    "pineapple", "watermelon", "grapes", "pomegranate",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(FINAL_CLASSES)}

# All known aliases for each canonical class
LABEL_ALIASES = {
    # apple
    "apple": "apple", "Apple": "apple", "apples": "apple", "APPLE": "apple",
    "apple_(fruit)": "apple", "red_apple": "apple", "green_apple": "apple",
    "red apple": "apple", "green apple": "apple", "bad_apple": "apple",
    "good_apple": "apple", "fresh apple": "apple", "rotten apple": "apple",
    "Apple Fresh": "apple", "Apple Rotten": "apple", "Apple Bad": "apple",
    "Apple Good": "apple",
    # banana
    "banana": "banana", "Banana": "banana", "bananas": "banana", "BANANA": "banana",
    "banana_ripe": "banana", "fresh banana": "banana", "rotten banana": "banana",
    "Banana Fresh": "banana", "Banana Rotten": "banana",
    # orange
    "orange": "orange", "Orange": "orange", "oranges": "orange", "ORANGE": "orange",
    "orange_(fruit)": "orange", "orange/orange fruit": "orange",
    "mandarin orange": "orange", "mandarin": "orange", "tangerine": "orange",
    "clementine": "orange", "fresh orange": "orange", "rotten orange": "orange",
    "Orange Fresh": "orange", "Orange Rotten": "orange",
    # mango
    "mango": "mango", "Mango": "mango", "mangos": "mango", "mangoes": "mango",
    "MANGO": "mango", "fresh mango": "mango", "ripe mango": "mango",
    "Mango Fresh": "mango", "Mango Rotten": "mango",
    "Hapus": "mango", "hapus": "mango",
    # pineapple
    "pineapple": "pineapple", "Pineapple": "pineapple", "pineapples": "pineapple",
    "PINEAPPLE": "pineapple", "fresh pineapple": "pineapple",
    # watermelon
    "watermelon": "watermelon", "Watermelon": "watermelon", "watermelons": "watermelon",
    "WATERMELON": "watermelon", "WaterMelon": "watermelon", "water_melon": "watermelon",
    "water melon": "watermelon", "fresh watermelon": "watermelon",
    # grapes
    "grapes": "grapes", "Grapes": "grapes", "grape": "grapes", "Grape": "grapes",
    "GRAPES": "grapes", "GRAPE": "grapes", "fresh grapes": "grapes",
    "Grapes Fresh": "grapes", "Grapes Rotten": "grapes",
    # pomegranate
    "pomegranate": "pomegranate", "Pomegranate": "pomegranate",
    "pomegranates": "pomegranate", "POMEGRANATE": "pomegranate",
    "fresh pomegranate": "pomegranate", "Pomegranate Fresh": "pomegranate",
}

# Fruit-360 folder names that map to our classes
FRUIT360_FOLDER_MAP = {
    # apple
    "Apple Braeburn": "apple", "Apple Crimson Snow": "apple", "Apple Golden 1": "apple",
    "Apple Golden 2": "apple", "Apple Golden 3": "apple", "Apple Granny Smith": "apple",
    "Apple Pink Lady": "apple", "Apple Red 1": "apple", "Apple Red 2": "apple",
    "Apple Red 3": "apple", "Apple Red Delicious": "apple", "Apple Red Yellow 1": "apple",
    "Apple Red Yellow 2": "apple",
    # banana
    "Banana": "banana", "Banana Lady Finger": "banana", "Banana Red": "banana",
    # orange
    "Orange": "orange", "Mandarine": "orange", "Clementine": "orange",
    "Tangelo": "orange",
    # mango
    "Mango": "mango", "Mango Red": "mango",
    # pineapple
    "Pineapple": "pineapple", "Pineapple Mini": "pineapple",
    # watermelon
    # (Fruit-360 has no watermelon -- it's too big to photograph as a whole fruit)
    # grapes
    "Grape Blue": "grapes", "Grape Pink": "grapes", "Grape White": "grapes",
    "Grape White 2": "grapes", "Grape White 3": "grapes", "Grape White 4": "grapes",
    # pomegranate
    "Pomegranate": "pomegranate",
}

TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO = 0.10
RANDOM_SEED = 42

RAW_DIR = Path("raw_datasets")

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# HELPERS
# ============================================================

def resolve_alias(name):
    """Map a raw class name to one of our 8 canonical classes. Returns None if unknown."""
    # Direct lookup
    canonical = LABEL_ALIASES.get(name)
    if canonical:
        return canonical
    # Case-insensitive lookup
    canonical = LABEL_ALIASES.get(name.lower())
    if canonical:
        return canonical
    # Substring scan (e.g. "Apple_Fresh_001" -> apple)
    name_lower = name.lower()
    for alias, canon in LABEL_ALIASES.items():
        if alias.lower() in name_lower:
            return canon
    return None


def file_hash(path):
    """MD5 of first 16KB — fast enough for deduplication."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(16384))
    return h.hexdigest()


def read_yaml_names(yaml_path):
    """Parse class names from a YOLO data.yaml file."""
    try:
        import yaml
        with open(yaml_path, encoding="utf-8", errors="ignore") as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]
        return names
    except Exception:
        return []


def find_yaml(directory):
    """Find a data.yaml in a directory."""
    for name in ["data.yaml", "dataset.yaml", "_annotations.yaml"]:
        p = Path(directory) / name
        if p.exists():
            return p
    found = list(Path(directory).rglob("data.yaml")) + list(Path(directory).rglob("dataset.yaml"))
    return found[0] if found else None


# ============================================================
# COLLECTION STAGE: returns list of (img_path, [(cls_id, cx, cy, w, h), ...])
# ============================================================

def collect_yolo_dataset(base_dir, source_name="?"):
    """Collect image/label pairs from a standard YOLO-format dataset."""
    base_dir = Path(base_dir)
    yaml_path = find_yaml(base_dir)
    if yaml_path is None:
        print(f"    [WARN] No data.yaml found in {base_dir}. Skipping.")
        return []

    source_classes = read_yaml_names(yaml_path)
    if not source_classes:
        print(f"    [WARN] Could not read class names from {yaml_path}. Skipping.")
        return []

    # Sanitize class names before printing (some datasets have emoji like 🍇)
    safe_names = [c.encode('ascii', errors='replace').decode('ascii') for c in source_classes[:8]]
    print(f"    Classes ({len(source_classes)}): {safe_names}{'...' if len(source_classes) > 8 else ''}")

    records = []

    # Find all image directories (avoids scanning non-image dirs)
    # Use targeted globs instead of rglob("*") over the entire tree
    img_dirs = set()
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        for p in base_dir.rglob(f"*{ext}"):
            img_dirs.add(p.parent)

    for img_dir in sorted(img_dirs):
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            # Standard YOLO: images/ -> labels/
            lbl_path = Path(str(img_path).replace("images", "labels", 1)).with_suffix(".txt")
            if not lbl_path.exists():
                lbl_path = img_path.with_suffix(".txt")
            if not lbl_path.exists():
                continue

            boxes = []
            with open(lbl_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        src_idx = int(parts[0])
                        if src_idx >= len(source_classes):
                            continue
                        src_name = source_classes[src_idx]
                        canonical = resolve_alias(src_name)
                        if canonical is None:
                            continue
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            continue
                        boxes.append((CLASS_TO_IDX[canonical], cx, cy, w, h))
                    except (ValueError, IndexError):
                        continue

            if boxes:
                records.append((img_path, boxes))

    print(f"    Found {len(records)} valid image-label pairs")
    return records


def collect_coco_dataset(base_dir, source_name="?"):
    """Collect from COCO JSON annotations format."""
    base_dir = Path(base_dir)
    records = []

    for ann_file in base_dir.rglob("_annotations.coco.json"):
        try:
            with open(ann_file) as f:
                coco = json.load(f)
        except Exception:
            continue

        img_dir = ann_file.parent
        cat_map = {c["id"]: c["name"] for c in coco.get("categories", [])}
        img_map = {img["id"]: img for img in coco.get("images", [])}

        boxes_by_img = defaultdict(list)
        for ann in coco.get("annotations", []):
            cat_name = cat_map.get(ann["category_id"], "")
            canonical = resolve_alias(cat_name)
            if canonical is None:
                continue
            # COCO bbox: [x, y, width, height] in pixels
            bx, by, bw, bh = ann["bbox"]
            img_info = img_map.get(ann["image_id"])
            if img_info is None:
                continue
            iw, ih = img_info["width"], img_info["height"]
            if iw == 0 or ih == 0:
                continue
            cx = (bx + bw / 2) / iw
            cy = (by + bh / 2) / ih
            nw = bw / iw
            nh = bh / ih
            if 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1:
                boxes_by_img[ann["image_id"]].append((CLASS_TO_IDX[canonical], cx, cy, nw, nh))

        for img_id, boxes in boxes_by_img.items():
            img_info = img_map[img_id]
            img_path = img_dir / img_info["file_name"]
            if img_path.exists() and boxes:
                records.append((img_path, boxes))

    if records:
        print(f"    Found {len(records)} COCO-format pairs in {base_dir}")
    return records


def collect_voc_dataset(base_dir, source_name="?"):
    """Collect from Pascal VOC XML annotation format."""
    base_dir = Path(base_dir)
    records = []

    for xml_path in base_dir.rglob("*.xml"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception:
            continue

        # Get image dimensions
        size = root.find("size")
        if size is None:
            continue
        try:
            iw = int(size.find("width").text)
            ih = int(size.find("height").text)
        except Exception:
            continue
        if iw == 0 or ih == 0:
            continue

        # Find corresponding image
        filename_el = root.find("filename")
        if filename_el is None:
            continue
        img_name = filename_el.text
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = xml_path.parent / (Path(img_name).stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        boxes = []
        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is None:
                continue
            canonical = resolve_alias(name_el.text or "")
            if canonical is None:
                continue
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            try:
                x1 = float(bndbox.find("xmin").text)
                y1 = float(bndbox.find("ymin").text)
                x2 = float(bndbox.find("xmax").text)
                y2 = float(bndbox.find("ymax").text)
            except Exception:
                continue
            cx = ((x1 + x2) / 2) / iw
            cy = ((y1 + y2) / 2) / ih
            w = (x2 - x1) / iw
            h = (y2 - y1) / ih
            if 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1:
                boxes.append((CLASS_TO_IDX[canonical], cx, cy, w, h))

        if boxes:
            records.append((img_path, boxes))

    if records:
        print(f"    Found {len(records)} VOC-format pairs in {base_dir}")
    return records


def collect_fruit360(base_dir, source_name="Fruit-360"):
    """
    Special handler for Fruit-360 classification dataset.

    Each subfolder is a fruit class. Each image contains ONE fruit, centered.
    We create a pseudo bounding box that covers 85% of the image.
    This is valid because Fruit-360 images are always centered, white-background,
    and the fruit fills most of the frame.
    """
    base_dir = Path(base_dir)
    records = []

    # Use iterdir() at each level — avoid rglob which walks the entire tree
    # before we know if a folder is relevant. Fruit-360 is only 2-3 levels deep.
    def _walk_dir(d, depth=0):
        if depth > 3:
            return
        try:
            entries = list(d.iterdir())
        except PermissionError:
            return
        for entry in entries:
            if not entry.is_dir():
                continue
            folder_name = entry.name
            canonical = FRUIT360_FOLDER_MAP.get(folder_name)
            if canonical is None:
                for key, val in FRUIT360_FOLDER_MAP.items():
                    if key.lower() in folder_name.lower():
                        canonical = val
                        break
            if canonical is not None:
                cls_id = CLASS_TO_IDX[canonical]
                for img_path in entry.iterdir():
                    if img_path.suffix.lower() in IMG_EXTENSIONS:
                        records.append((img_path, [(cls_id, 0.5, 0.5, 0.85, 0.85)]))
            else:
                # Not a known class folder — may be a split folder like "Training"
                _walk_dir(entry, depth + 1)

    _walk_dir(base_dir)

    if records:
        print(f"    Fruit-360: {len(records)} pseudo-labeled images")
    return records


# ============================================================
# COLLECTION ROUTER: decides which handler to use per dataset folder
# ============================================================

DATASET_HANDLERS = [
    # (folder_name_pattern, handler_function, description)
    ("fruits_360", collect_fruit360, "Fruit-360 pseudo-labeler"),
    ("fruit_quality", None, "Fruit Quality - auto-detect format"),
    ("fruit_veg_retail", None, "Retail Fruits - auto-detect format"),
    ("mango_detection", None, "Mango Detection - auto-detect format"),
    ("fruit_detection_v2", None, "Fruit Detection v2 - auto-detect format"),
    # Roboflow datasets (standard YOLO format)
    ("rf_", None, "Roboflow dataset - YOLO format"),
    # Existing v2 base datasets
    ("fruit_detection_kaggle", None, "Base Kaggle dataset"),
    ("lvis_fruits", None, "LVIS fruits"),
    ("fruit-lpwjt", None, "Roboflow Fruit v1"),
    ("watermelon-muqdf", None, "Roboflow Watermelon"),
    ("fruit-detection-dnwrs", None, "Roboflow Fruit v2"),
]


def auto_detect_and_collect(folder, source_name):
    """Try format handlers, but only those whose marker files exist (fast early-exit)."""
    records = []
    folder = Path(folder)

    # --- YOLO: only attempt if a data.yaml exists ---
    if find_yaml(folder) is not None:
        r = collect_yolo_dataset(folder, source_name)
        records.extend(r)

    # --- COCO: only attempt if any _annotations.coco.json exists ---
    has_coco = any(folder.rglob("_annotations.coco.json"))
    if has_coco:
        r = collect_coco_dataset(folder, source_name)
        records.extend(r)

    # --- VOC: only attempt if any .xml files exist ---
    has_voc = any(True for _ in folder.rglob("*.xml"))
    if has_voc and not records:  # VOC is least common; skip if already got data
        r = collect_voc_dataset(folder, source_name)
        records.extend(r)

    # Deduplicate within this dataset (same image may appear in multiple splits)
    seen = set()
    unique = []
    for img_path, boxes in records:
        key = str(img_path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append((img_path, boxes))

    return unique


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare v3 fruit detection dataset")
    parser.add_argument("--output", default="dataset_v3_raw",
                        help="Output directory for merged dataset (default: dataset_v3_raw)")
    parser.add_argument("--max-per-source", type=int, default=15000,
                        help="Max images to take from a single source (default: 15000)")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    output_dir = Path(args.output)

    print("=" * 60)
    print("  FRUIT DETECTION - DATASET PREPARATION v3")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # ----------------------------------------------------------
    # Step 1: Collect from all sources
    # ----------------------------------------------------------
    print("\n[1/4] Collecting from all dataset sources...\n")

    all_records = []
    seen_hashes = set()
    source_stats = {}

    for source_dir in sorted(RAW_DIR.iterdir()):
        if not source_dir.is_dir():
            continue

        print(f"\n  -- {source_dir.name} --")
        source_name = source_dir.name

        # Choose handler
        if "fruits_360" in source_name or "fruits-262" in source_name:
            records = collect_fruit360(source_dir, source_name)
        else:
            records = auto_detect_and_collect(source_dir, source_name)

        # Apply per-source cap
        if len(records) > args.max_per_source:
            random.shuffle(records)
            records = records[:args.max_per_source]
            print(f"    Capped to {args.max_per_source} images from this source")

        # Global deduplication by image hash
        kept = 0
        for img_path, boxes in records:
            try:
                h = file_hash(img_path)
            except Exception:
                continue
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            all_records.append((img_path, boxes))
            kept += 1

        source_stats[source_name] = kept
        print(f"    Kept {kept} unique images from {source_name}")

    # Also include existing v2 dataset if present
    v2_dir = Path("dataset_v2")
    if v2_dir.exists():
        print(f"\n  -- dataset_v2 (existing) --")
        v2_records = []
        for split in ["train", "valid", "test"]:
            img_dir = v2_dir / split / "images"
            lbl_dir = v2_dir / split / "labels"
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in IMG_EXTENSIONS:
                    continue
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    continue
                boxes = []
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            try:
                                boxes.append((int(parts[0]),
                                              float(parts[1]), float(parts[2]),
                                              float(parts[3]), float(parts[4])))
                            except ValueError:
                                pass
                if boxes:
                    try:
                        h = file_hash(img_path)
                    except Exception:
                        continue
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        v2_records.append((img_path, boxes))

        source_stats["dataset_v2"] = len(v2_records)
        all_records.extend(v2_records)
        print(f"    Added {len(v2_records)} unique images from dataset_v2")

    print(f"\n  Total unique images collected: {len(all_records):,}")
    for src, count in sorted(source_stats.items(), key=lambda x: -x[1]):
        print(f"    {src:45s} {count:6,}")

    if len(all_records) == 0:
        print("\n[ERROR] No images collected. Download datasets first:")
        print("  python download_datasets.py --kaggle-user YOUR_USER --kaggle-key YOUR_KEY")
        sys.exit(1)

    # ----------------------------------------------------------
    # Step 2: Per-class distribution check
    # ----------------------------------------------------------
    print("\n[2/4] Class distribution:")
    class_counts = defaultdict(int)
    for _, boxes in all_records:
        for cls_id, *_ in boxes:
            class_counts[cls_id] += 1

    total_boxes = sum(class_counts.values())
    for i, name in enumerate(FINAL_CLASSES):
        c = class_counts[i]
        pct = 100 * c / max(total_boxes, 1)
        bar = "#" * int(pct / 2)
        print(f"  {i}: {name:15s} {c:7,} ({pct:5.1f}%)  {bar}")

    # ----------------------------------------------------------
    # Step 3: Shuffle and split
    # ----------------------------------------------------------
    print(f"\n[3/4] Splitting {len(all_records):,} images 70/20/10...")
    random.shuffle(all_records)

    n_train = int(len(all_records) * TRAIN_RATIO)
    n_valid = int(len(all_records) * VALID_RATIO)

    splits = {
        "train": all_records[:n_train],
        "valid": all_records[n_train:n_train + n_valid],
        "test":  all_records[n_train + n_valid:],
    }

    # ----------------------------------------------------------
    # Step 4: Write output (parallelized)
    # ----------------------------------------------------------
    print(f"\n[4/4] Writing to {output_dir}/ (parallel copy with 8 threads)...")

    if output_dir.exists():
        print(f"  Removing old {output_dir}/...")
        shutil.rmtree(output_dir)

    def _copy_one(args):
        idx, img_path, boxes, img_dir, lbl_dir = args
        suffix = img_path.suffix.lower()
        new_img = img_dir / f"img_{idx:07d}{suffix}"
        new_lbl = lbl_dir / f"img_{idx:07d}.txt"
        try:
            shutil.copy2(img_path, new_img)
            with open(new_lbl, "w") as f:
                for b in boxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
            return True
        except Exception as e:
            print(f"  [WARN] Could not copy {img_path.name}: {e}")
            return False

    idx_counter = 0
    for split_name, records in splits.items():
        img_dir = output_dir / split_name / "images"
        lbl_dir = output_dir / split_name / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        tasks = []
        for img_path, boxes in records:
            tasks.append((idx_counter, img_path, boxes, img_dir, lbl_dir))
            idx_counter += 1

        copied = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            for result in executor.map(_copy_one, tasks):
                if result:
                    copied += 1

        print(f"  {split_name}: {copied:,} images written")

    # Write data yaml
    yaml_path = Path("data_v3_raw.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""# Fruit Detection Dataset v3 (Raw, pre-balancing)
# Generated by prepare_dataset_v3.py

path: ./{output_dir}

train: train/images
val:   valid/images
test:  test/images

nc: {len(FINAL_CLASSES)}

names:
""")
        for i, name in enumerate(FINAL_CLASSES):
            f.write(f"  {i}: {name}\n")

    print(f"\n  Config written to: {yaml_path}")

    print("\n" + "=" * 60)
    print("  PREPARATION COMPLETE")
    print(f"  Total images: {idx_counter:,}")
    print(f"  Output: {output_dir}/")
    print("\n  Next steps:")
    print("  1. python balance_dataset.py \\")
    print(f"       --source {output_dir} --out dataset_v3 \\")
    print("       --min_boxes 2500 --max_boxes 5000")
    print("  2. python train.py --model yolov8s.pt --data data_v3.yaml \\")
    print("       --name fruit_v3 --epochs 120 --augment --batch 8")
    print("=" * 60)


if __name__ == "__main__":
    main()
