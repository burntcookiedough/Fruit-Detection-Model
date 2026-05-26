"""
download_github_datasets.py
============================
Downloads high-priority fruit detection datasets from GitHub into raw_datasets/.

Datasets:
  1. PG-YOLO-Dataset          → 13,840 pomegranate images (LforikC)
  2. dataset_fruits_detection → 8,479 multi-fruit images (lightly-ai)
  3. OBJECT_DETECTION         → 9-class YOLO dataset (WOLFGAIZER)
  4. Pastai                   → ~4,700 watermelon images (AlkaSaliss)

Usage:
    python download_github_datasets.py
    python download_github_datasets.py --force   # re-download existing
"""

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path("raw_datasets")

DATASETS = [
    {
        "name": "pg_yolo",
        "repo": "LforikC/PG-YOLO-Dataset",
        "description": "13,840 pomegranate images, YOLO format, real orchard scenes",
        "nested_path": "PG-YOLO-Dataset-main",
    },
    {
        "name": "lightly_fruits",
        "repo": "lightly-ai/dataset_fruits_detection",
        "description": "8,479 images, YOLOv8 format, 6 fruit classes",
        "nested_path": "dataset_fruits_detection-main",
    },
    {
        "name": "wolfgaizer_fruits",
        "repo": "WOLFGAIZER/OBJECT_DETECTION",
        "description": "9-class YOLO dataset incl. pineapple, mango, kiwi",
        "nested_path": "OBJECT_DETECTION-main",
    },
    {
        "name": "pastai_watermelon",
        "repo": "AlkaSaliss/Pastai",
        "description": "~4,700 watermelon images, YOLOv5 format, real field photos",
        "nested_path": "Pastai-main",
    },
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download_and_extract(repo, dest_folder, nested_path, force=False):
    dest = RAW_DIR / dest_folder
    if dest.exists() and any(dest.iterdir()) and not force:
        print(f"  [SKIP] {dest_folder} already exists. Use --force to re-download.")
        return True

    zip_url = f"https://github.com/{repo}/archive/refs/heads/main.zip"
    print(f"\n  Downloading: {repo}")

    if dest.exists():
        print(f"  Removing old {dest}...")
        shutil.rmtree(dest, ignore_errors=True)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            print(f"  Downloading ZIP...")
            urllib.request.urlretrieve(zip_url, tmp_path)
            print(f"  Extracting...")
            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(RAW_DIR)
            tmp_path.unlink()
            tmp_path = None
    except Exception as e:
        zip_url_master = f"https://github.com/{repo}/archive/refs/heads/master.zip"
        print(f"  main branch failed ({e}), trying master...")
        try:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                urllib.request.urlretrieve(zip_url_master, tmp_path)
                with zipfile.ZipFile(tmp_path, "r") as zf:
                    zf.extractall(RAW_DIR)
                tmp_path.unlink()
                tmp_path = None
        except Exception as e2:
            print(f"  [ERROR] Download failed: {e2}")
            return False
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()

    extracted = RAW_DIR / nested_path
    if extracted.exists():
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        extracted.rename(dest)
        print(f"  [OK] Extracted to {dest}")
    else:
        candidates = list(RAW_DIR.glob(f"{dest_folder}*"))
        if candidates:
            for c in candidates:
                if c != dest and c.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest, ignore_errors=True)
                    c.rename(dest)
                    break
        print(f"  [OK] Extracted to {dest}")

    return True


def validate_dataset(folder_name):
    folder = RAW_DIR / folder_name
    if not folder.exists():
        print(f"  [WARN] {folder_name} not found.")
        return 0, 0

    images = []
    for ext in IMG_EXTS:
        images.extend(folder.rglob(f"*{ext}"))

    labels = list(folder.rglob("*.txt"))

    return len(images), len(labels)


def main():
    parser = argparse.ArgumentParser(description="Download fruit datasets from GitHub")
    parser.add_argument("--force", action="store_true", help="Re-download existing datasets")
    args = parser.parse_args()

    print("=" * 60)
    print("  FRUIT DETECTION — GITHUB DATASET DOWNLOADER")
    print("=" * 60)

    RAW_DIR.mkdir(exist_ok=True)

    success = 0
    for ds in DATASETS:
        ok = download_and_extract(ds["repo"], ds["name"], ds["nested_path"], force=args.force)
        if ok:
            success += 1
            print(f"  -> {ds['description']}")

    print(f"\n  {success}/{len(DATASETS)} datasets processed.")

    print("\n" + "=" * 60)
    print("  VALIDATION")
    print("=" * 60)

    total_images = 0
    total_labels = 0
    for ds in DATASETS:
        n_img, n_lbl = validate_dataset(ds["name"])
        total_images += n_img
        total_labels += n_lbl
        status = "[OK]" if n_img > 0 else "[!!]"
        print(f"  {status} {ds['name']:30s} {n_img:6,} images  {n_lbl:6,} labels  —  {ds['description']}")

    print(f"\n  Total: {total_images:,} images, {total_labels:,} labels")
    print("\n  Next step: python prepare_dataset_v3.py --output dataset_v4_raw --max-per-source 20000")
    print("=" * 60)


if __name__ == "__main__":
    main()
