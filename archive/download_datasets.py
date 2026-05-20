"""
Fruit Detection - Dataset Downloader v3
Downloads a curated set of real-world fruit datasets from Kaggle and Roboflow.

Usage:
    python download_datasets.py
    python download_datasets.py --kaggle-user YOUR_USERNAME --kaggle-key YOUR_KEY
    python download_datasets.py --roboflow-key YOUR_KEY
"""

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# ============================================================
# DATASET CATALOGUE
# ============================================================

# Each entry: (kaggle_dataset_id, local_folder_name, description, priority)
KAGGLE_DATASETS = [
    # ---- TIER 1: YOLO-format, real-world, multi-class (download these first) ----

    # Fruits by YOLO - 9 classes incl. mango, grapes; YOLO format; 2,974 images; rating 1.0
    ("kapturovalexander/fruits-by-yolo-fruits-detection", "fruits_by_yolo",
     "2,974 imgs, YOLO format, 9 classes incl. mango+grapes"),

    # Fruit detection YOLO - recent 2025, usability 1.0
    ("itsmeaman03/fruit-detection-yolo", "fruit_detection_yolo_2025",
     "YOLO format 2025, diverse fruit detection"),

    # Fruits Images Dataset: Object Detection (real photos, diverse)
    ("afsananadia/fruits-images-dataset-object-detection", "fruits_obj_detection",
     "Object detection format, real photos"),

    # Fruit detection for YOLOv8 (cubeai dataset)
    ("cubeai/fruit-detection-for-yolov8", "fruit_detection_yolov8",
     "YOLOv8 format, 503MB, diverse fruit detection"),

    # Fruit Ripeness Object Detection - diverse real-world, rating 0.9375
    ("killa92/fruit-ripeness-object-detection-dataset", "fruit_ripeness_detection",
     "Fruit ripeness detection, real-world diverse backgrounds"),

    # ---- TIER 2: Large classification datasets for pseudo-labeling ----

    # Fruits-360 (moltean) - 90,000+ images, 262 types incl. mango, pomegranate, grapes
    # These get pseudo-labeled (centered bbox) since they're classification images
    ("moltean/fruits", "fruits_360",
     "90k classification images, 262 types -> pseudo-label for detection"),

    # Fruits-262 (aelchimminut) - similar to Fruits-360 but different angles
    ("aelchimminut/fruits262", "fruits_262",
     "262 fruit classes, classification -> pseudo-label"),

    # Fruit and Vegetable Image Recognition - 61k downloads, 516 votes
    ("kritikseth/fruit-and-vegetable-image-recognition", "fruit_veg_recognition",
     "Classification dataset, 61k downloads, diverse fruits+vegetables"),

    # ---- TIER 3: Minority class targeted datasets ----

    # MangoFruitDDS - mango detection with diverse backgrounds
    ("warcoder/mangofruitdds", "mango_dds",
     "Mango detection diverse backgrounds"),

    # Mango ripeness YOLOv8 - real-world mango images with annotations
    ("rifansaputra/mango-ripeness-detection-yolov8", "mango_ripeness_yolo",
     "Mango ripeness YOLOv8 format, real photos"),

    # Pomegranate Ripeness Detection YOLOv8
    ("cubeai/pomegranate-ripeness-detection-for-yolov8", "pomegranate_yolo",
     "Pomegranate YOLOv8 format - fills pomegranate gap"),

    # Pomegranate Fruit Dataset - 1385 downloads, real images
    ("kumararun37/pomegranate-fruit-dataset", "pomegranate_dataset",
     "Pomegranate fruit dataset, 1385 downloads"),

    # FruitNet: Indian Fruits Dataset with Quality (pomegranate + more)
    ("shashwatwork/fruitnet-indian-fruits-dataset-with-quality", "fruitnet_indian",
     "Indian fruits incl. pomegranate, quality labels, real photos"),

    # Fruit Quality Dataset (abrars2) - usability 1.0, real phone camera images
    ("abrars2/fruit-quality-classificaltion-and-detection", "fruit_quality_detection",
     "Fruit quality detection, 1.0 usability, real phone camera"),

    # Fruits-360-YOLO (pre-converted YOLO format)
    ("miguelcruz67/fruits-360-yolo", "fruits_360_yolo",
     "Fruits-360 pre-converted to YOLO format"),
]

# Roboflow Universe public datasets to download via API
# Format: (workspace, project, version, local_folder, description)
ROBOFLOW_DATASETS = [
    # Multi-class fruit detection - real world diverse
    ("fruit-detection-hxoan", "fruit-detection-8", 3,
     "rf_fruit_detection_8class",
     "8-class real-world fruit detection"),

    # Mango detection - fills the mango gap
    ("mango-detection-9hgqy", "mango-detection-jluq6", 1,
     "rf_mango",
     "Mango-specific detection, diverse backgrounds"),

    # Pomegranate detection - fills the pomegranate gap
    ("roboflow-100", "pomegranate-ph4rd", 1,
     "rf_pomegranate",
     "Pomegranate detection dataset"),

    # Grapes detection
    ("roboflow-100", "grape-detection-berries", 1,
     "rf_grapes",
     "Grape/berry detection in natural settings"),

    # Fruit detection with real-world variety
    ("fruit-detection-7yxnl", "fruit-detection-5", 2,
     "rf_fruit_detection_varied",
     "Fruit detection with market/natural backgrounds"),

    # Combined Vegetables & Fruits - 42K images, 47 classes (all ours included)
    ("yolo-jpkho", "combined-vegetables-fruits", 1,
     "rf_combined_veg_fruits",
     "42K images, 47 classes incl. all 8 of our fruits"),
]

RAW_DIR = Path("raw_datasets")


# ============================================================
# CREDENTIAL SETUP
# ============================================================

def setup_kaggle_credentials(username, key):
    """Write kaggle.json to the right place."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"

    creds = {"username": username, "key": key}
    with open(kaggle_json, "w") as f:
        json.dump(creds, f)

    # Kaggle requires 600 permissions on unix; on Windows this is a no-op
    try:
        kaggle_json.chmod(0o600)
    except Exception:
        pass

    print(f"  [OK] Kaggle credentials written to {kaggle_json}")


def find_roboflow_key(cli_key=None):
    """Find Roboflow API key from CLI arg, env var, or config file."""
    if cli_key:
        return cli_key

    # Environment variable
    env_key = os.environ.get("ROBOFLOW_API_KEY")
    if env_key:
        return env_key

    # Config file locations
    for cfg_path in [
        Path.home() / ".config" / "roboflow" / "config.json",
        Path.home() / ".roboflow" / "config.json",
        Path("roboflow_config.json"),
    ]:
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    data = json.load(f)
                    key = data.get("api_key") or data.get("apiKey")
                    if key:
                        print(f"  [OK] Roboflow key loaded from {cfg_path}")
                        return key
            except Exception:
                pass

    return None


# ============================================================
# KAGGLE DOWNLOAD
# ============================================================

def download_kaggle_dataset(dataset_id, dest_folder, description):
    """Download and extract a single Kaggle dataset."""
    dest = RAW_DIR / dest_folder

    if dest.exists() and any(dest.iterdir()):
        print(f"  [SKIP] {dest_folder} already exists.")
        return True

    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n  Downloading: {dataset_id}")
    print(f"  -> {description}")

    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", dataset_id, "-p", str(dest), "--unzip"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"  [ERROR] Failed: {result.stderr.strip()}")
        # Try without --unzip and unzip manually
        result2 = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download",
             "-d", dataset_id, "-p", str(dest)],
            capture_output=True, text=True
        )
        if result2.returncode == 0:
            # Manual unzip
            for zf in dest.glob("*.zip"):
                print(f"  Extracting {zf.name}...")
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(dest)
            return True
        print(f"  [ERROR] Download failed for {dataset_id}. Skipping.")
        return False

    print(f"  [OK] Downloaded and extracted to {dest}")
    return True


def download_all_kaggle():
    print("\n" + "=" * 60)
    print("  KAGGLE DATASETS")
    print("=" * 60)

    success_count = 0
    for dataset_id, folder, desc in KAGGLE_DATASETS:
        ok = download_kaggle_dataset(dataset_id, folder, desc)
        if ok:
            success_count += 1

    print(f"\n  Kaggle: {success_count}/{len(KAGGLE_DATASETS)} datasets ready.")


# ============================================================
# ROBOFLOW DOWNLOAD
# ============================================================

def download_roboflow_dataset(workspace, project, version, folder, description, api_key):
    """Download a single Roboflow dataset via the Python SDK."""
    dest = RAW_DIR / folder

    if dest.exists() and any(dest.iterdir()):
        print(f"  [SKIP] {folder} already exists.")
        return True

    print(f"\n  Downloading Roboflow: {workspace}/{project} v{version}")
    print(f"  -> {description}")

    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download("yolov8", location=str(dest))
        print(f"  [OK] Downloaded to {dest}")
        return True
    except Exception as e:
        print(f"  [ERROR] Roboflow download failed: {e}")
        # Try alternative: direct REST API
        return download_roboflow_direct(workspace, project, version, dest, api_key)


def download_roboflow_direct(workspace, project, version, dest, api_key):
    """Fallback: download via Roboflow REST API."""
    import urllib.request
    import json as jsonlib

    url = f"https://api.roboflow.com/{workspace}/{project}/{version}/yolov8?api_key={api_key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Python/3"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = jsonlib.loads(resp.read())

        export_link = data.get("export", {}).get("link")
        if not export_link:
            print(f"  [ERROR] No export link in API response")
            return False

        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / "dataset.zip"
        urllib.request.urlretrieve(export_link, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest)
        zip_path.unlink()
        print(f"  [OK] Downloaded via REST to {dest}")
        return True
    except Exception as e:
        print(f"  [ERROR] REST API also failed: {e}")
        return False


def download_all_roboflow(api_key):
    print("\n" + "=" * 60)
    print("  ROBOFLOW UNIVERSE DATASETS")
    print("=" * 60)

    if not api_key:
        print("  [WARN] No Roboflow API key found. Skipping Roboflow downloads.")
        print("  Set ROBOFLOW_API_KEY env var or pass --roboflow-key")
        return

    success_count = 0
    for workspace, project, version, folder, desc in ROBOFLOW_DATASETS:
        ok = download_roboflow_dataset(workspace, project, version, folder, desc, api_key)
        if ok:
            success_count += 1

    print(f"\n  Roboflow: {success_count}/{len(ROBOFLOW_DATASETS)} datasets ready.")


# ============================================================
# SUMMARY
# ============================================================

def print_summary():
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)

    total_images = 0
    for folder_name in list(d[1] for d in KAGGLE_DATASETS) + list(d[3] for d in ROBOFLOW_DATASETS):
        folder = RAW_DIR / folder_name
        if folder.exists():
            imgs = list(folder.rglob("*.jpg")) + list(folder.rglob("*.png")) + list(folder.rglob("*.jpeg"))
            count = len(imgs)
            total_images += count
            print(f"  {folder_name:40s} {count:6d} images")
        else:
            print(f"  {folder_name:40s}  NOT downloaded")

    print(f"\n  Total raw images available: ~{total_images:,}")
    print("\n  Next step: python prepare_dataset_v3.py")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download fruit detection datasets from Kaggle and Roboflow"
    )
    parser.add_argument("--kaggle-user", help="Kaggle username")
    parser.add_argument("--kaggle-key", help="Kaggle API key")
    parser.add_argument("--roboflow-key", help="Roboflow API key")
    parser.add_argument("--kaggle-only", action="store_true", help="Only download Kaggle datasets")
    parser.add_argument("--roboflow-only", action="store_true", help="Only download Roboflow datasets")
    args = parser.parse_args()

    print("=" * 60)
    print("  FRUIT DETECTION - DATASET DOWNLOADER v3")
    print("=" * 60)

    # --- Setup Kaggle credentials ---
    kaggle_key = args.kaggle_key or os.environ.get("KAGGLE_KEY", "")
    kaggle_user = args.kaggle_user or os.environ.get("KAGGLE_USERNAME", "")

    if kaggle_key and kaggle_user:
        print(f"\n  Setting up Kaggle credentials for user: {kaggle_user}")
        setup_kaggle_credentials(kaggle_user, kaggle_key)
    else:
        # Check if kaggle.json already exists
        existing = Path.home() / ".kaggle" / "kaggle.json"
        if existing.exists():
            print(f"  [OK] Using existing Kaggle credentials at {existing}")
        else:
            print("\n  [ERROR] Kaggle credentials not set.")
            print("  Provide: --kaggle-user YOUR_USERNAME --kaggle-key YOUR_KEY")
            print("  Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
            sys.exit(1)

    # --- Find Roboflow key ---
    rf_key = find_roboflow_key(args.roboflow_key)

    # --- Ensure roboflow package is installed ---
    try:
        import roboflow
    except ImportError:
        print("\n  Installing roboflow package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "roboflow", "-q"], check=True)

    RAW_DIR.mkdir(exist_ok=True)

    # --- Download ---
    if not args.roboflow_only:
        download_all_kaggle()

    if not args.kaggle_only:
        download_all_roboflow(rf_key)

    print_summary()


if __name__ == "__main__":
    main()
