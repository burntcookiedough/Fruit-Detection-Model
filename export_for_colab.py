"""
Fruit Detection - Export Dataset for Colab/Kaggle Upload
Packages dataset_v3 into a zip file ready to upload.

Usage:
    python export_for_colab.py
    python export_for_colab.py --output my_dataset.zip
"""

import argparse
import zipfile
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="dataset_v3",
                        help="Dataset directory to zip (default: dataset_v3)")
    parser.add_argument("--output", default="dataset_v3_colab.zip",
                        help="Output zip file name (default: dataset_v3_colab.zip)")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        print(f"[ERROR] {source} not found. Run balance_dataset.py first.")
        return

    # Count files
    all_files = list(source.rglob("*"))
    img_files = [f for f in all_files if f.suffix.lower() in {".jpg", ".png", ".jpeg"}]
    lbl_files = [f for f in all_files if f.suffix == ".txt"]

    print("=" * 60)
    print("  FRUIT DETECTION - COLAB EXPORT")
    print("=" * 60)
    print(f"  Source   : {source}/")
    print(f"  Output   : {output}")
    print(f"  Images   : {len(img_files):,}")
    print(f"  Labels   : {len(lbl_files):,}")
    print(f"  Estimated size: ~{len(img_files) * 25 // 1000} MB")
    print()

    start = time.time()

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for i, file_path in enumerate(sorted(all_files)):
            if not file_path.is_file():
                continue
            arc_name = str(file_path)  # preserves dataset_v3/train/images/... path
            zf.write(file_path, arc_name)

            if (i + 1) % 2000 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (len(all_files) - i - 1) / rate
                print(f"  {i+1:,}/{len(all_files):,} files | {elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")

    elapsed = time.time() - start
    size_mb = output.stat().st_size / 1e6

    print()
    print("=" * 60)
    print(f"  EXPORT COMPLETE in {elapsed:.0f}s")
    print(f"  Output: {output}")
    print(f"  Size  : {size_mb:.1f} MB")
    print()
    print("  Next steps:")
    print("  1. Upload dataset_v3_colab.zip to Google Colab or Kaggle")
    print("  2. Open Fruit_Detection_v3_Training.ipynb in Colab")
    print("  3. Follow the notebook steps")
    print("=" * 60)


if __name__ == "__main__":
    main()
