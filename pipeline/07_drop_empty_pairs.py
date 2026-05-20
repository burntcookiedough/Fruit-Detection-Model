"""
drop_empty_yolo_images.py
=========================
Remove YOLO image/label pairs whose label file is empty after filtering.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop empty-label YOLO image pairs")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--quarantine", default=None)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    quarantine = Path(args.quarantine) if args.quarantine else Path(f"{args.dataset}_empty_quarantine")
    removed = 0
    for split in ["train", "valid", "test"]:
        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"
        if not image_dir.exists():
            continue
        for label in sorted(label_dir.glob("*.txt")):
            if label.read_text(encoding="utf-8", errors="ignore").strip():
                continue
            image = next((image_dir / f"{label.stem}{ext}" for ext in IMG_EXTS if (image_dir / f"{label.stem}{ext}").exists()), None)
            q_label = quarantine / split / "labels" / label.name
            q_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(label), str(q_label))
            if image is not None:
                q_image = quarantine / split / "images" / image.name
                q_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(image), str(q_image))
            removed += 1
    print(f"Removed empty-label image pairs: {removed}")
    print(f"Quarantine: {quarantine.resolve()}")


if __name__ == "__main__":
    main()
