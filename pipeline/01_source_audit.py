"""
source_audit_v4.py
==================
Audit raw_datasets/ before building Fruit Detection V4.

The goal is to classify each raw source before it can affect training:
  - eligible_yolo: already has usable YOLO labels and an explicit class map
  - needs_pseudo_label_review: image-only or unlabeled source
  - needs_conversion: labels exist, but not in directly trusted YOLO format
  - exclude: known synthetic/unsupported source

This script writes source_audit_v4.csv and prints a compact report. It does not
modify datasets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ELIGIBLE_SOURCE_TAGS = {
    "fruit_detection_kaggle",
    "fruit_detection_yolov8",
    "fruit_detection_yolo_2025",
    "fruit_quality_detection",
    "fruit_ripeness_detection",
    "fruit-detection-dnwrs",
    "fruit-lpwjt",
    "lvis_fruits",
    "mango_ripeness_yolo",
    "pomegranate_yolo",
    "watermelon-muqdf",
}

NEEDS_REVIEW = {
    "PG-YOLO-Dataset-master",
    "pomegranate_dataset",
    "mango_dds",
}

NEEDS_CONVERSION = {
    "fruits_by_yolo",
    "fruits_obj_detection",
}

EXCLUDE = {
    "fruits_360",
    "fruits_262",
    "fruits_360_yolo",
}


def count_files(root: Path, exts: set[str]) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def classify(name: str, images: int, labels: int) -> tuple[str, str]:
    if name in EXCLUDE or "fruits_360" in name.lower() or "fruits-360" in name.lower():
        return "exclude", "synthetic_or_explicitly_excluded"
    if name in ELIGIBLE_SOURCE_TAGS:
        if images == 0:
            return "exclude", "no_images"
        if labels == 0:
            return "needs_pseudo_label_review", "eligible_name_but_no_yolo_labels"
        return "eligible_yolo", "explicit_source_rule"
    if name in NEEDS_REVIEW:
        return "needs_pseudo_label_review", "image_only_or_untrusted_labels"
    if name in NEEDS_CONVERSION:
        return "needs_conversion", "nonstandard_annotation_layout"
    if images > 0 and labels == 0:
        return "needs_pseudo_label_review", "image_only"
    if images > 0 and labels > 0:
        return "needs_conversion", "unknown_label_schema"
    return "exclude", "no_usable_images"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw sources for V4 dataset build")
    parser.add_argument("--raw", default="raw_datasets", help="Raw datasets directory")
    parser.add_argument("--out", default="source_audit_v4.csv", help="CSV report path")
    args = parser.parse_args()

    raw_root = Path(args.raw)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw datasets directory not found: {raw_root}")

    rows = []
    for source in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        images = count_files(source, IMG_EXTS)
        labels = count_files(source, {".txt"})
        yamls = count_files(source, {".yaml", ".yml"})
        status, reason = classify(source.name, images, labels)
        rows.append({
            "source": source.name,
            "status": status,
            "reason": reason,
            "images": images,
            "txt_labels": labels,
            "yaml_files": yamls,
        })

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "status", "reason", "images", "txt_labels", "yaml_files"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 72)
    print("  FRUIT DETECTION V4 - SOURCE AUDIT")
    print("=" * 72)
    for row in rows:
        print(
            f"{row['source']:<32} {row['status']:<26} "
            f"{row['images']:>7} images  {row['txt_labels']:>7} labels  {row['reason']}"
        )
    print("=" * 72)
    print(f"Wrote: {out.resolve()}")


if __name__ == "__main__":
    main()
