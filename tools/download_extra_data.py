"""
Download extra real Open Images V7 data for scarce fruit classes.

This script adds only the underrepresented classes to dataset_v3_raw/train:
  - pomegranate -> class 7
  - watermelon  -> class 5
  - pineapple   -> class 4

Typical usage:
    python download_extra_data.py --download --integrate
    python download_extra_data.py --download --integrate --pomegranate 3500 --watermelon 1700 --pineapple 1400

After this finishes, rebuild the prepared dataset:
    python prepare_pod.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path


TARGET_CLASSES = {
    "Pomegranate": 7,
    "Watermelon": 5,
    "Pineapple": 4,
}

DEFAULT_SAMPLES_PER_CLASS = {
    "Pomegranate": 3500,
    "Watermelon": 1700,
    "Pineapple": 1400,
}

OUTPUT_RAW_DIR = Path("dataset_v3_raw")
FO_EXPORT_DIR = Path("fiftyone_yolo_export")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and integrate scarce fruit classes from Open Images V7"
    )
    parser.add_argument("--download", action="store_true", help="Download/export Open Images data via FiftyOne")
    parser.add_argument("--integrate", action="store_true", help="Integrate exported YOLO data into dataset_v3_raw/train")
    parser.add_argument("--clean-export", action="store_true", help="Delete temporary FiftyOne YOLO export after integration")
    parser.add_argument("--export-dir", type=Path, default=FO_EXPORT_DIR, help="Temporary YOLO export directory")
    parser.add_argument("--raw-dir", type=Path, default=OUTPUT_RAW_DIR, help="Raw dataset directory to append to")
    parser.add_argument("--pomegranate", type=int, default=DEFAULT_SAMPLES_PER_CLASS["Pomegranate"])
    parser.add_argument("--watermelon", type=int, default=DEFAULT_SAMPLES_PER_CLASS["Watermelon"])
    parser.add_argument("--pineapple", type=int, default=DEFAULT_SAMPLES_PER_CLASS["Pineapple"])
    return parser.parse_args()


def requested_samples(args: argparse.Namespace) -> dict[str, int]:
    return {
        "Pomegranate": args.pomegranate,
        "Watermelon": args.watermelon,
        "Pineapple": args.pineapple,
    }


def download_and_export(export_dir: Path, samples_per_class: dict[str, int]) -> None:
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise SystemExit(
            "FiftyOne is required for Open Images downloads.\n"
            "Install it with: python -m pip install fiftyone"
        ) from exc

    print("=" * 60)
    print("  DOWNLOADING SCARCE CLASSES FROM OPEN IMAGES V7")
    print("=" * 60)

    if export_dir.exists():
        print(f"  Removing old export dir: {export_dir}")
        shutil.rmtree(export_dir, ignore_errors=True)

    for class_name, max_samples in samples_per_class.items():
        if max_samples <= 0:
            print(f"\n  [SKIP] {class_name}: requested 0 samples")
            continue

        dataset_name = f"fruit-extra-{class_name.lower()}"
        class_export_dir = export_dir / class_name.lower()

        print(f"\n  Downloading {class_name}: up to {max_samples} images")
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=[class_name],
            max_samples=max_samples,
            dataset_name=dataset_name,
            overwrite=True,
        )

        print(f"  [OK] Downloaded {len(dataset)} images for {class_name}")
        print(f"  Exporting YOLO labels to: {class_export_dir}")
        dataset.export(
            export_dir=str(class_export_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            classes=[class_name],
        )

    print("\n[OK] Open Images export complete.")


def load_export_class_map(dataset_yaml: Path) -> dict[int, str]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Install dependencies from requirements.txt") from exc

    with open(dataset_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    names = cfg.get("names", {})
    if isinstance(names, list):
        return {idx: name for idx, name in enumerate(names)}
    return {int(idx): name for idx, name in names.items()}


def find_image_for_label(images_root: Path, split_name: str, stem: str) -> Path | None:
    candidates = [images_root / split_name / f"{stem}{ext}" for ext in IMG_EXTS]
    candidates.extend(images_root.rglob(f"{stem}.*"))
    for img in candidates:
        if img.exists() and img.suffix.lower() in IMG_EXTS:
            return img
    return None


def next_output_stem(out_img_dir: Path) -> str:
    existing = len(list(out_img_dir.glob("oi_v7_*.jpg")))
    while True:
        stem = f"oi_v7_{existing:06d}"
        if not any((out_img_dir / f"{stem}{ext}").exists() for ext in IMG_EXTS):
            return stem
        existing += 1


def integrate_one_export(class_export_dir: Path, raw_dir: Path) -> tuple[int, dict[int, int]]:
    dataset_yaml = class_export_dir / "dataset.yaml"
    labels_root = class_export_dir / "labels"
    images_root = class_export_dir / "images"

    if not dataset_yaml.exists() or not labels_root.exists() or not images_root.exists():
        print(f"  [WARN] Skipping incomplete export: {class_export_dir}")
        return 0, {}

    exported_names = load_export_class_map(dataset_yaml)
    export_to_ours = {
        export_id: TARGET_CLASSES[name]
        for export_id, name in exported_names.items()
        if name in TARGET_CLASSES
    }

    if not export_to_ours:
        print(f"  [WARN] No target classes found in: {dataset_yaml}")
        return 0, {}

    out_img_dir = raw_dir / "train" / "images"
    out_lbl_dir = raw_dir / "train" / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    added_imgs = 0
    added_boxes: dict[int, int] = defaultdict(int)

    for label_file in labels_root.rglob("*.txt"):
        split_name = label_file.parent.name
        image_file = find_image_for_label(images_root, split_name, label_file.stem)
        if image_file is None:
            continue

        new_boxes = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                export_id = int(float(parts[0]))
                if export_id not in export_to_ours:
                    continue
                our_id = export_to_ours[export_id]
                new_boxes.append(f"{our_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

        if not new_boxes:
            continue

        new_stem = next_output_stem(out_img_dir)
        out_image = out_img_dir / f"{new_stem}{image_file.suffix.lower()}"
        out_label = out_lbl_dir / f"{new_stem}.txt"

        shutil.copy2(image_file, out_image)
        with open(out_label, "w", encoding="utf-8") as f:
            f.write("\n".join(new_boxes) + "\n")

        added_imgs += 1
        for box in new_boxes:
            added_boxes[int(box.split()[0])] += 1

    return added_imgs, added_boxes


def integrate_into_raw(export_dir: Path, raw_dir: Path) -> None:
    print("=" * 60)
    print("  INTEGRATING NEW DATA INTO dataset_v3_raw")
    print("=" * 60)

    if not export_dir.exists():
        raise SystemExit(f"Export directory not found: {export_dir}")

    export_roots = [
        p for p in sorted(export_dir.iterdir())
        if p.is_dir() and (p / "dataset.yaml").exists()
    ]
    if (export_dir / "dataset.yaml").exists():
        export_roots.insert(0, export_dir)

    if not export_roots:
        raise SystemExit(f"No YOLO exports found under: {export_dir}")

    total_imgs = 0
    total_boxes: dict[int, int] = defaultdict(int)

    for class_export_dir in export_roots:
        print(f"\n  Integrating export: {class_export_dir}")
        added_imgs, added_boxes = integrate_one_export(class_export_dir, raw_dir)
        total_imgs += added_imgs
        for cls_id, count in added_boxes.items():
            total_boxes[cls_id] += count
        print(f"  [OK] Added {added_imgs} images")

    id_to_name = {v: k for k, v in TARGET_CLASSES.items()}
    print("\n[OK] Integration complete.")
    print(f"  Added {total_imgs} real images to {raw_dir / 'train'}")
    for cls_id in sorted(total_boxes):
        print(f"  Added {total_boxes[cls_id]} {id_to_name[cls_id]} boxes")


def main() -> None:
    args = parse_args()

    if not args.download and not args.integrate:
        print("No action selected. Use --download, --integrate, or both.")
        sys.exit(2)

    if args.download:
        download_and_export(args.export_dir, requested_samples(args))

    if args.integrate:
        integrate_into_raw(args.export_dir, args.raw_dir)
        if args.clean_export:
            print(f"\n  Removing temporary export dir: {args.export_dir}")
            shutil.rmtree(args.export_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
