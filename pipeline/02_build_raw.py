"""
build_v4_raw.py
===============
Build dataset_v4_raw/ from trusted YOLO-style sources in raw_datasets/.

Strict rules:
  - Only explicitly mapped sources are included.
  - Unlabeled/image-only sources are skipped until reviewed labels exist.
  - All labels are remapped to the canonical 8 fruit classes.
  - Exact duplicate images are removed by MD5.
  - A manifest is written for traceability.
  - The build fails if any canonical class has fewer than 200 raw boxes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

CLASSES = ["apple", "banana", "orange", "mango", "pineapple", "watermelon", "grapes", "pomegranate"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SKIP_SOURCE_NAMES = {
    "fruits_360",
    "fruits_262",
    "fruits_360_yolo",
    "fruits_by_yolo",
    "fruits_obj_detection",
    "PG-YOLO-Dataset-master",
    "pomegranate_dataset",
    "mango_dds",
}


@dataclass(frozen=True)
class SourceSpec:
    tag: str
    root: Path
    names: list[str] | dict[int, str]
    note: str = ""


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not path.exists():
        return boxes
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:])
            except ValueError:
                continue
            boxes.append((cls_id, cx, cy, w, h))
    return boxes


def write_boxes(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cls_id, cx, cy, w, h in boxes:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def normalize_name(name: str) -> str:
    s = name.strip().lower()
    for token in ["bad_", "good_", " overripe", " ripe", " rotten", " unripe", "-0 day"]:
        s = s.replace(token, " ")
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = " ".join(s.split())
    return s


def canonical_from_name(name: str) -> str | None:
    n = normalize_name(name)
    aliases = {
        "apple": "apple",
        "banana": "banana",
        "orange": "orange",
        "orange fruit": "orange",
        "mango": "mango",
        "pineapple": "pineapple",
        "watermelon": "watermelon",
        "grape": "grapes",
        "grapes": "grapes",
        "pomegranate": "pomegranate",
    }
    if n in aliases:
        return aliases[n]
    for key, value in aliases.items():
        if key in n:
            return value
    return None


def source_names_from_yaml(path: Path) -> list[str] | dict[int, str]:
    with path.open(encoding="utf-8", errors="ignore") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return [str(v) for v in names]


def hardcoded_sources(raw_root: Path) -> list[SourceSpec]:
    return [
        SourceSpec(
            "fruit_detection_kaggle",
            raw_root / "fruit_detection_kaggle" / "extracted" / "Fruits-detection",
            ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon"],
        ),
        SourceSpec(
            "fruit_detection_yolov8",
            raw_root / "fruit_detection_yolov8",
            ["apple", "banana", "grapes", "orange", "pineapple", "watermelon"],
            "Chinese class names in YAML; hardcoded to inspected order.",
        ),
        SourceSpec(
            "fruit_detection_yolo_2025",
            raw_root / "fruit_detection_yolo_2025" / "fruit-detection-dataset" / "fruit-detection-dataset",
            ["apple", "avacado", "banana", "guava", "kiwi", "mango", "orange", "peach", "pineapple"],
            "YAML is malformed; hardcoded to declared order.",
        ),
        SourceSpec(
            "fruit_quality_detection",
            raw_root / "fruit_quality_detection" / "Fruit Quality Classification",
            [
                "Bad_Apple 0-1 day", "Bad_Banana-0 day", "Bad_Guava 0 day", "Bad_Lime-0-1-day",
                "Bad_Orange-0-1 day", "Bad_Pomegranate-0-1-day", "Good_Apple 10-21 days",
                "Good_Banana-2-3 days", "Good_Guava 5-7 days", "Good_Lime-10-21 days",
                "Good_Lime-10-21-days", "Good_Orange-15-21 days", "Good_Pomegranate 50-60-days",
                "Good_Pomegranate-50-60-days",
            ],
        ),
        SourceSpec(
            "fruit_ripeness_detection",
            raw_root / "fruit_ripeness_detection",
            source_names_from_yaml(raw_root / "fruit_ripeness_detection" / "data.yaml"),
        ),
        SourceSpec(
            "fruit-detection-dnwrs",
            raw_root / "fruit-detection-dnwrs",
            source_names_from_yaml(raw_root / "fruit-detection-dnwrs" / "data.yaml"),
        ),
        SourceSpec(
            "fruit-lpwjt",
            raw_root / "fruit-lpwjt",
            source_names_from_yaml(raw_root / "fruit-lpwjt" / "data.yaml"),
        ),
        SourceSpec(
            "lvis_fruits",
            raw_root / "lvis_fruits" / "extracted" / "LVIS_Fruits_And_Vegetables",
            source_names_from_yaml(raw_root / "lvis_fruits" / "extracted" / "LVIS_Fruits_And_Vegetables" / "data.yaml"),
        ),
        SourceSpec(
            "mango_ripeness_yolo",
            raw_root / "mango_ripeness_yolo",
            ["Half-Ripe", "Not_Mango", "OverRipe", "Ripe", "Unripe"],
        ),
        SourceSpec(
            "pomegranate_yolo",
            raw_root / "pomegranate_yolo",
            {0: "pomegranate_bud_skip", 1: "pomegranate_flower_skip", 2: "pomegranate", 3: "pomegranate", 4: "pomegranate"},
            "Only fruit-stage labels are mapped; buds and flowers are skipped.",
        ),
        SourceSpec(
            "watermelon-muqdf",
            raw_root / "watermelon-muqdf",
            ["watermelon", "watermelon-peel"],
        ),
    ]


def name_for_class(names: list[str] | dict[int, str], cls_id: int) -> str | None:
    if isinstance(names, dict):
        return names.get(cls_id)
    if 0 <= cls_id < len(names):
        return names[cls_id]
    return None


def class_map_for_source(spec: SourceSpec) -> dict[int, int]:
    if spec.tag == "mango_ripeness_yolo":
        return {0: CLASS_TO_ID["mango"], 2: CLASS_TO_ID["mango"], 3: CLASS_TO_ID["mango"], 4: CLASS_TO_ID["mango"]}
    if spec.tag == "watermelon-muqdf":
        return {0: CLASS_TO_ID["watermelon"]}

    mapping = {}
    ids = spec.names.keys() if isinstance(spec.names, dict) else range(len(spec.names))
    for cls_id in ids:
        raw_name = name_for_class(spec.names, int(cls_id))
        if raw_name is None or raw_name.endswith("_skip") or raw_name == "Not_Mango":
            continue
        canonical = canonical_from_name(raw_name)
        if canonical is not None:
            mapping[int(cls_id)] = CLASS_TO_ID[canonical]
    return mapping


def iter_images(spec: SourceSpec):
    for img in sorted(p for p in spec.root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS):
        yield img


def label_for_image(img_path: Path) -> Path:
    parts = list(img_path.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img_path.with_suffix(".txt")


def valid_box(box: tuple[int, float, float, float, float]) -> bool:
    _, cx, cy, w, h = box
    return 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1


def build(raw_root: Path, output: Path, manifest_path: Path, min_boxes: int, clean: bool) -> None:
    if clean and output.exists():
        remove_tree(output)
    img_out = output / "train" / "images"
    lbl_out = output / "train" / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    seen_md5: set[str] = set()
    counts = Counter()
    source_counts = defaultdict(int)
    stats = Counter()
    rows = []

    for spec in hardcoded_sources(raw_root):
        if not spec.root.exists():
            stats["missing_source"] += 1
            print(f"[SKIP] {spec.tag}: missing root {spec.root}", flush=True)
            continue
        mapping = class_map_for_source(spec)
        if not mapping:
            stats["empty_class_map"] += 1
            print(f"[SKIP] {spec.tag}: no mapped classes", flush=True)
            continue

        print(f"[SOURCE] {spec.tag}: mapped class ids {sorted(mapping)}", flush=True)
        for img_path in iter_images(spec):
            lbl_path = label_for_image(img_path)
            if not lbl_path.exists():
                stats["missing_label"] += 1
                continue
            raw_boxes = read_boxes(lbl_path)
            mapped = []
            raw_classes = []
            for box in raw_boxes:
                raw_cls = box[0]
                raw_name = name_for_class(spec.names, raw_cls) or f"class_{raw_cls}"
                raw_classes.append(raw_name)
                if raw_cls not in mapping:
                    continue
                new_box = (mapping[raw_cls], *box[1:])
                if valid_box(new_box):
                    mapped.append(new_box)
                else:
                    stats["invalid_box"] += 1
            if not mapped:
                stats["empty_after_mapping"] += 1
                continue

            digest = md5_file(img_path)
            if digest in seen_md5:
                stats["duplicate_md5"] += 1
                continue
            seen_md5.add(digest)

            new_stem = f"{spec.tag}_{digest[:8]}"
            new_img = img_out / f"{new_stem}{img_path.suffix.lower()}"
            new_lbl = lbl_out / f"{new_stem}.txt"
            shutil.copy2(img_path, new_img)
            write_boxes(new_lbl, mapped)

            for cls_id, *_ in mapped:
                counts[cls_id] += 1
            source_counts[spec.tag] += 1
            stats["kept_images"] += 1
            if stats["kept_images"] % 1000 == 0:
                print(f"  kept {stats['kept_images']:,} images...", flush=True)
            rows.append({
                "source": spec.tag,
                "original_image": str(img_path),
                "original_label": str(lbl_path),
                "output_image": str(new_img),
                "output_label": str(new_lbl),
                "md5": digest,
                "raw_classes": "|".join(sorted(set(raw_classes))),
                "mapped_classes": "|".join(CLASSES[i] for i in sorted({b[0] for b in mapped})),
                "boxes": len(mapped),
                "note": spec.note,
            })

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source", "original_image", "original_label", "output_image", "output_label",
                "md5", "raw_classes", "mapped_classes", "boxes", "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 72)
    print("  DATASET V4 RAW BUILD REPORT")
    print("=" * 72)
    print(f"Kept images: {stats['kept_images']:,}")
    print(f"Duplicate images skipped: {stats['duplicate_md5']:,}")
    print(f"Missing labels skipped: {stats['missing_label']:,}")
    print(f"Empty after mapping skipped: {stats['empty_after_mapping']:,}")
    print(f"Invalid boxes skipped: {stats['invalid_box']:,}")
    print("\nPer-source kept images:")
    for source, n in sorted(source_counts.items()):
        print(f"  {source:<28} {n:>7}")
    print("\nPer-class raw box counts:")
    failed = []
    for cls_id, name in enumerate(CLASSES):
        n = counts[cls_id]
        print(f"  {name:<14} {n:>7}")
        if n < min_boxes:
            failed.append((name, n))

    print(f"\nWrote dataset: {output.resolve()}")
    print(f"Wrote manifest: {manifest_path.resolve()}")
    print("=" * 72)
    if failed:
        msg = ", ".join(f"{name}={n}" for name, n in failed)
        raise SystemExit(f"[FAIL] Raw class count gate failed (<{min_boxes} boxes): {msg}")
    print("[PASS] Raw class count gate passed.")


def remove_tree(path: Path) -> None:
    def onexc(func, p, exc):
        try:
            os.chmod(p, 0o700)
            func(p)
        except OSError:
            pass

    for _ in range(5):
        try:
            shutil.rmtree(path, onexc=onexc)
        except TypeError:
            shutil.rmtree(path, onerror=lambda func, p, exc: onexc(func, p, exc))
        except OSError:
            time.sleep(0.5)
            continue
        if not path.exists():
            return
    if path.exists():
        raise OSError(f"Could not remove existing output directory: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical V4 raw YOLO dataset")
    parser.add_argument("--raw", default="raw_datasets")
    parser.add_argument("--output", default="dataset_v4_raw")
    parser.add_argument("--manifest", default="dataset_v4_raw_manifest.csv")
    parser.add_argument("--min-boxes", type=int, default=200)
    parser.add_argument("--no-clean", action="store_true", help="Do not remove existing output first")
    args = parser.parse_args()

    build(
        raw_root=Path(args.raw),
        output=Path(args.output),
        manifest_path=Path(args.manifest),
        min_boxes=args.min_boxes,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()
