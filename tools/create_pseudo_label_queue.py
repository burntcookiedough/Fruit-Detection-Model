"""
create_pseudo_label_queue.py
============================
Create a manual review queue for image-only sources.

Images are copied into dataset_v4_label_queue/images and Phase A predictions
are written to dataset_v4_label_queue/predicted_labels. These labels are drafts
only. They must be manually corrected and copied to reviewed_labels before the
images are allowed into training.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_queue_images(sources: list[Path], queue_root: Path, max_per_source: int | None) -> list[Path]:
    image_dir = queue_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    seen = set()
    for source in sources:
        if not source.exists():
            print(f"[SKIP] Missing source: {source}")
            continue
        count = 0
        tag = source.name
        for image in sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS):
            digest = md5_file(image)
            if digest in seen:
                continue
            seen.add(digest)
            dest = image_dir / f"{tag}_{digest[:8]}{image.suffix.lower()}"
            shutil.copy2(image, dest)
            copied.append(dest)
            count += 1
            if max_per_source is not None and count >= max_per_source:
                break
    return copied


def write_prediction_label(label_path: Path, result) -> int:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        label_path.write_text("", encoding="utf-8")
        return 0
    lines = []
    for box in boxes:
        cls_id = int(box.cls[0])
        cx, cy, w, h = box.xywhn[0].tolist()
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create pseudo-label review queue")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[
            "raw_datasets/PG-YOLO-Dataset-master",
            "raw_datasets/pomegranate_dataset",
            "raw_datasets/mango_dds",
        ],
        help="Image-only source folders to queue",
    )
    parser.add_argument("--model", default="runs/fruit_v4_s_local/weights/best.pt")
    parser.add_argument("--queue", default="dataset_v4_label_queue")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-per-source", type=int, default=None)
    args = parser.parse_args()

    queue_root = Path(args.queue)
    reviewed_dir = queue_root / "reviewed_labels"
    predicted_dir = queue_root / "predicted_labels"
    reviewed_dir.mkdir(parents=True, exist_ok=True)
    predicted_dir.mkdir(parents=True, exist_ok=True)

    copied = copy_queue_images([Path(p) for p in args.sources], queue_root, args.max_per_source)
    if not copied:
        raise SystemExit("[FAIL] No images copied into the review queue.")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Pseudo-label model not found: {model_path}")

    model = YOLO(str(model_path))
    total_boxes = 0
    for result in model.predict(copied, imgsz=args.imgsz, conf=args.conf, verbose=False):
        image_path = Path(result.path)
        label_path = predicted_dir / f"{image_path.stem}.txt"
        total_boxes += write_prediction_label(label_path, result)

    print("=" * 72)
    print("  V4 PSEUDO-LABEL REVIEW QUEUE")
    print("=" * 72)
    print(f"Queued images: {len(copied):,}")
    print(f"Draft boxes:   {total_boxes:,}")
    print(f"Images:        {(queue_root / 'images').resolve()}")
    print(f"Draft labels:  {predicted_dir.resolve()}")
    print(f"Review labels: {reviewed_dir.resolve()}")
    print("\nManual gate: copy corrected YOLO labels into reviewed_labels before merging.")


if __name__ == "__main__":
    main()
