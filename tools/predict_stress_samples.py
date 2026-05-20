"""
predict_stress_samples.py
=========================
Save annotated predictions for quick visual inspection of a YOLO dataset.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

CLASSES = {
    0: "apple", 1: "banana", 2: "orange", 3: "mango",
    4: "pineapple", 5: "watermelon", 6: "grapes", 7: "pomegranate",
}
COLORS = {
    0: (0, 200, 0),
    1: (0, 230, 255),
    2: (0, 140, 255),
    3: (0, 180, 255),
    4: (30, 200, 220),
    5: (80, 80, 220),
    6: (180, 80, 180),
    7: (60, 60, 200),
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_classes(label: Path):
    classes = set()
    if not label.exists():
        return classes
    with label.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    classes.add(int(parts[0]))
                except ValueError:
                    pass
    return classes


def annotate(frame, results):
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = CLASSES.get(cls_id, f"class_{cls_id}")
            color = COLORS.get(cls_id, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            y0 = max(0, y1 - lh - bl - 4)
            cv2.rectangle(frame, (x1, y0), (x1 + lw, y1), color, -1)
            cv2.putText(frame, label, (x1, max(12, y1 - bl - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Save annotated stress-test predictions")
    parser.add_argument("--model", default="runs/fruit_v4_s_local/weights/best.pt")
    parser.add_argument("--dataset", default="synthetic_webcam_holdout")
    parser.add_argument("--output", default="runs/synthetic_webcam_predictions")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = Path(args.model)
    dataset = Path(args.dataset)
    image_dir = dataset / "images"
    label_dir = dataset / "labels"
    output = Path(args.output)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    by_priority = defaultdict(list)
    all_images = []
    for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        all_images.append(image)
        classes = read_classes(label_dir / f"{image.stem}.txt")
        for cls_id in classes:
            by_priority[cls_id].append(image)

    rng = random.Random(args.seed)
    selected = []
    selected_set = set()
    for cls_id in [4, 5, 7, 3]:
        candidates = list(by_priority.get(cls_id, []))
        rng.shuffle(candidates)
        for image in candidates[: max(1, args.count // 8)]:
            if image not in selected_set:
                selected.append(image)
                selected_set.add(image)
    remaining = [p for p in all_images if p not in selected_set]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, args.count - len(selected))])

    model = YOLO(str(model_path))
    for idx, image in enumerate(selected[: args.count]):
        frame = cv2.imread(str(image))
        if frame is None:
            continue
        results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        annotated = annotate(frame, results)
        cv2.imwrite(str(output / f"{idx:03d}_{image.name}"), annotated)

    print("=" * 72)
    print("  STRESS SAMPLE PREDICTIONS")
    print("=" * 72)
    print(f"Saved: {min(len(selected), args.count)} images")
    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    main()
