"""
audit_banana_style.py
=====================
Detects annotation inconsistency in the banana class:
  - images labeled per-finger  (many small boxes)
  - images labeled per-bunch   (1-3 large boxes)

Mixed supervision is the silent killer for YOLO banana recall.

Usage
-----
  python audit_banana_style.py
  python audit_banana_style.py --split val
  python audit_banana_style.py --show_images  # opens OpenCV windows for review
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

DATA_YAML    = Path("data_v3.yaml")
BANANA_ID    = 1          # 0-indexed in data_v3.yaml
FINGER_MAX_W = 0.08       # single finger width  <  8% image width
FINGER_MAX_H = 0.15       # single finger height < 15% image height
BUNCH_AREA   = 0.04       # a bunch box  >= 4% image area

# P3 stride=8 → minimum detectable feature = 8px × 8px at input 640
YOLO_MIN_PX  = 8
INPUT_SZ     = 640


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def read_boxes(lbl: Path):
    if not lbl.exists():
        return []
    rows = []
    with open(lbl) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    rows.append((int(parts[0]), *map(float, parts[1:])))
                except ValueError:
                    pass
    return rows


def classify_banana_image(banana_boxes: list) -> str:
    """
    Returns:
      'bunch'   — 1-3 large boxes (good labeling)
      'finger'  — many tiny boxes (degenerate labeling)
      'mixed'   — mix of large and tiny (worst case)
      'empty'   — no banana boxes
    """
    if not banana_boxes:
        return "empty"
    areas = [w * h for _, _, _, w, h in banana_boxes]
    is_finger = [a < BUNCH_AREA for a in areas]
    n_finger = sum(is_finger)
    n_bunch  = len(banana_boxes) - n_finger
    if n_finger > 0 and n_bunch > 0:
        return "mixed"
    if n_finger == len(banana_boxes):
        return "finger"
    return "bunch"


def px_size(w, h):
    """Return (w_px, h_px) at YOLO input size."""
    return w * INPUT_SZ, h * INPUT_SZ


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default=str(DATA_YAML))
    parser.add_argument("--split",  default="train", choices=["train","val","test"])
    parser.add_argument("--show_images", action="store_true",
                        help="Open OpenCV windows for sampled images (press any key to advance)")
    args = parser.parse_args()

    cfg  = load_yaml(args.data)
    root = Path(cfg["path"])
    img_dir = root / cfg[args.split]
    lbl_dir = Path(str(img_dir).replace("images", "labels"))

    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"})

    style_counter = Counter()
    boxes_per_img = []
    area_hist      = []
    min_px_hist    = []
    below_yolo_min = 0
    total_banana_boxes = 0

    print("=" * 60)
    print(f"  BANANA ANNOTATION STYLE AUDIT  [{args.split.upper()}]")
    print("=" * 60)
    print(f"  Scanning {len(imgs)} images ...")

    show_samples = {"finger": [], "bunch": [], "mixed": []}

    for img_path in imgs:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        boxes = read_boxes(lbl_path)
        banana = [(cls, cx, cy, w, h) for cls, cx, cy, w, h in boxes if cls == BANANA_ID]
        if not banana:
            continue

        style = classify_banana_image(banana)
        style_counter[style] += 1
        boxes_per_img.append(len(banana))
        total_banana_boxes += len(banana)

        for _, _, _, w, h in banana:
            a = w * h
            area_hist.append(a * 100)  # percent
            wp, hp = px_size(w, h)
            min_px = min(wp, hp)
            min_px_hist.append(min_px)
            if min(wp, hp) < YOLO_MIN_PX:
                below_yolo_min += 1

        if len(show_samples[style]) < 5:
            show_samples[style].append(img_path)

    # ── Summary ──────────────────────────────────────────────────
    total_banana_imgs = sum(style_counter.values())
    arr = np.array(boxes_per_img)
    area_arr = np.array(area_hist)
    px_arr   = np.array(min_px_hist)

    print(f"\n  Total images with banana  : {total_banana_imgs}")
    print(f"  Total banana boxes        : {total_banana_boxes}")
    print(f"  Mean boxes/image          : {arr.mean():.1f}")
    print(f"  Median boxes/image        : {np.median(arr):.1f}")
    print(f"  Max boxes in one image    : {arr.max()}")
    print(f"\n  Below YOLO min ({YOLO_MIN_PX}px) : {below_yolo_min}  ({100*below_yolo_min/total_banana_boxes:.1f}% of boxes)")

    print(f"\n  Annotation style breakdown:")
    print(f"  {'Style':<10} {'Count':>7} {'%':>6}")
    print("  " + "-" * 28)
    for style in ["bunch", "finger", "mixed"]:
        n = style_counter[style]
        pct = 100 * n / total_banana_imgs if total_banana_imgs else 0
        bar = "#" * int(pct / 2)
        print(f"  {style:<10} {n:>7}  {pct:>5.1f}%  {bar}")

    print(f"\n  Box area distribution (% of image):")
    for lo, hi in [(0, 0.1), (0.1, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, 100)]:
        n = int(((area_arr >= lo) & (area_arr < hi)).sum())
        pct = 100 * n / len(area_arr) if len(area_arr) else 0
        bar = "#" * int(pct / 2)
        label = f"{lo}-{hi}%"
        print(f"    {label:<12} {n:>6}  {pct:>5.1f}%  {bar}")

    print(f"\n  Min-dimension distribution (pixels at 640px input):")
    for lo, hi in [(0,4),(4,8),(8,16),(16,32),(32,640)]:
        n = int(((px_arr >= lo) & (px_arr < hi)).sum())
        pct = 100 * n / len(px_arr) if len(px_arr) else 0
        bar = "#" * int(pct / 2)
        label = f"{lo}-{hi}px"
        print(f"    {label:<12} {n:>6}  {pct:>5.1f}%  {bar}")

    # ── Inconsistency warning ─────────────────────────────────────
    if style_counter["finger"] > 0 and style_counter["bunch"] > 0:
        print("\n  *** INCONSISTENCY DETECTED ***")
        print(f"  {style_counter['bunch']} images use BUNCH-style labels")
        print(f"  {style_counter['finger']} images use FINGER-style labels")
        print(f"  {style_counter['mixed']} images use MIXED labels")
        print("  The model is receiving contradictory supervision.")
        print("  Standardise ALL banana images to per-bunch labeling.")

    # ── Optional visual review ────────────────────────────────────
    if args.show_images:
        for style, paths in show_samples.items():
            if not paths:
                continue
            print(f"\n  Showing {len(paths)} {style.upper()} examples (press any key)...")
            for img_path in paths:
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                H, W = img.shape[:2]
                boxes = read_boxes(lbl_path)
                for cls, cx, cy, w, h in boxes:
                    if cls != BANANA_ID:
                        continue
                    x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
                    x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
                    col = (0,200,0) if w*h >= BUNCH_AREA else (0,0,220)
                    cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
                    area_pct = w * h * 100
                    cv2.putText(img, f"{area_pct:.2f}%", (x1, y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
                cv2.imshow(f"Banana [{style}] - {img_path.name}", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    print("\n  [OK] Audit complete.")


if __name__ == "__main__":
    main()
