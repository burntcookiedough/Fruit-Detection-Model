"""
filter_dataset.py
=================
Creates a cleaned copy of the dataset with degenerate boxes removed.

Filtering rule (per-object, in PIXEL space at YOLO input size):
  DISCARD if  min(w_px, h_px) < MIN_SIDE_PX
  This is more principled than area thresholds for elongated objects.

For each split (train / val / test):
  - Copies all images as-is
  - Rewrites label files with bad boxes removed
  - Reports per-class counts before/after

Usage
-----
  python filter_dataset.py                  # default MIN_SIDE_PX=8, out=dataset_v3_clean
  python filter_dataset.py --min_px 10      # stricter
  python filter_dataset.py --dry_run        # count only, no files written
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

DATA_YAML    = Path("data_v3.yaml")
INPUT_SZ     = 640        # YOLO input resolution
DEFAULT_MIN_PX = 8        # minimum side length in pixels

CLASSES = ["apple","banana","orange","mango","pineapple","watermelon","grapes","pomegranate"]


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def read_boxes(lbl: Path):
    if not lbl.exists():
        return []
    boxes = []
    with open(lbl) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    boxes.append((int(parts[0]), *map(float, parts[1:])))
                except ValueError:
                    pass
    return boxes


def write_boxes(lbl: Path, boxes):
    lbl.parent.mkdir(parents=True, exist_ok=True)
    with open(lbl, "w") as f:
        for cls_id, cx, cy, w, h in boxes:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def keep_box(cx, cy, w, h, min_px: int) -> bool:
    """True if min side >= min_px in pixel space."""
    w_px = w * INPUT_SZ
    h_px = h * INPUT_SZ
    # Clamp to valid range first
    if not (0 < w <= 1 and 0 < h <= 1):
        return False
    if not (0 <= cx <= 1 and 0 <= cy <= 1):
        return False
    return min(w_px, h_px) >= min_px


def process_split(split_name: str, src_img_dir: Path, src_lbl_dir: Path,
                  dst_img_dir: Path, dst_lbl_dir: Path,
                  min_px: int, dry_run: bool) -> dict:
    if not src_img_dir.exists():
        print(f"  [{split_name}] Source not found: {src_img_dir}, skipping.")
        return {}

    imgs = sorted(p for p in src_img_dir.iterdir()
                  if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"})

    kept   = defaultdict(int)
    removed = defaultdict(int)
    empty_after = 0

    for img_path in imgs:
        lbl_path = src_lbl_dir / (img_path.stem + ".txt")
        boxes    = read_boxes(lbl_path)
        good     = [b for b in boxes if keep_box(*b[1:], min_px)]
        bad      = [b for b in boxes if not keep_box(*b[1:], min_px)]

        for b in good:   kept[b[0]]    += 1
        for b in bad:    removed[b[0]] += 1
        if not good:
            empty_after += 1

        if not dry_run:
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            write_boxes(dst_lbl_dir / (img_path.stem + ".txt"), good)

    return {"kept": dict(kept), "removed": dict(removed),
            "images": len(imgs), "empty_after": empty_after}


def print_split_report(split_name: str, result: dict):
    k = result["kept"]
    r = result["removed"]
    all_cls = sorted(set(list(k.keys()) + list(r.keys())))
    print(f"\n  [{split_name.upper()}]  {result['images']} images  "
          f"(images left empty after filter: {result['empty_after']})")
    print(f"  {'Class':<14} {'Kept':>7} {'Removed':>9} {'% removed':>10}")
    print("  " + "-"*45)
    total_k = total_r = 0
    for cls_id in all_cls:
        name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls_{cls_id}"
        kv = k.get(cls_id, 0); rv = r.get(cls_id, 0)
        total_k += kv; total_r += rv
        pct = 100 * rv / (kv + rv) if (kv + rv) else 0
        flag = "  <-- check" if pct > 30 else ""
        print(f"  {name:<14} {kv:>7} {rv:>9} {pct:>9.1f}%{flag}")
    pct_total = 100 * total_r / (total_k + total_r) if (total_k + total_r) else 0
    print("  " + "-"*45)
    print(f"  {'TOTAL':<14} {total_k:>7} {total_r:>9} {pct_total:>9.1f}%")


def write_clean_yaml(src_yaml_path: Path, dst_dataset_path: Path, out_yaml_path: Path):
    with open(src_yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(dst_dataset_path.resolve())
    with open(out_yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\n  Wrote clean data yaml: {out_yaml_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default=str(DATA_YAML))
    parser.add_argument("--out",     default="dataset_v3_clean",
                        help="Output cleaned dataset directory")
    parser.add_argument("--min_px",  type=int, default=DEFAULT_MIN_PX,
                        help=f"Min side in pixels (default={DEFAULT_MIN_PX})")
    parser.add_argument("--dry_run", action="store_true",
                        help="Count only — do not write any files")
    args = parser.parse_args()

    cfg      = load_yaml(args.data)
    src_root = Path(cfg["path"])
    dst_root = Path(args.out)

    print("=" * 60)
    print("  FRUIT DETECTION — TINY BOX FILTER")
    print(f"  Min side threshold : {args.min_px} px  (at {INPUT_SZ}px input)")
    print(f"  Source             : {src_root}")
    print(f"  Destination        : {dst_root}")
    print(f"  Dry run            : {args.dry_run}")
    print("=" * 60)

    split_keys = {
        "train": cfg.get("train", "train/images"),
        "val":   cfg.get("val",   "val/images"),
        "test":  cfg.get("test",  "test/images"),
    }

    for split_name, rel_path in split_keys.items():
        src_img = src_root / rel_path
        src_lbl = Path(str(src_img).replace("images", "labels"))
        dst_img = dst_root / rel_path
        dst_lbl = Path(str(dst_img).replace("images", "labels"))

        result = process_split(split_name, src_img, src_lbl,
                               dst_img, dst_lbl, args.min_px, args.dry_run)
        if result:
            print_split_report(split_name, result)

    if not args.dry_run:
        out_yaml = Path("data_v3_clean.yaml")
        write_clean_yaml(Path(args.data), dst_root, out_yaml)
        print("\n  Next steps:")
        print(f"    python train.py --data {out_yaml} --name fruit_v3_clean")
    else:
        print("\n  [DRY RUN] No files written. Remove --dry_run to apply.")

    print("\n  [OK] Filter complete.")


if __name__ == "__main__":
    main()
