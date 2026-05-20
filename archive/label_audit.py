"""
label_audit.py — Fruit Detection: Label Quality & Leakage Audit
===============================================================
Phase 2 – Bounding box quality checks:
  • Loose / near-full-image boxes (>85% area)
  • Tiny / degenerate boxes    (<0.5% area)
  • Extreme aspect ratios       (>10:1)
  • Missing or empty label files
  • Boxes with coordinates outside [0,1]

Phase 4 – Train/Val/Test leakage:
  • Exact filename duplicates across splits
  • Near-duplicate detection via perceptual hash (pHash)

Usage
-----
  python label_audit.py                 # all 3 splits
  python label_audit.py --no_phash      # skip slow pHash step
  python label_audit.py --split train   # single split
"""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

CLASSES  = ["apple","banana","orange","mango","pineapple","watermelon","grapes","pomegranate"]
DATA_YAML = Path("data_v3.yaml")

SPLIT_DIRS = {          # key used inside data_v3.yaml
    "train": "train",
    "val":   "val",
    "test":  "test",
}

# Thresholds
LOOSE_AREA   = 0.85    # box occupies >85% of image → suspicious
TINY_AREA    = 0.005   # box occupies <0.5% of image → suspicious
ASPECT_MAX   = 8.0     # w/h or h/w > 8 → suspicious
PHASH_DIST   = 8       # hamming distance for near-dup


# ────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def read_boxes(lbl: Path):
    boxes = []
    if not lbl.exists():
        return boxes
    with open(lbl) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    boxes.append((int(parts[0]), *map(float, parts[1:])))
                except ValueError:
                    pass
    return boxes


def phash(img, size=8):
    """Simple perceptual hash → 64-bit int."""
    gray = cv2.cvtColor(cv2.resize(img, (size*4, size*4)), cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    mean = small.mean()
    bits = (small > mean).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming(a, b):
    return bin(a ^ b).count("1")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ────────────────────────────────────────────
#  PER-SPLIT LABEL AUDIT
# ────────────────────────────────────────────

def audit_split(split_name: str, img_dir: Path, lbl_dir: Path) -> dict:
    issues = defaultdict(list)
    stats  = defaultdict(int)

    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"})

    for img_path in imgs:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        stats["total_images"] += 1

        # Missing label
        if not lbl_path.exists():
            issues["missing_label"].append(img_path.name)
            continue

        boxes = read_boxes(lbl_path)

        # Empty label
        if not boxes:
            issues["empty_label"].append(img_path.name)
            stats["empty_labels"] += 1
            continue

        stats["total_boxes"] += len(boxes)

        for cls_id, cx, cy, w, h in boxes:
            stats[f"class_{CLASSES[cls_id] if cls_id < len(CLASSES) else cls_id}"] += 1

            # Out-of-range coordinates
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                issues["oob_coords"].append(f"{img_path.name} cls={cls_id} [{cx:.3f},{cy:.3f},{w:.3f},{h:.3f}]")

            area = w * h
            # Loose (near full-image) box
            if area > LOOSE_AREA:
                issues["loose_box"].append(f"{img_path.name} cls={CLASSES[cls_id] if cls_id < len(CLASSES) else cls_id} area={area:.3f}")
                stats["loose_boxes"] += 1

            # Tiny box
            if area < TINY_AREA:
                issues["tiny_box"].append(f"{img_path.name} cls={CLASSES[cls_id] if cls_id < len(CLASSES) else cls_id} area={area:.5f}")
                stats["tiny_boxes"] += 1

            # Extreme aspect ratio
            if w > 0 and h > 0:
                ratio = max(w/h, h/w)
                if ratio > ASPECT_MAX:
                    issues["extreme_aspect"].append(
                        f"{img_path.name} cls={CLASSES[cls_id] if cls_id < len(CLASSES) else cls_id} "
                        f"w={w:.3f} h={h:.3f} ratio={ratio:.1f}"
                    )
                    stats["extreme_aspect"] += 1

    return {"stats": dict(stats), "issues": {k: v[:30] for k,v in issues.items()}}


# ────────────────────────────────────────────
#  CROSS-SPLIT LEAKAGE CHECK
# ────────────────────────────────────────────

def check_leakage(split_img_dirs: dict, use_phash: bool) -> dict:
    """Check filename & content duplicates across all pairs of splits."""
    # Collect filenames and hashes per split
    split_data = {}
    for split, img_dir in split_img_dirs.items():
        imgs = sorted(p for p in img_dir.iterdir()
                      if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"})
        split_data[split] = imgs

    leakage = {}

    splits = list(split_data.keys())
    for i in range(len(splits)):
        for j in range(i+1, len(splits)):
            sa, sb = splits[i], splits[j]
            names_a = {p.name for p in split_data[sa]}
            names_b = {p.name for p in split_data[sb]}
            exact = sorted(names_a & names_b)
            key = f"{sa}_vs_{sb}"
            leakage[key] = {"exact_filename_duplicates": len(exact),
                            "examples": exact[:10]}

            if use_phash and not exact:
                # Near-dup via pHash (slower)
                print(f"    pHash: hashing {sa} ({len(split_data[sa])}) ...")
                hashes_a = {}
                for p in split_data[sa]:
                    img = cv2.imread(str(p))
                    if img is not None:
                        hashes_a[p.name] = phash(img)

                print(f"    pHash: hashing {sb} ({len(split_data[sb])}) ...")
                near_dups = []
                for p in split_data[sb]:
                    img = cv2.imread(str(p))
                    if img is None: continue
                    h = phash(img)
                    for name_a, ha in hashes_a.items():
                        if hamming(h, ha) <= PHASH_DIST:
                            near_dups.append(f"{p.name} ≈ {name_a}")
                leakage[key]["near_duplicates_phash"] = len(near_dups)
                leakage[key]["near_dup_examples"] = near_dups[:10]

    return leakage


# ────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Label quality & leakage audit")
    p.add_argument("--data",    default=str(DATA_YAML))
    p.add_argument("--split",   default="all", choices=["all","train","val","test"])
    p.add_argument("--no_phash", action="store_true",
                   help="Skip slow perceptual hash near-dup check")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_yaml(args.data)
    root = Path(cfg["path"])

    # Map split name → img relative path
    split_map = {
        "train": cfg.get("train","train/images"),
        "val":   cfg.get("val",  "val/images"),
        "test":  cfg.get("test", "test/images"),
    }
    targets = list(split_map.items()) if args.split == "all" else [
        (args.split, split_map[args.split])
    ]

    print("="*60)
    print("  FRUIT DETECTION — LABEL AUDIT")
    print("="*60)

    audit_results = {}
    split_img_dirs = {}

    for split_name, img_rel in targets:
        img_dir = root / img_rel
        lbl_dir = img_dir.parent.parent / "labels"   # images/../labels
        # handle "train/images" → parent is "train", go up to root then down to labels
        # Robust: replace /images with /labels in path
        lbl_dir = Path(str(img_dir).replace("images","labels"))

        if not img_dir.exists():
            print(f"\n  [{split_name.upper()}] ⚠  Not found: {img_dir}")
            continue

        print(f"\n  [{split_name.upper()}] Auditing {img_dir} ...")
        result = audit_split(split_name, img_dir, lbl_dir)
        audit_results[split_name] = result
        split_img_dirs[split_name] = img_dir

        s = result["stats"]
        print(f"    Total images : {s.get('total_images',0)}")
        print(f"    Total boxes  : {s.get('total_boxes',0)}")
        print(f"    Empty labels : {s.get('empty_labels',0)}")
        print(f"    Loose boxes  : {s.get('loose_boxes',0)}  (>{LOOSE_AREA*100:.0f}% image area)")
        print(f"    Tiny boxes   : {s.get('tiny_boxes',0)}   (<{TINY_AREA*100:.1f}% image area)")
        print(f"    Extreme AR   : {s.get('extreme_aspect',0)}")

        print(f"\n    Class distribution:")
        for cls in CLASSES:
            n = s.get(f"class_{cls}", 0)
            bar = "#" * min(40, int(n / max(s.get("total_boxes",1), 1) * 120))
            print(f"      {cls:<14} {n:>6}  {bar}")

        iss = result["issues"]
        for issue_type, examples in iss.items():
            if examples:
                print(f"\n    ⚠  {issue_type} ({len(examples)} shown, may be more):")
                for ex in examples[:5]:
                    print(f"       {ex}")

    # ── Cross-split leakage ─────────────────────────────────────────
    if args.split == "all" and len(split_img_dirs) >= 2:
        print("\n" + "="*60)
        print("  CROSS-SPLIT LEAKAGE CHECK")
        print("="*60)
        leakage = check_leakage(split_img_dirs, use_phash=not args.no_phash)
        for pair, info in leakage.items():
            print(f"\n  {pair}:")
            print(f"    Exact filename duplicates : {info['exact_filename_duplicates']}")
            if info["examples"]:
                for e in info["examples"][:5]:
                    print(f"      {e}")
            nd = info.get("near_duplicates_phash", "skipped")
            print(f"    Near-duplicates (pHash)   : {nd}")
            if isinstance(nd, int) and nd > 0:
                for e in info.get("near_dup_examples",[])[:5]:
                    print(f"      {e}")

    print("\n  [OK] Audit complete.")


if __name__ == "__main__":
    main()
