"""
prepare_pod.py — End-to-end dataset preparation for RunPod cloud training.

Pipeline (runs automatically in order):
  1. FILTER  : Remove degenerate tiny boxes from dataset_v3_raw → dataset_v3_filtered
  2. BALANCE : Cap majority classes, augment minority → dataset_v3_final
  3. VERIFY  : Count images/labels per split, assert no empty label files
  4. YAML    : Write data_v3_final.yaml with correct absolute path for RunPod (/workspace)
  5. PACK    : Zip everything needed for the pod into fruit_detection_pod.zip

Usage (local Windows, before deploying RunPod):
    python prepare_pod.py
    python prepare_pod.py --min_px 10 --max_boxes 4000 --min_boxes 2000

Output:
    dataset_v3_final/    — the final clean, balanced dataset
    data_v3_final.yaml   — local yaml (absolute Windows path)
    data_v3_final_pod.yaml — pod yaml (/workspace path)
    fruit_detection_pod.zip  — upload this to RunPod
"""

import argparse
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

# Force UTF-8 output on Windows terminals (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLASSES = ["apple", "banana", "orange", "mango",
           "pineapple", "watermelon", "grapes", "pomegranate"]
NC = len(CLASSES)

SOURCE_RAW       = Path("dataset_v3_raw")
FILTERED_DIR     = Path("dataset_v3_filtered")   # intermediate (filter output)
FINAL_DIR        = Path("dataset_v3_final")       # final output
LOCAL_YAML       = Path("data_v3_final.yaml")
POD_YAML         = Path("data_v3_final_pod.yaml")
ZIP_OUT          = Path("fruit_detection_pod.zip")
POD_WORKSPACE    = "/workspace"

SPLITS           = ["train", "valid", "test"]
IMG_EXTS         = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INPUT_SZ         = 640        # YOLO input resolution (used for pixel-space filter)
RANDOM_SEED      = 42

# Balance defaults — calibrated for this dataset
DEFAULT_MIN_PX   = 8          # min box side in pixels at 640px
DEFAULT_MAX_BOXES = 4000      # cap majority classes (boxes in train)
DEFAULT_MIN_BOXES = 2000      # floor minority classes (boxes in train)

# ---------------------------------------------------------------------------
# Shared I/O helpers
# ---------------------------------------------------------------------------

def read_boxes(lbl: Path):
    """Read YOLO label file → list of (cls_id, cx, cy, w, h)."""
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


def collect_split(root: Path, split: str):
    """Returns list of (img_path, lbl_path, boxes, classes_set)."""
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    if not img_dir.exists():
        return []
    records = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        boxes = read_boxes(lbl)
        if not boxes:
            continue
        records.append((img, lbl, boxes, {b[0] for b in boxes}))
    return records


def count_boxes(records):
    counts = defaultdict(int)
    for _, _, boxes, _ in records:
        for b in boxes:
            counts[b[0]] += 1
    return counts


def print_distribution(label, counts, n_images):
    max_c = max(counts.values(), default=1)
    print(f"\n  [{label}]  {n_images} images")
    for i, name in enumerate(CLASSES):
        c = counts.get(i, 0)
        bar_len = int(c / max(max_c, 1) * 28)
        bar = "|" * bar_len
        print(f"    {i}: {name:<14} {c:>5}  {bar}")


# ---------------------------------------------------------------------------
# Step 1 — Filter (remove tiny boxes)
# ---------------------------------------------------------------------------

def keep_box(cx, cy, w, h, min_px: int) -> bool:
    if not (0 < w <= 1 and 0 < h <= 1):
        return False
    if not (0 <= cx <= 1 and 0 <= cy <= 1):
        return False
    return min(w * INPUT_SZ, h * INPUT_SZ) >= min_px


def run_filter(src: Path, dst: Path, min_px: int):
    print(f"\n{'='*60}")
    print(f"  STEP 1/4 — FILTER (min_side >= {min_px}px at {INPUT_SZ}px)")
    print(f"  Source : {src}")
    print(f"  Output : {dst}")
    print(f"{'='*60}")

    if not src.exists():
        sys.exit(f"\n[ERROR] Source dataset not found: {src}\n"
                 "Run prepare_dataset_v3.py first to create dataset_v3_raw.")

    if dst.exists():
        print(f"  Removing old {dst} ...")
        shutil.rmtree(dst, ignore_errors=True)

    total_kept = total_removed = 0
    for split in SPLITS:
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        dst_img = dst / split / "images"
        dst_lbl = dst / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists():
            print(f"  [{split}] Not found — skipping.")
            continue

        kept_cls = defaultdict(int)
        removed_cls = defaultdict(int)

        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            boxes = read_boxes(lbl)
            good = [b for b in boxes if keep_box(*b[1:], min_px)]
            bad  = [b for b in boxes if not keep_box(*b[1:], min_px)]
            for b in good: kept_cls[b[0]] += 1
            for b in bad:  removed_cls[b[0]] += 1
            shutil.copy2(img, dst_img / img.name)
            write_boxes(dst_lbl / (img.stem + ".txt"), good)

        kept_sum    = sum(kept_cls.values())
        removed_sum = sum(removed_cls.values())
        total_kept    += kept_sum
        total_removed += removed_sum
        imgs_n = len(list(dst_img.iterdir()))
        print(f"\n  [{split.upper()}] {imgs_n} images — kept {kept_sum} boxes, "
              f"removed {removed_sum} boxes ({100*removed_sum/max(kept_sum+removed_sum,1):.1f}%)")
        for i, name in enumerate(CLASSES):
            k = kept_cls.get(i, 0)
            r = removed_cls.get(i, 0)
            flag = "  < check" if r > 0 and (r / max(k + r, 1)) > 0.30 else ""
            print(f"    {name:<14} kept={k:<5} removed={r}{flag}")

    print(f"\n  Filter complete. Total removed: {total_removed} boxes "
          f"({100*total_removed/max(total_kept+total_removed,1):.1f}% of all boxes)")


# ---------------------------------------------------------------------------
# Step 2 — Balance (cap + augment)
# ---------------------------------------------------------------------------

def aug_brightness_contrast(img, alpha_range=(0.5, 1.5), beta_range=(-40, 40)):
    alpha = random.uniform(*alpha_range)
    beta  = random.uniform(*beta_range)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def aug_noise(img, sigma_range=(5, 25)):
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def aug_blur(img):
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)

def aug_hflip(img, boxes):
    return cv2.flip(img, 1), [(c, 1.0 - cx, cy, w, h) for c, cx, cy, w, h in boxes]

def aug_rotate(img, boxes, angle_range=(-10, 10)):
    angle = random.uniform(*angle_range)
    H, W  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    new_boxes = []
    for c, cx, cy, bw, bh in boxes:
        x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
        x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
        corners = np.array([[x1,y1,1],[x2,y1,1],[x1,y2,1],[x2,y2,1]], dtype=np.float32)
        rc = (M @ corners.T).T
        rx1, rx2 = np.clip(rc[:,0].min(),0,W), np.clip(rc[:,0].max(),0,W)
        ry1, ry2 = np.clip(rc[:,1].min(),0,H), np.clip(rc[:,1].max(),0,H)
        ncx, ncy = ((rx1+rx2)/2)/W, ((ry1+ry2)/2)/H
        nbw, nbh = (rx2-rx1)/W,    (ry2-ry1)/H
        if 0 < nbw <= 1 and 0 < nbh <= 1:
            new_boxes.append((c, ncx, ncy, nbw, nbh))
    return rotated, new_boxes


def greedy_cap(records, target_max):
    """Keep all minority images; stop adding images once a majority class is capped."""
    random.shuffle(records)
    minority = {3, 4, 5, 7}
    selected, remainder = [], []
    counts = defaultdict(int)
    for rec in records:
        if rec[3] & minority:
            selected.append(rec)
            for b in rec[2]: counts[b[0]] += 1
        else:
            remainder.append(rec)
    for rec in remainder:
        if any(counts[b[0]] < target_max for b in rec[2]):
            selected.append(rec)
            for b in rec[2]: counts[b[0]] += 1
    return selected, counts


def augment_to_floor(records, counts, target_min, img_dir, lbl_dir, prefix):
    needs = {i for i in range(NC) if counts[i] < target_min}
    if not needs:
        print("  No classes need augmentation — already balanced.")
        return
    print(f"  Classes below floor: {[CLASSES[i] for i in sorted(needs)]}")
    candidates = [r for r in records if r[3] & needs]
    orig_counts = dict(counts)
    ceiling = {i: max(orig_counts.get(i, 1) * 3, target_min) for i in range(NC)}
    aug_idx = 0
    OPS = ["brightness", "noise", "blur", "hflip", "rotate"]
    for _ in range(60000):
        still = {i for i in needs if counts[i] < target_min and counts[i] < ceiling[i]}
        if not still:
            break
        rec = random.choice(candidates)
        img_path, _, boxes, classes = rec
        if not (classes & still):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        aimg, aboxes = img.copy(), list(boxes)
        for op in random.sample(OPS, k=random.randint(2, 4)):
            if op == "brightness": aimg = aug_brightness_contrast(aimg)
            elif op == "noise":    aimg = aug_noise(aimg)
            elif op == "blur":     aimg = aug_blur(aimg)
            elif op == "hflip":    aimg, aboxes = aug_hflip(aimg, aboxes)
            elif op == "rotate":   aimg, aboxes = aug_rotate(aimg, aboxes)
        if not aboxes:
            continue
        name = f"{prefix}_aug_{aug_idx:05d}.jpg"
        cv2.imwrite(str(img_dir / name), aimg)
        write_boxes(lbl_dir / f"{prefix}_aug_{aug_idx:05d}.txt", aboxes)
        for b in aboxes: counts[b[0]] += 1
        aug_idx += 1
    print(f"  Generated {aug_idx} augmented images.")


def run_balance(src: Path, dst: Path, max_boxes: int, min_boxes: int):
    print(f"\n{'='*60}")
    print(f"  STEP 2/4 — BALANCE  (cap={max_boxes}, floor={min_boxes})")
    print(f"  Source : {src}")
    print(f"  Output : {dst}")
    print(f"{'='*60}")

    if dst.exists():
        print(f"  Removing old {dst} ...")
        shutil.rmtree(dst, ignore_errors=True)

    # --- TRAIN: cap + augment ---
    print("\n  [TRAIN] Collecting records...")
    train_records = collect_split(src, "train")
    raw_counts = count_boxes(train_records)
    print_distribution("TRAIN raw", raw_counts, len(train_records))

    selected, sel_counts = greedy_cap(train_records, max_boxes)
    print(f"\n  After cap: {len(selected)} images retained.")
    print_distribution("TRAIN after cap", sel_counts, len(selected))

    train_img_dir = dst / "train" / "images"
    train_lbl_dir = dst / "train" / "labels"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path, _, boxes, _ in selected:
        shutil.copy2(img_path, train_img_dir / img_path.name)
        write_boxes(train_lbl_dir / (img_path.stem + ".txt"), boxes)

    print("\n  Augmenting minority classes...")
    augment_to_floor(selected, sel_counts, min_boxes, train_img_dir, train_lbl_dir, "tr")

    final_train = collect_split(dst, "train")
    print_distribution("TRAIN final", count_boxes(final_train), len(final_train))

    # --- VALID + TEST: copy as-is (no balancing; representative sample) ---
    for split in ["valid", "test"]:
        print(f"\n  [{split.upper()}] Copying as-is...")
        records = collect_split(src, split)
        out_img = dst / split / "images"
        out_lbl = dst / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        for img_path, _, boxes, _ in records:
            shutil.copy2(img_path, out_img / img_path.name)
            write_boxes(out_lbl / (img_path.stem + ".txt"), boxes)
        print_distribution(split.upper(), count_boxes(records), len(records))

    print("\n  Balance complete.")


# ---------------------------------------------------------------------------
# Step 3 — Verify (zero-tolerance sanity check)
# ---------------------------------------------------------------------------

def run_verify(dataset: Path):
    print(f"\n{'='*60}")
    print(f"  STEP 3/4 — VERIFY")
    print(f"  Dataset : {dataset}")
    print(f"{'='*60}")

    errors = []
    total_images = 0
    for split in SPLITS:
        img_dir = dataset / split / "images"
        lbl_dir = dataset / split / "labels"
        if not img_dir.exists():
            errors.append(f"  [ERROR] Missing split directory: {img_dir}")
            continue
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        total_images += len(imgs)
        empty_labels = 0
        orphan_imgs  = 0
        for img in imgs:
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                orphan_imgs += 1
            else:
                boxes = read_boxes(lbl)
                if not boxes:
                    empty_labels += 1
        n = len(imgs)
        ok = orphan_imgs == 0 and empty_labels == 0
        status = "[OK]" if ok else "[!!]"
        print(f"  {status} [{split:<6}]  {n:>5} images  |  "
              f"empty labels: {empty_labels}  |  orphan images: {orphan_imgs}")
        if orphan_imgs > 0:
            errors.append(f"[{split}] {orphan_imgs} images have no label file.")
        if empty_labels > n * 0.05:   # more than 5% empty is suspicious
            errors.append(f"[{split}] {empty_labels} images have empty label files "
                          f"({100*empty_labels/n:.1f}% of split).")

    print(f"\n  Total images across all splits: {total_images}")

    if errors:
        print("\n  [!!] VERIFY WARNINGS:")
        for e in errors:
            print(f"    {e}")
        print("\n  These are warnings, not blockers. Review before training.")
    else:
        print("\n  [OK] All checks passed. Dataset is clean and ready.")

    return total_images


# ---------------------------------------------------------------------------
# Step 4 — Write YAMLs + Zip
# ---------------------------------------------------------------------------

def write_yaml(dataset_path: str, out_path: Path):
    content = f"""# Fruit Detection Dataset — Final (Filtered + Balanced)
# Generated by prepare_pod.py

path: {dataset_path}
train: train/images
val:   valid/images
test:  test/images

nc: {NC}

names:
"""
    for i, name in enumerate(CLASSES):
        content += f"  {i}: {name}\n"
    with open(out_path, "w") as f:
        f.write(content)
    print(f"  Wrote: {out_path}")


def run_pack(final_dir: Path, local_yaml: Path, pod_yaml: Path, zip_out: Path):
    print(f"\n{'='*60}")
    print(f"  STEP 4/4 — PACK  → {zip_out}")
    print(f"{'='*60}")

    # Write local yaml (absolute Windows path)
    write_yaml(str(final_dir.resolve()), local_yaml)

    # Write pod yaml (/workspace path for RunPod Linux)
    write_yaml(f"{POD_WORKSPACE}/dataset_v3_final", pod_yaml)

    # Files to include in zip
    files_to_zip = [
        final_dir,
        pod_yaml,
        Path("train.py"),
        Path("config.py"),
        Path("requirements.txt"),
    ]

    missing = [str(p) for p in files_to_zip if not p.exists()]
    if missing:
        sys.exit(f"\n[ERROR] Cannot zip — missing files:\n  " + "\n  ".join(missing))

    if zip_out.exists():
        zip_out.unlink()

    total_files = 0
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for item in files_to_zip:
            item = Path(item)
            if item.is_dir():
                for file in sorted(item.rglob("*")):
                    if file.is_file():
                        zf.write(file, file.relative_to(item.parent))
                        total_files += 1
            else:
                zf.write(item, item.name)
                total_files += 1

    size_mb = zip_out.stat().st_size / (1024 ** 2)
    print(f"\n  [OK] Packed {total_files} files -> {zip_out}  ({size_mb:.1f} MB)")
    print(f"\n  Upload command (replace <ip> and <port>):")
    print(f"    scp -P <port> {zip_out.resolve()} root@<ip>:/workspace/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end dataset prep: filter → balance → verify → pack"
    )
    parser.add_argument("--min_px",    type=int, default=DEFAULT_MIN_PX,
                        help=f"Min box side in pixels at {INPUT_SZ}px (default={DEFAULT_MIN_PX})")
    parser.add_argument("--max_boxes", type=int, default=DEFAULT_MAX_BOXES,
                        help=f"Cap majority classes in train (default={DEFAULT_MAX_BOXES})")
    parser.add_argument("--min_boxes", type=int, default=DEFAULT_MIN_BOXES,
                        help=f"Floor minority classes in train (default={DEFAULT_MIN_BOXES})")
    parser.add_argument("--skip_pack", action="store_true",
                        help="Skip zipping (useful for local-only reruns)")
    parser.add_argument("--seed",      type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "=" * 60)
    print("  FRUIT DETECTION -- PREPARE POD DATASET")
    print("  filter -> balance -> verify -> pack")
    print("=" * 60)
    print(f"  min_px    = {args.min_px}")
    print(f"  max_boxes = {args.max_boxes}")
    print(f"  min_boxes = {args.min_boxes}")

    # 1. Filter
    run_filter(SOURCE_RAW, FILTERED_DIR, args.min_px)

    # 2. Balance
    run_balance(FILTERED_DIR, FINAL_DIR, args.max_boxes, args.min_boxes)

    # 3. Verify
    total = run_verify(FINAL_DIR)

    # 4. Pack
    if not args.skip_pack:
        run_pack(FINAL_DIR, LOCAL_YAML, POD_YAML, ZIP_OUT)

    print("\n" + "=" * 60)
    print("  [DONE] DATASET READY")
    print(f"  Final dataset  : {FINAL_DIR.resolve()}")
    print(f"  Local yaml     : {LOCAL_YAML.resolve()}")
    print(f"  Pod yaml       : {POD_YAML.resolve()}")
    if not args.skip_pack:
        print(f"  Pod zip        : {ZIP_OUT.resolve()}")
    print(f"  Total images   : {total}")
    print("=" * 60)
    print("\n  NEXT STEP: Upload fruit_detection_pod.zip to RunPod")
    print("  Then run on the pod:")
    print("    python train.py --model yolov8s.pt \\")
    print("      --data /workspace/data_v3_final_pod.yaml \\")
    print("      --batch 64 --workers 4 --name fruit_v4_s --clean")
    print("=" * 60)


if __name__ == "__main__":
    main()
