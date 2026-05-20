"""
check_split_leakage.py
======================
Hard-gate train/valid/test split leakage for YOLO datasets.

Checks:
  - exact duplicate image content by MD5
  - perceptual near-duplicates by simple pHash

The script exits non-zero if any cross-split duplicate or near-duplicate is
found, unless --no-phash is used to skip the near-duplicate gate.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "valid", "test"]


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(path: Path, size: int = 8, highfreq_factor: int = 4) -> int | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    img_size = size * highfreq_factor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low_freq = dct[:size, :size].copy()
    coeffs = low_freq.flatten()[1:]
    median = float(np.median(coeffs))
    bits = (low_freq.flatten() > median)
    bits[0] = False
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class BKTree:
    def __init__(self):
        self.root = None

    def add(self, value: int, item: Path) -> None:
        node = (value, [item], {})
        if self.root is None:
            self.root = node
            return
        cur = self.root
        while True:
            dist = hamming(value, cur[0])
            if dist == 0:
                cur[1].append(item)
                return
            children = cur[2]
            if dist not in children:
                children[dist] = node
                return
            cur = children[dist]

    def query(self, value: int, max_dist: int):
        if self.root is None:
            return []
        out = []
        stack = [self.root]
        while stack:
            cur = stack.pop()
            dist = hamming(value, cur[0])
            if dist <= max_dist:
                out.extend((item, dist) for item in cur[1])
            low, high = dist - max_dist, dist + max_dist
            for edge_dist, child in cur[2].items():
                if low <= edge_dist <= high:
                    stack.append(child)
        return out


def split_images(dataset: Path) -> dict[str, list[Path]]:
    result = {}
    for split in SPLITS:
        img_dir = dataset / split / "images"
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing split image directory: {img_dir}")
        result[split] = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    return result


def check_md5(images_by_split: dict[str, list[Path]]):
    by_hash = defaultdict(list)
    for split, images in images_by_split.items():
        for image in images:
            by_hash[md5_file(image)].append((split, image))
    leaks = []
    for digest, items in by_hash.items():
        splits = {split for split, _ in items}
        if len(splits) > 1:
            leaks.append((digest, items))
    return leaks


def check_phash(images_by_split: dict[str, list[Path]], threshold: int):
    leaks = []
    trees: dict[str, BKTree] = {}
    hashes: dict[str, dict[Path, int]] = {}
    for split, images in images_by_split.items():
        tree = BKTree()
        hashes[split] = {}
        print(f"Hashing {split}: {len(images):,} images")
        for image in images:
            h = phash(image)
            if h is None:
                continue
            hashes[split][image] = h
            tree.add(h, image)
        trees[split] = tree

    checked_pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    for left, right in checked_pairs:
        print(f"Checking pHash leakage: {left} vs {right}")
        for image, h in hashes[right].items():
            matches = trees[left].query(h, threshold)
            for match, dist in matches:
                leaks.append((left, match, right, image, dist))
                if len(leaks) >= 100:
                    return leaks
    return leaks


def main() -> None:
    parser = argparse.ArgumentParser(description="Check cross-split leakage")
    parser.add_argument("--dataset", default="dataset_v4_quality")
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--no-phash", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    images_by_split = split_images(dataset)

    print("=" * 72)
    print("  DATASET V4 SPLIT LEAKAGE CHECK")
    print("=" * 72)
    for split, images in images_by_split.items():
        print(f"{split:<8} {len(images):>7} images")

    md5_leaks = check_md5(images_by_split)
    print(f"\nExact MD5 cross-split duplicates: {len(md5_leaks)}")
    for digest, items in md5_leaks[:10]:
        joined = " | ".join(f"{split}:{path.name}" for split, path in items)
        print(f"  {digest[:12]}  {joined}")

    phash_leaks = []
    if not args.no_phash:
        phash_leaks = check_phash(images_by_split, args.phash_threshold)
        print(f"\npHash near-duplicates across splits: {len(phash_leaks)}")
        for left, left_path, right, right_path, dist in phash_leaks[:10]:
            print(f"  dist={dist:<2} {left}:{left_path.name}  <->  {right}:{right_path.name}")
    else:
        print("\npHash near-duplicate check skipped.")

    if md5_leaks or phash_leaks:
        raise SystemExit("[FAIL] Split leakage gate failed.")
    print("\n[PASS] No cross-split leakage found.")


if __name__ == "__main__":
    main()
