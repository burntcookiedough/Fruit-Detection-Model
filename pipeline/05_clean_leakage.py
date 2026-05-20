"""
clean_split_leakage.py
======================
Quarantine cross-split duplicate and near-duplicate images from a YOLO dataset.

Removal priority preserves evaluation integrity:
  - train vs valid/test: move the train image out
  - valid vs test: move the valid image out

Images and labels are moved into <dataset>_leakage_quarantine/.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

from check_split_leakage import BKTree, hamming, md5_file, phash

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "valid", "test"]


def images_for_split(dataset: Path, split: str) -> list[Path]:
    img_dir = dataset / split / "images"
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def label_for_image(image: Path) -> Path:
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def move_pair(dataset: Path, quarantine: Path, image: Path) -> bool:
    if not image.exists():
        return False
    split = image.parent.parent.name
    dest_img = quarantine / split / "images" / image.name
    dest_lbl = quarantine / split / "labels" / f"{image.stem}.txt"
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    dest_lbl.parent.mkdir(parents=True, exist_ok=True)
    label = label_for_image(image)
    shutil.move(str(image), str(dest_img))
    if label.exists():
        shutil.move(str(label), str(dest_lbl))
    return True


def exact_duplicate_removals(dataset: Path) -> set[Path]:
    by_hash = defaultdict(list)
    for split in SPLITS:
        for image in images_for_split(dataset, split):
            by_hash[md5_file(image)].append((split, image))

    remove = set()
    for items in by_hash.values():
        splits = {split for split, _ in items}
        if len(splits) <= 1:
            continue
        for split, image in items:
            if split == "train":
                remove.add(image)
        if not any(split == "train" for split, _ in items):
            for split, image in items:
                if split == "valid":
                    remove.add(image)
    return remove


def phash_duplicate_removals(dataset: Path, threshold: int, max_passes: int) -> set[Path]:
    all_remove = set()
    for _ in range(max_passes):
        by_split = {}
        trees = {}
        for split in SPLITS:
            hashes = {}
            tree = BKTree()
            for image in images_for_split(dataset, split):
                if image in all_remove:
                    continue
                h = phash(image)
                if h is None:
                    continue
                hashes[image] = h
                tree.add(h, image)
            by_split[split] = hashes
            trees[split] = tree

        remove = set()
        for left, right, remove_side in [
            ("train", "valid", "train"),
            ("train", "test", "train"),
            ("valid", "test", "valid"),
        ]:
            for right_img, right_hash in by_split[right].items():
                for left_img, _dist in trees[left].query(right_hash, threshold):
                    remove.add(left_img if remove_side == left else right_img)

        remove -= all_remove
        if not remove:
            break
        all_remove |= remove
    return all_remove


def main() -> None:
    parser = argparse.ArgumentParser(description="Quarantine V4 cross-split leakage")
    parser.add_argument("--dataset", default="dataset_v4_quality")
    parser.add_argument("--quarantine", default=None)
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--max-passes", type=int, default=3)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    quarantine = Path(args.quarantine) if args.quarantine else Path(f"{args.dataset}_leakage_quarantine")
    if not dataset.exists():
        raise FileNotFoundError(dataset)

    exact_remove = exact_duplicate_removals(dataset)
    phash_remove = phash_duplicate_removals(dataset, args.phash_threshold, args.max_passes)
    to_remove = sorted(exact_remove | phash_remove)

    moved = 0
    for image in to_remove:
        if move_pair(dataset, quarantine, image):
            moved += 1

    print("=" * 72)
    print("  V4 SPLIT LEAKAGE CLEANUP")
    print("=" * 72)
    print(f"Exact duplicate removals: {len(exact_remove):,}")
    print(f"pHash duplicate removals: {len(phash_remove):,}")
    print(f"Moved to quarantine:      {moved:,}")
    print(f"Quarantine: {quarantine.resolve()}")


if __name__ == "__main__":
    main()
