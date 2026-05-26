"""PyTorch Dataset for YOLO-format labels."""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF


class FruitDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=416, augment=False, cache_dir=None,
                 cache_images=False, return_meta=False):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.img_size = img_size
        self.augment = augment
        self.cache_dir = cache_dir
        self.cache_images = cache_images
        self.return_meta = return_meta
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.isdir(lbl_dir):
            raise FileNotFoundError(f"Label directory not found: {lbl_dir}")
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not self.img_files:
            raise RuntimeError(f"No images found in: {img_dir}")

    def __len__(self):
        return len(self.img_files)

    def cache_path(self, fname):
        if not self.cache_dir:
            return None
        stem = os.path.splitext(fname)[0]
        return os.path.join(self.cache_dir, f"{stem}.npy")

    def load_image(self, fname):
        img_path = os.path.join(self.img_dir, fname)
        cache_path = self.cache_path(fname)
        if self.cache_images and cache_path and os.path.exists(cache_path):
            arr = np.load(cache_path)
            return TF.to_tensor(arr)

        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        return TF.to_tensor(img)

    def build_cache(self, overwrite=False, verbose=True):
        if not self.cache_dir:
            raise ValueError("cache_dir must be set before building image cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        written = 0
        skipped = 0
        for idx, fname in enumerate(self.img_files, start=1):
            cache_path = self.cache_path(fname)
            if os.path.exists(cache_path) and not overwrite:
                skipped += 1
                continue
            img_path = os.path.join(self.img_dir, fname)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)
            tmp_path = cache_path + ".tmp.npy"
            np.save(tmp_path, arr)
            os.replace(tmp_path, cache_path)
            written += 1
            if verbose and (idx % 1000 == 0 or idx == len(self.img_files)):
                print(f"  cached {idx}/{len(self.img_files)} images -> {self.cache_dir}")
        return {"written": written, "skipped": skipped, "total": len(self.img_files)}

    def __getitem__(self, idx):
        fname = self.img_files[idx]
        lbl_path = os.path.join(self.lbl_dir, os.path.splitext(fname)[0] + '.txt')

        boxes = []
        labels = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx = float(parts[1]) * self.img_size
                        cy = float(parts[2]) * self.img_size
                        w = float(parts[3]) * self.img_size
                        h = float(parts[4]) * self.img_size
                        if w > 0 and h > 0:
                            boxes.append([cx, cy, w, h])
                            labels.append(cls_id)

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        img = self.load_image(fname)
        flipped = False

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
                flipped = True
                if boxes.numel() > 0:
                    boxes[:, 0] = self.img_size - boxes[:, 0]
            if random.random() < 0.3:
                img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
            if random.random() < 0.3:
                img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
            if random.random() < 0.3:
                img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
            if random.random() < 0.1:
                img = TF.to_pil_image(img)
                img = img.filter(ImageFilter.GaussianBlur(radius=random.choice([1, 2])))
                img = TF.to_tensor(img)

        if self.return_meta:
            stem = os.path.splitext(fname)[0]
            return img, boxes, labels, stem, flipped
        return img, boxes, labels


def collate_fn(batch):
    images = []
    boxes_list = []
    labels_list = []
    sample_keys = []
    has_meta = len(batch[0]) == 5
    for item in batch:
        if has_meta:
            img, boxes, labels, stem, flipped = item
            sample_keys.append(f"{stem}__flip{int(flipped)}")
        else:
            img, boxes, labels = item
        images.append(img)
        boxes_list.append(boxes)
        labels_list.append(labels)
    images = torch.stack(images, dim=0)
    if has_meta:
        return images, boxes_list, labels_list, sample_keys
    return images, boxes_list, labels_list
