"""PyTorch Dataset for YOLO-format labels."""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF

# ImageNet channel-wise normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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

    def _mosaic(self, idx):
        """Combine 4 images into one mosaic composite."""
        img_size = self.img_size
        # Random center point for the mosaic
        cx = random.randint(img_size // 4, 3 * img_size // 4)
        cy = random.randint(img_size // 4, 3 * img_size // 4)

        indices = [idx] + random.choices(range(len(self)), k=3)
        all_boxes = []
        all_labels = []

        # Create canvas
        mosaic_img = torch.zeros(3, img_size, img_size)

        for i, idx_i in enumerate(indices):
            img_i = self.load_image(self.img_files[idx_i])
            lbl_path = os.path.join(self.lbl_dir, os.path.splitext(self.img_files[idx_i])[0] + '.txt')
            boxes_i = []
            labels_i = []
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            bx = float(parts[1]) * img_size
                            by = float(parts[2]) * img_size
                            bw = float(parts[3]) * img_size
                            bh = float(parts[4]) * img_size
                            if bw > 0 and bh > 0:
                                boxes_i.append([bx, by, bw, bh])
                                labels_i.append(cls_id)

            _, h, w = img_i.shape
            # Place each quadrant
            if i == 0:    # top-left
                x1s, y1s, x1e, y1e = max(cx - w, 0), max(cy - h, 0), cx, cy
                x2s, y2s, x2e, y2e = w - (x1e - x1s), h - (y1e - y1s), w, h
            elif i == 1:  # top-right
                x1s, y1s, x1e, y1e = cx, max(cy - h, 0), min(cx + w, img_size), cy
                x2s, y2s, x2e, y2e = 0, h - (y1e - y1s), min(w, x1e - x1s), h
            elif i == 2:  # bottom-left
                x1s, y1s, x1e, y1e = max(cx - w, 0), cy, cx, min(cy + h, img_size)
                x2s, y2s, x2e, y2e = w - (x1e - x1s), 0, w, min(h, y1e - y1s)
            else:         # bottom-right
                x1s, y1s, x1e, y1e = cx, cy, min(cx + w, img_size), min(cy + h, img_size)
                x2s, y2s, x2e, y2e = 0, 0, min(w, x1e - x1s), min(h, y1e - y1s)

            mosaic_img[:, y1s:y1e, x1s:x1e] = img_i[:, y2s:y2e, x2s:x2e]

            # Shift boxes: convert cxcywh to xyxy, shift, clip, convert back
            for bi, (bx, by, bw, bh) in enumerate(boxes_i):
                # Original box in xyxy
                ox1 = bx - bw / 2
                oy1 = by - bh / 2
                ox2 = bx + bw / 2
                oy2 = by + bh / 2
                # Shift to mosaic coords
                shift_x = x1s - x2s
                shift_y = y1s - y2s
                nx1 = max(ox1 + shift_x, x1s)
                ny1 = max(oy1 + shift_y, y1s)
                nx2 = min(ox2 + shift_x, x1e)
                ny2 = min(oy2 + shift_y, y1e)
                # Check if box is still valid (min area)
                if nx2 - nx1 > 2 and ny2 - ny1 > 2:
                    new_cx = (nx1 + nx2) / 2
                    new_cy = (ny1 + ny2) / 2
                    new_w = nx2 - nx1
                    new_h = ny2 - ny1
                    all_boxes.append([new_cx, new_cy, new_w, new_h])
                    all_labels.append(labels_i[bi])

        boxes = torch.tensor(all_boxes, dtype=torch.float32) if all_boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(all_labels, dtype=torch.long) if all_labels else torch.zeros((0,), dtype=torch.long)
        return mosaic_img, boxes, labels

    def __getitem__(self, idx):
        fname = self.img_files[idx]

        # Mosaic augmentation: combine 4 images with 50% probability during training
        if self.augment and random.random() < 0.5:
            img, boxes, labels = self._mosaic(idx)
        else:
            # Standard single-image path with label loading
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
            # Existing augmentations: horizontal flip, color jitter, blur
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

            # Cutout: erase a random rectangular patch (30% probability)
            if random.random() < 0.3:
                img_size = self.img_size
                eh = random.randint(img_size // 10, img_size // 4)
                ew = random.randint(img_size // 10, img_size // 4)
                ex = random.randint(0, img_size - ew)
                ey = random.randint(0, img_size - eh)
                img[:, ey:ey+eh, ex:ex+ew] = 0.0

            # Random vertical flip (10% probability)
            if random.random() < 0.1:
                img = TF.vflip(img)
                if boxes.numel() > 0:
                    boxes[:, 1] = self.img_size - boxes[:, 1]

        # ImageNet normalization (always applied, after all augmentations)
        img = TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)

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
