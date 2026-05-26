"""Phase 1 gate: visualize 4 train images with GT boxes."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from torch.utils.data import DataLoader

from custom_config import TRAIN_IMG_DIR, TRAIN_LBL_DIR, IMG_SIZE, CLASS_NAMES
from custom_detector.dataset import FruitDataset, collate_fn
from custom_detector.utils import draw_boxes, cxcywh_to_xyxy


def main():
    ds = FruitDataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR, IMG_SIZE, augment=False)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    images, boxes_list, labels_list = next(iter(loader))

    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    for i in range(4):
        img = images[i].permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        boxes = boxes_list[i]
        labels = labels_list[i]
        if boxes.numel() > 0:
            boxes_xyxy = cxcywh_to_xyxy(boxes).numpy()
            img = draw_boxes(img, boxes_xyxy, labels.numpy(), class_names=CLASS_NAMES)
        out_path = os.path.join(debug_dir, f'phase1_sample_{i}.jpg')
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f'Saved {out_path}')
    print('Phase 1 gate: check debug/ folder — boxes should align with fruits.')


if __name__ == '__main__':
    main()
