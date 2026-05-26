"""Phase 3 gate: overfit 1 image for 300 steps."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.nn.utils import clip_grad_norm_

from custom_config import (
    TRAIN_IMG_DIR, TRAIN_LBL_DIR, IMG_SIZE, NUM_CLASSES, GRAD_CLIP,
    ANCHOR_SCALES, ANCHOR_RATIOS, FM_SIZES, FOCAL_GAMMA, FOCAL_ALPHA,
    POS_IOU, NEG_IOU, NEG_POS_RATIO, MATCHER_TYPE
)
from custom_detector.dataset import FruitDataset
from custom_detector.model import FruitDetector
from custom_detector.loss import DetectionLoss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ds = FruitDataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR, IMG_SIZE, augment=False)

    # Grab 1 image
    img, boxes, labels = ds[0]
    images = img.unsqueeze(0).to(device)
    boxes_list = [boxes.to(device)]
    labels_list = [labels.to(device)]

    model = FruitDetector(NUM_CLASSES, IMG_SIZE, ANCHOR_SCALES, ANCHOR_RATIOS, FM_SIZES).to(device)
    criterion = DetectionLoss(
        NUM_CLASSES, POS_IOU, NEG_IOU, FOCAL_GAMMA, FOCAL_ALPHA,
        matcher_type=MATCHER_TYPE, img_size=IMG_SIZE,
        fm_sizes=FM_SIZES, ratios=ANCHOR_RATIOS,
        neg_pos_ratio=NEG_POS_RATIO
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print('Starting overfit test on 1 image...')
    for step in range(300):
        cls_pred, box_pred, anchors = model(images)
        loss_dict = criterion(cls_pred, box_pred, anchors, boxes_list, labels_list)
        loss = loss_dict['total']

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % 50 == 0:
            print(f'Step {step:03d} | loss={loss.item():.4f} cls={loss_dict["cls"].item():.4f} box={loss_dict["box"].item():.4f} num_pos={loss_dict["num_pos"]}')

    print(f'Final loss: {loss.item():.4f}')
    if loss.item() < 0.5:
        print('PASS: Loss dropped below 0.5')
    else:
        print('FAIL: Loss did not drop below 0.5 — debug before training.')


if __name__ == '__main__':
    main()
