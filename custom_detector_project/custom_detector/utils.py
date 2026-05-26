"""Box format conversions and visualization."""
import torch
import cv2
import numpy as np


def cxcywh_to_xyxy(boxes):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_cxcywh(boxes):
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def draw_boxes(img, boxes, labels=None, scores=None, class_names=None, color=(0, 255, 0)):
    """Draw boxes on a numpy image (H, W, C) in BGR."""
    img = img.copy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = ''
        if labels is not None:
            lbl = int(labels[i])
            name = class_names[lbl] if class_names else str(lbl)
            text = name
        if scores is not None:
            text += f' {scores[i]:.2f}' if text else f'{scores[i]:.2f}'
        if text:
            cv2.putText(img, text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img
