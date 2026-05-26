"""Inference on images or webcam."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from custom_config import (
    IMG_SIZE, NUM_CLASSES, CLASS_NAMES, CONF_THRESH, NMS_IOU,
    ANCHOR_SCALES, ANCHOR_RATIOS, PRE_NMS_TOPK, MAX_DETECTIONS,
    WEIGHTS_DIR, FM_SIZES
)
from custom_detector.model import FruitDetector
from custom_detector.utils import draw_boxes


def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    orig = np.array(img)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    t = transforms.ToTensor()(img).unsqueeze(0)
    return t, orig


def preprocess_webcam(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    t = transforms.ToTensor()(pil).unsqueeze(0)
    return t


def run_image(model, img_path, device, anchors):
    from train_custom import decode_predictions
    t, orig = preprocess(img_path)
    t = t.to(device)
    with torch.no_grad():
        cls_pred, box_pred, _ = model(t)
        boxes, labels, scores = decode_predictions(
            cls_pred[0], box_pred[0], anchors, CONF_THRESH, NMS_IOU,
            PRE_NMS_TOPK, MAX_DETECTIONS
        )

    h, w = orig.shape[:2]
    scale_x = w / IMG_SIZE
    scale_y = h / IMG_SIZE
    if boxes.numel() > 0:
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        out = draw_boxes(orig, boxes.cpu().numpy(), labels.cpu().numpy(), scores.cpu().numpy(), CLASS_NAMES)
    else:
        out = orig

    out_path = os.path.splitext(img_path)[0] + '_detected.jpg'
    cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"Saved: {out_path}")


def run_webcam(model, device, anchors):
    from train_custom import decode_predictions
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    print("Press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = preprocess_webcam(frame).to(device)
        with torch.no_grad():
            cls_pred, box_pred, _ = model(t)
            boxes, labels, scores = decode_predictions(
                cls_pred[0], box_pred[0], anchors, CONF_THRESH, NMS_IOU,
                PRE_NMS_TOPK, MAX_DETECTIONS
            )

        h, w = frame.shape[:2]
        scale_x = w / IMG_SIZE
        scale_y = h / IMG_SIZE
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            for i, box in enumerate(boxes.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                txt = f"{CLASS_NAMES[int(labels[i].item())]} {scores[i].item():.2f}"
                cv2.putText(frame, txt, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Fruit Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['image', 'webcam'], required=True)
    parser.add_argument('--input', type=str, default='', help='Image path for image mode')
    parser.add_argument('--weights', type=str, default=os.path.join(WEIGHTS_DIR, 'best_map50.pt'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FruitDetector(NUM_CLASSES, IMG_SIZE, ANCHOR_SCALES, ANCHOR_RATIOS, FM_SIZES).to(device)

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    anchors = model.anchors.to(device)

    if args.mode == 'image':
        if not args.input:
            print("Provide --input for image mode")
            return
        run_image(model, args.input, device, anchors)
    else:
        run_webcam(model, device, anchors)


if __name__ == '__main__':
    main()
