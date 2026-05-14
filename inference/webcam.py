"""
Fruit Detection - Live Webcam Inference

Usage:
    python inference/webcam.py
    python inference/webcam.py --model models/best.pt --camera 0
    Press 'q' to quit.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
from ultralytics import YOLO

COLORS = {
    0: (0,200,0), 1: (0,230,255), 2: (0,140,255), 3: (0,180,255),
    4: (30,200,220), 5: (80,80,220), 6: (180,80,180), 7: (60,60,200),
}

def annotate(frame, results):
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cid = int(box.cls[0])
            name = r.names[cid]
            color = COLORS.get(cid, (200,200,200))
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"{name} {conf:.2f}"
            (lw,lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1,y1-lh-bl-4), (x1+lw,y1), color, -1)
            cv2.putText(frame, label, (x1,y1-bl-2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return frame

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--camera", type=int, default=0, help="Camera index")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}"); sys.exit(1)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}"); sys.exit(1)

    print(f"Webcam started (camera {args.camera}). Press 'q' to quit.")
    fps_smooth = 0.0

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed"); break

        results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        annotated = annotate(frame, results)

        # FPS counter
        dt = time.time() - t0
        fps = 1.0 / max(dt, 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps
        cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Fruit Detection - Live", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stopped.")

if __name__ == "__main__":
    main()
