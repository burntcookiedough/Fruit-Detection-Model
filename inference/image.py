"""
Fruit Detection - Single Image Inference

Usage:
    python inference/image.py --source path/to/image.jpg
    python inference/image.py --source path/to/image.jpg --save
"""

import argparse
import sys
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
    p.add_argument("--source", required=True, help="Image path or directory")
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}"); sys.exit(1)

    model = YOLO(args.model)
    src = Path(args.source)
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    imgs = sorted(f for f in (src.iterdir() if src.is_dir() else [src]) if f.suffix.lower() in exts)

    out_dir = Path("output")
    if args.save: out_dir.mkdir(exist_ok=True)

    for img_path in imgs:
        frame = cv2.imread(str(img_path))
        if frame is None: continue
        results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        annotated = annotate(frame.copy(), results)
        n = sum(len(r.boxes) for r in results if r.boxes is not None)
        print(f"{img_path.name}: {n} detection(s)")
        if args.save:
            cv2.imwrite(str(out_dir / f"det_{img_path.name}"), annotated)
        cv2.imshow("Fruit Detection", annotated)
        if cv2.waitKey(0) == ord("q"): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
