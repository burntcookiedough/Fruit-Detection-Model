import argparse
import sys
import time
import random
from pathlib import Path
import cv2
import numpy as np
import urllib.request
from ultralytics import YOLO
import torch

# --- PyTorch 2.6 Compatibility Workaround ---
# PyTorch 2.6 defaults to weights_only=True, which breaks ultralytics weight loading.
# Since we trust our own trained model, we temporarily patch torch.load to bypass this.
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# --------------------------------------------

# A list of fruit images from Wikimedia Commons to serve as random "real world" internet tests
SAMPLE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/2/25/Fruit_bowl.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/f4/Honeycrisp.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/c/c4/Orange-Fruit-Pieces.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/90/Hapus_Mango.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/c/cb/Pineapple_and_cross_section.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/a/a2/Watermelon.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/b/bb/Table_grapes_on_white.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/fa/Pomegranate_split.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/4/4c/Bananas.jpg"
]

# Colors for the 8 classes: apple, banana, orange, mango, pineapple, watermelon, grapes, pomegranate
COLORS = {
    0: (0, 200, 0),    1: (0, 230, 255),  2: (0, 140, 255),  3: (0, 180, 255),
    4: (30, 200, 220), 5: (80, 80, 220),  6: (180, 80, 180), 7: (60, 60, 200),
}

def annotate(frame, results):
    """Draws bounding boxes and labels on the image"""
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cid = int(box.cls[0])
            name = r.names[cid]
            color = COLORS.get(cid, (200, 200, 200))
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background and text
            label = f"{name} {conf:.2f}"
            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - lh - bl - 4), (x1 + lw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - bl - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

def run_internet(model, url=None, conf=0.25, imgsz=640):
    """Downloads an image from the internet and runs inference"""
    if url is None:
        url = random.choice(SAMPLE_URLS)
        print(f"\n[INFO] No URL provided. Picked a random sample URL:\n{url}")
    else:
        print(f"\n[INFO] Downloading image from:\n{url}")

    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        response = urllib.request.urlopen(req)
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(arr, -1)
        if frame is None:
            print("[ERROR] Failed to decode image from URL.")
            return
    except Exception as e:
        print(f"[ERROR] Could not download or load image: {e}")
        return

    print("[INFO] Running inference...")
    results = model(frame, imgsz=imgsz, conf=conf, verbose=False)
    annotated = annotate(frame, results)
    
    # Scale down for viewing if the image is too large for the screen
    h, w = annotated.shape[:2]
    if max(h, w) > 1000:
        scale = 1000 / max(h, w)
        annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))
    
    print("\n>>> PRESS ANY KEY in the image window to close it <<<")
    cv2.imshow("Fruit Detection - Internet Image", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_webcam(model, camera_id=0, conf=0.25, imgsz=640):
    """Runs live inference on a connected webcam"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera with ID {camera_id}. Make sure it is connected and not used by another app.")
        return

    print(f"\n[INFO] Webcam started on camera {camera_id}.")
    print(">>> PRESS 'q' in the video window to quit <<<")
    
    fps_smooth = 0.0

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed")
            break

        # Run inference
        results = model(frame, imgsz=imgsz, conf=conf, verbose=False)
        annotated = annotate(frame, results)

        # Calculate and draw FPS
        dt = time.time() - t0
        fps = 1.0 / max(dt, 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps
        cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Fruit Detection - Live Webcam", annotated)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam stopped.")

def main():
    p = argparse.ArgumentParser(description="Fruit Detection Demo: Internet Images & Webcam")
    p.add_argument("--mode", choices=["internet", "webcam"], default="internet", 
                   help="Choose 'internet' to test an image URL, or 'webcam' for live camera.")
    p.add_argument("--url", type=str, help="Specific image URL (for internet mode). If omitted, a random URL is picked.")
    p.add_argument("--camera", type=int, default=0, help="Camera index (for webcam mode), usually 0 or 1.")
    p.add_argument("--model", default="models/best.pt", help="Path to your trained YOLOv8 model")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold to show boxes")
    p.add_argument("--imgsz", type=int, default=640, help="Image size for inference")
    args = p.parse_args()

    if not Path(args.model).exists():
        print(f"[ERROR] Model file not found at: {args.model}")
        print("Please train the model first or check the path.")
        sys.exit(1)

    print(f"[INFO] Loading model weights from {args.model}...")
    model = YOLO(args.model)

    if args.mode == "internet":
        run_internet(model, args.url, args.conf, args.imgsz)
    elif args.mode == "webcam":
        run_webcam(model, args.camera, args.conf, args.imgsz)

if __name__ == "__main__":
    main()
