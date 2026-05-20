import argparse
import sys
import time
import random
from pathlib import Path
import cv2
import numpy as np
import urllib.request
from ultralytics import YOLO

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

def preprocess_webcam_frame(frame):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalisation) to improve
    detection in dark or unevenly lit webcam frames.

    CLAHE works in LAB colour space so it only touches luminance -- colours
    remain natural but local contrast is dramatically improved.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def run_webcam(model, camera_id=0, conf=0.15, imgsz=640, enhance=True):
    """Runs live inference on a connected webcam.

    Args:
        model       : Loaded YOLO model.
        camera_id   : Camera index (default 0).
        conf        : Confidence threshold.  Lower = more detections.
                      0.10-0.20 works best for webcam.  Default: 0.15
        imgsz       : Inference image size.  320 is faster; 640 is more accurate.
        enhance     : Apply CLAHE preprocessing (helps a lot in dark rooms).
    """
    # Warm up the model with a dummy frame first so the webcam buffer doesn't timeout
    print("[INFO] Warming up model...")
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model(dummy, imgsz=imgsz, conf=conf, verbose=False)

    # Robust Windows fallback system to handle MSMF/DirectShow and custom resolution issues
    cap = None
    backends_to_try = [
        ("MSMF with 1280x720", None, True),
        ("MSMF with default resolution", None, False),
        ("DirectShow with 1280x720", cv2.CAP_DSHOW, True),
        ("DirectShow with default resolution", cv2.CAP_DSHOW, False)
    ]

    for label, backend, set_res in backends_to_try:
        print(f"[INFO] Attempting to open webcam via {label}...")
        if backend is None:
            temp_cap = cv2.VideoCapture(camera_id)
        else:
            temp_cap = cv2.VideoCapture(camera_id, backend)

        if not temp_cap.isOpened():
            print("  -> Backend initialization failed.")
            temp_cap.release()
            continue

        if set_res:
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Test if we can actually read a frame
        ret, frame = temp_cap.read()
        if ret and frame is not None and frame.size > 0:
            print("  [OK] Successfully connected and captured a test frame!")
            cap = temp_cap
            break
        else:
            print("  -> Failed to grab frame, trying next fallback.")
            temp_cap.release()

    if cap is None:
        print(f"[ERROR] Cannot open camera with ID {camera_id}.")
        print("Please verify:\n1. The webcam is connected.\n2. No other app (Zoom, Teams, Discord, browser, OBS) is currently using the camera.")
        return

    print(f"\n[INFO] Webcam started on camera {camera_id}.")
    print(f"[INFO] Confidence threshold : {conf}  (lower = more detections)")
    print(f"[INFO] CLAHE enhancement    : {'ON' if enhance else 'OFF'}")
    print(">>>> PRESS 'q' in the video window to quit <<<<")

    fps_smooth = 0.0
    fail_count = 0

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count > 10:
                print("[ERROR] Frame capture failed repeatedly. Exiting.")
                break
            time.sleep(0.1)
            continue
        fail_count = 0  # reset on success

        # Apply CLAHE to boost contrast in poor lighting
        inference_frame = preprocess_webcam_frame(frame) if enhance else frame

        # Run inference
        results = model(inference_frame, imgsz=imgsz, conf=conf, verbose=False)

        # Annotate original (un-processed) frame so colours look natural
        annotated = annotate(frame, results)

        # Calculate and draw FPS
        dt = time.time() - t0
        fps = 1.0 / max(dt, 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps
        cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(annotated, f"conf={conf:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

        cv2.imshow("Fruit Detection - Live Webcam", annotated)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam stopped.")

def main():
    p = argparse.ArgumentParser(
        description="Fruit Detection Demo: Internet Images & Webcam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                                  # random internet image
  python demo.py --mode webcam                    # webcam (CLAHE + conf=0.15)
  python demo.py --mode webcam --conf 0.10        # lower conf = more detections
  python demo.py --mode webcam --no-enhance       # disable CLAHE preprocessing
  python demo.py --url https://...image.jpg       # specific URL
"""
    )
    p.add_argument("--mode", choices=["internet", "webcam"], default="internet",
                   help="'internet' to test an image URL, or 'webcam' for live camera.")
    p.add_argument("--url", type=str,
                   help="Specific image URL (internet mode). If omitted, a random URL is picked.")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index (webcam mode), usually 0 or 1.")
    p.add_argument("--model", default="models/best.pt",
                   help="Path to your trained YOLOv8 model (default: models/best.pt)")
    p.add_argument("--conf", type=float, default=None,
                   help="Confidence threshold. Default: 0.25 for internet, 0.15 for webcam.")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Image size for inference (default: 640)")
    p.add_argument("--no-enhance", dest="enhance", action="store_false", default=True,
                   help="Disable CLAHE preprocessing for webcam mode (enabled by default)")
    args = p.parse_args()

    if not Path(args.model).exists():
        print(f"[ERROR] Model file not found at: {args.model}")
        print("Please train the model first or check the path.")
        sys.exit(1)

    print(f"[INFO] Loading model weights from {args.model}...")
    model = YOLO(args.model)

    if args.mode == "internet":
        conf = args.conf if args.conf is not None else 0.25
        run_internet(model, args.url, conf, args.imgsz)
    elif args.mode == "webcam":
        conf = args.conf if args.conf is not None else 0.15
        run_webcam(model, args.camera, conf, args.imgsz, enhance=args.enhance)


if __name__ == "__main__":
    main()
