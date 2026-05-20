import os
import random
import shutil
from pathlib import Path
import cv2
from ultralytics import YOLO

# Classes mapping from data_v3_final.yaml
CLASSES = {
    0: "apple", 1: "banana", 2: "orange", 3: "mango", 
    4: "pineapple", 5: "watermelon", 6: "grapes", 7: "pomegranate"
}

# Distinguishable colors for bounding boxes
COLORS = {
    0: (0, 200, 0),      # apple: green
    1: (0, 230, 255),    # banana: yellow
    2: (0, 140, 255),    # orange: orange
    3: (0, 180, 255),    # mango: light orange
    4: (30, 200, 220),   # pineapple: gold
    5: (80, 80, 220),    # watermelon: red/pink
    6: (180, 80, 180),   # grapes: purple
    7: (60, 60, 200),    # pomegranate: deep red
}

def annotate(frame, results):
    """Draws bounding boxes and labels on the image"""
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cid = int(box.cls[0])
            name = CLASSES.get(cid, r.names.get(cid, f"Class {cid}"))
            color = COLORS.get(cid, (200, 200, 200))
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background and text
            label = f"{name} {conf:.2f}"
            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - lh - bl - 4), (x1 + lw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - bl - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

def main():
    model_path = Path("models/best.pt")
    if not model_path.exists():
        print(f"[ERROR] Trained model not found at {model_path}. Please train a model first.")
        return
        
    val_images_dir = Path("dataset_v3_final/valid/images")
    if not val_images_dir.exists():
        print(f"[ERROR] Validation images directory not found at {val_images_dir}.")
        return

    output_dir = Path("runs/sample_predictions")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of validation images
    images = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png")) + list(val_images_dir.glob("*.jpeg"))
    if not images:
        print(f"[ERROR] No images found in {val_images_dir}.")
        return

    # Select 10 random images
    num_samples = min(10, len(images))
    sampled_images = random.sample(images, num_samples)

    print(f"[INFO] Loading model from {model_path}...")
    model = YOLO(str(model_path))

    print(f"[INFO] Running predictions on {num_samples} random validation images...")
    print("=" * 60)

    for i, img_path in enumerate(sampled_images):
        # Load image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARNING] Could not read image: {img_path.name}")
            continue

        # Run inference
        # Use conf=0.15 matching the webcam threshold to see how it performs
        results = model(frame, imgsz=640, conf=0.15, verbose=False)

        # Count detections
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cid = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = CLASSES.get(cid, r.names.get(cid, f"Class {cid}"))
                    detections.append(f"{name} ({conf:.2%})")

        # Annotate and save
        annotated = annotate(frame, results)
        dest_path = output_dir / img_path.name
        cv2.imwrite(str(dest_path), annotated)

        det_str = ", ".join(detections) if detections else "No objects detected"
        print(f"[{i+1}/{num_samples}] {img_path.name}: {det_str}")
        print(f"      Saved to: {dest_path}")

    print("=" * 60)
    print(f"[OK] Completed! All prediction samples saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
