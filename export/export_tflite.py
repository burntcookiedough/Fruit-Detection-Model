"""
Fruit Detection - Export to TFLite

Usage:
    python export/export_tflite.py
    python export/export_tflite.py --model models/best.pt

Note: Requires tensorflow. Install with:
    pip install tensorflow>=2.13.0
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}"); sys.exit(1)

    print(f"Exporting {args.model} to TFLite...")
    model = YOLO(args.model)
    export_path = model.export(format="tflite", imgsz=args.imgsz)

    if export_path:
        print(f"\n[OK] Exported to {export_path}")
    else:
        print("[FAIL] Export failed")


if __name__ == "__main__":
    main()
