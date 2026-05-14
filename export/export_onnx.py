"""
Fruit Detection - Export to ONNX

Usage:
    python export/export_onnx.py
    python export/export_onnx.py --model models/best.pt
"""

import argparse
import shutil
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

    print(f"Exporting {args.model} to ONNX...")
    model = YOLO(args.model)
    export_path = model.export(format="onnx", imgsz=args.imgsz, simplify=True)

    # Copy to models/ directory (if not already there)
    dest = Path("models") / "best.onnx"
    dest.parent.mkdir(exist_ok=True)
    if export_path and Path(export_path).exists():
        src = Path(export_path).resolve()
        dst = dest.resolve()
        if src != dst:
            shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / (1024 * 1024)
        print(f"\n[OK] Exported to {dest} ({size_mb:.1f} MB)")
    else:
        print("[FAIL] Export failed")


if __name__ == "__main__":
    main()
