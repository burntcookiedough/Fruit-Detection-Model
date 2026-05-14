"""
Fruit Detection Model - Evaluation Script
Runs the trained model on the held-out test set and prints metrics.

Usage:
    python evaluate.py
    python evaluate.py --model models/best.pt --split test
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

# --- PyTorch 2.6 Compatibility Workaround ---
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fruit detection model")
    parser.add_argument("--model", type=str, default="models/best.pt",
                        help="Path to trained model (default: models/best.pt)")
    parser.add_argument("--data", type=str, default="data.yaml",
                        help="Path to data.yaml (default: data.yaml)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device (default: 0)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run train.py first, or specify --model path."
        )

    data_path = str(Path(args.data).resolve())

    print("=" * 60)
    print("  FRUIT DETECTION MODEL - EVALUATION")
    print("=" * 60)
    print(f"  Model   : {model_path}")
    print(f"  Data    : {data_path}")
    print(f"  Split   : {args.split}")
    print(f"  Conf    : {args.conf}")
    print("=" * 60)

    # Load trained model
    model = YOLO(str(model_path))

    # Run validation on the specified split
    results = model.val(
        data=data_path,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Precision     : {results.box.mp:.4f}")
    print(f"  Recall        : {results.box.mr:.4f}")
    print(f"  mAP@50        : {results.box.map50:.4f}")
    print(f"  mAP@50-95     : {results.box.map:.4f}")
    print("=" * 60)

    # Per-class breakdown
    names = results.names
    print("\n  Per-class mAP@50:")
    print("  " + "-" * 35)
    for i, ap50 in enumerate(results.box.ap50):
        class_name = names.get(i, f"class_{i}")
        print(f"    {class_name:<15s} : {ap50:.4f}")
    print("  " + "-" * 35)

    print("\n[OK] Evaluation complete!")
    print("  Confusion matrix and plots saved by ultralytics to runs/")


if __name__ == "__main__":
    main()
