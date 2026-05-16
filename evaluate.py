"""
evaluate.py — Fruit Detection Model: Evaluation entry point.

Runs the trained model against a dataset split and prints per-class metrics.
All defaults are inherited from config.py.

Usage
-----
Evaluate on the test split (default):
    python evaluate.py

Evaluate on the validation split:
    python evaluate.py --split val

Evaluate a specific checkpoint:
    python evaluate.py --model runs/fruit_v3/weights/best.pt
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

import config


def resolve_device(override: str | None) -> str:
    """Return the best available device, respecting an explicit override."""
    if override is not None:
        return override
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained fruit detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", type=str, default=str(config.MODELS_DIR / "best.pt"),
        help="Path to trained weights (default: models/best.pt)",
    )
    parser.add_argument(
        "--data", type=str, default=str(config.DATA_YAML),
        help="Path to data.yaml (default: data_v3.yaml)",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=config.IMGSZ,
        help=f"Input image size (default: {config.IMGSZ})",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: '0' for GPU, 'cpu'. Auto-detects if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at: {model_path}\n"
            "Run train.py first, or pass --model <path>."
        )

    data_path = str(Path(args.data).resolve())

    print("=" * 60)
    print("  FRUIT DETECTION MODEL — EVALUATION")
    print("=" * 60)
    print(f"  Model  : {model_path}")
    print(f"  Data   : {data_path}")
    print(f"  Split  : {args.split}")
    print(f"  Conf   : {args.conf}")
    print(f"  Device : {device}")
    print("=" * 60)

    model = YOLO(str(model_path))

    results = model.val(
        data=data_path,
        split=args.split,
        imgsz=args.imgsz,
        device=device,
        conf=args.conf,
    )

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Precision  : {results.box.mp:.4f}")
    print(f"  Recall     : {results.box.mr:.4f}")
    print(f"  mAP@50     : {results.box.map50:.4f}")
    print(f"  mAP@50-95  : {results.box.map:.4f}")

    # Per-class breakdown
    print("\n  Per-class mAP@50:")
    print("  " + "-" * 32)
    for i, ap50 in enumerate(results.box.ap50):
        name = results.names.get(i, f"class_{i}")
        bar = "█" * int(ap50 * 20)
        print(f"    {name:<14s} {ap50:.4f}  {bar}")
    print("  " + "-" * 32)

    print("\n  [OK] Evaluation complete.")
    print("  Confusion matrix and plots saved by ultralytics to runs/")


if __name__ == "__main__":
    main()
