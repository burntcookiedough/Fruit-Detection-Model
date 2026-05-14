"""
Fruit Detection Model - Training Script
Fine-tunes YOLOv8n on the merged fruit dataset.

Usage:
    python train.py
    python train.py --epochs 100 --batch 8 --imgsz 640
"""

import argparse
import shutil
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
    parser = argparse.ArgumentParser(description="Train YOLOv8n fruit detector")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Pretrained model to fine-tune (default: yolov8n.pt)")
    parser.add_argument("--data", type=str, default="data_v2.yaml",
                        help="Path to data.yaml (default: data_v2.yaml)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs (default: 80)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16, reduce to 8 if OOM)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: '0' for GPU, 'cpu' for CPU (default: 0)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Dataloader workers (default: 2)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--name", type=str, default="fruit_v2",
                        help="Run name (default: fruit_v2)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve data.yaml to absolute path so YOLO finds it correctly
    data_path = str(Path(args.data).resolve())

    print("=" * 60)
    print("  FRUIT DETECTION MODEL - TRAINING")
    print("=" * 60)
    print(f"  Model       : {args.model}")
    print(f"  Data        : {data_path}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Image size  : {args.imgsz}")
    print(f"  Batch size  : {args.batch}")
    print(f"  Device      : {args.device}")
    print(f"  Patience    : {args.patience}")
    print(f"  Run name    : {args.name}")
    print("=" * 60)

    # Load pretrained model
    if args.resume:
        # Resume from last checkpoint
        last_ckpt = Path("runs") / args.name / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"No checkpoint found at {last_ckpt}")
        model = YOLO(str(last_ckpt))
        print(f"\nResuming from {last_ckpt}")
    else:
        model = YOLO(args.model)
        print(f"\nLoaded pretrained weights: {args.model}")

    # Train
    project_dir = str(Path(__file__).resolve().parent / "runs")
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        cache=True,
        project=project_dir,
        name=args.name,
        exist_ok=True,
    )

    # Copy best weights to models/ for easy access
    best_pt = Path(project_dir) / args.name / "weights" / "best.pt"
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True)

    if best_pt.exists():
        dest = models_dir / "best.pt"
        shutil.copy2(best_pt, dest)
        print(f"\n[OK] Best weights copied to {dest}")
    else:
        print(f"\n[!] best.pt not found at {best_pt}")

    print("\n[OK] Training complete!")
    print(f"  Results saved to: {project_dir}/{args.name}/")
    print(f"  Best weights: {models_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
