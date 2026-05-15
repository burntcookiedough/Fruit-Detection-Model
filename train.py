"""
Fruit Detection Model - Training Script (v3)
Fine-tunes YOLOv8 on the merged/balanced fruit dataset.

Usage:
    # Recommended: YOLOv8m + balanced dataset + webcam augmentations
    python train.py --data data_v3.yaml --name fruit_v3 --epochs 120 --augment

    # On Colab / Kaggle (already has GPU, uses 4 workers)
    python train.py --data /path/to/data_v3.yaml --name fruit_v3 --epochs 120 --augment --batch 16

    # Resume a run
    python train.py --name fruit_v3 --resume
"""

import os
import platform
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


def auto_device():
    """Pick the best available device automatically."""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        print(f"  GPU detected: {gpu}")
        return "0"
    print("  No GPU detected — using CPU (training will be slow)")
    return "cpu"


def auto_workers():
    """Use 4 workers on Linux/Colab, 0 on Windows to prevent Paging File crashes."""
    if platform.system() == "Windows":
        return 0
    return min(4, os.cpu_count() or 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 fruit detector")
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                        help="Pretrained model to fine-tune (default: yolov8m.pt)")
    parser.add_argument("--data", type=str, default="data_v3.yaml",
                        help="Path to data.yaml (default: data_v3.yaml)")
    parser.add_argument("--epochs", type=int, default=120,
                        help="Number of training epochs (default: 120)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size (-1=auto-detect max that fits in VRAM, default: -1)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override: '0' for GPU, 'cpu'. Auto-detects if not set.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Dataloader workers. Auto: 4 on Linux, 2 on Windows.")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience (default: 25)")
    parser.add_argument("--name", type=str, default="fruit_v3",
                        help="Run name (default: fruit_v3)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--augment", action="store_true",
                        help="Enable aggressive webcam-style augmentations (recommended for v3)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-detect device and workers if not set
    device = args.device if args.device is not None else auto_device()
    workers = args.workers if args.workers is not None else auto_workers()

    # Resolve data.yaml to absolute path so YOLO finds it correctly
    # (critical: on Colab the cwd may differ from where the yaml is)
    data_path = str(Path(args.data).resolve())
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"data.yaml not found at: {data_path}\n"
            f"Run balance_dataset.py first to generate it."
        )

    print("=" * 60)
    print("  FRUIT DETECTION MODEL - TRAINING")
    print("=" * 60)
    print(f"  Model       : {args.model}")
    print(f"  Data        : {data_path}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Image size  : {args.imgsz}")
    print(f"  Batch size  : {args.batch}")
    print(f"  Device      : {device}")
    print(f"  Workers     : {workers}")
    print(f"  Patience    : {args.patience}")
    print(f"  Run name    : {args.name}")
    print(f"  Augment     : {'YES (webcam-optimised)' if args.augment else 'NO (default)'}")
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

    # -------------------------------------------------------
    # Build training kwargs
    # -------------------------------------------------------
    train_kwargs = dict(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=workers,
        patience=args.patience,
        cache="disk",     # Use fast local SSD cache instead of RAM (prevents 16GB RAM OOM)
        project=str(Path(__file__).resolve().parent / "runs"),
        name=args.name,
        exist_ok=True,
        # --- Quality improvements ---
        label_smoothing=0.1,   # prevents overconfidence, improves generalization
        cos_lr=True,           # cosine LR decay: better convergence than step decay
        close_mosaic=10,       # disable mosaic for last 10 epochs for stable fine-tuning
        val=True,              # always validate
        save_period=10,        # checkpoint every 10 epochs
    )

    if args.augment:
        # ---- Webcam robustness augmentations ----
        # These simulate noisy real-world / webcam conditions:
        #   hsv_*     : colour jitter handles dark/bright/yellow-lit rooms
        #   degrees   : mild rotation (fruit held at angle)
        #   translate : partial in-frame fruit
        #   scale     : varying distance to fruit
        #   blur_limit: webcam lens blur / motion blur
        #   erasing   : partial occlusion (hand covering fruit)
        #   mosaic    : trains on multi-object scenes (bowl of fruits)
        #   mixup     : improves generalisation on minority classes
        print("\n  Augmentation: webcam-optimised profile active")
        train_kwargs.update(dict(
            hsv_h=0.020,   # hue shift: handles colour casts from room lighting
            hsv_s=0.80,    # saturation: over/under-saturated webcam feeds
            hsv_v=0.50,    # brightness: dark rooms, backlit windows
            degrees=12,    # rotation: tilted fruit / tilted camera
            translate=0.12,
            scale=0.60,    # zoom in/out: different distances from camera
            shear=4.0,     # slight perspective distortion
            perspective=0.0003,
            flipud=0.0,    # fruit is never upside-down in real life
            fliplr=0.5,
            mosaic=1.0,    # mosaic always on
            mixup=0.15,    # small mixup helps minority classes
            erasing=0.40,  # 40% chance of random erase (simulates occlusion)
            # blur_limit removed -- not a standard ultralytics kwarg (handled via augment pipeline)
        ))

    # Train
    project_dir = str(Path(__file__).resolve().parent / "runs")
    results = model.train(**train_kwargs)

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
