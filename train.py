"""
train.py — Fruit Detection Model: Training entry point.

Fine-tunes a YOLOv8 model on the balanced fruit dataset (v3).
All hyperparameters and paths are defined in config.py.

Usage
-----
Fresh training (recommended):
    python train.py

Resume a paused run (restores epoch, optimizer, and LR scheduler state):
    python train.py --resume

Override any default from config.py:
    python train.py --epochs 150 --batch 16 --name my_run
"""

import argparse
import platform
import os
import shutil
from pathlib import Path

from ultralytics import YOLO
import torch

import config


# ---------------------------------------------------------------------------
# Device / worker helpers
# ---------------------------------------------------------------------------

def resolve_device(override: str | None) -> str:
    """
    Return the best available compute device.

    Priority: explicit override → CUDA GPU → CPU.
    MPS (Apple Silicon) is intentionally omitted — not relevant here.
    """
    if override is not None:
        return override

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU : {name}  ({vram:.1f} GB VRAM)")
        return "0"

    print("  WARNING: No CUDA GPU detected — falling back to CPU. Training will be very slow.")
    return "cpu"


def resolve_workers(override: int | None) -> int:
    """
    Return the number of DataLoader worker processes.

    Windows spawns new Python processes per worker; with a small dataset
    cache the overhead exceeds the benefit, so we use 0 (main process only).
    Linux / Colab can safely use multiple workers.
    """
    if override is not None:
        return override
    return 0 if platform.system() == "Windows" else min(4, os.cpu_count() or 2)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 fruit detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model", type=str, default=config.BASE_MODEL,
        help=f"Pretrained weights to fine-tune (default: {config.BASE_MODEL})",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to data.yaml. Defaults to data_v3_clean.yaml when --clean, else data_v3.yaml.",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS,
        help=f"Training epochs (default: {config.EPOCHS})",
    )
    parser.add_argument(
        "--imgsz", type=int, default=config.IMGSZ,
        help=f"Input image size (default: {config.IMGSZ})",
    )
    parser.add_argument(
        "--batch", type=int, default=config.BATCH,
        help="Batch size. -1 = AutoBatch (fills VRAM safely). (default: -1)",
    )
    parser.add_argument(
        "--patience", type=int, default=config.PATIENCE,
        help=f"Early-stop patience in epochs (default: {config.PATIENCE})",
    )
    parser.add_argument(
        "--name", type=str, default=config.RUN_NAME,
        help=f"Run name under runs/ (default: {config.RUN_NAME})",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override: '0' for GPU, 'cpu'. Auto-detects if omitted.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="DataLoader workers. Auto: 0 on Windows, 4 on Linux.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the last checkpoint (restores full training state).",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help=(
            "Use clean dataset + reduced augmentation profile for the post-fix "
            "debugging retrain. Switches data to data_v3_clean.yaml and augmentations "
            "to CLEAN_AUGMENT_KWARGS (lower mosaic, no mixup, gentler scale). "
            "Run filter_dataset.py first."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device  = resolve_device(args.device)
    workers = resolve_workers(args.workers)

    # ------------------------------------------------------------------
    # Select augmentation profile and data yaml
    # ------------------------------------------------------------------
    if args.clean:
        aug_profile  = config.CLEAN_AUGMENT_KWARGS
        default_data = str(config.DATA_YAML_CLEAN)
        profile_name = "clean (post-fix debugging)"
    else:
        aug_profile  = config.AUGMENT_KWARGS
        default_data = str(config.DATA_YAML)
        profile_name = "full webcam-optimised"

    data_yaml = args.data if args.data is not None else default_data

    # Resolve data.yaml to an absolute path.
    # Critical on Colab / Kaggle where the working directory differs.
    data_path = str(Path(data_yaml).resolve())
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"data.yaml not found at: {data_path}\n"
            "Run prepare_dataset_v3.py and balance_dataset.py first."
        )

    print("=" * 60)
    print("  FRUIT DETECTION MODEL — TRAINING")
    print("=" * 60)
    print(f"  Model   : {args.model}")
    print(f"  Data    : {data_path}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  ImgSz   : {args.imgsz}")
    print(f"  Batch   : {'auto' if args.batch == -1 else args.batch}")
    print(f"  Device  : {device}")
    print(f"  Workers : {workers}")
    print(f"  Patience: {args.patience}")
    print(f"  Run     : {args.name}")
    print(f"  Resume  : {args.resume}")
    print(f"  Profile : {profile_name}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Resume path — restores full training state (epoch, optimizer, LR).
    # Note: when resume=True, ultralytics reads all hyperparams from the
    # checkpoint and ignores kwargs passed to model.train().  We therefore
    # call model.train(resume=True) with no other arguments.
    # ------------------------------------------------------------------
    if args.resume:
        last_ckpt = config.RUNS_DIR / args.name / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(
                f"No checkpoint found at: {last_ckpt}\n"
                "Start a fresh run without --resume, or check the run name."
            )
        print(f"\n  Resuming from: {last_ckpt}")
        model = YOLO(str(last_ckpt))

        # FIX: Bypass Ultralytics YOLOv8 disk space check bug during resume
        # It doesn't account for existing cache files, so we artificially report 1TB free space.
        import shutil
        from collections import namedtuple
        original_disk_usage = shutil.disk_usage
        def mock_disk_usage(path):
            usage = original_disk_usage(path)
            return namedtuple('usage', 'total used free')(usage.total, usage.used, 1000 * 1024**3)
        shutil.disk_usage = mock_disk_usage

        results = model.train(resume=True)
        
        # Restore original function just in case
        shutil.disk_usage = original_disk_usage

    # ------------------------------------------------------------------
    # Fresh training path
    # ------------------------------------------------------------------
    else:
        model = YOLO(args.model)
        print(f"\n  Loaded pretrained weights: {args.model}")

        train_kwargs = {
            "data":         data_path,
            "epochs":       args.epochs,
            "imgsz":        args.imgsz,
            "batch":        args.batch,
            "device":       device,
            "workers":      workers,
            "patience":     args.patience,
            "project":      str(config.RUNS_DIR),
            "name":         args.name,
            "exist_ok":     False,   # Prevent silently overwriting a finished run.
                                     # Use --resume to continue, or change --name.
            "save_period":  config.SAVE_PERIOD,
            **config.TRAIN_QUALITY_KWARGS,
            **aug_profile,           # AUGMENT_KWARGS or CLEAN_AUGMENT_KWARGS
        }

        print(f"\n  Augmentation profile: {profile_name} (see config.py)")
        results = model.train(**train_kwargs)

    # ------------------------------------------------------------------
    # Post-training: copy best weights to models/ for easy access
    # ------------------------------------------------------------------
    best_pt = config.RUNS_DIR / args.name / "weights" / "best.pt"
    config.MODELS_DIR.mkdir(exist_ok=True)

    if best_pt.exists():
        dest = config.MODELS_DIR / "best.pt"
        shutil.copy2(best_pt, dest)
        print(f"\n  [OK] Best weights saved to: {dest}")
    else:
        print(f"\n  [!] best.pt not found at {best_pt} — check training logs.")

    print(f"\n  [OK] Training complete.")
    print(f"       Results : {config.RUNS_DIR / args.name}/")
    print(f"       Weights : {config.MODELS_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
