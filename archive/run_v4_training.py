"""
run_v4_training.py
==================
Guarded local training orchestrator for Fruit Detection V4.

This script refuses to train until dataset quality gates are present. It can run
worker benchmarks, smoke runs, final training, and evaluation commands.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DATA_YAML = Path("data_v4_balanced.yaml")
QUALITY_REPORT = Path("quality_report_v4.txt")
BALANCE_REPORT = Path("balance_report_v4.txt")
MANIFEST = Path("dataset_v4_raw_manifest.csv")
DATASET = Path("dataset_v4_balanced")
HOLDOUT_YAML = Path("webcam_holdout/data_holdout.yaml")
SYNTHETIC_HOLDOUT_YAML = Path("synthetic_webcam_holdout/data_synthetic_holdout.yaml")
CLASSES = ["apple", "banana", "orange", "mango", "pineapple", "watermelon", "grapes", "pomegranate"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def require_file(path: Path, message: str) -> None:
    if not path.exists():
        raise SystemExit(f"[BLOCKED] {message}: {path}")


def read_label_classes(path: Path) -> set[int]:
    classes = set()
    if not path.exists():
        return classes
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    classes.add(int(parts[0]))
                except ValueError:
                    pass
    return classes


def require_labeled_holdout(
    yaml_path: Path,
    image_dir: Path,
    label_dir: Path,
    min_images_per_class: int,
    name: str,
) -> None:
    require_file(yaml_path, f"{name} YAML missing")
    require_file(image_dir, f"{name} images directory missing")
    require_file(label_dir, f"{name} labels directory missing")

    per_class_images = {i: 0 for i in range(len(CLASSES))}
    missing_labels = []
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            missing_labels.append(image.name)
            continue
        for cls_id in read_label_classes(label):
            if cls_id in per_class_images:
                per_class_images[cls_id] += 1

    failures = [
        f"{CLASSES[cls_id]}={count}"
        for cls_id, count in per_class_images.items()
        if count < min_images_per_class
    ]
    if missing_labels:
        raise SystemExit(f"[BLOCKED] {name} has missing labels, first examples: {missing_labels[:5]}")
    if failures:
        raise SystemExit(
            f"[BLOCKED] {name} is not ready. Need at least "
            f"{min_images_per_class} labeled images per class; current: {', '.join(failures)}"
        )


def require_real_holdout_ready() -> None:
    require_labeled_holdout(
        yaml_path=HOLDOUT_YAML,
        image_dir=Path("webcam_holdout/images"),
        label_dir=Path("webcam_holdout/labels"),
        min_images_per_class=25,
        name="Real webcam holdout",
    )


def require_synthetic_holdout_ready() -> None:
    require_labeled_holdout(
        yaml_path=SYNTHETIC_HOLDOUT_YAML,
        image_dir=Path("synthetic_webcam_holdout/images"),
        label_dir=Path("synthetic_webcam_holdout/labels"),
        min_images_per_class=20,
        name="Synthetic webcam holdout",
    )


def preflight(holdout_mode: str = "none") -> None:
    require_file(MANIFEST, "Raw manifest missing. Run build_v4_raw.py first")
    require_file(DATA_YAML, "Balanced data YAML missing. Run balance_v4_train.py first")
    require_file(QUALITY_REPORT, "Quality report missing. Run prepare_v4_quality.py first")
    require_file(BALANCE_REPORT, "Balance report missing. Run balance_v4_train.py first")
    require_file(DATASET / "train" / "images", "Quality train split missing")
    require_file(DATASET / "valid" / "images", "Quality valid split missing")
    require_file(DATASET / "test" / "images", "Quality test split missing")
    report = QUALITY_REPORT.read_text(encoding="utf-8", errors="ignore")
    if "READY FOR TRAINING" not in report:
        raise SystemExit("[BLOCKED] quality_report_v4.txt does not say READY FOR TRAINING.")
    balance_report = BALANCE_REPORT.read_text(encoding="utf-8", errors="ignore")
    if "[PASS] Training balance gate passed." not in balance_report:
        raise SystemExit("[BLOCKED] balance_report_v4.txt does not show a passed balance gate.")
    run([sys.executable, "check_split_leakage.py", "--dataset", str(DATASET)])
    if holdout_mode == "real":
        require_real_holdout_ready()
    elif holdout_mode == "synthetic":
        require_synthetic_holdout_ready()


def benchmark(py: str, workers: list[int], tag: str) -> None:
    preflight(holdout_mode="synthetic")
    rows = []
    for model, batch in [("yolov8n.pt", "16"), ("yolov8s.pt", "8")]:
        for worker in workers:
            name = f"fruit_v4_bench_{tag}_{Path(model).stem}_w{worker}"
            cmd = [
                py, "train.py", "--model", model, "--data", str(DATA_YAML), "--name", name,
                "--epochs", "3", "--batch", batch, "--patience", "3", "--workers", str(worker),
            ]
            try:
                run(cmd)
                status = "ok"
            except subprocess.CalledProcessError as exc:
                status = f"failed:{exc.returncode}"
            rows.append({"model": model, "batch": batch, "workers": worker, "run": name, "status": status})
    with Path(f"v4_worker_benchmark_{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "batch", "workers", "run", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote v4_worker_benchmark_{tag}.csv. Choose the fastest stable worker count from Ultralytics logs.")


def smoke(py: str, workers: int) -> None:
    preflight(holdout_mode="synthetic")
    run([py, "train.py", "--model", "yolov8n.pt", "--data", str(DATA_YAML), "--name", "fruit_v4_nano_smoke", "--epochs", "10", "--batch", "16", "--patience", "5", "--workers", str(workers)])
    run([py, "train.py", "--model", "yolov8s.pt", "--data", str(DATA_YAML), "--name", "fruit_v4_quality_smoke", "--epochs", "10", "--batch", "8", "--patience", "5", "--workers", str(workers)])


def final_train(py: str, workers: int, nano_batch: int) -> None:
    preflight(holdout_mode="real")
    run([py, "evaluate.py", "--model", "runs/fruit_v4_s_local/weights/best.pt", "--data", str(HOLDOUT_YAML), "--split", "test"])
    run([py, "train.py", "--model", "yolov8s.pt", "--data", str(DATA_YAML), "--name", "fruit_v4_quality", "--epochs", "120", "--batch", "8", "--patience", "25", "--workers", str(workers)])
    run([py, "train.py", "--model", "yolov8n.pt", "--data", str(DATA_YAML), "--name", "fruit_v4_nano", "--epochs", "120", "--batch", str(nano_batch), "--patience", "25", "--workers", str(workers)])


def evaluate(py: str) -> None:
    preflight(holdout_mode="real")
    for run_name in ["fruit_v4_quality", "fruit_v4_nano"]:
        weights = Path("runs") / run_name / "weights" / "best.pt"
        require_file(weights, f"Missing weights for {run_name}")
        run([py, "evaluate.py", "--model", str(weights), "--data", str(DATA_YAML), "--split", "test"])
        run([py, "evaluate.py", "--model", str(weights), "--data", str(HOLDOUT_YAML), "--split", "test"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Guarded V4 laptop training runner")
    parser.add_argument("--python", default=str(Path("venv") / "Scripts" / "python.exe"))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--nano-batch", type=int, default=16)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--benchmark-workers", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument("--tag", default=None, help="Run tag for benchmark names. Defaults to timestamp.")
    args = parser.parse_args()

    py = args.python
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.benchmark:
        benchmark(py, args.benchmark_workers, tag)
    elif args.smoke:
        smoke(py, args.workers)
    elif args.final:
        final_train(py, args.workers, args.nano_batch)
    elif args.evaluate:
        evaluate(py)
    else:
        preflight(holdout_mode="synthetic")
        print("[PASS] Training preflight gates passed. Choose --benchmark, --smoke, --final, or --evaluate.")


if __name__ == "__main__":
    main()
