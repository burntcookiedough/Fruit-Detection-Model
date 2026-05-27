"""Pre-flight checks — run before training to verify environment."""
import os, sys, shutil

def check_dataset():
    base = os.path.join(os.path.dirname(__file__), "..", "dataset_v4_balanced")
    splits = {"train": ("images", "labels"), "valid": ("images", "labels"), "test": ("images", "labels")}
    all_ok = True
    for split, (img_sub, lbl_sub) in splits.items():
        img_dir = os.path.join(base, split, img_sub)
        lbl_dir = os.path.join(base, split, lbl_sub)
        img_count = len([f for f in os.listdir(img_dir) if f.endswith((".jpg",".jpeg",".png"))]) if os.path.isdir(img_dir) else 0
        lbl_count = len([f for f in os.listdir(lbl_dir) if f.endswith(".txt")]) if os.path.isdir(lbl_dir) else 0
        status = "OK" if img_count > 0 and lbl_count > 0 else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {split:6s} | images={img_count:6d}  labels={lbl_count:6d}  [{status}]")
    result = "PASSED" if all_ok else "FAILED"
    print(f"\nDataset check: {result}")
    return all_ok

def check_disk():
    usage = shutil.disk_usage(os.path.dirname(__file__))
    free_gb = usage.free / 1e9
    print(f"Disk free: {free_gb:.1f} GB")
    print(f"Required:  ~5 GB (cache + checkpoints)")
    status = "OK" if free_gb > 5 else "WARNING: Low disk space!"
    print(f"Status:    {status}")
    return free_gb > 5

def check_gpu():
    import torch
    print(f"PyTorch:   {torch.__version__}")
    print(f"CUDA:      {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:       {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM:      {props.total_memory / 1e9:.1f} GB")
        print(f"CUDA ver:  {torch.version.cuda}")
        print(f"cuDNN ver: {torch.backends.cudnn.version()}")
    else:
        print("WARNING: No CUDA GPU — training will be extremely slow!")
    return torch.cuda.is_available()

def check_config():
    sys.path.insert(0, os.path.dirname(__file__))
    from custom_config import (IMG_SIZE, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, LR,
                                ANCHOR_SCALES, FM_SIZES, MATCHER_TYPE, NUM_WORKERS,
                                CLASS_NAMES, RUNS_DIR, WEIGHTS_DIR)
    print(f"IMG_SIZE:      {IMG_SIZE}")
    print(f"NUM_CLASSES:   {NUM_CLASSES}")
    print(f"BATCH_SIZE:    {BATCH_SIZE}")
    print(f"NUM_EPOCHS:    {NUM_EPOCHS}")
    print(f"LR:            {LR}")
    print(f"ANCHOR_SCALES: {ANCHOR_SCALES}")
    print(f"FM_SIZES:      {FM_SIZES}")
    print(f"MATCHER_TYPE:  {MATCHER_TYPE}")
    print(f"NUM_WORKERS:   {NUM_WORKERS}")
    print(f"CLASS_NAMES:   {CLASS_NAMES}")
    print(f"RUNS_DIR:      {RUNS_DIR}")
    print(f"WEIGHTS_DIR:   {WEIGHTS_DIR}")
    return True

if __name__ == "__main__":
    checks = [
        ("GPU / CUDA", check_gpu),
        ("Dataset", check_dataset),
        ("Config", check_config),
        ("Disk Space", check_disk),
    ]
    print("=" * 60)
    print("  PRE-FLIGHT CHECKS")
    print("=" * 60)
    results = {}
    for name, fn in checks:
        print(f"\n--- {name} ---")
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}")
    all_pass = all(results.values())
    verdict = "ALL CHECKS PASSED — ready to train!" if all_pass else "SOME CHECKS FAILED — fix issues above."
    print(f"\n  {verdict}")
    print("=" * 60)
