"""
debug_analysis.py — Fruit Detection: Comprehensive Failure Analysis
====================================================================
Covers:
  Phase 1  – Visual failure analysis (FN / FP / low-confidence image dumps)
  Phase 3  – Confidence threshold sweep  (conf 0.1 → 0.5)
  Phase 6  – Object size distribution
  Phase 8  – Per-class metrics dashboard

Usage
-----
  python debug_analysis.py                      # full analysis, test split
  python debug_analysis.py --split val          # run on val split
  python debug_analysis.py --max_samples 200    # limit images processed
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CLASSES     = ["apple", "banana", "orange", "mango",
               "pineapple", "watermelon", "grapes", "pomegranate"]
MODEL_PATH  = Path("models/best.pt")
DATA_YAML   = Path("data_v3.yaml")
IOU_THRESH  = 0.45          # IoU to match pred → GT
THRESHOLDS  = [0.10, 0.20, 0.30, 0.40, 0.50]
PRIMARY_CONF = 0.25

DEBUG_DIR    = Path("debug")
FN_DIR       = DEBUG_DIR / "false_negatives"
FP_DIR       = DEBUG_DIR / "false_positives"
LC_DIR       = DEBUG_DIR / "low_confidence"
OK_DIR       = DEBUG_DIR / "correct_predictions"
REPORT_PATH  = DEBUG_DIR / "analysis_report.json"


# ─────────────────────────────────────────────
#  I/O HELPERS
# ─────────────────────────────────────────────

def load_data_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def read_gt_boxes(label_path: Path) -> list[tuple]:
    """Returns list of (cls_id, cx, cy, w, h) in YOLO normalised coords."""
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def yolo2xyxy(cx, cy, w, h, W, H):
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    return x1, y1, x2, y2


def iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def collect_images(data_cfg: dict, split: str, max_samples: int) -> list[Path]:
    root = Path(data_cfg["path"])
    split_key = "val" if split == "val" else split
    img_dir = root / data_cfg[split_key]
    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"})
    return imgs[:max_samples] if max_samples else imgs


def label_dir_for(img_path: Path) -> Path:
    return img_path.parent.parent / "labels" / (img_path.stem + ".txt")


# ─────────────────────────────────────────────
#  ANNOTATED CROP HELPER
# ─────────────────────────────────────────────

def save_annotated(img, gt_boxes, pred_boxes, pred_confs, pred_cls, out_path: Path, H, W):
    """Draw GT (green) and predictions (red) on image and save."""
    vis = img.copy()
    for (cls_id, cx, cy, bw, bh) in gt_boxes:
        x1,y1,x2,y2 = map(int, yolo2xyxy(cx, cy, bw, bh, W, H))
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,200,0), 2)
        cv2.putText(vis, f"GT:{CLASSES[cls_id]}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,0), 1)
    for (x1,y1,x2,y2), conf, cls_id in zip(pred_boxes, pred_confs, pred_cls):
        x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,220), 2)
        cv2.putText(vis, f"{CLASSES[int(cls_id)]}:{conf:.2f}", (x1, y2+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,220), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


# ─────────────────────────────────────────────
#  THRESHOLD SWEEP
# ─────────────────────────────────────────────

def evaluate_at_threshold(results_cache: list[dict], conf_thresh: float) -> dict:
    """results_cache: precomputed per-image dicts with all raw preds."""
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)

    for item in results_cache:
        gt  = item["gt"]            # list of (cls_id, cx,cy,w,h)
        preds = [(x1,y1,x2,y2,c,cl)
                 for x1,y1,x2,y2,c,cl in item["preds"] if c >= conf_thresh]
        H, W = item["H"], item["W"]

        gt_xyxy = [(cls_id, *yolo2xyxy(cx,cy,bw,bh,W,H))
                   for cls_id,cx,cy,bw,bh in gt]
        pred_xyxy = [(int(cl), x1,y1,x2,y2) for x1,y1,x2,y2,c,cl in preds]

        matched_gt  = set()
        matched_pred= set()

        for pi,(pcls,px1,py1,px2,py2) in enumerate(pred_xyxy):
            best_iou = 0; best_gi = -1
            for gi,(gcls,gx1,gy1,gx2,gy2) in enumerate(gt_xyxy):
                if gi in matched_gt: continue
                if gcls != pcls: continue
                v = iou((px1,py1,px2,py2),(gx1,gy1,gx2,gy2))
                if v > best_iou:
                    best_iou = v; best_gi = gi
            if best_gi >= 0 and best_iou >= IOU_THRESH:
                tp[pcls] += 1
                matched_gt.add(best_gi); matched_pred.add(pi)
            else:
                fp[pcls] += 1

        for gi,(gcls,*_) in enumerate(gt_xyxy):
            if gi not in matched_gt:
                fn[gcls] += 1

    out = {}
    for i, name in enumerate(CLASSES):
        t = tp[i]; f = fp[i]; n = fn[i]
        prec = t/(t+f) if (t+f) else 0.0
        rec  = t/(t+n) if (t+n) else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        out[name] = {"TP":t,"FP":f,"FN":n,
                     "Precision":round(prec,4),
                     "Recall":round(rec,4),
                     "F1":round(f1,4)}
    return out


# ─────────────────────────────────────────────
#  OBJECT SIZE ANALYSIS
# ─────────────────────────────────────────────

def analyze_sizes(results_cache: list[dict]) -> dict:
    areas = defaultdict(list)
    for item in results_cache:
        H, W = item["H"], item["W"]
        for cls_id, cx, cy, bw, bh in item["gt"]:
            area_pct = bw * bh * 100   # percentage of image area
            areas[CLASSES[cls_id]].append(round(area_pct, 4))
    summary = {}
    for name, vals in areas.items():
        arr = np.array(vals)
        summary[name] = {
            "count": len(arr),
            "mean_area_%": round(float(arr.mean()), 3) if len(arr) else 0,
            "median_area_%": round(float(np.median(arr)), 3) if len(arr) else 0,
            "tiny_<2%": int((arr < 2).sum()),
            "small_2-5%": int(((arr >= 2) & (arr < 5)).sum()),
            "medium_5-15%": int(((arr >= 5) & (arr < 15)).sum()),
            "large_>15%": int((arr >= 15).sum()),
        }
    return summary


# ─────────────────────────────────────────────
#  SIZE-BUCKETED RECALL  (Priority 5)
# ─────────────────────────────────────────────

# Buckets defined by GT box area (% of image)
SIZE_BUCKETS = [
    ("tiny   <2%",   0.0,  2.0),
    ("small  2-5%",  2.0,  5.0),
    ("medium 5-15%", 5.0, 15.0),
    ("large  >15%", 15.0, 101.),
]


def size_bucketed_recall(results_cache: list[dict], conf: float) -> dict:
    """
    For each size bucket × class, compute:
      TP, FN, Recall
    This proves whether tiny objects drive the recall collapse.
    """
    # bucket → class → {tp, fn}
    stats = {label: {name: {"tp": 0, "fn": 0}
                     for name in CLASSES}
             for label, _, _ in SIZE_BUCKETS}

    for item in results_cache:
        gt    = item["gt"]
        preds = [(x1,y1,x2,y2,c,cl)
                 for x1,y1,x2,y2,c,cl in item["preds"] if c >= conf]
        H, W  = item["H"], item["W"]

        gt_xyxy   = [(cls_id, *yolo2xyxy(cx,cy,bw,bh,W,H), bw*bh*100)
                     for cls_id,cx,cy,bw,bh in gt]
        pred_xyxy = [(int(cl), x1,y1,x2,y2) for x1,y1,x2,y2,c,cl in preds]

        matched_gt = set()
        for pi,(pcls,px1,py1,px2,py2) in enumerate(pred_xyxy):
            best_iou = 0; best_gi = -1
            for gi,(gcls,gx1,gy1,gx2,gy2,_) in enumerate(gt_xyxy):
                if gi in matched_gt or gcls != pcls: continue
                v = iou((px1,py1,px2,py2),(gx1,gy1,gx2,gy2))
                if v > best_iou: best_iou=v; best_gi=gi
            if best_gi >= 0 and best_iou >= IOU_THRESH:
                matched_gt.add(best_gi)

        for gi,(gcls,gx1,gy1,gx2,gy2,area_pct) in enumerate(gt_xyxy):
            name = CLASSES[gcls] if gcls < len(CLASSES) else f"cls_{gcls}"
            for label, lo, hi in SIZE_BUCKETS:
                if lo <= area_pct < hi:
                    if gi in matched_gt:
                        stats[label][name]["tp"] += 1
                    else:
                        stats[label][name]["fn"] += 1
                    break

    return stats


def print_bucketed_recall(stats: dict):
    print("\n" + "="*80)
    print("  SIZE-BUCKETED RECALL  (confirms tiny-object failure)")
    print("="*80)
    header = f"  {'Bucket':<14}" + "".join(f" {n[:6]:>8}" for n in CLASSES)
    print(header)
    print("  " + "-"*76)
    for label, _, _ in SIZE_BUCKETS:
        bucket_stats = stats[label]
        row = f"  {label:<14}"
        for name in CLASSES:
            tp = bucket_stats[name]["tp"]
            fn = bucket_stats[name]["fn"]
            rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            if tp + fn == 0:
                row += f" {'  -':>8}"
            elif rec < 0.4:
                row += f" {rec:>7.2f}!"
            else:
                row += f" {rec:>8.2f}"
        print(row)
    print("  (! = recall < 0.40, - = no GT boxes in this bucket)")


# ─────────────────────────────────────────────
#  VISUAL DUMP  (Phase 1)
# ─────────────────────────────────────────────

def dump_visual_failures(results_cache: list[dict], conf: float, max_per_class: int = 60):
    """Save annotated images for FN, FP, low-conf, correct per class."""
    # reset dirs
    for d in [FN_DIR, FP_DIR, LC_DIR, OK_DIR]:
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True)

    fn_counts  = defaultdict(int)
    fp_counts  = defaultdict(int)
    ok_counts  = defaultdict(int)
    lc_counts  = defaultdict(int)

    for item in results_cache:
        img_path = Path(item["img_path"])
        gt       = item["gt"]
        all_preds = item["preds"]     # (x1,y1,x2,y2,conf,cls)
        H, W     = item["H"], item["W"]

        # Split preds by threshold
        preds_above = [(x1,y1,x2,y2,c,cl) for x1,y1,x2,y2,c,cl in all_preds if c >= conf]
        preds_low   = [(x1,y1,x2,y2,c,cl) for x1,y1,x2,y2,c,cl in all_preds if 0.10 <= c < conf]

        gt_xyxy = [(cls_id, *yolo2xyxy(cx,cy,bw,bh,W,H)) for cls_id,cx,cy,bw,bh in gt]
        pred_above_xyxy = [(int(cl), x1,y1,x2,y2) for x1,y1,x2,y2,c,cl in preds_above]
        pred_low_xyxy   = [(int(cl), x1,y1,x2,y2) for x1,y1,x2,y2,c,cl in preds_low]

        matched_gt = set()
        matched_pred = set()

        for pi,(pcls,px1,py1,px2,py2) in enumerate(pred_above_xyxy):
            best_iou = 0; best_gi = -1
            for gi,(gcls,gx1,gy1,gx2,gy2) in enumerate(gt_xyxy):
                if gi in matched_gt or gcls != pcls: continue
                v = iou((px1,py1,px2,py2),(gx1,gy1,gx2,gy2))
                if v > best_iou: best_iou=v; best_gi=gi
            if best_gi >= 0 and best_iou >= IOU_THRESH:
                matched_gt.add(best_gi); matched_pred.add(pi)

        img_loaded = None

        def _img():
            nonlocal img_loaded
            if img_loaded is None:
                img_loaded = cv2.imread(str(img_path))
            return img_loaded

        pred_boxes  = [(x1,y1,x2,y2) for x1,y1,x2,y2,_,_ in preds_above]
        pred_confs  = [c for _,_,_,_,c,_ in preds_above]
        pred_cls    = [cl for _,_,_,_,_,cl in preds_above]

        # --- False Negatives (missed GT boxes) ---
        for gi,(gcls,gx1,gy1,gx2,gy2) in enumerate(gt_xyxy):
            if gi in matched_gt: continue
            name = CLASSES[gcls]
            if fn_counts[name] >= max_per_class: continue
            # Check if it's a low-conf detection (between 0.1 and conf)
            rescued = any(int(cl)==gcls and
                          iou((x1,y1,x2,y2),(gx1,gy1,gx2,gy2)) >= IOU_THRESH
                          for cl,x1,y1,x2,y2 in pred_low_xyxy)
            subdir = LC_DIR / name if rescued else FN_DIR / name
            counter = lc_counts if rescued else fn_counts
            out = subdir / f"{img_path.stem}_fn{gi}.jpg"
            save_annotated(_img(), gt, pred_boxes, pred_confs, pred_cls, out, H, W)
            counter[name] += 1

        # --- False Positives (pred not matched) ---
        for pi,(pcls,px1,py1,px2,py2) in enumerate(pred_above_xyxy):
            if pi in matched_pred: continue
            name = CLASSES[pcls]
            if fp_counts[name] >= max_per_class: continue
            out = FP_DIR / name / f"{img_path.stem}_fp{pi}.jpg"
            save_annotated(_img(), gt, pred_boxes, pred_confs, pred_cls, out, H, W)
            fp_counts[name] += 1

        # --- Correct predictions (tp sample) ---
        if matched_gt and ok_counts[CLASSES[gt[0][0]]] < 20:
            gcls = gt[0][0]; name = CLASSES[gcls]
            out = OK_DIR / name / f"{img_path.stem}.jpg"
            save_annotated(_img(), gt, pred_boxes, pred_confs, pred_cls, out, H, W)
            ok_counts[name] += 1

    print(f"\n  Visual dumps written to: {DEBUG_DIR}/")
    for name in CLASSES:
        print(f"    {name:<14s}  FN={fn_counts[name]}  FP={fp_counts[name]}"
              f"  LowConf={lc_counts[name]}  OK={ok_counts[name]}")


# ─────────────────────────────────────────────
#  PRETTY PRINT HELPERS
# ─────────────────────────────────────────────

def print_threshold_table(sweep: dict):
    print("\n" + "="*80)
    print("  CONFIDENCE THRESHOLD SWEEP")
    print("="*80)
    for thresh, metrics in sweep.items():
        print(f"\n  conf={thresh}")
        print(f"    {'Class':<14} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
        print("    " + "-"*52)
        for cls, m in metrics.items():
            print(f"    {cls:<14} {m['Precision']:>6.3f} {m['Recall']:>6.3f} "
                  f"{m['F1']:>6.3f} {m['TP']:>5} {m['FP']:>5} {m['FN']:>5}")


def print_size_table(sizes: dict):
    print("\n" + "="*80)
    print("  OBJECT SIZE DISTRIBUTION  (% of image area)")
    print("="*80)
    print(f"  {'Class':<14} {'Count':>6} {'Mean%':>7} {'Median%':>8} "
          f"{'Tiny<2%':>8} {'Small':>7} {'Med':>5} {'Large>15%':>10}")
    print("  " + "-"*70)
    for name, s in sizes.items():
        print(f"  {name:<14} {s['count']:>6} {s['mean_area_%']:>7.2f} {s['median_area_%']:>8.2f} "
              f"{s['tiny_<2%']:>8} {s['small_2-5%']:>7} {s['medium_5-15%']:>5} {s['large_>15%']:>10}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fruit detection failure analysis")
    p.add_argument("--model",  default=str(MODEL_PATH))
    p.add_argument("--data",   default=str(DATA_YAML))
    p.add_argument("--split",  default="test", choices=["val","test"])
    p.add_argument("--max_samples", type=int, default=0,
                   help="Limit images (0=all). Use 200 for a quick run.")
    p.add_argument("--no_dump", action="store_true",
                   help="Skip visual image dumps (faster)")
    return p.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("  FRUIT DETECTION — FAILURE ANALYSIS")
    print("="*60)

    model = YOLO(args.model)
    data_cfg = load_data_yaml(args.data)

    imgs = collect_images(data_cfg, args.split, args.max_samples)
    print(f"  Images to process: {len(imgs)}")

    # ── Step 1: Run inference once at lowest threshold, cache all preds ──
    print("\n  Running inference (conf=0.01 to cache all predictions)...")
    results_cache = []

    for img_path in imgs:
        res = model.predict(str(img_path), conf=0.01, iou=0.45, verbose=False)[0]
        H, W = res.orig_shape[:2]
        gt = read_gt_boxes(label_dir_for(img_path))
        boxes = res.boxes
        preds = []
        if boxes is not None and len(boxes):
            xyxy  = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss  = boxes.cls.cpu().numpy()
            for (x1,y1,x2,y2), c, cl in zip(xyxy, confs, clss):
                preds.append((float(x1),float(y1),float(x2),float(y2),float(c),int(cl)))

        results_cache.append({
            "img_path": str(img_path),
            "gt": gt,
            "preds": preds,
            "H": H, "W": W,
        })

    # ── Phase 3: Threshold sweep ──────────────────────────────────────
    print("\n  Running threshold sweep...")
    sweep = {}
    for thresh in THRESHOLDS:
        sweep[thresh] = evaluate_at_threshold(results_cache, thresh)
    print_threshold_table(sweep)

    # ── Phase 6: Object size analysis ────────────────────────────────
    print("\n  Analysing object sizes...")
    sizes = analyze_sizes(results_cache)
    print_size_table(sizes)

    # ── Priority 5: Size-bucketed recall ─────────────────────────────
    bucket_stats = size_bucketed_recall(results_cache, PRIMARY_CONF)
    print_bucketed_recall(bucket_stats)

    # ── Phase 8: Per-class dashboard @ primary conf ──────────────────
    primary_metrics = sweep.get(PRIMARY_CONF) or evaluate_at_threshold(results_cache, PRIMARY_CONF)
    print("\n" + "="*60)
    print(f"  PER-CLASS METRICS DASHBOARD  (conf={PRIMARY_CONF})")
    print("="*60)
    print(f"  {'Class':<14} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'FN Rate':>8}")
    print("  " + "-"*48)
    for cls, m in primary_metrics.items():
        fn_rate = m["FN"] / (m["TP"] + m["FN"]) if (m["TP"]+m["FN"]) > 0 else 0
        flag = "  ← LOW RECALL" if m["Recall"] < 0.55 else ""
        print(f"  {cls:<14} {m['Precision']:>6.3f} {m['Recall']:>6.3f} "
              f"{m['F1']:>6.3f}  {fn_rate:>8.3f}{flag}")

    # ── Phase 1: Visual dump ──────────────────────────────────────────
    if not args.no_dump:
        print("\n  Saving visual failure images...")
        dump_visual_failures(results_cache, conf=PRIMARY_CONF, max_per_class=60)

    # ── Save JSON report ─────────────────────────────────────────────
    DEBUG_DIR.mkdir(exist_ok=True)
    report = {
        "model": args.model,
        "split": args.split,
        "images_evaluated": len(imgs),
        "threshold_sweep": {str(k): v for k,v in sweep.items()},
        "size_analysis": sizes,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  JSON report saved → {REPORT_PATH}")
    print("\n  [OK] Analysis complete.")


if __name__ == "__main__":
    main()
