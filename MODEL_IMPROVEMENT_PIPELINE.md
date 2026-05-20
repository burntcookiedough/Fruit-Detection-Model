# Fruit Detector Improvement Pipeline

Goal: build the best fruit detector we can train on this laptop, then export a model that is realistic for embedded hardware.

Hardware assumption:
- Laptop GPU: RTX 3060 Laptop, 6 GB VRAM
- Stable training ceiling: nano/small YOLO models
- Avoid medium/large models for local training unless you accept very long runs and OOM risk

Primary rule: improve data first, then compare model families. A clean small model beats a larger model trained on noisy labels.

---

## 0. Current Baseline

Let the current run finish unless it clearly stalls.

Current serious baseline:

```powershell
$env:YOLO_CONFIG_DIR="F:\Fruit Detection Model\Ultralytics"
.\venv\Scripts\python.exe train.py --model yolov8s.pt --data data_v3_final.yaml --batch 8 --workers 4 --name fruit_v4_s_local --clean --epochs 80 --patience 15
```

After training:

```powershell
.\venv\Scripts\python.exe evaluate.py --model runs/fruit_v4_s_local/weights/best.pt --data data_v3_final.yaml --split test
```

Keep the best checkpoint even if the final epoch is worse:

```text
runs/fruit_v4_s_local/weights/best.pt
```

---

## 1. Build A Real Webcam Holdout Set

This is the most important step for a model that feels good in the real app.

Create a small private test set that is never used for training:

```text
webcam_holdout/
  images/
  labels/
```

Target size:
- Minimum useful: 200 labeled images
- Good: 500 labeled images
- Excellent: 1,000 labeled images

Capture conditions:
- fruit held in hand
- fruit on table
- mixed fruits in frame
- dim/yellow indoor lighting
- daylight/backlit scenes
- partial occlusion
- near and far distance
- cluttered background

Acceptance gate:

```text
Do not claim the model is good until it works on webcam_holdout.
```

---

## 2. Failure-Driven Data Loop

After every serious run:

1. Run evaluation/debug analysis.
2. Inspect false negatives and false positives.
3. Add real images for the failure modes.
4. Rebuild `dataset_v3_final`.
5. Train the next candidate.

Priority data gaps:

```text
1. Pomegranate: still the weakest class by real-data volume.
2. Watermelon: needs more varied shape/lighting/context.
3. Pineapple: needs more real-world angles and occlusions.
4. Small bananas/apples: improve only with better labels and real images.
```

Rebuild after adding data:

```powershell
.\venv\Scripts\python.exe prepare_pod.py
```

Dataset gate before training:

```text
No empty labels.
No invalid boxes.
No missing image/label pairs.
Train class counts should be roughly balanced.
Validation/test should remain real and not augmented.
```

---

## 3. Laptop Model Ladder

Run candidates in this order. Do not jump straight to every model; compare one variable at a time.

Hugging Face-style lesson:

```text
Do not search for one magic model.
Run a ladder of candidates on the same dataset, same validation split, same metrics, then choose the smallest model that hits the target.
```

### A. Embedded-Speed Candidate

Use this if the target device is weak: Raspberry Pi CPU, mobile CPU, browser CPU, or low-end NPU.

```powershell
$env:YOLO_CONFIG_DIR="F:\Fruit Detection Model\Ultralytics"
.\venv\Scripts\python.exe train.py --model yolo11n.pt --data data_v3_final.yaml --batch 16 --workers 4 --name fruit_v5_11n_edge --clean --epochs 100 --patience 20
```

Fallback if `yolo11n.pt` is unavailable:

```powershell
.\venv\Scripts\python.exe train.py --model yolov8n.pt --data data_v3_final.yaml --batch 16 --workers 4 --name fruit_v5_8n_edge --clean --epochs 100 --patience 20
```

### B. Best Laptop-Trainable Quality Candidate

Use this if embedded hardware is stronger: Jetson, Intel NPU, Coral-class accelerator, laptop inference, or server-assisted edge.

```powershell
$env:YOLO_CONFIG_DIR="F:\Fruit Detection Model\Ultralytics"
.\venv\Scripts\python.exe train.py --model yolo11s.pt --data data_v3_final.yaml --batch 8 --workers 4 --name fruit_v5_11s_quality --clean --epochs 120 --patience 25
```

Fallback if `yolo11s.pt` is unavailable:

```powershell
.\venv\Scripts\python.exe train.py --model yolov8s.pt --data data_v3_final.yaml --batch 8 --workers 4 --name fruit_v5_8s_quality --clean --epochs 120 --patience 25
```

### C. Hugging Face Transformer Detector Benchmark

Hugging Face's strongest object-detection workflows commonly compare modern COCO-pretrained detectors such as:

```text
ustc-community/dfine-small-coco    10.4M params
PekingU/rtdetr_v2_r18vd            20.2M params
PekingU/rtdetr_v2_r50vd            43.0M params
```

These are useful as quality references, especially D-FINE Small and RT-DETR v2 R18.

But for this project, they are **benchmark candidates**, not the first embedded deployment choice:

```text
YOLO export path is already simple: ONNX / TFLite / embedded runtimes.
HF Transformer detectors may need a separate COCO-format training/export/inference stack.
```

Use them only after the YOLO ladder has a strong baseline, or if YOLO plateaus below target quality.

If we add this path locally, first convert the dataset to COCO/HF `objects` format and run a small proof training run. Do not replace the YOLO pipeline blindly.

### D. Final Fine-Tune Pass

Only run this after picking the best candidate from A/B.

```powershell
.\venv\Scripts\python.exe train.py --model runs/fruit_v5_11s_quality/weights/best.pt --data data_v3_final.yaml --batch 8 --workers 4 --name fruit_v5_11s_finetune --clean --epochs 40 --patience 10
```

If the selected model is nano, replace the model path with that nano run.

---

## 4. When To Use `--clean`

Default to `--clean` for this project.

Why:
- lots of small fruit boxes
- elongated bananas
- pomegranate is data-limited
- full mosaic/mixup previously hurt recall stability

Only test full augmentation after a strong clean baseline:

```powershell
.\venv\Scripts\python.exe train.py --model yolo11s.pt --data data_v3_final.yaml --batch 8 --workers 4 --name fruit_v5_11s_fullaug --epochs 80 --patience 15
```

Keep the full-augmentation run only if it beats clean on the webcam holdout set, not just public validation.

---

## 5. Evaluation Matrix

Every candidate must be judged on the same tests.

Required:

```powershell
.\venv\Scripts\python.exe evaluate.py --model <weights.pt> --data data_v3_final.yaml --split test
```

Also run a webcam-holdout evaluation once the holdout YAML exists.

Track:

```text
Model name
Weights path
Training time
Best epoch
mAP50
mAP50-95
Precision
Recall
Per-class AP
Webcam holdout mAP50
Webcam false positives
Webcam false negatives
Export format
Embedded FPS
Embedded memory usage
```

Selection rule:

```text
Choose the smallest model that meets quality and latency targets.
```

For embedded deployment, a model that is 2% lower mAP but twice as fast may be the better product model.

Hugging Face-style experiment discipline:

```text
1. Validate dataset format before long training.
2. Save every run with a unique name.
3. Load the best checkpoint at the end, not the last checkpoint.
4. Track mAP/mAR or mAP50/mAP50-95, not just loss.
5. Keep only 2-3 best checkpoints to save disk.
6. Compare models on the same holdout split.
7. Push/persist final artifacts somewhere durable.
```

Local equivalent for this repo:

```text
dataset validation      -> prepare_pod.py + label audit/debug scripts
unique run names        -> fruit_v5_11s_quality, fruit_v5_11n_edge, etc.
best checkpoint         -> runs/<run>/weights/best.pt
metrics                 -> results.csv + evaluate.py
artifact persistence    -> models/ + exported ONNX/TFLite
```

---

## 6. Embedded Export Pipeline

Export the selected checkpoint.

ONNX:

```powershell
.\venv\Scripts\python.exe export\export_onnx.py --model runs/fruit_v5_11s_quality/weights/best.pt --imgsz 640
```

TFLite:

```powershell
.\venv\Scripts\python.exe export\export_tflite.py --model runs/fruit_v5_11s_quality/weights/best.pt --imgsz 640
```

For weaker embedded hardware, also test 512:

```powershell
.\venv\Scripts\python.exe export\export_onnx.py --model runs/fruit_v5_11n_edge/weights/best.pt --imgsz 512
```

Embedded comparison targets:

```text
High quality target: YOLO11s or YOLOv8s at 640
Balanced target: YOLO11n or YOLOv8n at 640
Fast target: YOLO11n or YOLOv8n at 512
```

Do not choose export size by accuracy alone. Measure latency on the real device.

---

## 7. Threshold Tuning

After choosing the model, tune confidence thresholds.

Expected behavior:

```text
Lower confidence => more recall, more false positives
Higher confidence => fewer false positives, more misses
```

For a user-facing detector, tune for the real app:

```text
If missing fruit is worse: lower confidence.
If wrong fruit labels are worse: raise confidence.
```

Classes may need different thresholds. Banana/pomegranate often need lower confidence than mango/pineapple.

---

## 8. Laptop Training Rules

Use these defaults:

```text
YOLO nano:  batch 16, workers 4
YOLO small: batch 8, workers 4
imgsz: 640
cache: disk
power plan: High performance
trainer priority: High
worker priority: AboveNormal
```

If training freezes or RAM pressure is high:

```text
Reduce workers from 4 to 2.
If still unstable, use workers 0.
```

If CUDA runs out of memory:

```text
Reduce batch first.
Do not reduce image size unless embedded speed is more important than small-fruit recall.
```

---

## 9. Definition Of “Very Good”

For this project, a very good embedded-ready model should meet most of these:

```text
Public test mAP50:      70%+
Public test mAP50-95:   55%+
Recall:                 60%+
Webcam holdout mAP50:   65%+
Stable pomegranate detections in real webcam scenes
Few wrong fruit labels on mixed-fruit scenes
Runs at acceptable FPS on target embedded hardware
```

If webcam holdout performance is poor, do not solve it by training longer. Add the missing real-world images.

---

## 10. Recommended Next Three Runs

After `fruit_v4_s_local` completes:

1. Newer small model quality comparison:

```powershell
.\venv\Scripts\python.exe train.py --model yolo11s.pt --data data_v3_final.yaml --batch 8 --workers 4 --name fruit_v5_11s_quality --clean --epochs 120 --patience 25
```

2. Embedded nano comparison:

```powershell
.\venv\Scripts\python.exe train.py --model yolo11n.pt --data data_v3_final.yaml --batch 16 --workers 4 --name fruit_v5_11n_edge --clean --epochs 100 --patience 20
```

3. Final fine-tune of the winner:

```powershell
.\venv\Scripts\python.exe train.py --model <winner_best.pt> --data data_v3_final.yaml --batch <same_batch> --workers 4 --name <winner>_finetune --clean --epochs 40 --patience 10
```

Do not run all experiments blindly. Stop and inspect failures after each serious run.

---

## 11. What To Borrow From Hugging Face

Hugging Face's good vision-model workflows are strong because they enforce process, not because they use one special architecture.

Borrow these habits:

```text
Dataset-first workflow:
- validate annotation format before training
- sanitize boxes
- remap categories deterministically
- keep image IDs stable
- never spend long training time before data validation passes

Training workflow:
- train from strong COCO-pretrained weights
- evaluate every epoch
- save every epoch or every few epochs
- load best model at end
- compare by mAP, not by vibes
- use a fixed validation split

Experiment workflow:
- run small proof runs first
- then full runs
- name every run clearly
- persist model artifacts
- log enough metadata to reproduce the run
```

Models worth knowing from the Hugging Face object-detection ecosystem:

```text
D-FINE Small:
  Repo: ustc-community/dfine-small-coco
  Params: 10.4M
  Why useful: modern efficient detector, good quality/size reference

RT-DETR v2 R18:
  Repo: PekingU/rtdetr_v2_r18vd
  Params: 20.2M
  Why useful: real-time transformer detector, stronger benchmark candidate

RT-DETR v2 R50:
  Repo: PekingU/rtdetr_v2_r50vd
  Params: 43.0M
  Why useful: higher-capacity quality benchmark, not ideal for laptop/embedded first
```

How this changes our plan:

```text
Primary embedded path remains YOLO11n/YOLO11s or YOLOv8n/YOLOv8s.
HF detectors become reference experiments if YOLO quality plateaus.
The immediate improvement is stricter experiment discipline, not switching stacks mid-run.
```

If we later test HF detectors locally, the required repo work is:

```text
1. Add YOLO-to-COCO conversion.
2. Add HF dataset-format inspector.
3. Add a small D-FINE/RT-DETR training script.
4. Evaluate against the same webcam holdout.
5. Export only if embedded inference is practical.
```
