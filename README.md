# 🚀 Object Detection (No GPU Required)

A **real-time object detection system** built using YOLOv8, optimized for **CPU-only systems** (Intel UHD Graphics / No CUDA).

---

## 🔥 Features

- ✅ Works on **CPU (no GPU required)**
- 🎥 Supports **Webcam + Video Input**
- ⚡ Optimized with:
  - Frame Skipping
  - Reduced Inference Size
- 📊 Live:
  - FPS counter
  - Detection count
- 💾 Optional video saving
- 🎯 Multiple YOLOv8 model sizes

---

## 🧠 Tech Stack

- Python 3.13+
- OpenCV
- Ultralytics YOLOv8

---

## 📦 Installation

### 1. Clone repo

```bash
git clone https://github.com/mehtadigisha/Object-Detection.git
cd Object-Detection
```

### 2. Create virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install CPU-only PyTorch **first**

> This guarantees you get the CPU-only build and don't accidentally pull in  
> a CUDA-dependent wheel that will fail on Intel UHD.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
````

### 5. Verify installation

```bash
python - <<'EOF'
import torch, cv2, ultralytics, numpy
print("torch      :", torch.__version__, " | CUDA:", torch.cuda.is_available())
print("opencv     :", cv2.__version__)
print("ultralytics:", ultralytics.__version__)
print("numpy      :", numpy.__version__)
EOF
```

Expected output (example):
```
torch      : 2.3.1+cpu  | CUDA: False
opencv     : 4.10.0
ultralytics: 8.3.18
numpy      : 2.0.1
```
---

### ▶️ Run the Project

```bash
python main.py
```

You will be asked:

```
  Use webcam or video file? [webcam/video]: _
  Choose model size [n/s/m] (default=s): _
  Save output video? [y/n]: _
  Change performance settings? [y/n]: _
```
## ⌨️ Command-Line Flags (advanced / automation)

Skip all prompts by passing flags directly:

```bash
# Webcam, nano model, no save
python main.py --source 0 --model yolov8n.pt

# Video file, small model, save output
python main.py --source traffic.mp4 --model yolov8s.pt --save out.mp4

# Custom confidence + inference size
python main.py --source 0 --conf 0.5 --infer-size 320

# Frame-skip (process every 2nd frame — faster on slow CPUs)
python main.py --source 0 --frame-skip 1
```

| Flag | Default | Description |
|---|---|---|
| `--source` | *(prompt)* | `0` = webcam, or path to video |
| `--model` | *(prompt)* | `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt` |
| `--conf` | `0.45` | Detection confidence threshold (0–1) |
| `--infer-size` | `416` | Inference image dimension in pixels |
| `--frame-skip` | `0` | Skip N frames between inferences |
| `--save` | *(prompt)* | Output video path |
| `--display-width` | `960` | Max display width in pixels |

---

## ⚡ Performance Optimisation Guide

### Intel UHD (no discrete GPU) — recommended settings

| Use-case | `--infer-size` | `--frame-skip` | Expected FPS |
|---|---|---|---|
| Real-time webcam (720p) | 320 | 1 | ~18–25 |
| Balanced | 416 | 0 | ~10–18 |
| High accuracy | 640 | 0 | ~5–10 |

### Tips

1. **Use `yolov8n`** — the nano model is ~4× faster than small on CPU, with  
   only a modest accuracy drop for common objects.

2. **Lower `--infer-size`** — `320` is plenty for detecting people and vehicles  
   in typical scenes; `640` adds detail but halves your FPS.

3. **Enable frame-skip** — `--frame-skip 1` runs inference every other frame,  
   re-using the last detection on skipped frames. Great for video files.

4. **Set `--display-width 640`** — halving the display resolution reduces  
   OpenCV rendering time on large monitors.

5. **Close background apps** — YOLO on CPU is memory-bandwidth bound;  
   closing browsers / other processes frees significant bandwidth.

6. **OMP_NUM_THREADS** — if you notice thermal throttling, cap CPU threads:
   ```bash
   # Windows PowerShell
   $env:OMP_NUM_THREADS = "4"; python main.py

   # macOS / Linux
   OMP_NUM_THREADS=4 python main.py
   ```

---

## 🏷 Detected Classes (COCO — 80 classes)

```
person  bicycle  car  motorcycle  airplane  bus  train  truck  boat
traffic light  fire hydrant  stop sign  parking meter  bench
bird  cat  dog  horse  sheep  cow  elephant  bear  zebra  giraffe
backpack  umbrella  handbag  tie  suitcase  frisbee  skis  snowboard
sports ball  kite  baseball bat  baseball glove  skateboard  surfboard
tennis racket  bottle  wine glass  cup  fork  knife  spoon  bowl
banana  apple  sandwich  orange  broccoli  carrot  hot dog  pizza
donut  cake  chair  couch  potted plant  bed  dining table  toilet
tv  laptop  mouse  remote  keyboard  cell phone  microwave  oven
toaster  sink  refrigerator  book  clock  vase  scissors  teddy bear
hair drier  toothbrush
```

---

## 🎬 Demo Example

### 📥 Input
[Input Video](https://github.com/user-attachments/assets/59a4c141-e06a-41c4-8333-e5153c1e1a68)

### 📤 Output
[Output Video](https://github.com/user-attachments/assets/b4298d0b-e381-4533-b8e7-2f2f1bfc90a7)

## 📄 License

This project template is released under the **MIT License**.  
YOLOv8 is © Ultralytics and licensed under AGPL-3.0.

---

> Built for Python 3.13 · Intel UHD CPU · Ultralytics YOLOv8
