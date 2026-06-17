# Banana Pi Vision Lab (YOLO BBox Tester)

A FastAPI app for Banana Pi M7 to test real-time object detection overlays with YOLO models.

The app serves a live detection page on port 3000 and a settings page where you can:

- Select model files from `/models`.
- Toggle detection classes (checkbox filters).
- Tune confidence, IoU, image size, and bbox opacity.
- Switch camera source (`uvc://`, `rtsp://`, `rtmp://`).
- Enable SAHI sliced inference for small-object experiments.

## Core behavior

- Main page (`/`) shows the MJPEG stream with bbox overlays.
- Settings page (`/settings`) controls runtime behavior without restarting, including class toggle checkboxes.
- Model catalog scans `/models` for these suffixes:
  - `.pt`
  - `.onnx`
  - `.rknn` (loadable when RKNN runtime is installed)
- Class labels are read from sidecar metadata if available:
  - `model.onnx.classes.json`
  - `model.classes.json`
  - `model.json`
  - `model.labels.txt`

## Convert a model to JSON

Use the helper script to extract class names from a model or sidecar labels file and write the JSON format the app already reads:

```bash
python convert_model_to_json.py models/yolo26n.pt
python convert_model_to_json.py models/yolo26n.rknn -o models/yolo26n.rknn.classes.json
python convert_model_to_json.py models/yolo26n.rknn --labels-file models/yolo26n.labels.txt -o models/yolo26n.rknn.classes.json
```

## Deployment model bootstrap (Docker Compose)

At container startup, `model_bootstrap.py` will:

- Import supported model files and sidecars from `/model-import` (mapped from `./model-import`) into `/models`.
- Generate missing class sidecars like `model.rknn.classes.json` when class names can be resolved.
- Write `/models/.default_model` so app startup can auto-select the most suitable model.

Recommended Banana Pi M7 flow:

1. Put exported models in `./model-import` (for example `yolo26n.rknn`).
2. Run `docker compose up --build -d`.
3. Open `/settings` and confirm/select the model from WebUI.

CLI helper for operators:

```bash
python select_model.py --host http://127.0.0.1:3000
python select_model.py --host http://127.0.0.1:3000 --select yolo26n.rknn
```

## Folder layout

```text
/models
  yolo11n.onnx
  yolo11n.onnx.classes.json
/data
  faces/
    alice/
      1.jpg
```

## RKNN Model Preparation

### Part A — Acquire a `.rknn` file

Use one of these paths:

**Option 1: Export from an existing YOLO `.pt` model on a development machine** (most common)

This requires `rknn-toolkit2` (full toolkit), not `rknn-toolkit-lite2` (lite2 is inference-only on device).

First export `.pt` to `.onnx` with Ultralytics:

```bash
yolo export model=yolo11n.pt format=onnx imgsz=640
```

Then convert ONNX to RKNN:

```python
from rknn.api import RKNN

rknn = RKNN()
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
rknn.load_onnx(model='yolo11n.onnx')   # export .pt to .onnx first via ultralytics
rknn.build(do_quantization=False)
rknn.export_rknn('./models/yolo11n.rknn')
rknn.release()
```

**Option 2: Download a pre-exported `.rknn` model**

You can download from community sources such as Rockchip model zoo: `rockchip-linux/rknn_model_zoo`.
Make sure the model target matches your NPU (`rk3588` for Banana Pi M7).

### Part B — Generate the required `.rknn.classes.json` sidecar

The `.rknn.classes.json` file is required. Without it, the model can load but detections may be silent or misclassified.
It contains the class label index mapping used by the app.

Use one of these methods (recommended order):

```bash
# If you have a labels .txt file (one class per line):
python convert_model_to_json.py models/yolo11n.rknn --labels-file models/yolo11n.labels.txt -o models/yolo11n.rknn.classes.json

# If a sibling .pt or .onnx is present (auto-extracts labels from the model):
python convert_model_to_json.py models/yolo11n.rknn -o models/yolo11n.rknn.classes.json

# Or write it manually — it's just a JSON index-to-name mapping:
# models/yolo11n.rknn.classes.json
```

Example structure (COCO-style, truncated):

```json
{
  "names": {
    "0": "person",
    "1": "bicycle",
    "2": "car",
    "...": "..."
  }
}
```

### Part C — Final expected folder state

```text
models/
  yolo11n.rknn
  yolo11n.rknn.classes.json
```

Both files must be present before starting the app.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 3000
```

## Banana Pi M7 RKNN deployment without Docker

This mode is supported. You can run the same WebUI directly on the host OS (Debian/Ubuntu based image) with RKNN runtime.

### 1) Install system dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv libgl1 libglib2.0-0
```

### 2) Create virtual environment and install app dependencies

```bash
cd /path/to/EchelonVision/Yolo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Install RKNN runtime wheel

Install the RKNN runtime package matching your Banana Pi image and Python version.

```bash
pip install /path/to/rknn_toolkit_lite2-<version>-cp3x-cp3x-linux_aarch64.whl
```

If RKNN runtime is not installed, `.rknn` models will appear in Settings but remain non-selectable.

### 4) Prepare model files

See [RKNN Model Preparation](#rknn-model-preparation) above for how to obtain or export the `.rknn` file and generate the required sidecar.

Put model files into `./models`, for example:

```text
models/
  yolo26n.rknn
  yolo26n.rknn.classes.json
```

Generate sidecar metadata when needed:

```bash
python convert_model_to_json.py models/yolo26n.rknn -o models/yolo26n.rknn.classes.json
python convert_model_to_json.py models/yolo26n.rknn --labels-file models/yolo26n.labels.txt -o models/yolo26n.rknn.classes.json
```

### 5) Start the app (no Docker)

```bash
cd /path/to/EchelonVision/Yolo
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 3000
```

Open:

- `http://<banana-pi-ip>:3000/`
- `http://<banana-pi-ip>:3000/settings`

### 6) Use the Installed Model selector

1. Open `/settings`.
2. Pick your `.rknn` file from **Installed Model**.
3. Click **Save Settings** to apply immediately.

No container or extra launcher script is required for manual run.

### Optional: auto-start with systemd

Create `/etc/systemd/system/echelon-vision.service`:

```ini
[Unit]
Description=Echelon Vision WebUI
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/path/to/EchelonVision/Yolo
Environment=PYTHONUNBUFFERED=1
ExecStart=/path/to/EchelonVision/Yolo/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 3000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable echelon-vision
sudo systemctl start echelon-vision
sudo systemctl status echelon-vision
```

Open:

- `http://<banana-pi-ip>:3000/`
- `http://<banana-pi-ip>:3000/settings`

## Docker deployment (recommended)

```bash
docker compose up --build
```

The compose setup maps:

- `./models` -> `/models`
- `./model-import` -> `/model-import` (read-only)
- `./data` -> `/data`
- `/dev/video0` for UVC camera

## Important environment variables

- `PORT` default `3000`
- `CAMERA_SOURCE` default `uvc://0`
- `MODELS_DIR` default `/models`
- `MODEL_IMPORT_DIR` default `/model-import`
- `DATA_DIR` default `/data`
- `YOLO_MODEL` default `yolov8n.pt`
- `YOLO_CONFIDENCE` default `0.35`
- `YOLO_IOU` default `0.45`
- `YOLO_IMAGE_SIZE` default `640`
- `BBOX_OPACITY` default `0.35`
- `SAHI_ENABLED` default `false`
- `FACES_DIR` default `/data/faces`
- `FACE_RECOGNITION_ENABLED` default `false`

## Banana Pi M7 notes

- Use ONNX models for CPU fallback testing.
- RKNN models are selectable only when `rknn-toolkit-lite2` is available on the device/runtime.
- If RKNN runtime is missing, RKNN models remain visible but disabled in Settings.
- For RTSP/RTMP streams, ensure LAN routing and stream credentials are valid.
- For USB camera, confirm device exists on host (`/dev/video0`, `/dev/video1`, etc.).

## APIs

- `GET /api/status` runtime and latest detections
- `GET /api/models` model catalog
- `POST /api/models/select` load model
- `POST /api/classes` set enabled class filters
- `POST /api/settings` update runtime settings
- `GET /api/debug/rknn` RKNN output/shape debug
- `GET /api/debug/bbox-trace` last-frame bbox trace (decode/remap/draw pipeline)

# EchelonVision
