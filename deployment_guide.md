# Raspberry Pi 5 AI Vision Deployment Guide

This guide describes how to deploy the current AI vision module on Raspberry Pi
5. The current hardware target is:

- Raspberry Pi 5.
- Raspberry Pi Camera Rev 1.3 connected through the CSI ribbon interface.
- Raspberry Pi OS 64-bit.
- Picamera2 as the default CSI camera backend.

The first deployment goal is a software smoke test: all four YOLO models can be
loaded, the CSI camera can capture frames, the live view can draw boxes and
labels, and the control loop can run with mock serial. Connect the real lower
control board only after these checks pass.

The most useful validation order is:

```text
1. Model-only smoke test
2. Camera-only Picamera2 test
3. Live-view test with real camera and real models
4. control/app.py with real camera, real models, and mock serial
5. deploy/run.py only after the real lower control board is ready
```

Remote repository:

```text
https://github.com/x1710927987/raspberrypi-smart-cart-ai-vision-module.git
```

## 1. Hardware and System

Recommended setup:

```text
Raspberry Pi 5
Raspberry Pi Camera Rev 1.3 / CSI ribbon camera
Raspberry Pi OS 64-bit
microSD card 64 GB or larger
Stable 5V/5A power supply
Optional USB-TTL / serial cable for the lower control board
```

Camera notes:

- Raspberry Pi Camera Rev 1.3 usually appears as `ov5647`.
- It is a CSI camera, not a USB camera.
- Use `--camera-backend picamera2` for the CSI camera.
- Use `--camera-backend opencv --camera 0` only for USB camera fallback.

## 2. Install System Dependencies

```bash
sudo apt update
sudo apt upgrade -y

sudo apt install -y \
  git \
  git-lfs \
  python3 \
  python3-venv \
  python3-pip \
  python3-picamera2 \
  rpicam-apps \
  libgl1 \
  libglib2.0-0 \
  v4l-utils \
  htop
```

Check the camera at system level:

```bash
rpicam-hello --list-cameras
rpicam-hello --camera 0 --timeout 5000
```

Expected result: the camera list includes `ov5647`, and the preview from
`rpicam-hello` has normal colors.

If this fails, check the CSI ribbon orientation, the connector latch, the camera
module, power supply, and whether the camera is attached to the intended CSI
port.

## 3. Clone Repository and Pull Model Weights

```bash
cd ~
git lfs install
git clone https://github.com/x1710927987/raspberrypi-smart-cart-ai-vision-module.git
cd raspberrypi-smart-cart-ai-vision-module
git lfs pull
```

If the repository is already cloned on the Raspberry Pi, update it with:

```bash
cd ~/raspberrypi-smart-cart-ai-vision-module
git status --short
git pull
git lfs pull
```

If there are local uncommitted changes on the Raspberry Pi, inspect them before
pulling. Do not overwrite local calibration or config changes blindly.

Confirm that Git LFS downloaded real model files:

```bash
git lfs ls-files
ls -lh models/weights/
```

The current deployment requires these four model weights:

```text
models/weights/smartcart_objects_yolov8n_combined_v3_pt_v1.pt
models/weights/smartcart_traffic_light_yolov8n_combined_v2_pt_v1.pt
models/weights/smartcart_laneseg_yolov8n_seg_roboflow_pt_v1.pt
models/weights/smartcart_hazard_yolov8n_roboflow_pt_v1.pt
```

Only the four files above are required by the current default runtime. Other
`.pt` files in `models/weights/` may be historical models or training seed
weights, for example older traffic-light/object models or `yolov8n.pt` /
`yolov8n-seg.pt`. They are useful for retraining or comparison, but they are not
required for normal Raspberry Pi deployment unless a manifest explicitly points
to them.

If a `.pt` file is only tens or hundreds of bytes, it is probably an LFS pointer
instead of the real weight file. Run:

```bash
git lfs install
git lfs pull
```

You can also check the artifact paths registered by manifests:

```bash
python tools/check_model_manifest.py \
  models/training/smartcart_objects_yolov8n_combined_v3_pt_v1.manifest.json \
  --require-artifact
```

## 4. Create Python Environment

Use Raspberry Pi OS Python 3.11 and `venv`. The `--system-site-packages` flag is
important because `picamera2` is installed by apt.

```bash
cd ~/raspberrypi-smart-cart-ai-vision-module
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install Python dependencies:

```bash
python -m pip install \
  numpy \
  pyyaml \
  pyserial \
  opencv-python \
  torch \
  torchvision \
  ultralytics
```

If OpenCV GUI support causes issues, run no-window checks first and use:

```bash
python -m pip install opencv-python-headless
```

If `torch`, `torchvision`, or `ultralytics` fails to install:

```bash
python --version
uname -m
getconf LONG_BIT
python -m pip --version
```

Expected Raspberry Pi values are usually Python 3.11, `aarch64`, and `64`.
If Python is too new or the OS is 32-bit, PyTorch wheels may be unavailable.
Use Raspberry Pi OS 64-bit and the system Python version whenever possible.

If GUI display is not needed yet, prefer no-window tests first. Do not spend
time debugging OpenCV windows before the model imports and no-window live view
work.

Verify imports:

```bash
python -c "import cv2; print(cv2.__version__)"
python -c "import torch; print(torch.__version__)"
python -c "import ultralytics; print(ultralytics.__version__)"
python -c "from picamera2 import Picamera2; print('picamera2 ok')"
```

## 5. Check Default Model Manifests

Current default manifests:

```text
models/training/smartcart_objects_yolov8n_combined_v3_pt_v1.manifest.json
models/training/smartcart_traffic_light_yolov8n_combined_v2_pt_v1.manifest.json
models/training/smartcart_laneseg_yolov8n_seg_roboflow_pt_v1.manifest.json
models/training/smartcart_hazard_yolov8n_roboflow_pt_v1.manifest.json
```

Run:

```bash
python tools/check_model_manifest.py \
  models/training/smartcart_objects_yolov8n_combined_v3_pt_v1.manifest.json \
  --task objects \
  --require-artifact

python tools/check_model_manifest.py \
  models/training/smartcart_traffic_light_yolov8n_combined_v2_pt_v1.manifest.json \
  --task traffic_light \
  --require-artifact

python tools/check_model_manifest.py \
  models/training/smartcart_laneseg_yolov8n_seg_roboflow_pt_v1.manifest.json \
  --task laneseg \
  --require-artifact

python tools/check_model_manifest.py \
  models/training/smartcart_hazard_yolov8n_roboflow_pt_v1.manifest.json \
  --task hazard \
  --require-artifact
```

Current hazard model coverage is:

```text
pothole, curb
```

Water-filled potholes or similar ground pits are handled as `pothole`. The
current default pipeline does not use a separate `water` class.

## 6. Unified Perception Smoke Test

Run the model-only smoke test before using the camera:

```bash
python tools/run_perception_pipeline_smoke.py --device cpu --limit 2
```

Expected result:

```text
status=ok
json=cache/evaluation/unified_pipeline_smoke_test.json
report=cache/evaluation/unified_pipeline_smoke_test.md
```

This command does not open the camera and does not touch the serial port. If it
fails, check Python imports, manifest paths, Git LFS weights, and model backend
loading first.

If you want to use a specific road image:

```bash
python tools/run_perception_pipeline_smoke.py \
  --device cpu \
  --image path/to/road_scene.jpg
```

## 7. Camera Validation

System-level check:

```bash
rpicam-hello --list-cameras
rpicam-hello --camera 0 --timeout 5000
```

Python Picamera2 check:

```bash
python -c "from picamera2 import Picamera2; cam=Picamera2(); cam.configure(cam.create_video_configuration(main={'format':'RGB888','size':(640,480)})); cam.start(); frame=cam.capture_array(); print(frame.shape); cam.stop(); cam.close()"
```

If Python capture hangs with `Camera frontend has timed out`, treat it as a
camera hardware, ribbon, port, power, or camera-ownership issue first. Do not
debug YOLO models until `rpicam-hello` and Picamera2 both work.

Useful ownership check:

```bash
sudo fuser -v /dev/video* /dev/media*
```

If another process is using the camera, stop that process before running the
live-view or control-loop tools.

## 8. Live Recognition View

Run this in the Raspberry Pi desktop or VNC terminal:

```bash
python tools/run_perception_live_view.py \
  --camera-backend picamera2 \
  --device cpu \
  --fps 3
```

The live view shows:

- Camera frames.
- Object boxes, class labels, and confidence.
- Traffic-light, laneseg, and hazard status text.
- Traffic-light, laneseg, and hazard boxes when the active model output
  includes `bbox`.
- FPS and inference latency.

Important visualization note: traffic light, laneseg, and hazard modules can
still be detected even when their exact box is not drawn. The live-view tool
draws boxes only when the corresponding `PerceptionOutput` block includes a
`bbox`. It always shows module status text for traffic light, laneseg, and
hazard.

For SSH or no-window validation:

```bash
python tools/run_perception_live_view.py \
  --camera-backend picamera2 \
  --device cpu \
  --fps 3 \
  --max-frames 10 \
  --no-window \
  --print-json-every 1 \
  --save-dir cache/live_view_frames \
  --save-every 1
```

If colors are swapped, keep the current defaults:

```text
pixel_format=RGB888
camera_color_order=bgr
```

You can also pass them explicitly:

```bash
python tools/run_perception_live_view.py \
  --camera-backend picamera2 \
  --pixel-format RGB888 \
  --camera-color-order bgr \
  --device cpu \
  --fps 3
```

## 9. Mock Serial Control Loop

Run the full camera -> perception -> decision -> serial path without connecting
the real lower control board:

```bash
python control/app.py \
  --camera-backend picamera2 \
  --mock-serial \
  --fps 3 \
  --max-frames 10
```

This opens the real CSI camera and the real four-model perception pipeline, but
uses a mock serial sender.

This is the preferred integration test before connecting the lower control
board. It validates the actual camera, the actual model chain, the decision
engine, and command generation while avoiding real motor output.

Success criteria:

```text
Camera starts without Picamera2 timeout
Perception pipeline initializes
Decision engine initializes
Mock serial initializes
The loop reaches --max-frames and exits normally
```

## 10. Real Serial Preparation

After the lower control board is connected, check serial devices:

```bash
ls /dev/ttyUSB* /dev/ttyACM*
dmesg | tail -40
```

Add the current user to the serial permission group:

```bash
sudo usermod -aG dialout $USER
sudo reboot
```

After reboot:

```bash
groups
```

If the serial device is `/dev/ttyACM0`, update config or pass `--port
/dev/ttyACM0`.

Only run real serial after bench safety checks:

```bash
python control/app.py \
  --camera-backend picamera2 \
  --port /dev/ttyUSB0 \
  --fps 3
```

## 11. `deploy/config.yaml`

Recommended Raspberry Pi 5 + Raspberry Pi Camera Rev 1.3 config:

```yaml
runtime:
  target_fps_min: 10.0
  safety_brake_on_lost_ms: 300
  mock_mode: true

perception:
  camera_backend: "picamera2"
  camera_index: 0
  camera_width: 640
  camera_height: 480
  camera_fps:
  camera_warmup_seconds: 1.0
  camera_read_timeout_seconds: 2.0
  camera_stop_timeout_seconds: 2.0
  pixel_format: "RGB888"
  camera_color_order: "bgr"
  device: "cpu"

serial:
  port: "/dev/ttyUSB0"
  baud: 115200
```

Keep `mock_mode: true` until camera, model loading, live view, and control loop
tests are stable. Switch to real serial only after bench validation:

```yaml
runtime:
  mock_mode: false
```

Run the deployment service:

```bash
python deploy/run.py --config deploy/config.yaml
```

Important behavior difference:

```text
control/app.py --mock-serial:
  Uses real camera + real perception + real decision engine + mock serial.
  Use this for software integration before the lower control board is connected.

deploy/run.py with runtime.mock_mode: true:
  Uses mock perception and mock serial. It is useful for service startup checks,
  but it does not test the real camera or real YOLO models.

deploy/run.py with runtime.mock_mode: false:
  Uses real camera + real perception + real serial. Use this only after the
  lower control board and bench safety setup are ready.
```

## 12. Common Issues

### `ModuleNotFoundError: No module named 'control'`

Run from the repository root:

```bash
cd ~/raspberrypi-smart-cart-ai-vision-module
source .venv/bin/activate
python control/app.py --camera-backend picamera2 --mock-serial --fps 3
```

Do not run the script from inside the `control/` or `deploy/` subdirectory.

### Picamera2 Capture Timeout

Check camera ownership:

```bash
sudo fuser -v /dev/video* /dev/media*
```

Check hardware:

```bash
rpicam-hello --camera 0 --timeout 5000
```

If `rpicam-hello` also fails, check the ribbon orientation, connector latch,
camera module, CSI port, and power supply.

### No VNC Window

Use no-window mode and save annotated frames:

```bash
python tools/run_perception_live_view.py \
  --camera-backend picamera2 \
  --device cpu \
  --no-window \
  --max-frames 10 \
  --save-dir cache/live_view_frames \
  --save-every 1
```

### Slow Inference

The checked-in `.pt` models are acceptable for stage validation, but Raspberry
Pi CPU inference can be slow. Options:

- Lower `--fps`.
- Lower capture resolution, for example:

```bash
python tools/run_perception_live_view.py \
  --camera-backend picamera2 \
  --camera-width 320 \
  --camera-height 240 \
  --device cpu \
  --fps 3
```

- Run heavier models less frequently.
- Add ROI filtering and temporal confirmation.
- Export ONNX/TFLite later and register new manifests.

### No Serial Device

If the lower control board is not connected, missing `/dev/ttyUSB*` or
`/dev/ttyACM*` is normal. Continue using `--mock-serial`.

### Git LFS Weight Still Missing

Symptoms:

```text
model file is very small
torch load fails
manifest check reports missing artifact
```

Fix:

```bash
git lfs install
git lfs pull
ls -lh models/weights/
python tools/check_model_manifest.py \
  models/training/smartcart_objects_yolov8n_combined_v3_pt_v1.manifest.json \
  --require-artifact
```

### Window Opens But No Boxes Are Drawn

Check the JSON printed by live view:

```bash
python tools/run_perception_live_view.py \
  --camera-backend picamera2 \
  --device cpu \
  --fps 3 \
  --print-json-every 1
```

If JSON contains objects but no visual boxes, inspect the drawing tool. If JSON
does not contain the expected module output, inspect the corresponding model,
threshold, or scene content.

### Real-Vehicle Safety Boundary

Before disabling mock serial:

```text
Vehicle wheels should be lifted, or motor power should be disconnected.
Manual emergency stop should be available.
Use low FPS and low speed first.
Keep one person watching the vehicle and one person watching logs.
```

## 13. Deployment Checklist

```text
[ ] Raspberry Pi 5 runs 64-bit Raspberry Pi OS
[ ] Raspberry Pi Camera Rev 1.3 is listed by rpicam-hello
[ ] git lfs pull has been run
[ ] Four deployment .pt weights exist under models/weights/
[ ] Python venv is created and activated
[ ] cv2 / torch / ultralytics / picamera2 / yaml / serial import correctly
[ ] Four manifests pass tools/check_model_manifest.py
[ ] tools/run_perception_pipeline_smoke.py reports status=ok
[ ] tools/run_perception_live_view.py shows or saves annotated live frames
[ ] control/app.py runs with --mock-serial
[ ] Real serial device exists before disabling mock serial
[ ] User belongs to dialout before real serial
[ ] Wheels are lifted or motor power is disconnected before real-serial bench test
[ ] Manual emergency stop is available before real-vehicle testing
```

## 14. Recommended Execution Order

```text
1. Install system dependencies
2. Clone repository and run git lfs pull
3. Create venv and install Python dependencies
4. Check four manifests and weights
5. Run unified perception smoke test
6. Check Raspberry Pi Camera Rev 1.3 with rpicam-hello
7. Run live view in VNC
8. Run control/app.py with --mock-serial
9. Connect lower control board and confirm serial device
10. Run real serial at low FPS with wheels lifted or motor power disconnected
```
