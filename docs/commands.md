# Commands

This file records the common commands for developing, validating, and deploying
the smart-cart AI vision module.

Current deployment hardware:

- Raspberry Pi 5.
- Raspberry Pi Camera Rev 1.3 connected through the CSI ribbon interface.
- Picamera2 is the default camera backend on Raspberry Pi.

## Environment Setup

Run these commands each time a new terminal is opened on Raspberry Pi:

```bash
cd ~/raspberrypi-smart-cart-ai-vision-module
source .venv/bin/activate
```

Update an existing Raspberry Pi checkout:

```bash
cd ~/raspberrypi-smart-cart-ai-vision-module
git status --short
git pull
git lfs pull
```

On the Windows development workstation:

```powershell
cd D:\Repository\internship_and_research\raspberrypi-smart-cart-ai-vision-module
conda activate smartcart-ai
```

## Camera Checks

Check that Raspberry Pi Camera Rev 1.3 is detected. It usually appears as
`ov5647`:

```bash
rpicam-hello --list-cameras
rpicam-hello --camera 0 --timeout 5000
```

If this works but Python capture does not, check Picamera2:

```bash
python -c "from picamera2 import Picamera2; cam=Picamera2(); cam.configure(cam.create_video_configuration(main={'format':'RGB888','size':(640,480)})); cam.start(); frame=cam.capture_array(); print(frame.shape); cam.stop(); cam.close()"
```

## Model Manifest Checks

```bash
git lfs ls-files
ls -lh models/weights/

python tools/check_model_manifest.py models/training/smartcart_objects_yolov8n_combined_v3_pt_v1.manifest.json --task objects --require-artifact
python tools/check_model_manifest.py models/training/smartcart_traffic_light_yolov8n_combined_v2_pt_v1.manifest.json --task traffic_light --require-artifact
python tools/check_model_manifest.py models/training/smartcart_laneseg_yolov8n_seg_roboflow_pt_v1.manifest.json --task laneseg --require-artifact
python tools/check_model_manifest.py models/training/smartcart_hazard_yolov8n_roboflow_pt_v1.manifest.json --task hazard --require-artifact
```

## Unified Perception Smoke Test

Run this after changing model manifests or weights:

```bash
python tools/run_perception_pipeline_smoke.py --device cpu
```

On the Windows workstation with CUDA:

```powershell
python tools\run_perception_pipeline_smoke.py --device 0
```

## Capture Annotated Frames

Capture 10 annotated images from Raspberry Pi Camera Rev 1.3 without opening a
GUI window:

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

## Real-Time Image Recognition

Run this in VNC to show the live annotated camera view:

```bash
python tools/run_perception_live_view.py \
  --camera-backend picamera2 \
  --device cpu \
  --fps 3 \
  --save-dir cache/live_view_frames \
  --save-every 30
```

If frame colors look swapped, keep the default Picamera2 settings from
`deploy/config.yaml`: `pixel_format: "RGB888"` and
`camera_color_order: "bgr"`.

For a USB camera fallback:

```bash
python tools/run_perception_live_view.py \
  --camera-backend opencv \
  --camera 0 \
  --device cpu \
  --fps 3
```

## Control Loop Smoke Test

Run the control loop with camera input but without a real serial device:

```bash
python control/app.py \
  --camera-backend picamera2 \
  --mock-serial \
  --fps 3 \
  --max-frames 10
```

This is the preferred real camera + real perception + mock serial integration
test before connecting the lower control board.

When the vehicle controller is connected, first confirm the serial device:

```bash
ls /dev/ttyUSB* /dev/ttyACM*
groups
```

Then run without `--mock-serial` only after bench safety checks:

```bash
python control/app.py \
  --camera-backend picamera2 \
  --port /dev/ttyUSB0 \
  --fps 3
```

## Deployment Service

Run the deployment service with the default Raspberry Pi 5 configuration:

```bash
python deploy/run.py --config deploy/config.yaml
```

Note: `deploy/run.py` with `runtime.mock_mode: true` uses mock perception and
mock serial. Use `control/app.py --mock-serial` when you need real camera and
real YOLO models without sending commands to the real serial port.

## Training Short Smoke Examples

Traffic light:

```powershell
python tools\train_traffic_light_yolo.py --epochs 3 --workers 0 --device cpu --name smoke_traffic_light --exist-ok
```

Objects:

```powershell
python tools\train_object_detection_yolo.py --epochs 3 --workers 0 --device cpu --name smoke_objects --exist-ok
```

Hazard:

```powershell
python tools\train_hazard_yolo.py --epochs 3 --workers 0 --device cpu --name smoke_hazard --exist-ok
```

LaneSeg:

```powershell
python tools\train_laneseg_yolo.py --epochs 3 --workers 0 --device cpu --name smoke_laneseg --exist-ok
```
