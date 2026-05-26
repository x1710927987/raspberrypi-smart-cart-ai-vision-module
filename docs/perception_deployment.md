# Perception Deployment Guide

This guide describes how to copy the perception module to a Raspberry Pi or a
similar edge device, install the minimum runtime dependencies, and run the
unified smoke test before connecting perception output to control logic.

The current default artifacts are Ultralytics `.pt` models. They are suitable
for local validation and first Raspberry Pi bring-up. If runtime speed is not
enough, export ONNX or TFLite artifacts later and register new manifests.

## Files to Copy

Copy these directories and files to the target device:

```text
perception/
models/training/smartcart_objects_yolov8n_combined_v3_pt_v1.manifest.json
models/training/smartcart_traffic_light_yolov8n_combined_v2_pt_v1.manifest.json
models/training/smartcart_laneseg_yolov8n_seg_roboflow_pt_v1.manifest.json
models/training/smartcart_hazard_yolov8n_roboflow_pt_v1.manifest.json
models/weights/smartcart_objects_yolov8n_combined_v3_pt_v1.pt
models/weights/smartcart_traffic_light_yolov8n_combined_v2_pt_v1.pt
models/weights/smartcart_laneseg_yolov8n_seg_roboflow_pt_v1.pt
models/weights/smartcart_hazard_yolov8n_roboflow_pt_v1.pt
tools/run_perception_pipeline_smoke.py
tools/run_perception_live_view.py
perception_delivery_report.md
docs/perception_deployment.md
```

For smoke testing with the default sample discovery, also copy a few test images
under these paths:

```text
data/external/objects_combined_v3_split/test/images/
data/external/traffic_light_combined_v2_split/test/images/
data/external/roboflow_sidewalk_v1_split/test/images/
data/external/roboflow_hazard_v1_split/test/images/
```

If storage is limited, copy only one representative image per directory. You can
also pass explicit sample images with `--image`, in which case the full
`data/external/...` tree is not required.

Do not copy training runs, cache folders, or dataset source pools unless the
target device is also used for training:

```text
cache/
runs/
models/training/object_detection_yolo_*/
models/training/traffic_light_yolo_*/
models/training/laneseg_yolo_*/
models/training/hazard_yolo_*/
data/external/*/train/
data/external/*/valid/
```

## Recommended Target Layout

Use the same relative layout as the repository:

```text
smartcart-perception/
  perception/
  models/
    training/
    weights/
  tools/
  docs/
  data/
    external/
      objects_combined_v3_split/test/images/
      traffic_light_combined_v2_split/test/images/
      roboflow_sidewalk_v1_split/test/images/
      roboflow_hazard_v1_split/test/images/
```

The manifest files use relative weight paths such as
`models/weights/smartcart_objects_yolov8n_combined_v3_pt_v1.pt`, so keeping this
layout avoids path changes.

## Minimum Runtime Dependencies

Create a Python 3.10+ environment on the target device. The exact commands
depend on your Raspberry Pi OS image and whether PyTorch/Ultralytics wheels are
available for your platform.

Minimum Python packages:

```text
numpy
opencv-python
ultralytics
torch
torchvision
pyyaml
```

Typical pip setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy opencv-python pyyaml ultralytics
```

If `opencv-python` is too heavy or fails on the Pi, use the system OpenCV
package or `opencv-python-headless`:

```bash
python -m pip install opencv-python-headless
```

If Ultralytics installs a PyTorch build that is too slow or unavailable for the
device, install the PyTorch wheel recommended for your Raspberry Pi OS and
Python version first, then install `ultralytics`.

## Smoke Test

Run the unified perception smoke test before connecting to control logic:

```bash
source .venv/bin/activate
python tools/run_perception_pipeline_smoke.py --device cpu
```

Expected summary:

```text
status=ok
sample_count=4
samples_with_objects=...
samples_with_laneseg=...
samples_with_traffic_light=...
samples_with_hazard=...
```

The command writes:

```text
cache/evaluation/unified_pipeline_smoke_test.json
cache/evaluation/unified_pipeline_smoke_test.md
```

For custom sample images:

```bash
python tools/run_perception_pipeline_smoke.py \
  --device cpu \
  --image samples/road_scene_001.jpg \
  --image samples/traffic_light_001.jpg
```

The smoke test is successful when:

- It prints `status=ok`.
- It writes the JSON and Markdown reports.
- Each saved `PerceptionOutput` can round-trip through JSON and pass runtime
  validation.

## Live Camera View

After the smoke test passes and a camera is available, use the live view tool on
the Raspberry Pi desktop or through VNC:

```bash
python tools/run_perception_live_view.py --camera-backend picamera2 --device cpu --fps 3
```

The window draws object boxes and labels from `output.objects`, and displays
`traffic_light`, `laneseg`, `hazard`, FPS, and inference latency in the corner.
Press `q` or `Esc` to exit.
Use `--camera-backend opencv --camera 0` for a USB camera.

For an SSH-only check without an OpenCV window:

```bash
python tools/run_perception_live_view.py \
  --camera-backend picamera2 \
  --camera 0 \
  --device cpu \
  --no-window \
  --max-frames 5 \
  --print-json-every 1 \
  --save-dir cache/live_view_frames \
  --save-every 1
```

## Runtime Integration Pattern

Control code should depend on `PerceptionOutput`, not on individual YOLO models.

Recommended data flow:

```text
camera frame
  -> PerceptionPipeline
  -> PerceptionOutput
  -> control decision logic
  -> control command / serial output
```

Minimal integration example:

```python
import cv2

from perception.camera_pipeline import (
    PerceptionPipeline,
    build_default_hazard_provider,
    build_default_laneseg_provider,
    build_default_object_detector,
    build_default_traffic_light_provider,
)


pipeline = PerceptionPipeline(
    detector=build_default_object_detector(device="cpu"),
    traffic_light_provider=build_default_traffic_light_provider(device="cpu"),
    laneseg_provider=build_default_laneseg_provider(device="cpu"),
    hazard_provider=build_default_hazard_provider(device="cpu"),
)

frame = cv2.imread("samples/road_scene_001.jpg")
output = pipeline.process_frame(frame)

print(output.to_json())
```

The control module should read these fields:

```text
output.objects
output.traffic_light
output.laneseg
output.hazard
```

## Performance Checks

Record these values on the target device:

```text
cold_start_time_seconds
first_frame_latency_ms
average_frame_latency_ms
peak_memory_mb
camera_resolution
device_temperature
```

Run several frames from a live camera before final integration. The first frame
is expected to be slower because models are loaded lazily.

## If PT Inference Is Too Slow

If the Raspberry Pi cannot meet the target FPS with `.pt` models:

1. Export each model to ONNX or TFLite.
2. Place converted artifacts under `models/weights/`.
3. Register new manifests with updated artifact paths and formats.
4. Re-run:

```bash
python tools/check_model_manifest.py models/training/<new-manifest>.manifest.json --task <task> --require-artifact
python tools/run_perception_pipeline_smoke.py --device cpu
```

Keep the `.pt` manifests as local-development baselines until the converted
runtime is fully validated.

## Handoff Checklist

Before giving the perception package to the control-logic team, confirm:

- The four default manifests exist on the target device.
- The four default weight files exist under `models/weights/`.
- `python tools/run_perception_pipeline_smoke.py --device cpu` reports
  `status=ok`.
- `python tools/run_perception_live_view.py --camera-backend picamera2 --device cpu --fps 3`
  can show annotated camera frames on VNC, or the same command with
  `--no-window --max-frames 5` can process frames on SSH.
- The control team has a sample `PerceptionOutput` JSON.
- The control team knows that hazard output may need ROI and temporal
  confirmation before triggering braking.
- Raspberry Pi performance numbers have been recorded or scheduled for testing.

## Current Known Limits

- Current deployed artifacts are `.pt`, not optimized ONNX/TFLite.
- Hazard currently covers only pothole and curb.
- Objects currently cover pedestrian, bicycle, car, scooter, and roadblock.
- Traffic light currently covers red, yellow, and green.
- LaneSeg currently returns schema-level `LaneSeg(mask_id, conf)`, not a full
  pixel mask.
- The perception pipeline is single-frame; temporal smoothing is expected to be
  handled later in perception or control logic.
