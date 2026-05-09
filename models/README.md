# Model Registry Guide

This directory stores model metadata, conversion notes, training notes, and local model weight placeholders for the smart-cart perception module.

Large model files are deployment artifacts. Keep them out of Git unless the team explicitly decides to version a tiny test model. Store real `.pt`, `.onnx`, `.tflite`, `.engine`, `.xml`, `.bin`, or `.h5` files under `models/weights/` locally, and record their metadata in a manifest.

## Directory Layout

```text
models/
  weights/                         # Local model artifacts, usually not committed
  training/                        # Training configs, class maps, reports, metrics
  conversion/                      # ONNX/TFLite conversion scripts and notes
  model_manifest.example.json      # Example model registry entry
  model_manifest.objects.example.json
  model_manifest.traffic_light.example.json
```

## Naming Convention

Use stable ASCII names:

```text
smartcart_<task>_<architecture>_<dataset>_<format>_v<major>.<ext>
```

Examples:

```text
smartcart_objects_yolov8n_roboflow_onnx_v1.onnx
smartcart_hazard_yolov8n_roboflow_tflite_v1.tflite
smartcart_traffic_light_yolov8n_s2tld_onnx_v1.onnx
smartcart_laneseg_deeplabv3_cityscapes_tflite_v1.tflite
```

Recommended task names:

```text
objects
traffic_light
hazard
laneseg
```

Recommended format names:

```text
pt
onnx
tflite
engine
openvino
```

## Class Mapping

Model output classes must be mapped into `schema.md`.

Object detection target classes:

```text
pedestrian, obstacle, bicycle, car, animal, stroller, wheelchair, bollard, scooter, unknown
```

Traffic-light target states:

```text
red, yellow, green, off, flashing, unknown
```

Hazard target types:

```text
pothole, step_up, step_down, speed_bump, water, debris, unknown
```

The manifest should keep both:

- `model_classes`: model-native output order, exactly as exported.
- `schema_mapping`: map model-native names or indices into project schema labels.

Example:

```json
{
  "model_classes": ["person", "bicycle", "car", "traffic cone"],
  "schema_mapping": {
    "person": "pedestrian",
    "bicycle": "bicycle",
    "car": "car",
    "traffic cone": "obstacle"
  }
}
```

## ONNX Registration

For ONNX models, record:

- opset version
- input tensor name and shape
- output tensor names
- preprocessing requirements
- confidence threshold
- NMS threshold
- execution provider preference

Typical Raspberry Pi options:

```text
CPUExecutionProvider
OpenVINOExecutionProvider
```

Use ONNX when you want a portable intermediate format and can run `onnxruntime` or OpenVINO on the target device.

## TFLite Registration

For TFLite models, record:

- input tensor shape
- quantization type: `float32`, `float16`, `int8`, or `uint8`
- normalization rule
- label order
- confidence threshold
- NMS threshold
- delegate preference

Typical Raspberry Pi options:

```text
cpu
edge_tpu
gpu
```

Use TFLite when the Raspberry Pi deployment stack favors TensorFlow Lite or when the model is quantized for lower latency.

## Expected Workflow

1. Train or export a model from Roboflow, Ultralytics, TensorFlow, or another training pipeline.
2. Place the model file under `models/weights/` locally.
3. Copy `model_manifest.example.json` to a real manifest name, for example:

```text
models/training/smartcart_objects_yolov8n_roboflow_onnx_v1.manifest.json
```

4. Fill in source dataset, labels, input/output tensors, thresholds, and deployment notes.
5. Run a small local inference smoke test before connecting the model to `perception.camera_pipeline.PerceptionPipeline`.

## Minimum Acceptance Criteria

A model is ready to integrate when:

- The manifest is complete.
- The model file exists at the registered path.
- The class mapping only uses labels from `schema.md`.
- A smoke test can run inference on at least one image from `data/raw`.
- The postprocessed output can be converted into `PerceptionOutput`.

## Current Baselines

Object detection:

```powershell
python tools/export_objects_yolo_dataset.py --overwrite
python tools/train_object_detection_yolo.py --dry-run
python tools/check_model_manifest.py models/model_manifest.objects.example.json --task objects
```

Traffic light:

```powershell
python tools/train_traffic_light_yolo.py --dry-run
python tools/check_model_manifest.py models/model_manifest.traffic_light.example.json --task traffic_light
```
