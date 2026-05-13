# Model Registry Guide

This directory stores model metadata, conversion notes, training notes, and model weight placeholders for the smart-cart perception module.

Large model files are deployment artifacts. Deployment-ready `.pt` weights are tracked with Git LFS under `models/weights/`, while local experiments, obsolete smoke weights, converted formats, and training-run outputs should stay ignored unless the team explicitly registers them. Every deployable artifact must also be recorded in a manifest.

## Directory Layout

```text
models/
  weights/                         # Git LFS deployment weights plus ignored local experiments
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
pedestrian, obstacle, roadblock, bicycle, car, animal, stroller, wheelchair, bollard, scooter, unknown
```

Traffic-light target states:

```text
red, yellow, green, off, flashing, unknown
```

Hazard target types:

```text
pothole, curb, step_up, step_down, speed_bump, water, debris, unknown
```

The manifest should keep both:

- `model_classes`: model-native output order, exactly as exported.
- `schema_mapping`: map model-native names or indices into project schema labels.

Example:

```json
{
  "model_classes": ["person", "bicycle", "car", "traffic cone", "roadblock"],
  "schema_mapping": {
    "person": "pedestrian",
    "bicycle": "bicycle",
    "car": "car",
    "traffic cone": "obstacle",
    "roadblock": "roadblock"
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
5. Run the manifest checker for the model task.
6. Run the unified perception pipeline smoke test before treating the model as integrated.

## Model Change Acceptance Workflow

Run these checks every time a model artifact or default manifest changes.

1. Verify the manifest and artifact path:

```powershell
conda activate smartcart-ai
python tools\check_model_manifest.py models\training\<manifest-name>.manifest.json --task <task> --require-artifact
```

2. If the model has a task-specific evaluator, run it on the corresponding test split. For example:

```powershell
python tools\evaluate_object_detection_model.py --device 0
python tools\evaluate_traffic_light_model.py --split test --device 0
python tools\evaluate_hazard_model.py --split test --device 0
```

3. Run the unified end-to-end smoke test:

```powershell
python tools\run_perception_pipeline_smoke.py --device 0
```

For CPU-only validation:

```powershell
python tools\run_perception_pipeline_smoke.py --device cpu
```

This writes:

```text
cache/evaluation/unified_pipeline_smoke_test.json
cache/evaluation/unified_pipeline_smoke_test.md
```

The smoke test must report `status=ok`. It also validates that the saved
`PerceptionOutput` payloads can round-trip through JSON and pass runtime schema
checks.

4. On Raspberry Pi or a VNC desktop, run the live camera view to verify visual
   output:

```powershell
python tools\run_perception_live_view.py --camera 0 --device cpu --fps 3
```

The live view draws `objects` bounding boxes and labels, and displays
traffic-light, laneseg, hazard, FPS, and inference latency status.

## Minimum Acceptance Criteria

A model is ready to integrate when:

- The manifest is complete.
- The model file exists at the registered path.
- The class mapping only uses labels from `schema.md`.
- Task-specific evaluation can run on the relevant test split, when an evaluator exists.
- `tools/run_perception_pipeline_smoke.py` reports `status=ok`.
- `tools/run_perception_live_view.py` can show annotated live camera output on the target device or VNC session.
- The postprocessed output can be converted into and validated as `PerceptionOutput`.

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
