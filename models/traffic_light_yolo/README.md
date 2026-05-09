# Traffic Light YOLO Model

This directory documents the traffic-light YOLO model used by the perception module.

The current prepared training dataset is:

```text
data/external/roboflow_traffic_light_v1_split/data.yaml
```

Use this YOLO dataset when training a detector that finds traffic-light boxes and classifies each box as `green`, `red`, or `yellow`.

## Expected Classes

The source dataset class order is:

```text
0: green
1: red
2: yellow
```

These labels map directly into the project traffic-light schema:

```text
green -> green
red -> red
yellow -> yellow
```

The full schema also supports `off`, `flashing`, and `unknown`, but the current Roboflow dataset does not contain those classes. Keep them out of `model_classes` unless a future dataset actually trains them.

## Training Output

Recommended local output names:

```text
models/weights/smartcart_traffic_light_yolov8n_roboflow_onnx_v1.onnx
models/weights/smartcart_traffic_light_yolov8n_roboflow_tflite_v1.tflite
models/weights/smartcart_traffic_light_yolov8n_roboflow_pt_v1.pt
```

Large model files should stay local and should not be committed unless the team explicitly decides to version a tiny test artifact.

## Registration

After training or exporting, copy:

```text
models/model_manifest.traffic_light.example.json
```

to a real manifest name, for example:

```text
models/training/smartcart_traffic_light_yolov8n_roboflow_onnx_v1.manifest.json
```

Then update:

- `artifact.path`
- `artifact.format`
- `artifact.sha256`
- `artifact.size_bytes`
- `source.dataset_version`
- `source.license`
- `source.exported_at`
- `evaluation.metrics`

Before connecting the model to `perception.camera_pipeline.PerceptionPipeline`, run:

```powershell
python tools\check_model_manifest.py models\training\smartcart_traffic_light_yolov8n_roboflow_onnx_v1.manifest.json --task traffic_light --require-artifact
```

For a manifest-only check before the model file exists:

```powershell
python tools\check_model_manifest.py models\model_manifest.traffic_light.example.json --task traffic_light
```

## Integration Notes

The traffic-light YOLO detector has two possible integration paths:

1. Use the detector directly to produce a `TrafficLight` result from the best traffic-light prediction.
2. Use YOLO only to crop traffic lights, then pass crops into a smaller classifier.

For the current project stage, path 1 is simpler and should be enough for a first Raspberry Pi smoke test.
