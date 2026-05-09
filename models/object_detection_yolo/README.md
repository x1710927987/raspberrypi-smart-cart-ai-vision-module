# Object Detection YOLO

This folder documents the baseline YOLO detector for smart-cart object perception.

## Scope

Current baseline classes:

```text
pedestrian, bicycle, car
```

These are the classes currently available from the imported external datasets. Keep the model narrow until we have enough labeled examples for the remaining `schema.md` object labels.

## Dataset Export

The project stores normalized object annotations as JSON under `data/annotations/objects`. Before Ultralytics training, export them to a YOLO directory:

```powershell
conda activate smartcart-ai
python tools/export_objects_yolo_dataset.py --overwrite
```

This creates:

```text
data/processed/objects_yolo_v1/
  data.yaml
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  test/images/
  test/labels/
```

The default class order is:

```text
0 pedestrian
1 bicycle
2 car
```

## Training

Dry-run the training plan:

```powershell
python tools/train_object_detection_yolo.py --dry-run
```

Run a short smoke test:

```powershell
python tools/train_object_detection_yolo.py `
  --epochs 3 `
  --batch 4 `
  --device cpu `
  --name smartcart_objects_yolov8n_smoke `
  --no-plots
```

Run a longer GPU training job:

```powershell
python tools/train_object_detection_yolo.py `
  --epochs 50 `
  --batch 16 `
  --device 0 `
  --name smartcart_objects_yolov8n_roboflow_v1_50epoch_gpu
```

## Registration

After training, register the best checkpoint:

```powershell
python tools/register_object_detection_model.py `
  --source-model models/training/object_detection_yolo_v1/smartcart_objects_yolov8n_roboflow_v1_50epoch_gpu/weights/best.pt `
  --run-dir models/training/object_detection_yolo_v1/smartcart_objects_yolov8n_roboflow_v1_50epoch_gpu `
  --model-id smartcart_objects_yolov8n_roboflow_pt_v1 `
  --overwrite
```

For Raspberry Pi deployment, export ONNX or TFLite after the PT checkpoint is validated locally.
