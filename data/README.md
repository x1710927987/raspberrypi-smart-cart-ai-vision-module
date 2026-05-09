# Data Collection Guide

This directory stores local datasets for the Raspberry Pi smart cart perception module.

The repository should keep only documentation, validation scripts, split files, annotation examples, and `.gitkeep` placeholders. Do not commit large raw images, videos, model outputs, or generated caches unless the team explicitly decides to version a tiny sample set.

## Directory Layout

```text
data/
  raw/
    objects/images/                 # Images for object detection annotation
    sidewalk/images/                # Road/sidewalk images for drivable-area segmentation
    traffic_light/red/              # Red traffic-light samples
    traffic_light/yellow/           # Yellow traffic-light samples
    traffic_light/green/            # Green traffic-light samples
    traffic_light/negative/         # No traffic light or unclear/off samples
    hazard/pothole/                 # Potholes and dark road holes
    hazard/step_up/                 # Curbs or upward steps
    hazard/step_down/               # Downward steps or sudden drops
    hazard/speed_bump/              # Speed bumps
    hazard/water/                   # Water patches
    hazard/debris/                  # Debris and scattered obstacles on the road
    hazard/negative/                # Clean road surface without hazards
  annotations/
    objects/                        # JSON annotations for object detection
    laneseg/masks/                  # Binary masks for drivable-area segmentation
    traffic_light/                  # Optional JSON annotations for traffic-light classification
    hazard/                         # Optional JSON annotations for hazard classification/detection
  manifests/
    sources.example.yaml            # Dataset source/license/provenance template
  processed/
    objects_yolo_v1/                # Generated Ultralytics training dataset, ignored by Git
  splits/
    train.txt
    val.txt
    test.txt
  validate_dataset.py
```

## Supported Labels

Object classes must match `schema.md`:

```text
pedestrian, obstacle, bicycle, car, animal, stroller, wheelchair, bollard, scooter, unknown
```

Traffic-light states must match `schema.md`:

```text
red, yellow, green, off, flashing, unknown
```

Hazard types must match `schema.md`:

```text
pothole, step_up, step_down, speed_bump, water, debris, unknown
```

Use `negative` only as a raw data folder name for classification negatives. In annotations, prefer `unknown` when a schema-compatible label is required.

## File Naming

Use stable ASCII names:

```text
<task>_<scene>_<yyyymmdd>_<hhmmss>_<index>.<ext>
```

Examples:

```text
objects_sidewalk_school_20260508_143015_0001.jpg
sidewalk_park_path_20260508_150430_0003.png
hazard_pothole_residential_20260508_161200_0002.jpg
traffic_light_red_crossing_20260508_170015_0004.jpg
```

Recommended image extensions are `.jpg`, `.jpeg`, and `.png`.

## Collection Rules

Collect images from the camera height and angle expected on the cart. Include different lighting conditions, weather, road textures, shadows, and crowded sidewalk scenes.

For each important class, collect both positive and negative samples. Negative samples are especially important for traffic lights and hazards because false braking is costly.

Avoid collecting personally identifiable faces where possible. If faces, license plates, or private addresses are visible, blur or crop them before sharing the dataset outside the local development machine.

## Annotation Notes

Object detection annotations should use pixel coordinates in `[x1, y1, x2, y2]` format, matching `schema.md`.

Segmentation masks should have the same width and height as the source image. Use non-zero pixels for drivable area and zero pixels for non-drivable area.

The validation script accepts JSON annotations with flexible keys, but these fields are recommended:

```json
{
  "image": "raw/objects/images/objects_sidewalk_school_20260508_143015_0001.jpg",
  "objects": [
    {"cls": "pedestrian", "bbox": [120, 80, 180, 260]}
  ]
}
```

```json
{
  "image": "raw/hazard/pothole/hazard_pothole_residential_20260508_161200_0002.jpg",
  "hazard": {"type": "pothole"}
}
```

```json
{
  "image": "raw/traffic_light/red/traffic_light_red_crossing_20260508_170015_0004.jpg",
  "traffic_light": {"state": "red"}
}
```

## Validation

Run from the repository root:

```powershell
conda activate smartcart-ai
python data/validate_dataset.py
```

Use strict mode when the first real dataset has been collected:

```powershell
python data/validate_dataset.py --strict
```

## Import Toolchain

The helper scripts live in `tools/` and should be run from the repository root.

Audit image readability, duplicates, and image-size distribution:

```powershell
python tools/audit_images.py --root data/raw
```

Create train/val/test split files from readable images under `data/raw`:

```powershell
python tools/make_splits.py --data-root data --ratios 0.7,0.2,0.1 --seed 42
```

Import a YOLO object-detection dataset into this repository layout:

```powershell
python tools/import_yolo_dataset.py `
  --images-dir data/external/example/images/train `
  --labels-dir data/external/example/labels/train `
  --names-file data/external/example/names.txt `
  --class-map data/external/example/class_map.json `
  --prefix example_train
```

The YOLO class map is a JSON object from source classes or source class indices to our schema classes:

```json
{
  "person": "pedestrian",
  "0": "pedestrian",
  "traffic cone": "obstacle"
}
```

For Roboflow-style YOLO datasets that already contain `data.yaml`, use the newer generic import flow. First audit the dataset:

```powershell
python tools/audit_yolo_dataset.py --root data/external/roboflow_pedestrian_v1 --class-map person=pedestrian
python tools/audit_yolo_dataset.py --root data/external/roboflow_vehicle_v1 --class-map bike=bicycle --class-map car=car
python tools/audit_yolo_dataset.py --root data/external/roboflow_sidewalk_v1 --class-map sidewalk=sidewalk
```

If the dataset only has `train/`, create a deterministic train/valid/test copy:

```powershell
python tools/split_yolo_dataset.py `
  --source-root data/external/roboflow_pedestrian_v1 `
  --output-root data/external/roboflow_pedestrian_v1_split `
  --ratios 0.7,0.2,0.1 `
  --seed 42
```

Then import mapped object-detection labels:

```powershell
python tools/import_perception_yolo_dataset.py `
  --dataset-root data/external/roboflow_pedestrian_v1_split `
  --task objects `
  --class-map person=pedestrian `
  --prefix roboflow_pedestrian_v1

python tools/import_perception_yolo_dataset.py `
  --dataset-root data/external/roboflow_vehicle_v1_split `
  --task objects `
  --class-map bike=bicycle `
  --class-map car=car `
  --prefix roboflow_vehicle_v1
```

For YOLO segmentation labels, import binary masks for the lane/sidewalk segmentation task:

```powershell
python tools/import_perception_yolo_dataset.py `
  --dataset-root data/external/roboflow_sidewalk_v1_split `
  --task laneseg `
  --class-map sidewalk=sidewalk `
  --prefix roboflow_sidewalk_v1
```

After importing object datasets, export the combined project annotations back to a YOLO training layout:

```powershell
python tools/export_objects_yolo_dataset.py --overwrite
```

This generated dataset is intentionally placed under `data/processed/` and ignored by Git.

Extract frames from locally recorded video:

```powershell
python tools/extract_video_frames.py `
  --video data/external/sidewalk_run_01.mp4 `
  --output-dir data/raw/sidewalk/images `
  --every-n-frames 15 `
  --prefix sidewalk_run_01
```

Before importing a real external dataset, copy `data/manifests/sources.example.yaml` to `data/manifests/sources.yaml` and fill in the real source URL, license, local path, and class mapping.
