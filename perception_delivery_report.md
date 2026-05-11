# Perception Module Delivery Report

Date: 2026-05-11

This report is the final delivery checklist for the perception module owned by
role A. It summarizes the currently integrated models, manifests, weights,
evaluation artifacts, smoke-test commands, known limitations, and remaining
handoff work.

## Delivery Status

The perception module is ready for current-stage acceptance if no additional
object or hazard categories are required.

Delivered core capabilities:

- Object detection for pedestrian, bicycle, car, scooter, and roadblock.
- Traffic-light recognition for red, yellow, and green.
- Sidewalk / drivable-area recognition.
- Hazard recognition for pothole and curb.
- Fusion of all perception submodules into `PerceptionOutput`.
- Real-model unified pipeline smoke test.
- Dataset import, audit, split, merge, label-fix, data-augmentation,
  evaluation, and error-visualization tooling.

The remaining work is mostly deployment validation, threshold/ROI policy,
performance testing, and handoff to the control-logic module.

## Default Model Inventory

| Module | Default model | Manifest | Weight file | Main output |
| --- | --- | --- | --- | --- |
| objects | `smartcart_objects_yolov8n_combined_v3_pt_v1` | `models/training/smartcart_objects_yolov8n_combined_v3_pt_v1.manifest.json` | `models/weights/smartcart_objects_yolov8n_combined_v3_pt_v1.pt` | `pedestrian`, `bicycle`, `car`, `scooter`, `roadblock` |
| traffic_light | `smartcart_traffic_light_yolov8n_combined_v2_pt_v1` | `models/training/smartcart_traffic_light_yolov8n_combined_v2_pt_v1.manifest.json` | `models/weights/smartcart_traffic_light_yolov8n_combined_v2_pt_v1.pt` | `green`, `red`, `yellow` |
| laneseg | `smartcart_laneseg_yolov8n_seg_roboflow_pt_v1` | `models/training/smartcart_laneseg_yolov8n_seg_roboflow_pt_v1.manifest.json` | `models/weights/smartcart_laneseg_yolov8n_seg_roboflow_pt_v1.pt` | `sidewalk` -> `LaneSeg(mask_id=1)` |
| hazard | `smartcart_hazard_yolov8n_roboflow_pt_v1` | `models/training/smartcart_hazard_yolov8n_roboflow_pt_v1.manifest.json` | `models/weights/smartcart_hazard_yolov8n_roboflow_pt_v1.pt` | `pothole`, `curb` |

Default integration entry points:

- `perception.camera_pipeline.DEFAULT_OBJECTS_MANIFEST`
- `perception.camera_pipeline.DEFAULT_TRAFFIC_LIGHT_MANIFEST`
- `perception.camera_pipeline.DEFAULT_LANESEG_MANIFEST`
- `perception.camera_pipeline.DEFAULT_HAZARD_MANIFEST`
- `tools/run_perception_pipeline_smoke.py`

## Model Metrics

### Objects

Manifest:
`models/training/smartcart_objects_yolov8n_combined_v3_pt_v1.manifest.json`

Dataset version:
`data/external/objects_combined_v3_split`

Training-run metrics from manifest:

| Metric | Value |
| --- | ---: |
| precision | 0.92744 |
| recall | 0.87087 |
| mAP50 | 0.92846 |
| mAP50-95 | 0.70561 |

Independent test-split evaluation:

```text
dataset_root=data/external/objects_combined_v3_split
split=test
total_images=891
correct_images=620
accuracy=0.6958
gt_boxes=2862
predicted_boxes=2862
true_positives=2580
false_positives=282
false_negatives=282
misclassified=1
```

Per-class test metrics:

| Class | Precision | Recall | TP |
| --- | ---: | ---: | ---: |
| bicycle | 0.9032 | 0.8863 | 569 |
| car | 0.9058 | 0.8612 | 856 |
| pedestrian | 0.8586 | 0.9126 | 407 |
| roadblock | 0.9347 | 0.9981 | 530 |
| scooter | 0.8862 | 0.8755 | 218 |

Evaluation artifacts:

- `cache/evaluation/object_detection_v3_test_mistakes.json`
- `cache/evaluation/object_detection_v3_error_report.md`
- `cache/evaluation/object_detection_error_gallery_v3/`
- `cache/evaluation/object_detection_scooter_v3_error_review_template.csv`
- `cache/evaluation/object_detection_scooter_v3_spot_check.md`

Conclusion:
objects v3 is acceptable as the current default model. Remaining scooter errors
are mostly dense parking, occlusion, edge crops, small targets, and
label/matching edge cases. There is no strong systematic wrong-class failure.

### Traffic Light

Manifest:
`models/training/smartcart_traffic_light_yolov8n_combined_v2_pt_v1.manifest.json`

Dataset version:
`data/external/traffic_light_combined_v2_split`

Manifest metrics:

| Metric | Value |
| --- | ---: |
| precision | 0.86599 |
| recall | 0.91106 |
| mAP50 | 0.90945 |
| mAP50-95 | 0.70388 |

Evaluation artifacts:

- `cache/evaluation/traffic_light_combined_v2_test_mistakes.json`
- `cache/evaluation/traffic_light_roboflow_test_mistakes_v2_model.json`
- `cache/evaluation/traffic_light_error_review_report_v2_remaining_final.md`
- `cache/evaluation/traffic_light_error_gallery_v2_remaining/`

Remaining reviewed errors:

```text
total_cases=5
add_similar_data=3
adjust_postprocess=1
ignore_sample=1
```

Conclusion:
traffic-light v2 is acceptable for current-stage delivery. Remaining issues are
small distant lights, low contrast, truncated edge lights, and pedestrian-signal
distractors. These can stay in the backlog.

### LaneSeg

Manifest:
`models/training/smartcart_laneseg_yolov8n_seg_roboflow_pt_v1.manifest.json`

Dataset version:
`data/external/roboflow_sidewalk_v1_split`

Manifest metrics:

| Metric | Value |
| --- | ---: |
| mask precision | 0.96419 |
| mask recall | 0.73723 |
| mask mAP50 | 0.84856 |
| mask mAP50-95 | 0.73539 |
| box mAP50 | 0.84509 |

Conclusion:
laneseg can produce schema-level `LaneSeg(mask_id=1, conf=...)` inside the
unified pipeline. The current runtime does not pass the full segmentation mask
to the control module. If pixel-level path planning is needed, the mask
interface should be extended later.

### Hazard

Manifest:
`models/training/smartcart_hazard_yolov8n_roboflow_pt_v1.manifest.json`

Dataset version:
`data/external/roboflow_hazard_v1_split`

Manifest metrics:

| Metric | Value |
| --- | ---: |
| precision | 0.67611 |
| recall | 0.71703 |
| mAP50 | 0.72687 |
| mAP50-95 | 0.41848 |

Evaluation artifacts:

- `cache/evaluation/hazard_test_mistakes.json`
- `cache/evaluation/hazard_error_gallery/`

Conclusion:
hazard can detect pothole and curb for stage acceptance. Because hazard output
can affect braking, deployment should add ROI filtering, temporal confirmation,
and confidence threshold policy before real vehicle control.

## Unified Pipeline Smoke Test

Required acceptance command:

```powershell
conda activate smartcart-ai
python tools\run_perception_pipeline_smoke.py --device 0
```

CPU-only validation:

```powershell
conda activate smartcart-ai
python tools\run_perception_pipeline_smoke.py --device cpu
```

Output files:

- `cache/evaluation/unified_pipeline_smoke_test.json`
- `cache/evaluation/unified_pipeline_smoke_test.md`

Latest run summary:

```text
status=ok
sample_count=4
modules_invoked=objects, traffic_light, laneseg, hazard
samples_with_objects=2
samples_with_laneseg=2
samples_with_traffic_light=1
samples_with_hazard=2
validated_outputs=4
```

Latest sample summary:

| Sample | objects | laneseg | traffic_light | hazard |
| --- | ---: | --- | --- | --- |
| objects test image | 1 | None | None | None |
| traffic-light test image | 1 | `mask_id=1` | `yellow` | `curb` |
| sidewalk test image | 0 | `mask_id=1` | None | None |
| hazard test image | 0 | None | None | `curb` |

Conclusion:
all four default real models can be invoked by the unified pipeline. Their
outputs can be fused into `PerceptionOutput`, serialized to JSON, read back,
and validated by the runtime schema checks.

## Tooling Inventory

### Data and Training

- `tools/audit_yolo_dataset.py`
- `tools/split_yolo_dataset.py`
- `tools/merge_objects_yolo_datasets.py`
- `tools/merge_hazard_yolo_datasets.py`
- `tools/train_object_detection_yolo.py`
- `tools/train_traffic_light_yolo.py`
- `tools/train_laneseg_yolo.py`
- `tools/train_hazard_yolo.py`

### Model Registration and Validation

- `tools/register_object_detection_model.py`
- `tools/register_traffic_light_model.py`
- `tools/register_laneseg_model.py`
- `tools/register_hazard_model.py`
- `tools/check_model_manifest.py`

### Evaluation and Error Review

- `tools/evaluate_object_detection_model.py`
- `tools/evaluate_traffic_light_model.py`
- `tools/evaluate_hazard_model.py`
- `tools/visualize_object_detection_errors.py`
- `tools/visualize_traffic_light_errors.py`
- `tools/visualize_hazard_errors.py`
- `tools/analyze_object_detection_errors.py`
- `tools/analyze_object_detection_error_review.py`
- `tools/analyze_traffic_light_error_review.py`

### Pipeline Acceptance

- `tools/run_perception_pipeline_smoke.py`

## Code Module Inventory

- `perception/runtime.py`: runtime data structures and `PerceptionOutput`
- `perception/preprocessing.py`: image preprocessing and bbox coordinate mapping
- `perception/detection.py`: object postprocessing and schema mapping
- `perception/traffic_light.py`: traffic-light interface and rule baseline
- `perception/laneseg.py`: sidewalk / drivable-area interface
- `perception/hazard.py`: hazard interface
- `perception/fusion.py`: multi-module result fusion
- `perception/camera_pipeline.py`: unified pipeline and default model entry points
- `perception/model_inference/`: manifest, backend, and provider framework

## Known Limitations

1. Current default model artifacts are `.pt` files. They are good for local
   validation, but Raspberry Pi deployment should export ONNX or TFLite and
   register new manifests.
2. The hazard model only covers `pothole` and `curb`. It does not yet cover
   `step_up`, `step_down`, `speed_bump`, `water`, or `debris`.
3. The objects model covers `pedestrian`, `bicycle`, `car`, `scooter`, and
   `roadblock`. It does not yet cover `animal`, `stroller`, `wheelchair`, or
   `bollard`.
4. LaneSeg currently outputs schema-level `LaneSeg(mask_id, conf)`, not a full
   pixel mask for downstream control.
5. Traffic-light recognition is trained for `red`, `yellow`, and `green`, not
   explicit `off`, `flashing`, or `unknown` states.
6. The unified pipeline is single-frame perception. It does not yet include
   temporal voting, multi-frame stability, ROI filtering, or control safety
   policies.
7. Local GPU/CPU smoke tests pass, but FPS, memory, camera input, and cold-start
   behavior still need target-device validation.
8. Some external datasets still have `license=unknown` in manifests. Dataset
   source and license records should be cleaned up before public release.

## Stage Acceptance Checklist

For a stage demo or handoff, show the following:

1. Run the unified smoke test:

```powershell
conda activate smartcart-ai
python tools\run_perception_pipeline_smoke.py --device 0
```

2. Show smoke-test outputs:

```text
cache/evaluation/unified_pipeline_smoke_test.json
cache/evaluation/unified_pipeline_smoke_test.md
```

3. Show the four default manifests:

```text
models/training/smartcart_objects_yolov8n_combined_v3_pt_v1.manifest.json
models/training/smartcart_traffic_light_yolov8n_combined_v2_pt_v1.manifest.json
models/training/smartcart_laneseg_yolov8n_seg_roboflow_pt_v1.manifest.json
models/training/smartcart_hazard_yolov8n_roboflow_pt_v1.manifest.json
```

4. Show evaluation and error-review artifacts:

```text
cache/evaluation/object_detection_v3_error_report.md
cache/evaluation/traffic_light_error_review_report_v2_remaining_final.md
cache/evaluation/hazard_test_mistakes.json
cache/evaluation/object_detection_error_gallery_v3/
cache/evaluation/traffic_light_error_gallery_v2_remaining/
cache/evaluation/hazard_error_gallery/
```

## Recommended Next Work

Recommended priorities after this delivery:

1. Test FPS, memory, cold-start time, and live camera input on the target device.
2. Export ONNX/TFLite and verify the Raspberry Pi runtime format.
3. Confirm the `PerceptionOutput` to control-command interface with the control
   logic owner.
4. Add ROI filtering, temporal confirmation, and braking thresholds for hazard.
5. Decide whether to collect and train additional classes such as `wheelchair`,
   `stroller`, `bollard`, `animal`, `step_up`, and `step_down` after real-road
   tests.

## Final Assessment

If no additional categories are required for the current milestone, role A's
perception algorithm module is basically complete. The next work is deployment
adaptation, target-device performance validation, control-interface joint
debugging, and final experiment-report preparation.
