from pathlib import Path

import numpy as np
import pytest

from perception.camera_pipeline import PerceptionPipeline, PipelineConfig
from perception.model_inference import (
    FixedPredictionBackend,
    ManifestHazardDetector,
    ManifestLaneSegmenter,
    ManifestObjectDetector,
    ManifestTrafficLightClassifier,
    ModelManifest,
    UltralyticsBackend,
    build_provider_from_manifest,
    load_model_manifest,
)
from perception.preprocessing import PreprocessConfig
from perception.runtime import Hazard, LaneSeg, ObjectBBox, TrafficLight


def test_load_model_manifest_example_and_validate_mapping():
    manifest = load_model_manifest("models/model_manifest.example.json")
    assert manifest.model_id == "smartcart_objects_yolov8n_roboflow_onnx_v1"
    assert manifest.task == "objects"
    assert manifest.artifact_format == "onnx"
    assert manifest.map_label("traffic cone") == "obstacle"
    assert manifest.map_label("traffic_cone") == "obstacle"
    assert manifest.artifact_path == Path.cwd() / "models" / "weights" / "smartcart_objects_yolov8n_roboflow_onnx_v1.onnx"
    with pytest.raises(FileNotFoundError):
        load_model_manifest("models/model_manifest.example.json", require_artifact=True)


def test_manifest_object_detector_maps_backend_predictions_into_schema():
    manifest = _manifest(task="objects", classes=["person", "traffic cone"], mapping={"person": "pedestrian", "traffic cone": "obstacle"})
    backend = FixedPredictionBackend(
        {
            "detections": [
                {"class_id": 0, "bbox": [10, 20, 110, 220], "confidence": 0.91},
                {"label": "traffic cone", "bbox": [200, 210, 250, 300], "score": 0.72},
                {"label": "person", "bbox": [1, 2, 3, 4], "confidence": 0.1},
            ]
        }
    )
    detector = ManifestObjectDetector(manifest, backend)
    objects = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert backend.calls == 1
    assert objects == [
        ObjectBBox("pedestrian", [10.0, 20.0, 110.0, 220.0], 0.91),
        ObjectBBox("obstacle", [200.0, 210.0, 250.0, 300.0], 0.72),
    ]


def test_manifest_providers_plug_into_perception_pipeline():
    object_manifest = _manifest(task="objects", classes=["person"], mapping={"person": "pedestrian"})
    lane_manifest = _manifest(task="laneseg", postprocessing={"confidence_threshold": 0.4})
    traffic_manifest = _manifest(task="traffic_light", classes=["red", "green"], mapping={"red": "red", "green": "green"})
    hazard_manifest = _manifest(task="hazard", classes=["pothole"], mapping={"pothole": "pothole"})

    pipeline = PerceptionPipeline(
        detector=build_provider_from_manifest(
            object_manifest,
            FixedPredictionBackend([{"label": "person", "bbox": [5, 10, 45, 80], "conf": 0.8}]),
        ),
        laneseg_provider=ManifestLaneSegmenter(lane_manifest, FixedPredictionBackend({"mask_id": 2, "conf": 0.88})),
        traffic_light_provider=ManifestTrafficLightClassifier(traffic_manifest, FixedPredictionBackend([{"label": "green", "conf": 0.93}])),
        hazard_provider=ManifestHazardDetector(hazard_manifest, FixedPredictionBackend({"label": "pothole", "conf": 0.81})),
        config=PipelineConfig(preprocess=PreprocessConfig(target_size=(64, 64))),
    )
    output = pipeline.process_frame(np.zeros((128, 128, 3), dtype=np.uint8), timestamp=1720000123.456)
    assert output.objects == [ObjectBBox("pedestrian", [5.0, 10.0, 45.0, 80.0], 0.8)]
    assert output.laneseg == LaneSeg(2, 0.88)
    assert output.traffic_light == TrafficLight("green", 0.93)
    assert output.hazard == Hazard("pothole", 0.81)


def test_ultralytics_backend_converts_yolo_boxes_to_detections():
    manifest = _manifest(task="traffic_light", classes=["green", "red", "yellow"], mapping={"green": "green", "red": "red", "yellow": "yellow"})
    backend = UltralyticsBackend(yolo_class=_fake_yolo_class(class_ids=[1, 2], confidences=[0.82, 0.41], boxes=[[10, 20, 30, 40], [50, 60, 70, 80]]))
    detections = backend.predict(np.zeros((96, 96, 3), dtype=np.uint8), None, manifest)
    assert detections == [
        {"class_id": 1, "label": "red", "bbox": [10.0, 20.0, 30.0, 40.0], "confidence": 0.82},
        {"class_id": 2, "label": "yellow", "bbox": [50.0, 60.0, 70.0, 80.0], "confidence": 0.41},
    ]


def test_ultralytics_backend_plugs_into_traffic_light_provider():
    manifest = _manifest(task="traffic_light", classes=["green", "red", "yellow"], mapping={"green": "green", "red": "red", "yellow": "yellow"})
    backend = UltralyticsBackend(yolo_class=_fake_yolo_class(class_ids=[0, 1], confidences=[0.51, 0.86], boxes=[[1, 2, 3, 4], [5, 6, 7, 8]]))
    provider = ManifestTrafficLightClassifier(manifest, backend)
    assert provider.detect(np.zeros((96, 96, 3), dtype=np.uint8)) == TrafficLight("red", 0.86, [5.0, 6.0, 7.0, 8.0])


def test_ultralytics_backend_converts_segmentation_result_to_laneseg():
    manifest = _manifest(
        task="laneseg",
        classes=["sidewalk"],
        postprocessing={"confidence_threshold": 0.35, "mask_id": 7, "mask_class": "sidewalk"},
    )
    backend = UltralyticsBackend(yolo_class=_fake_yolo_class(class_ids=[0], confidences=[0.74], boxes=[[1, 2, 30, 40]], has_masks=True))
    provider = ManifestLaneSegmenter(manifest, backend)
    assert provider.segment(np.zeros((96, 96, 3), dtype=np.uint8)) == LaneSeg(7, 0.74, [1.0, 2.0, 30.0, 40.0])


def test_manifest_validation_rejects_bad_task_and_bad_schema_label():
    bad_task = _manifest(task="classification", validate=False)
    with pytest.raises(ValueError, match="unsupported model task"):
        bad_task.validate()

    bad_label = _manifest(task="hazard", classes=["crack"], mapping={"crack": "crack"}, validate=False)
    with pytest.raises(ValueError, match="not valid for task"):
        bad_label.validate()


def test_manifest_accepts_curb_as_hazard_schema_label():
    manifest = _manifest(task="hazard", classes=["curb"], mapping={"curb": "curb"})
    assert manifest.map_label("curb") == "curb"


def test_manifest_accepts_roadblock_as_object_schema_label():
    manifest = _manifest(task="objects", classes=["cone"], mapping={"cone": "roadblock"})
    assert manifest.map_label("cone") == "roadblock"


def _manifest(*, task, classes=None, mapping=None, postprocessing=None, validate=True):
    payload = {
        "schema_version": "0.1",
        "model_id": f"test_{task}_model",
        "task": task,
        "artifact": {
            "path": f"models/weights/test_{task}.mock",
            "format": "mock",
            "architecture": "fixed",
            "version": "v0",
        },
        "postprocessing": {"confidence_threshold": 0.35, "max_detections": 50},
        "model_classes": classes or [],
        "schema_mapping": mapping or {},
    }
    payload["postprocessing"].update(postprocessing or {})
    manifest = ModelManifest(Path(f"models/training/test_{task}.manifest.json"), payload)
    if validate:
        manifest.validate()
    return manifest


def _fake_yolo_class(*, class_ids, confidences, boxes, has_masks=False):
    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path

        def predict(self, frame, **kwargs):
            return [_FakeResult(class_ids, confidences, boxes, has_masks=has_masks)]

    return FakeYOLO


class _FakeResult:
    def __init__(self, class_ids, confidences, boxes, *, has_masks=False):
        self.boxes = _FakeBoxes(class_ids, confidences, boxes)
        self.masks = _FakeMasks() if has_masks else None


class _FakeBoxes:
    def __init__(self, class_ids, confidences, boxes):
        self.cls = class_ids
        self.conf = confidences
        self.xyxy = boxes


class _FakeMasks:
    data = [[[1]]]
