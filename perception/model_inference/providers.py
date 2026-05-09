from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np

from perception.detection import DetectionConfig, ModelDetection, postprocess_detections
from perception.model_inference.backends import InferenceBackend
from perception.model_inference.manifest import ModelManifest, load_model_manifest
from perception.preprocessing import PreprocessResult, validate_frame
from perception.runtime import Hazard, LaneSeg, ObjectBBox, TrafficLight


class ManifestObjectDetector:
    def __init__(self, manifest: ModelManifest, backend: InferenceBackend):
        _ensure_task(manifest, "objects")
        self.manifest = manifest
        self.backend = backend

    def detect(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> list[ObjectBBox]:
        validate_frame(frame)
        raw = self.backend.predict(frame, preprocess_result, self.manifest)
        detections = [_coerce_detection(item, self.manifest) for item in _as_iterable(raw)]
        return postprocess_detections(
            detections,
            config=DetectionConfig(
                conf_threshold=self.manifest.confidence_threshold,
                class_map=self.manifest.schema_mapping,
                keep_unknown=True,
            ),
            preprocess_result=preprocess_result,
            bbox_space=self.manifest.bbox_space,
        )[: self.manifest.max_detections]


class ManifestTrafficLightClassifier:
    def __init__(self, manifest: ModelManifest, backend: InferenceBackend):
        _ensure_task(manifest, "traffic_light")
        self.manifest = manifest
        self.backend = backend

    def detect(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> Optional[TrafficLight]:
        validate_frame(frame)
        raw = self.backend.predict(frame, preprocess_result, self.manifest)
        item = _best_item(raw)
        if item is None:
            return None
        label, conf = _label_and_conf(item)
        if conf < self.manifest.confidence_threshold:
            return None
        return TrafficLight(str(self.manifest.map_label(label)), round(conf, 4))

    def classify(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> Optional[TrafficLight]:
        return self.detect(frame, preprocess_result)

    def predict(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> Optional[TrafficLight]:
        return self.detect(frame, preprocess_result)


class ManifestHazardDetector:
    def __init__(self, manifest: ModelManifest, backend: InferenceBackend):
        _ensure_task(manifest, "hazard")
        self.manifest = manifest
        self.backend = backend

    def detect(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> Optional[Hazard]:
        validate_frame(frame)
        raw = self.backend.predict(frame, preprocess_result, self.manifest)
        item = _best_item(raw)
        if item is None:
            return None
        label, conf = _label_and_conf(item)
        if conf < self.manifest.confidence_threshold:
            return None
        return Hazard(str(self.manifest.map_label(label)), round(conf, 4))

    def classify(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> Optional[Hazard]:
        return self.detect(frame, preprocess_result)

    def predict(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> Optional[Hazard]:
        return self.detect(frame, preprocess_result)


class ManifestLaneSegmenter:
    def __init__(self, manifest: ModelManifest, backend: InferenceBackend):
        _ensure_task(manifest, "laneseg")
        self.manifest = manifest
        self.backend = backend

    def segment(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> Optional[LaneSeg]:
        validate_frame(frame)
        raw = self.backend.predict(frame, preprocess_result, self.manifest)
        if raw is None:
            return None
        if isinstance(raw, LaneSeg):
            return raw if raw.conf >= self.manifest.confidence_threshold else None
        if not isinstance(raw, Mapping):
            raise ValueError("laneseg backend output must be LaneSeg or mapping")
        conf = float(raw.get("conf", raw.get("confidence", 0.0)))
        if conf < self.manifest.confidence_threshold:
            return None
        return LaneSeg(int(raw.get("mask_id", 0)), round(conf, 4))

    def detect(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> Optional[LaneSeg]:
        return self.segment(frame, preprocess_result)

    def predict(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> Optional[LaneSeg]:
        return self.segment(frame, preprocess_result)


def build_provider_from_manifest(path_or_manifest: str | Path | ModelManifest, backend: InferenceBackend):
    manifest = load_model_manifest(path_or_manifest) if isinstance(path_or_manifest, (str, Path)) else path_or_manifest
    if manifest.task == "objects":
        return ManifestObjectDetector(manifest, backend)
    if manifest.task == "traffic_light":
        return ManifestTrafficLightClassifier(manifest, backend)
    if manifest.task == "hazard":
        return ManifestHazardDetector(manifest, backend)
    if manifest.task == "laneseg":
        return ManifestLaneSegmenter(manifest, backend)
    raise ValueError(f"unsupported model task: {manifest.task!r}")


def _coerce_detection(item: Any, manifest: ModelManifest) -> ModelDetection:
    if isinstance(item, ModelDetection):
        return item
    if not isinstance(item, Mapping):
        raise ValueError("object backend output entries must be ModelDetection or mapping")
    label_value = item.get("label", item.get("cls", item.get("class", item.get("class_id"))))
    if label_value is None:
        raise ValueError("object backend output is missing label/class_id")
    bbox = item.get("bbox")
    if bbox is None:
        raise ValueError("object backend output is missing bbox")
    conf = item.get("conf", item.get("confidence", item.get("score", 0.0)))
    label = manifest.class_name(label_value)
    return ModelDetection(str(label), [float(value) for value in bbox], float(conf))


def _as_iterable(raw: Any) -> Iterable[Any]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        if "detections" in raw:
            value = raw["detections"]
            if not isinstance(value, Iterable):
                raise ValueError("detections must be iterable")
            return value
        return [raw]
    if isinstance(raw, (str, bytes)):
        raise ValueError("backend output must not be text")
    return raw


def _best_item(raw: Any) -> Optional[Any]:
    items = list(_as_iterable(raw))
    if not items:
        return None
    return max(items, key=lambda item: _label_and_conf(item)[1])


def _label_and_conf(item: Any) -> tuple[int | str, float]:
    if not isinstance(item, Mapping):
        raise ValueError("classification backend output entries must be mappings")
    label = item.get("label", item.get("state", item.get("type", item.get("class_id"))))
    if label is None:
        raise ValueError("classification backend output is missing label/state/type/class_id")
    conf = item.get("conf", item.get("confidence", item.get("score", 0.0)))
    return label, float(conf)


def _ensure_task(manifest: ModelManifest, task: str) -> None:
    if manifest.task != task:
        raise ValueError(f"manifest task must be {task!r}, got {manifest.task!r}")
