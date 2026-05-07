from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

import numpy as np

from perception.preprocessing import PreprocessResult, validate_frame
from perception.runtime import ObjectBBox

SCHEMA_OBJECT_CLASSES = (
    "pedestrian",
    "obstacle",
    "bicycle",
    "car",
    "animal",
    "stroller",
    "wheelchair",
    "bollard",
    "scooter",
    "unknown",
)
DEFAULT_CLASS_MAP: Dict[str, str] = {
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "car": "car",
    "truck": "car",
    "bus": "car",
    "motorbike": "scooter",
    "motorcycle": "scooter",
    "scooter": "scooter",
    "dog": "animal",
    "cat": "animal",
    "bird": "animal",
    "animal": "animal",
    "stroller": "stroller",
    "wheelchair": "wheelchair",
    "bollard": "bollard",
    "traffic_cone": "obstacle",
    "cone": "obstacle",
    "bench": "obstacle",
    "chair": "obstacle",
    "backpack": "obstacle",
    "suitcase": "obstacle",
    "obstacle": "obstacle",
    "unknown": "unknown",
}


@dataclass(frozen=True)
class ModelDetection:
    label: str
    bbox: List[float]
    conf: float


@dataclass(frozen=True)
class DetectionConfig:
    conf_threshold: float = 0.35
    class_map: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_CLASS_MAP))
    keep_unknown: bool = False
    round_digits: int = 1


class ObjectDetector(Protocol):
    def detect(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> List[ObjectBBox]:
        ...


class EmptyObjectDetector:
    def detect(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> List[ObjectBBox]:
        validate_frame(frame)
        return []


class DummyObjectDetector:
    def __init__(self, detections: Iterable[ModelDetection], *, config: Optional[DetectionConfig] = None, bbox_space: str = "original"):
        self._detections = list(detections)
        self._config = config or DetectionConfig()
        self._bbox_space = bbox_space

    def detect(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> List[ObjectBBox]:
        validate_frame(frame)
        return postprocess_detections(
            self._detections,
            config=self._config,
            preprocess_result=preprocess_result,
            bbox_space=self._bbox_space,
        )


def postprocess_detections(
    detections: Iterable[ModelDetection | Mapping[str, object]],
    *,
    config: Optional[DetectionConfig] = None,
    preprocess_result: Optional[PreprocessResult] = None,
    bbox_space: str = "original",
) -> List[ObjectBBox]:
    cfg = config or DetectionConfig()
    _validate_config(cfg)
    bbox_space_key = bbox_space.strip().lower()
    if bbox_space_key not in {"original", "processed"}:
        raise ValueError("bbox_space must be either 'original' or 'processed'")
    if bbox_space_key == "processed" and preprocess_result is None:
        raise ValueError("preprocess_result is required when bbox_space='processed'")

    objects: List[ObjectBBox] = []
    for detection in detections:
        model_detection = _coerce_model_detection(detection)
        if model_detection.conf < cfg.conf_threshold:
            continue
        schema_class = map_model_class(model_detection.label, class_map=cfg.class_map, keep_unknown=cfg.keep_unknown)
        if schema_class is None:
            continue
        bbox = preprocess_result.bbox_to_original(model_detection.bbox) if bbox_space_key == "processed" else model_detection.bbox
        objects.append(ObjectBBox(schema_class, [round(float(v), cfg.round_digits) for v in bbox], round(float(model_detection.conf), 4)))
    return objects


def map_model_class(label: str, *, class_map: Mapping[str, str] = DEFAULT_CLASS_MAP, keep_unknown: bool = False) -> Optional[str]:
    mapped = class_map.get(normalize_label(label))
    if mapped is None:
        return "unknown" if keep_unknown else None
    if mapped not in SCHEMA_OBJECT_CLASSES:
        raise ValueError(f"mapped class is not supported by schema: {mapped!r}")
    return mapped


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("-", "_").replace(" ", "_")


def _coerce_model_detection(detection: ModelDetection | Mapping[str, object]) -> ModelDetection:
    if isinstance(detection, ModelDetection):
        model_detection = detection
    else:
        try:
            model_detection = ModelDetection(str(detection["label"]), [float(v) for v in detection["bbox"]], float(detection["conf"]))  # type: ignore[index]
        except KeyError as exc:
            raise ValueError(f"detection is missing required key: {exc.args[0]}") from exc
        except TypeError as exc:
            raise ValueError("detection bbox must be an iterable of 4 numeric values") from exc
    _validate_bbox(model_detection.bbox)
    _validate_conf(model_detection.conf)
    if not model_detection.label.strip():
        raise ValueError("detection label must not be empty")
    return model_detection


def _validate_config(config: DetectionConfig) -> None:
    _validate_conf(config.conf_threshold)
    if config.round_digits < 0:
        raise ValueError("round_digits must be non-negative")


def _validate_conf(value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError("confidence must be in [0.0, 1.0]")


def _validate_bbox(bbox: Sequence[float]) -> None:
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly 4 values")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x2 > x1 and y2 > y1")
    if min(x1, y1, x2, y2) < 0.0:
        raise ValueError("bbox must use non-negative pixel coordinates")
