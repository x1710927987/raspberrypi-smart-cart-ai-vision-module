from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from perception.model_inference.manifest import ModelManifest
from perception.preprocessing import PreprocessResult


class InferenceBackend(Protocol):
    def predict(self, frame: np.ndarray, preprocess_result: PreprocessResult | None, manifest: ModelManifest) -> Any:
        ...


class FixedPredictionBackend:
    """Small backend for tests, demos, and wiring a pipeline before real model runtime exists."""

    def __init__(self, prediction: Any):
        self.prediction = prediction
        self.calls = 0

    def predict(self, frame: np.ndarray, preprocess_result: PreprocessResult | None, manifest: ModelManifest) -> Any:
        self.calls += 1
        return self.prediction


class UnavailableBackend:
    def __init__(self, backend_name: str, install_hint: str):
        self.backend_name = backend_name
        self.install_hint = install_hint

    def predict(self, frame: np.ndarray, preprocess_result: PreprocessResult | None, manifest: ModelManifest) -> Any:
        raise RuntimeError(f"{self.backend_name} backend is not available. {self.install_hint}")


class UltralyticsBackend:
    """Runtime backend for YOLO `.pt` artifacts loaded through Ultralytics."""

    def __init__(self, model_path: str | None = None, *, confidence: float | None = None, device: str | None = None, yolo_class: Any | None = None):
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self._yolo_class = yolo_class
        self._model: Any | None = None

    def predict(self, frame: np.ndarray, preprocess_result: PreprocessResult | None, manifest: ModelManifest) -> Any:
        model = self._load_model(manifest)
        kwargs: dict[str, Any] = {
            "verbose": False,
            "conf": self.confidence if self.confidence is not None else manifest.confidence_threshold,
        }
        if self.device:
            kwargs["device"] = self.device
        results = model.predict(frame, **kwargs)
        if manifest.task == "laneseg":
            return _ultralytics_results_to_laneseg(results, manifest)
        return _ultralytics_results_to_detections(results, manifest)

    def _load_model(self, manifest: ModelManifest) -> Any:
        if self._model is None:
            yolo_class = self._yolo_class or _import_yolo()
            self._model = yolo_class(self.model_path or str(manifest.artifact_path))
        return self._model


def _import_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics backend is not available. Install ultralytics in the active environment.") from exc
    return YOLO


def _ultralytics_results_to_detections(results: Any, manifest: ModelManifest) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for result in _as_result_list(results):
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for index in range(_box_count(boxes)):
            class_id = int(round(_item_value(getattr(boxes, "cls"), index)))
            conf = float(_item_value(getattr(boxes, "conf"), index))
            bbox = _sequence_values(_item_row(getattr(boxes, "xyxy"), index))
            detections.append(
                {
                    "class_id": class_id,
                    "label": manifest.class_name(class_id),
                    "bbox": bbox,
                    "confidence": conf,
                }
            )
    return detections[: manifest.max_detections]


def _ultralytics_results_to_laneseg(results: Any, manifest: ModelManifest) -> dict[str, Any] | None:
    mask_class = str(manifest.postprocessing.get("mask_class", "")).strip()
    mask_id = int(manifest.postprocessing.get("mask_id", 1))
    best_conf = -1.0
    best_class_id: int | None = None
    best_bbox: list[float] | None = None
    for result in _as_result_list(results):
        if getattr(result, "masks", None) is None:
            continue
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for index in range(_box_count(boxes)):
            class_id = int(round(_item_value(getattr(boxes, "cls"), index)))
            label = manifest.class_name(class_id)
            if mask_class and label != mask_class:
                continue
            conf = float(_item_value(getattr(boxes, "conf"), index))
            if conf > best_conf:
                best_conf = conf
                best_class_id = class_id
                best_bbox = _sequence_values(_item_row(getattr(boxes, "xyxy"), index))
    if best_class_id is None:
        return None
    return {
        "mask_id": mask_id,
        "class_id": best_class_id,
        "label": manifest.class_name(best_class_id),
        "confidence": best_conf,
        "bbox": best_bbox,
    }


def _as_result_list(results: Any) -> list[Any]:
    if results is None:
        return []
    if isinstance(results, list):
        return results
    return list(results)


def _box_count(boxes: Any) -> int:
    cls_values = getattr(boxes, "cls")
    try:
        return len(cls_values)
    except TypeError:
        return int(getattr(cls_values, "shape", [0])[0])


def _item_value(values: Any, index: int) -> float:
    item = values[index]
    if hasattr(item, "item"):
        return float(item.item())
    return float(item)


def _item_row(values: Any, index: int) -> Any:
    return values[index]


def _sequence_values(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]
