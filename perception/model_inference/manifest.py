from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


TASKS = {"objects", "traffic_light", "hazard", "laneseg"}
FORMATS = {"pt", "onnx", "tflite", "engine", "openvino", "mock"}
SCHEMA_LABELS = {
    "objects": {
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
    },
    "traffic_light": {"red", "yellow", "green", "off", "flashing", "unknown"},
    "hazard": {"pothole", "step_up", "step_down", "speed_bump", "water", "debris", "unknown"},
    "laneseg": set(),
}


@dataclass(frozen=True)
class ModelManifest:
    path: Path
    data: Mapping[str, Any]

    @property
    def model_id(self) -> str:
        return str(_required(self.data, "model_id"))

    @property
    def task(self) -> str:
        return str(_required(self.data, "task"))

    @property
    def artifact_path(self) -> Path:
        artifact = _mapping(_required(self.data, "artifact"), "artifact")
        raw_path = Path(str(_required(artifact, "path")))
        if raw_path.is_absolute():
            return raw_path
        return self.repo_root / raw_path

    @property
    def artifact_format(self) -> str:
        artifact = _mapping(_required(self.data, "artifact"), "artifact")
        return str(_required(artifact, "format")).lower()

    @property
    def repo_root(self) -> Path:
        for candidate in (self.path.resolve().parent, *self.path.resolve().parents):
            if (candidate / "schema.md").exists() and (candidate / "models").exists():
                return candidate
        return Path.cwd()

    @property
    def model_classes(self) -> list[str]:
        classes = self.data.get("model_classes", [])
        if not isinstance(classes, list):
            raise ValueError("model_classes must be a list")
        return [str(item) for item in classes]

    @property
    def schema_mapping(self) -> Mapping[str, str]:
        mapping = self.data.get("schema_mapping", {})
        if not isinstance(mapping, Mapping):
            raise ValueError("schema_mapping must be an object")
        normalized: dict[str, str] = {}
        for key, value in mapping.items():
            normalized[str(key)] = str(value)
            normalized[_normalize_label(str(key))] = str(value)
        return normalized

    @property
    def postprocessing(self) -> Mapping[str, Any]:
        value = self.data.get("postprocessing", {})
        return _mapping(value, "postprocessing")

    @property
    def confidence_threshold(self) -> float:
        return float(self.postprocessing.get("confidence_threshold", 0.0))

    @property
    def max_detections(self) -> int:
        return int(self.postprocessing.get("max_detections", 50))

    @property
    def bbox_space(self) -> str:
        value = str(self.postprocessing.get("bbox_coordinate_space", "original_image")).strip().lower()
        if value in {"processed", "processed_image", "model_input"}:
            return "processed"
        return "original"

    def class_name(self, class_id: int | str) -> str:
        if isinstance(class_id, str) and not class_id.strip().isdigit():
            return class_id
        index = int(class_id)
        classes = self.model_classes
        if index < 0 or index >= len(classes):
            raise ValueError(f"class index out of range for {self.model_id}: {index}")
        return classes[index]

    def map_label(self, label_or_index: int | str, *, default_unknown: bool = True) -> Optional[str]:
        source_label = self.class_name(label_or_index)
        mapping = self.schema_mapping
        mapped = mapping.get(source_label)
        if mapped is None:
            mapped = mapping.get(_normalize_label(source_label))
        if mapped is None:
            mapped = mapping.get(str(label_or_index))
        if mapped is None:
            return "unknown" if default_unknown else None
        _validate_schema_label(self.task, mapped)
        return mapped

    def validate(self, *, require_artifact: bool = False) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unsupported model task: {self.task!r}")
        if self.artifact_format not in FORMATS:
            raise ValueError(f"unsupported model artifact format: {self.artifact_format!r}")
        if require_artifact and not self.artifact_path.exists():
            raise FileNotFoundError(f"model artifact does not exist: {self.artifact_path}")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("postprocessing.confidence_threshold must be in [0.0, 1.0]")
        if self.max_detections < 0:
            raise ValueError("postprocessing.max_detections must be non-negative")
        for target in self.schema_mapping.values():
            _validate_schema_label(self.task, target)


def load_model_manifest(path: str | Path, *, require_artifact: bool = False) -> ModelManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("model manifest root must be a JSON object")
    manifest = ModelManifest(manifest_path, payload)
    manifest.validate(require_artifact=require_artifact)
    return manifest


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"model manifest missing required field: {key}")
    return mapping[key]


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _validate_schema_label(task: str, label: str) -> None:
    allowed = SCHEMA_LABELS.get(task, set())
    if allowed and label not in allowed:
        raise ValueError(f"label {label!r} is not valid for task {task!r}")


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("-", "_").replace(" ", "_")
