from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.model_inference.manifest import FORMATS, TASKS, ModelManifest, load_model_manifest


MODEL_ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*_v[0-9]+$")


@dataclass
class ManifestCheckResult:
    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def check_manifest_file(path: str | Path, *, expected_task: str | None = None, require_artifact: bool = False) -> ManifestCheckResult:
    manifest_path = Path(path)
    result = ManifestCheckResult(manifest_path)
    try:
        manifest = load_model_manifest(manifest_path, require_artifact=require_artifact)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        result.errors.append(str(exc))
        return result

    _check_basic_manifest(manifest, result, expected_task=expected_task)
    _check_artifact(manifest, result)
    _check_preprocessing(manifest.data.get("preprocessing", {}), result)
    _check_postprocessing(manifest.data.get("postprocessing", {}), result)
    _check_class_mapping(manifest, result)
    _check_backend_sections(manifest, result)
    return result


def print_result(result: ManifestCheckResult) -> None:
    print(f"manifest={result.path}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    for error in result.errors:
        print(f"error: {error}")
    print("status=ok" if result.ok else "status=failed")


def _check_basic_manifest(manifest: ModelManifest, result: ManifestCheckResult, *, expected_task: str | None) -> None:
    if expected_task is not None and expected_task not in TASKS:
        result.errors.append(f"unknown expected task: {expected_task!r}")
    if expected_task is not None and manifest.task != expected_task:
        result.errors.append(f"manifest task {manifest.task!r} does not match expected task {expected_task!r}")
    if not MODEL_ID_RE.match(manifest.model_id):
        result.warnings.append("model_id should use lowercase ASCII words and end with _v<version>")
    status = str(manifest.data.get("status", "")).strip().lower()
    if status == "example":
        result.warnings.append("manifest status is example; copy it before registering a real trained model")


def _check_artifact(manifest: ModelManifest, result: ManifestCheckResult) -> None:
    artifact = _as_mapping(manifest.data.get("artifact", {}), "artifact", result)
    if not artifact:
        return
    artifact_format = manifest.artifact_format
    if artifact_format not in FORMATS:
        result.errors.append(f"unsupported artifact format: {artifact_format!r}")
    suffix = manifest.artifact_path.suffix.lower().lstrip(".")
    if artifact_format != "openvino" and suffix and suffix != artifact_format:
        result.errors.append(f"artifact path suffix .{suffix} does not match format {artifact_format!r}")
    if artifact_format == "openvino" and suffix not in {"xml", "bin"}:
        result.warnings.append("OpenVINO artifacts usually register the .xml model file")


def _check_preprocessing(value: Any, result: ManifestCheckResult) -> None:
    preprocessing = _as_mapping(value, "preprocessing", result)
    if not preprocessing:
        result.warnings.append("missing preprocessing section")
        return
    resize = _as_mapping(preprocessing.get("resize", {}), "preprocessing.resize", result)
    if resize:
        for key in ("width", "height"):
            raw = resize.get(key)
            if not isinstance(raw, int) or raw <= 0:
                result.errors.append(f"preprocessing.resize.{key} must be a positive integer")
    normalize = _as_mapping(preprocessing.get("normalize", {}), "preprocessing.normalize", result)
    if normalize:
        scale = normalize.get("scale")
        if not isinstance(scale, (int, float)) or float(scale) <= 0.0:
            result.errors.append("preprocessing.normalize.scale must be a positive number")


def _check_postprocessing(value: Any, result: ManifestCheckResult) -> None:
    postprocessing = _as_mapping(value, "postprocessing", result)
    if not postprocessing:
        result.warnings.append("missing postprocessing section")
        return
    nms = postprocessing.get("nms_iou_threshold")
    if nms is not None and (not isinstance(nms, (int, float)) or not 0.0 <= float(nms) <= 1.0):
        result.errors.append("postprocessing.nms_iou_threshold must be in [0.0, 1.0]")


def _check_class_mapping(manifest: ModelManifest, result: ManifestCheckResult) -> None:
    classes = manifest.model_classes
    mapping = manifest.schema_mapping
    if manifest.task != "laneseg" and not classes:
        result.errors.append("model_classes must not be empty for classification or detection tasks")
    for class_name in classes:
        if class_name not in mapping and _normalize_label(class_name) not in mapping:
            result.errors.append(f"schema_mapping is missing model class {class_name!r}")
    for source_label in mapping:
        if source_label.isdigit():
            continue
        if source_label not in classes and _normalize_label(source_label) not in {_normalize_label(item) for item in classes}:
            result.warnings.append(f"schema_mapping contains source label not present in model_classes: {source_label!r}")


def _check_backend_sections(manifest: ModelManifest, result: ManifestCheckResult) -> None:
    artifact_format = manifest.artifact_format
    if artifact_format == "onnx":
        onnx = _as_mapping(manifest.data.get("onnx", {}), "onnx", result)
        _check_tensor_list(onnx.get("input_tensors"), "onnx.input_tensors", result)
        _check_tensor_list(onnx.get("output_tensors"), "onnx.output_tensors", result)
    elif artifact_format == "tflite":
        tflite = _as_mapping(manifest.data.get("tflite", {}), "tflite", result)
        if not tflite.get("enabled", False):
            result.errors.append("tflite.enabled must be true when artifact.format is tflite")


def _check_tensor_list(value: Any, field_name: str, result: ManifestCheckResult) -> None:
    if not isinstance(value, list) or not value:
        result.errors.append(f"{field_name} must be a non-empty list")
        return
    for index, item in enumerate(value):
        tensor = _as_mapping(item, f"{field_name}[{index}]", result)
        if not tensor:
            continue
        if not tensor.get("name"):
            result.errors.append(f"{field_name}[{index}].name is required")
        shape = tensor.get("shape")
        if not isinstance(shape, list) or not shape:
            result.errors.append(f"{field_name}[{index}].shape must be a non-empty list")
        if not tensor.get("dtype"):
            result.errors.append(f"{field_name}[{index}].dtype is required")


def _as_mapping(value: Any, field_name: str, result: ManifestCheckResult) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        result.errors.append(f"{field_name} must be an object")
        return {}
    return value


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("-", "_").replace(" ", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate smart-cart model manifest files.")
    parser.add_argument("manifests", nargs="+", type=Path, help="Manifest JSON files to check.")
    parser.add_argument("--task", choices=sorted(TASKS), help="Expected manifest task.")
    parser.add_argument("--require-artifact", action="store_true", help="Fail when the registered model file does not exist.")
    args = parser.parse_args()

    results = [
        check_manifest_file(path, expected_task=args.task, require_artifact=args.require_artifact)
        for path in args.manifests
    ]
    for result in results:
        print_result(result)
    print(f"checked={len(results)}")
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
