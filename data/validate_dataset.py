from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
OBJECT_CLASSES = {
    "pedestrian",
    "obstacle",
    "roadblock",
    "bicycle",
    "car",
    "animal",
    "stroller",
    "wheelchair",
    "bollard",
    "scooter",
    "unknown",
}
TRAFFIC_LIGHT_STATES = {"red", "yellow", "green", "off", "flashing", "unknown"}
HAZARD_TYPES = {"pothole", "curb", "step_up", "step_down", "speed_bump", "water", "debris", "unknown"}

REQUIRED_DIRS = [
    "raw/objects/images",
    "raw/sidewalk/images",
    "raw/traffic_light/red",
    "raw/traffic_light/yellow",
    "raw/traffic_light/green",
    "raw/traffic_light/negative",
    "raw/hazard/pothole",
    "raw/hazard/curb",
    "raw/hazard/step_up",
    "raw/hazard/step_down",
    "raw/hazard/speed_bump",
    "raw/hazard/water",
    "raw/hazard/debris",
    "raw/hazard/negative",
    "annotations/objects",
    "annotations/laneseg/masks",
    "annotations/traffic_light",
    "annotations/hazard",
    "manifests",
    "splits",
]


@dataclass
class ValidationReport:
    images_checked: int = 0
    annotations_checked: int = 0
    warnings: list[str] | None = None
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        self.warnings = [] if self.warnings is None else self.warnings
        self.errors = [] if self.errors is None else self.errors

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_dataset(root: Path, *, strict: bool = False) -> ValidationReport:
    data_root = root.resolve()
    report = ValidationReport()
    _check_required_dirs(data_root, report)
    _check_images(data_root, report)
    _check_annotations(data_root, report)
    _check_splits(data_root, report)
    if strict and report.images_checked == 0:
        report.error("strict mode requires at least one readable image under data/raw")
    return report


def _check_required_dirs(data_root: Path, report: ValidationReport) -> None:
    for relative in REQUIRED_DIRS:
        path = data_root / relative
        if not path.is_dir():
            report.error(f"missing required directory: {path}")


def _check_images(data_root: Path, report: ValidationReport) -> None:
    raw_root = data_root / "raw"
    if not raw_root.exists():
        return
    for image_path in sorted(raw_root.rglob("*")):
        if not image_path.is_file() or image_path.name == ".gitkeep":
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            report.warn(f"ignored non-image file under raw data: {image_path}")
            continue
        image = _read_image(image_path)
        if image is None or image.size == 0:
            report.error(f"unreadable image: {image_path}")
            continue
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            report.error(f"invalid image dimensions: {image_path}")
            continue
        report.images_checked += 1


def _check_annotations(data_root: Path, report: ValidationReport) -> None:
    annotations_root = data_root / "annotations"
    if not annotations_root.exists():
        return
    for json_path in sorted(annotations_root.rglob("*.json")):
        report.annotations_checked += 1
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.error(f"invalid json annotation {json_path}: {exc}")
            continue
        if not isinstance(payload, dict):
            report.error(f"annotation root must be a JSON object: {json_path}")
            continue
        _check_referenced_paths(data_root, json_path, payload, report)
        _check_object_annotations(json_path, payload, report)
        _check_traffic_light_annotations(json_path, payload, report)
        _check_hazard_annotations(json_path, payload, report)


def _check_referenced_paths(data_root: Path, json_path: Path, payload: dict[str, Any], report: ValidationReport) -> None:
    for key in ("image", "mask"):
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            report.error(f"{json_path}: {key} must be a string path")
            continue
        path = data_root / value
        if not path.exists():
            report.error(f"{json_path}: referenced {key} does not exist: {path}")


def _check_object_annotations(json_path: Path, payload: dict[str, Any], report: ValidationReport) -> None:
    objects = payload.get("objects")
    if objects is None:
        return
    if not isinstance(objects, list):
        report.error(f"{json_path}: objects must be a list")
        return
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            report.error(f"{json_path}: objects[{index}] must be an object")
            continue
        cls = obj.get("cls")
        if cls not in OBJECT_CLASSES:
            report.error(f"{json_path}: unsupported objects[{index}].cls: {cls!r}")
        bbox = obj.get("bbox")
        if not _is_valid_bbox(bbox):
            report.error(f"{json_path}: invalid objects[{index}].bbox: {bbox!r}")


def _check_traffic_light_annotations(json_path: Path, payload: dict[str, Any], report: ValidationReport) -> None:
    state = _nested_label(payload, "traffic_light", "state")
    if state is None:
        state = payload.get("traffic_light_state")
    if state is not None and state not in TRAFFIC_LIGHT_STATES:
        report.error(f"{json_path}: unsupported traffic light state: {state!r}")


def _check_hazard_annotations(json_path: Path, payload: dict[str, Any], report: ValidationReport) -> None:
    hazard_type = _nested_label(payload, "hazard", "type")
    if hazard_type is None:
        hazard_type = payload.get("hazard_type")
    if hazard_type is not None and hazard_type not in HAZARD_TYPES:
        report.error(f"{json_path}: unsupported hazard type: {hazard_type!r}")


def _check_splits(data_root: Path, report: ValidationReport) -> None:
    splits_root = data_root / "splits"
    if not splits_root.exists():
        return
    for split_name in ("train.txt", "val.txt", "test.txt"):
        split_path = splits_root / split_name
        if not split_path.exists():
            report.warn(f"missing optional split file: {split_path}")
            continue
        for line_no, raw_line in enumerate(split_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not (data_root / line).exists():
                report.error(f"{split_path}:{line_no}: split entry does not exist: {line}")


def _nested_label(payload: dict[str, Any], parent_key: str, label_key: str) -> Any:
    parent = payload.get(parent_key)
    if isinstance(parent, dict):
        return parent.get(label_key)
    return None


def _is_valid_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return min(x1, y1, x2, y2) >= 0.0 and x2 > x1 and y2 > y1


def _print_report(report: ValidationReport) -> None:
    print(f"images_checked={report.images_checked}")
    print(f"annotations_checked={report.annotations_checked}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    print("status=ok" if report.ok else "status=failed")


def _read_image(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate smart-cart perception dataset structure and annotations.")
    parser.add_argument("--root", default=Path(__file__).resolve().parent, type=Path, help="Path to the data directory.")
    parser.add_argument("--strict", action="store_true", help="Fail when the dataset contains no readable images.")
    args = parser.parse_args()
    report = validate_dataset(args.root, strict=args.strict)
    _print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
