from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from perception.model_inference import UltralyticsBackend, load_model_manifest
from yolo_dataset_utils import IMAGE_EXTENSIONS, parse_yolo_rows, read_yolo_config, row_to_pixel_bbox


DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "external" / "roboflow_hazard_v1_split"
DEFAULT_MANIFEST = REPO_ROOT / "models" / "training" / "smartcart_hazard_yolov8n_roboflow_pt_v1.manifest.json"
SCHEMA_TYPES = ("pothole", "curb", "unknown")


@dataclass(frozen=True)
class HazardBox:
    type: str
    bbox: list[float]


@dataclass(frozen=True)
class ImageEvaluation:
    image: str
    gt_hazards: list[str]
    primary_gt: str
    predicted_type: str
    confidence: float
    correct: bool
    gt_boxes: list[HazardBox]
    predicted_bbox: list[float] | None


@dataclass(frozen=True)
class EvaluationResult:
    dataset_root: str
    split: str
    manifest: str
    total_images: int
    evaluated_images: int
    correct_images: int
    accuracy: float
    no_detection: int
    missing_labels: int
    unreadable_images: int
    class_metrics: dict[str, dict[str, float | int]]
    confusion_matrix: dict[str, dict[str, int]]
    mistakes: list[ImageEvaluation]


def evaluate_hazard_model(
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    split: str = "test",
    manifest_path: str | Path = DEFAULT_MANIFEST,
    backend: Any | None = None,
    device: str | None = None,
    limit: int | None = None,
) -> EvaluationResult:
    dataset_root = Path(dataset_root)
    manifest_path = Path(manifest_path)
    config = read_yolo_config(dataset_root / "data.yaml")
    images_dir = dataset_root / split / "images"
    labels_dir = dataset_root / split / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"missing split images directory: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"missing split labels directory: {labels_dir}")

    manifest = load_model_manifest(manifest_path, require_artifact=backend is None)
    predictor = backend or UltralyticsBackend(device=device)
    image_paths = [path for path in sorted(images_dir.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if limit is not None:
        image_paths = image_paths[:limit]

    evaluations: list[ImageEvaluation] = []
    mistakes: list[ImageEvaluation] = []
    confusion = _empty_confusion()
    gt_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    true_positive_counts: Counter[str] = Counter()
    missing_labels = 0
    unreadable_images = 0
    no_detection = 0

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue
        frame = read_image(image_path)
        if frame is None:
            unreadable_images += 1
            continue
        height, width = frame.shape[:2]
        gt_boxes = parse_ground_truth_boxes(label_path, config.names, width=width, height=height)
        if not gt_boxes:
            missing_labels += 1
            continue

        prediction = predict_primary_hazard(frame, manifest=manifest, backend=predictor)
        predicted_type = prediction["type"] if prediction is not None else "unknown"
        confidence = round(float(prediction["confidence"]), 4) if prediction is not None else 0.0
        predicted_bbox = prediction["bbox"] if prediction is not None else None
        if prediction is None:
            no_detection += 1

        gt_hazards = _unique_preserve_order(box.type for box in gt_boxes)
        primary_gt = choose_primary_gt(gt_boxes)
        correct = predicted_type in set(gt_hazards)

        for hazard_type in set(gt_hazards):
            gt_counts[hazard_type] += 1
        pred_counts[predicted_type] += 1
        if correct:
            true_positive_counts[predicted_type] += 1
        confusion.setdefault(primary_gt, _empty_confusion_row())[predicted_type] += 1

        evaluation = ImageEvaluation(
            image=_repo_relative_posix(image_path),
            gt_hazards=gt_hazards,
            primary_gt=primary_gt,
            predicted_type=predicted_type,
            confidence=confidence,
            correct=correct,
            gt_boxes=gt_boxes,
            predicted_bbox=predicted_bbox,
        )
        evaluations.append(evaluation)
        if not correct:
            mistakes.append(evaluation)

    evaluated_images = len(evaluations)
    correct_images = sum(1 for item in evaluations if item.correct)
    return EvaluationResult(
        dataset_root=_repo_relative_posix(dataset_root),
        split=split,
        manifest=_repo_relative_posix(manifest_path),
        total_images=len(image_paths),
        evaluated_images=evaluated_images,
        correct_images=correct_images,
        accuracy=round(correct_images / evaluated_images, 4) if evaluated_images else 0.0,
        no_detection=no_detection,
        missing_labels=missing_labels,
        unreadable_images=unreadable_images,
        class_metrics=_class_metrics(gt_counts, pred_counts, true_positive_counts),
        confusion_matrix=confusion,
        mistakes=mistakes,
    )


def parse_ground_truth_boxes(label_path: Path, names: list[str], *, width: int, height: int) -> list[HazardBox]:
    rows, _invalids = parse_yolo_rows(label_path, names)
    boxes: list[HazardBox] = []
    for row in rows:
        hazard_type = _normalize_type(row.source_cls)
        boxes.append(HazardBox(type=hazard_type, bbox=row_to_pixel_bbox(row, width, height)))
    return boxes


def choose_primary_gt(gt_boxes: list[HazardBox]) -> str:
    if not gt_boxes:
        return "unknown"
    largest = max(gt_boxes, key=lambda item: _bbox_area(item.bbox))
    return largest.type


def predict_primary_hazard(frame: np.ndarray, *, manifest: Any, backend: Any) -> dict[str, Any] | None:
    raw = backend.predict(frame, None, manifest)
    best: dict[str, Any] | None = None
    for item in _as_iterable(raw):
        if not isinstance(item, dict):
            continue
        label_value = item.get("label", item.get("type", item.get("class_id", item.get("class"))))
        if label_value is None:
            continue
        conf = float(item.get("conf", item.get("confidence", item.get("score", 0.0))))
        if conf < manifest.confidence_threshold:
            continue
        hazard_type = _normalize_type(str(manifest.map_label(manifest.class_name(label_value))))
        bbox = item.get("bbox")
        prediction = {
            "type": hazard_type,
            "confidence": conf,
            "bbox": [round(float(value), 2) for value in bbox] if bbox is not None else None,
        }
        if best is None or prediction["confidence"] > best["confidence"]:
            best = prediction
    return best


def read_image(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def print_evaluation(result: EvaluationResult, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(_result_to_dict(result), ensure_ascii=False, separators=(",", ":")))
        return
    print(f"dataset_root={result.dataset_root}")
    print(f"split={result.split}")
    print(f"manifest={result.manifest}")
    print(f"total_images={result.total_images}")
    print(f"evaluated_images={result.evaluated_images}")
    print(f"correct_images={result.correct_images}")
    print(f"accuracy={result.accuracy:.4f}")
    print(f"no_detection={result.no_detection}")
    print(f"missing_labels={result.missing_labels}")
    print(f"unreadable_images={result.unreadable_images}")
    print("class_metrics=" + json.dumps(result.class_metrics, ensure_ascii=False, sort_keys=True))
    print("confusion_matrix=" + json.dumps(result.confusion_matrix, ensure_ascii=False, sort_keys=True))
    print(f"mistakes={len(result.mistakes)}")
    print("status=ok")


def write_mistakes(path: Path, mistakes: list[ImageEvaluation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(item) for item in mistakes], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _class_metrics(gt_counts: Counter[str], pred_counts: Counter[str], tp_counts: Counter[str]) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for hazard_type in SCHEMA_TYPES:
        tp = tp_counts[hazard_type]
        predicted = pred_counts[hazard_type]
        actual = gt_counts[hazard_type]
        metrics[hazard_type] = {
            "tp": tp,
            "predicted": predicted,
            "actual": actual,
            "precision": round(tp / predicted, 4) if predicted else 0.0,
            "recall": round(tp / actual, 4) if actual else 0.0,
        }
    return metrics


def _empty_confusion() -> dict[str, dict[str, int]]:
    return {hazard_type: _empty_confusion_row() for hazard_type in SCHEMA_TYPES}


def _empty_confusion_row() -> dict[str, int]:
    return {hazard_type: 0 for hazard_type in SCHEMA_TYPES}


def _normalize_type(label: str) -> str:
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in SCHEMA_TYPES else "unknown"


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _as_iterable(raw: Any) -> Iterable[Any]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        if "detections" in raw:
            return raw["detections"]
        return [raw]
    if isinstance(raw, (str, bytes)):
        return []
    return raw


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _result_to_dict(result: EvaluationResult) -> dict[str, Any]:
    return asdict(result)


def _repo_relative_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a hazard YOLO model on a YOLO split.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument("--device", help="Ultralytics device value, for example cpu, 0, or 0,1.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N images.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    parser.add_argument("--errors-out", type=Path, help="Optional JSON file for incorrect predictions.")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        result = evaluate_hazard_model(
            dataset_root=_resolve_path(args.dataset_root),
            split=args.split,
            manifest_path=_resolve_path(args.manifest),
            device=args.device,
            limit=args.limit,
        )
        if args.errors_out:
            write_mistakes(_resolve_path(args.errors_out), result.mistakes)
        print_evaluation(result, as_json=args.json)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
