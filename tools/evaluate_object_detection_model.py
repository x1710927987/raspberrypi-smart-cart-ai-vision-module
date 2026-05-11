from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from perception.model_inference import ManifestObjectDetector, UltralyticsBackend, load_model_manifest
from yolo_dataset_utils import IMAGE_EXTENSIONS, parse_yolo_rows, read_yolo_config, row_to_pixel_bbox


DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "external" / "objects_combined_v3_split"
DEFAULT_MANIFEST = REPO_ROOT / "models" / "training" / "smartcart_objects_yolov8n_combined_v3_pt_v1.manifest.json"
DEFAULT_IOU_THRESHOLD = 0.5
OBJECT_CLASSES = ("pedestrian", "bicycle", "car", "scooter", "roadblock", "unknown")


@dataclass(frozen=True)
class ObjectBox:
    cls: str
    bbox: list[float]
    conf: float | None = None


@dataclass(frozen=True)
class ObjectMatch:
    gt: ObjectBox | None
    pred: ObjectBox | None
    iou: float
    status: str


@dataclass(frozen=True)
class ImageEvaluation:
    image: str
    correct: bool
    gt_boxes: list[ObjectBox]
    predicted_boxes: list[ObjectBox]
    matches: list[ObjectMatch]
    false_negatives: list[ObjectBox]
    false_positives: list[ObjectBox]
    misclassified: list[ObjectMatch]


@dataclass(frozen=True)
class EvaluationResult:
    dataset_root: str
    split: str
    manifest: str
    iou_threshold: float
    total_images: int
    evaluated_images: int
    correct_images: int
    accuracy: float
    missing_labels: int
    unreadable_images: int
    gt_boxes: int
    predicted_boxes: int
    true_positives: int
    false_positives: int
    false_negatives: int
    misclassified: int
    class_metrics: dict[str, dict[str, float | int]]
    confusion_matrix: dict[str, dict[str, int]]
    mistakes: list[ImageEvaluation]


def evaluate_object_detection_model(
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    split: str = "test",
    manifest_path: str | Path = DEFAULT_MANIFEST,
    backend: Any | None = None,
    device: str | None = None,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
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
    detector = ManifestObjectDetector(manifest, backend or UltralyticsBackend(device=device))
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
    total_gt_boxes = 0
    total_predicted_boxes = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_misclassified = 0

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
        predictions = [
            ObjectBox(cls=_normalize_class(item.cls), bbox=[round(float(value), 2) for value in item.bbox], conf=round(float(item.conf), 4))
            for item in detector.detect(frame)
        ]
        matches, false_negatives, false_positives, misclassified = match_detections(gt_boxes, predictions, iou_threshold=iou_threshold)

        for gt_box in gt_boxes:
            gt_counts[gt_box.cls] += 1
        for pred_box in predictions:
            pred_counts[pred_box.cls] += 1
        for match in matches:
            if match.status == "tp" and match.gt is not None:
                true_positive_counts[match.gt.cls] += 1
                confusion.setdefault(match.gt.cls, _empty_confusion_row())[match.gt.cls] += 1
            elif match.status == "misclassified" and match.gt is not None and match.pred is not None:
                confusion.setdefault(match.gt.cls, _empty_confusion_row())[match.pred.cls] += 1
        for gt_box in false_negatives:
            confusion.setdefault(gt_box.cls, _empty_confusion_row())["unknown"] += 1
        for pred_box in false_positives:
            confusion.setdefault("unknown", _empty_confusion_row())[pred_box.cls] += 1

        total_gt_boxes += len(gt_boxes)
        total_predicted_boxes += len(predictions)
        total_false_positives += len(false_positives)
        total_false_negatives += len(false_negatives)
        total_misclassified += len(misclassified)
        correct = not false_negatives and not false_positives and not misclassified

        evaluation = ImageEvaluation(
            image=_repo_relative_posix(image_path),
            correct=correct,
            gt_boxes=gt_boxes,
            predicted_boxes=predictions,
            matches=matches,
            false_negatives=false_negatives,
            false_positives=false_positives,
            misclassified=misclassified,
        )
        evaluations.append(evaluation)
        if not correct:
            mistakes.append(evaluation)

    evaluated_images = len(evaluations)
    correct_images = sum(1 for item in evaluations if item.correct)
    true_positives = sum(true_positive_counts.values())
    return EvaluationResult(
        dataset_root=_repo_relative_posix(dataset_root),
        split=split,
        manifest=_repo_relative_posix(manifest_path),
        iou_threshold=round(float(iou_threshold), 4),
        total_images=len(image_paths),
        evaluated_images=evaluated_images,
        correct_images=correct_images,
        accuracy=round(correct_images / evaluated_images, 4) if evaluated_images else 0.0,
        missing_labels=missing_labels,
        unreadable_images=unreadable_images,
        gt_boxes=total_gt_boxes,
        predicted_boxes=total_predicted_boxes,
        true_positives=true_positives,
        false_positives=total_false_positives + total_misclassified,
        false_negatives=total_false_negatives + total_misclassified,
        misclassified=total_misclassified,
        class_metrics=_class_metrics(gt_counts, pred_counts, true_positive_counts),
        confusion_matrix=confusion,
        mistakes=mistakes,
    )


def parse_ground_truth_boxes(label_path: Path, names: list[str], *, width: int, height: int) -> list[ObjectBox]:
    rows, _invalids = parse_yolo_rows(label_path, names)
    boxes: list[ObjectBox] = []
    for row in rows:
        boxes.append(ObjectBox(cls=_normalize_class(row.source_cls), bbox=row_to_pixel_bbox(row, width, height), conf=None))
    return boxes


def match_detections(
    gt_boxes: list[ObjectBox],
    predictions: list[ObjectBox],
    *,
    iou_threshold: float,
) -> tuple[list[ObjectMatch], list[ObjectBox], list[ObjectBox], list[ObjectMatch]]:
    pairs: list[tuple[float, int, int]] = []
    for gt_index, gt_box in enumerate(gt_boxes):
        for pred_index, pred_box in enumerate(predictions):
            iou = bbox_iou(gt_box.bbox, pred_box.bbox)
            if iou >= iou_threshold:
                pairs.append((iou, gt_index, pred_index))
    pairs.sort(reverse=True, key=lambda item: item[0])

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[ObjectMatch] = []
    misclassified: list[ObjectMatch] = []
    for iou, gt_index, pred_index in pairs:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        gt_box = gt_boxes[gt_index]
        pred_box = predictions[pred_index]
        status = "tp" if gt_box.cls == pred_box.cls else "misclassified"
        match = ObjectMatch(gt=gt_box, pred=pred_box, iou=round(iou, 4), status=status)
        matches.append(match)
        if status == "misclassified":
            misclassified.append(match)

    false_negatives = [gt_box for index, gt_box in enumerate(gt_boxes) if index not in used_gt]
    false_positives = [pred_box for index, pred_box in enumerate(predictions) if index not in used_pred]
    for gt_box in false_negatives:
        matches.append(ObjectMatch(gt=gt_box, pred=None, iou=0.0, status="fn"))
    for pred_box in false_positives:
        matches.append(ObjectMatch(gt=None, pred=pred_box, iou=0.0, status="fp"))
    return matches, false_negatives, false_positives, misclassified


def bbox_iou(left: list[float], right: list[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    left_area = _bbox_area(left)
    right_area = _bbox_area(right)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


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
    print(f"iou_threshold={result.iou_threshold:.4f}")
    print(f"total_images={result.total_images}")
    print(f"evaluated_images={result.evaluated_images}")
    print(f"correct_images={result.correct_images}")
    print(f"accuracy={result.accuracy:.4f}")
    print(f"gt_boxes={result.gt_boxes}")
    print(f"predicted_boxes={result.predicted_boxes}")
    print(f"true_positives={result.true_positives}")
    print(f"false_positives={result.false_positives}")
    print(f"false_negatives={result.false_negatives}")
    print(f"misclassified={result.misclassified}")
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
    for object_class in OBJECT_CLASSES:
        tp = tp_counts[object_class]
        predicted = pred_counts[object_class]
        actual = gt_counts[object_class]
        metrics[object_class] = {
            "tp": tp,
            "predicted": predicted,
            "actual": actual,
            "precision": round(tp / predicted, 4) if predicted else 0.0,
            "recall": round(tp / actual, 4) if actual else 0.0,
        }
    return metrics


def _empty_confusion() -> dict[str, dict[str, int]]:
    return {object_class: _empty_confusion_row() for object_class in OBJECT_CLASSES}


def _empty_confusion_row() -> dict[str, int]:
    return {object_class: 0 for object_class in OBJECT_CLASSES}


def _normalize_class(label: str) -> str:
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "person": "pedestrian",
        "people": "pedestrian",
        "e_scooter": "scooter",
        "electric_scooter": "scooter",
        "cone": "roadblock",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in OBJECT_CLASSES else "unknown"


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


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
    parser = argparse.ArgumentParser(description="Evaluate an object-detection YOLO model on a YOLO split.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument("--device", help="Ultralytics device value, for example cpu, 0, or 0,1.")
    parser.add_argument("--iou-threshold", default=DEFAULT_IOU_THRESHOLD, type=float)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N images.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    parser.add_argument("--errors-out", type=Path, help="Optional JSON file for incorrect predictions.")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        result = evaluate_object_detection_model(
            dataset_root=_resolve_path(args.dataset_root),
            split=args.split,
            manifest_path=_resolve_path(args.manifest),
            device=args.device,
            iou_threshold=args.iou_threshold,
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
