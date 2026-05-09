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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.model_inference import ManifestTrafficLightClassifier, UltralyticsBackend, load_model_manifest
from perception.runtime import TrafficLight


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
SCHEMA_STATES = ("green", "red", "yellow", "unknown")
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "external" / "roboflow_traffic_light_v1_split"
DEFAULT_MANIFEST = REPO_ROOT / "models" / "training" / "smartcart_traffic_light_yolov8n_smoke_pt_v1.manifest.json"


@dataclass(frozen=True)
class ImageEvaluation:
    image: str
    gt_states: list[str]
    primary_gt: str
    predicted_state: str
    confidence: float
    correct: bool


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


def evaluate_traffic_light_model(
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    split: str = "valid",
    manifest_path: str | Path = DEFAULT_MANIFEST,
    backend: Any | None = None,
    device: str | None = None,
    limit: int | None = None,
) -> EvaluationResult:
    dataset_root = Path(dataset_root)
    manifest_path = Path(manifest_path)
    names = read_yolo_names(dataset_root / "data.yaml")
    images_dir = dataset_root / split / "images"
    labels_dir = dataset_root / split / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"missing split images directory: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"missing split labels directory: {labels_dir}")

    manifest = load_model_manifest(manifest_path, require_artifact=backend is None)
    provider = ManifestTrafficLightClassifier(manifest, backend or UltralyticsBackend(device=device))
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
        gt_states = parse_label_states(label_path, names)
        if not gt_states:
            missing_labels += 1
            continue
        frame = read_image(image_path)
        if frame is None:
            unreadable_images += 1
            continue

        prediction = provider.detect(frame)
        predicted_state = prediction.state if prediction is not None else "unknown"
        confidence = round(float(prediction.conf), 4) if prediction is not None else 0.0
        if prediction is None:
            no_detection += 1
        gt_set = set(gt_states)
        primary_gt = gt_states[0]
        correct = predicted_state in gt_set

        for state in gt_set:
            gt_counts[state] += 1
        pred_counts[predicted_state] += 1
        if correct:
            true_positive_counts[predicted_state] += 1
        confusion.setdefault(primary_gt, _empty_confusion_row())[predicted_state] += 1

        evaluation = ImageEvaluation(
            image=_repo_relative_posix(image_path),
            gt_states=gt_states,
            primary_gt=primary_gt,
            predicted_state=predicted_state,
            confidence=confidence,
            correct=correct,
        )
        evaluations.append(evaluation)
        if not correct:
            mistakes.append(evaluation)

    evaluated_images = len(evaluations)
    correct_images = sum(1 for item in evaluations if item.correct)
    accuracy = round(correct_images / evaluated_images, 4) if evaluated_images else 0.0
    return EvaluationResult(
        dataset_root=_repo_relative_posix(dataset_root),
        split=split,
        manifest=_repo_relative_posix(manifest_path),
        total_images=len(image_paths),
        evaluated_images=evaluated_images,
        correct_images=correct_images,
        accuracy=accuracy,
        no_detection=no_detection,
        missing_labels=missing_labels,
        unreadable_images=unreadable_images,
        class_metrics=_class_metrics(gt_counts, pred_counts, true_positive_counts),
        confusion_matrix=confusion,
        mistakes=mistakes,
    )


def parse_label_states(label_path: Path, names: list[str]) -> list[str]:
    states: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            class_id = int(float(line.split()[0]))
        except (IndexError, ValueError):
            continue
        if 0 <= class_id < len(names):
            state = names[class_id].strip().lower()
            states.append(state if state in SCHEMA_STATES else "unknown")
    return states


def read_yolo_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"missing data.yaml: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("names:"):
            return _parse_names(line.split(":", 1)[1].strip())
    raise ValueError(f"{path}: missing names")


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
    payload = [asdict(item) for item in mistakes]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _class_metrics(gt_counts: Counter[str], pred_counts: Counter[str], tp_counts: Counter[str]) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for state in SCHEMA_STATES:
        tp = tp_counts[state]
        predicted = pred_counts[state]
        actual = gt_counts[state]
        precision = round(tp / predicted, 4) if predicted else 0.0
        recall = round(tp / actual, 4) if actual else 0.0
        metrics[state] = {
            "tp": tp,
            "predicted": predicted,
            "actual": actual,
            "precision": precision,
            "recall": recall,
        }
    return metrics


def _empty_confusion() -> dict[str, dict[str, int]]:
    return {state: _empty_confusion_row() for state in SCHEMA_STATES}


def _empty_confusion_row() -> dict[str, int]:
    return {state: 0 for state in SCHEMA_STATES}


def _parse_names(value: str) -> list[str]:
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]


def _result_to_dict(result: EvaluationResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["mistakes"] = [asdict(item) for item in result.mistakes]
    return payload


def _repo_relative_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a traffic-light model on a YOLO split.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, type=Path)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
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
        result = evaluate_traffic_light_model(
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
