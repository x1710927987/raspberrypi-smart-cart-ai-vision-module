from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from yolo_dataset_utils import format_yolo_row, parse_yolo_rows, read_image, read_yolo_config


DEFAULT_REVIEW_CSV = REPO_ROOT / "cache" / "evaluation" / "object_detection_scooter_error_review_template.csv"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "external" / "objects_combined_v2_split"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "external" / "objects_scooter_add_similar_v1"
DEFAULT_CLASSES = ("pedestrian", "bicycle", "car", "scooter", "roadblock")
IMAGE_SUFFIX = ".jpg"


@dataclass(frozen=True)
class ReviewCase:
    case_id: str
    error_type: str
    source_image: Path
    reason: str
    action: str
    priority: str
    notes: str


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float


@dataclass(frozen=True)
class GeneratedSample:
    case_id: str
    source_image: str
    output_image: str
    output_label: str
    transform: str
    error_type: str
    reason: str
    label_rows: int


def augment_add_similar_data(
    *,
    review_csv: str | Path = DEFAULT_REVIEW_CSV,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    seed: int = 42,
    max_per_case: int = 2,
    overwrite: bool = False,
) -> list[GeneratedSample]:
    review_path = _resolve_path(Path(review_csv))
    dataset_root = _resolve_path(Path(dataset_root))
    output_root = _resolve_path(Path(output_root))
    config = read_yolo_config(dataset_root / "data.yaml")
    if tuple(config.names) != DEFAULT_CLASSES:
        raise ValueError(f"expected object classes {DEFAULT_CLASSES}, got {config.names}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_root}")
        shutil.rmtree(output_root)

    cases = read_review_cases(review_path)
    rng = random.Random(seed)
    samples: list[GeneratedSample] = []
    for case in cases:
        source_image = _resolve_path(case.source_image)
        image = read_image(source_image)
        if image is None or image.size == 0:
            raise ValueError(f"unreadable source image for {case.case_id}: {source_image}")
        label_path = image_to_label_path(source_image)
        labels = read_labels(label_path, config.names)
        if not labels:
            raise ValueError(f"no usable labels for {case.case_id}: {label_path}")
        transforms = choose_transforms(case, max_per_case=max_per_case, rng=rng)
        for index, transform_name in enumerate(transforms, start=1):
            augmented, augmented_labels = apply_transform(image, labels, transform_name, rng)
            if not augmented_labels:
                continue
            stem = f"{case.case_id}_{index:02d}_{transform_name}"
            image_path = output_root / "train" / "images" / f"{stem}{IMAGE_SUFFIX}"
            label_out_path = output_root / "train" / "labels" / f"{stem}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_out_path.parent.mkdir(parents=True, exist_ok=True)
            write_image(image_path, augmented)
            label_out_path.write_text(format_labels(augmented_labels), encoding="utf-8")
            samples.append(
                GeneratedSample(
                    case_id=case.case_id,
                    source_image=_repo_relative_posix(source_image),
                    output_image=_repo_relative_posix(image_path),
                    output_label=_repo_relative_posix(label_out_path),
                    transform=transform_name,
                    error_type=case.error_type,
                    reason=case.reason,
                    label_rows=len(augmented_labels),
                )
            )

    write_data_yaml(output_root / "data.yaml", DEFAULT_CLASSES)
    write_manifest(output_root / "augmentation_manifest.csv", samples)
    write_summary(output_root / "augmentation_summary.json", review_path, samples, seed=seed, max_per_case=max_per_case)
    return samples


def read_review_cases(path: Path) -> list[ReviewCase]:
    if not path.exists():
        raise FileNotFoundError(f"review CSV does not exist: {path}")
    cases: list[ReviewCase] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("action", "").strip() != "add_similar_data":
                continue
            source_image = row.get("source_image", "").strip()
            if not source_image:
                continue
            cases.append(
                ReviewCase(
                    case_id=row.get("case_id", "").strip(),
                    error_type=row.get("error_type", "").strip(),
                    source_image=Path(source_image),
                    reason=row.get("reason", "").strip(),
                    action=row.get("action", "").strip(),
                    priority=row.get("priority", "").strip(),
                    notes=row.get("notes", "").strip(),
                )
            )
    if not cases:
        raise ValueError(f"no add_similar_data cases found in {path}")
    return cases


def read_labels(path: Path, names: list[str]) -> list[YoloBox]:
    rows, invalids = parse_yolo_rows(path, names)
    if invalids:
        raise ValueError(f"{path}: invalid YOLO rows: {invalids[:3]}")
    boxes: list[YoloBox] = []
    for row in rows:
        if row.row_type != "bbox":
            continue
        cx, cy, box_w, box_h = row.coords
        boxes.append(YoloBox(row.class_id, cx, cy, box_w, box_h))
    return boxes


def choose_transforms(case: ReviewCase, *, max_per_case: int, rng: random.Random) -> list[str]:
    focus = f"{case.error_type} {case.reason} {case.notes}".lower()
    if "background_confusion" in focus:
        pool = ["identity", "color_jitter", "blur_noise", "shadow_contrast"]
    elif "small_distant" in focus:
        pool = ["small_canvas", "small_canvas", "blur_noise", "shadow_contrast", "horizontal_flip"]
    elif "unusual_view" in focus:
        pool = ["horizontal_flip", "color_jitter", "shadow_contrast", "blur_noise"]
    elif "bicycle_scooter_ambiguity" in focus:
        pool = ["identity", "horizontal_flip", "color_jitter", "blur_noise"]
    else:
        pool = ["color_jitter", "blur_noise", "horizontal_flip", "shadow_contrast", "small_canvas"]
    selected: list[str] = []
    while len(selected) < max_per_case:
        selected.append(rng.choice(pool))
    return selected


def apply_transform(image: np.ndarray, labels: list[YoloBox], transform_name: str, rng: random.Random) -> tuple[np.ndarray, list[YoloBox]]:
    if transform_name == "horizontal_flip":
        return horizontal_flip(image, labels)
    if transform_name == "small_canvas":
        return small_canvas(image, labels, rng)
    if transform_name == "blur_noise":
        return blur_noise(image, labels, rng)
    if transform_name == "shadow_contrast":
        return shadow_contrast(image, labels, rng)
    if transform_name == "color_jitter":
        return color_jitter(image, labels, rng)
    return image.copy(), list(labels)


def horizontal_flip(image: np.ndarray, labels: list[YoloBox]) -> tuple[np.ndarray, list[YoloBox]]:
    flipped = cv2.flip(image, 1)
    return flipped, [YoloBox(label.class_id, 1.0 - label.cx, label.cy, label.width, label.height) for label in labels]


def color_jitter(image: np.ndarray, labels: list[YoloBox], rng: random.Random) -> tuple[np.ndarray, list[YoloBox]]:
    alpha = rng.uniform(0.76, 1.24)
    beta = rng.uniform(-30, 30)
    output = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if rng.random() < 0.45:
        hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= rng.uniform(0.75, 1.25)
        hsv[..., 2] *= rng.uniform(0.82, 1.18)
        hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
        output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return output, list(labels)


def blur_noise(image: np.ndarray, labels: list[YoloBox], rng: random.Random) -> tuple[np.ndarray, list[YoloBox]]:
    output, out_labels = color_jitter(image, labels, rng)
    if rng.random() < 0.7:
        kernel = rng.choice([3, 5])
        output = cv2.GaussianBlur(output, (kernel, kernel), 0)
    noise = np.random.default_rng(rng.randrange(1_000_000_000)).normal(0, rng.uniform(2.0, 8.0), output.shape)
    output = np.clip(output.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return output, out_labels


def shadow_contrast(image: np.ndarray, labels: list[YoloBox], rng: random.Random) -> tuple[np.ndarray, list[YoloBox]]:
    output, out_labels = color_jitter(image, labels, rng)
    height, width = output.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    x1 = rng.randint(0, max(1, width - 1))
    x2 = rng.randint(0, max(1, width - 1))
    cv2.line(mask, (x1, 0), (x2, height - 1), 1.0, thickness=max(8, width // 7))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(12.0, width / 20.0))
    factor = rng.uniform(0.62, 0.86)
    output = (output.astype(np.float32) * (1.0 - mask[..., None] * (1.0 - factor))).clip(0, 255).astype(np.uint8)
    return output, out_labels


def small_canvas(image: np.ndarray, labels: list[YoloBox], rng: random.Random) -> tuple[np.ndarray, list[YoloBox]]:
    height, width = image.shape[:2]
    scale = rng.uniform(0.62, 0.86)
    new_w = max(2, int(width * scale))
    new_h = max(2, int(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full_like(image, 114)
    max_x = max(0, width - new_w)
    max_y = max(0, height - new_h)
    x0 = rng.randint(0, max_x) if max_x else 0
    y0 = rng.randint(0, max_y) if max_y else 0
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    transformed: list[YoloBox] = []
    for label in labels:
        x1 = (label.cx - label.width / 2.0) * scale + x0 / width
        y1 = (label.cy - label.height / 2.0) * scale + y0 / height
        x2 = (label.cx + label.width / 2.0) * scale + x0 / width
        y2 = (label.cy + label.height / 2.0) * scale + y0 / height
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        transformed.append(YoloBox(label.class_id, (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1))
    return canvas, transformed


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        images_index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"cannot derive label path from image path without images directory: {image_path}") from exc
    parts[images_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def format_labels(labels: Iterable[YoloBox]) -> str:
    rows = [
        format_yolo_row(
            type(
                "Row",
                (),
                {
                    "class_id": label.class_id,
                    "coords": [label.cx, label.cy, label.width, label.height],
                },
            )()
        )
        for label in labels
    ]
    return "\n".join(rows) + ("\n" if rows else "")


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or IMAGE_SUFFIX, image)
    if not ok:
        raise ValueError(f"failed to encode image: {path}")
    encoded.tofile(str(path))


def write_data_yaml(path: Path, classes: tuple[str, ...]) -> None:
    names_text = "[" + ", ".join(repr(name) for name in classes) + "]"
    path.write_text(
        "\n".join(
            [
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "",
                f"nc: {len(classes)}",
                f"names: {names_text}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_manifest(path: Path, samples: list[GeneratedSample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GeneratedSample.__dataclass_fields__.keys()))
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.__dict__)


def write_summary(path: Path, review_path: Path, samples: list[GeneratedSample], *, seed: int, max_per_case: int) -> None:
    payload = {
        "review_csv": _repo_relative_posix(review_path),
        "seed": seed,
        "max_per_case": max_per_case,
        "generated_samples": len(samples),
        "cases": len({sample.case_id for sample in samples}),
        "transforms": _count(sample.transform for sample in samples),
        "reasons": _count(sample.reason for sample in samples),
        "error_types": _count(sample.error_type for sample in samples),
        "note": "Generated from reviewed add_similar_data cases. Re-split before training to avoid evaluating against harvested source images.",
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_summary(samples: list[GeneratedSample], output_root: Path) -> None:
    print(f"output_root={_repo_relative_posix(output_root)}")
    print(f"generated_samples={len(samples)}")
    print(f"cases={len({sample.case_id for sample in samples})}")
    print("transforms=" + json.dumps(_count(sample.transform for sample in samples), ensure_ascii=False, sort_keys=True))
    print("reasons=" + json.dumps(_count(sample.reason for sample in samples), ensure_ascii=False, sort_keys=True))
    print("error_types=" + json.dumps(_count(sample.error_type for sample in samples), ensure_ascii=False, sort_keys=True))
    print("status=ok")


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_relative_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an object-detection add-similar-data YOLO source from reviewed scooter mistakes.")
    parser.add_argument("--review-csv", default=DEFAULT_REVIEW_CSV, type=Path)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, type=Path)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-per-case", default=2, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        samples = augment_add_similar_data(
            review_csv=args.review_csv,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            seed=args.seed,
            max_per_case=args.max_per_case,
            overwrite=args.overwrite,
        )
        print_summary(samples, _resolve_path(args.output_root))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
