from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from yolo_dataset_utils import read_image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "processed" / "objects_yolo_v1"
DEFAULT_CLASSES = ["pedestrian", "bicycle", "car"]
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class ExportSummary:
    output_root: Path
    classes: list[str]
    images: dict[str, int]
    objects: dict[str, int]
    skipped_annotations: int = 0
    skipped_objects: int = 0


def export_objects_yolo_dataset(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    classes: list[str] | None = None,
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42,
    overwrite: bool = False,
) -> ExportSummary:
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    class_names = list(classes or DEFAULT_CLASSES)
    if len(class_names) != len(set(class_names)):
        raise ValueError("classes must not contain duplicates")
    annotations = _load_annotations(data_root)
    assignments = _assign_splits(annotations, ratios=ratios, seed=seed)

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_root}")
        shutil.rmtree(output_root)
    for split in SPLITS:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

    images_count = {split: 0 for split in SPLITS}
    objects_count = {split: 0 for split in SPLITS}
    skipped_annotations = 0
    skipped_objects = 0

    for annotation_path, payload, split in assignments:
        image_path = _resolve_data_path(data_root, payload.get("image"))
        image = read_image(image_path) if image_path else None
        if image is None or image.size == 0:
            skipped_annotations += 1
            continue
        height, width = image.shape[:2]
        label_rows: list[str] = []
        for obj in payload.get("objects", []):
            cls = str(obj.get("cls", ""))
            if cls not in class_names:
                skipped_objects += 1
                continue
            bbox = obj.get("bbox")
            yolo = _bbox_to_yolo_row(class_names.index(cls), bbox, width, height)
            if yolo is None:
                skipped_objects += 1
                continue
            label_rows.append(yolo)
        if not label_rows:
            skipped_annotations += 1
            continue

        output_stem = annotation_path.stem
        output_image = output_root / split / "images" / f"{output_stem}{image_path.suffix.lower()}"
        output_label = output_root / split / "labels" / f"{output_stem}.txt"
        shutil.copy2(image_path, output_image)
        output_label.write_text("\n".join(label_rows) + "\n", encoding="utf-8")
        images_count[split] += 1
        objects_count[split] += len(label_rows)

    _write_data_yaml(output_root / "data.yaml", class_names)
    _write_readme(output_root / "README.generated.txt", data_root, class_names, images_count, objects_count)
    return ExportSummary(
        output_root=output_root,
        classes=class_names,
        images=images_count,
        objects=objects_count,
        skipped_annotations=skipped_annotations,
        skipped_objects=skipped_objects,
    )


def print_summary(summary: ExportSummary) -> None:
    print(f"output_root={summary.output_root}")
    print(f"classes={summary.classes}")
    for split in SPLITS:
        print(f"{split}_images={summary.images[split]}")
        print(f"{split}_objects={summary.objects[split]}")
    print(f"skipped_annotations={summary.skipped_annotations}")
    print(f"skipped_objects={summary.skipped_objects}")
    print("status=ok")


def _load_annotations(data_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    annotations_dir = data_root / "annotations" / "objects"
    if not annotations_dir.exists():
        raise FileNotFoundError(f"missing object annotations directory: {annotations_dir}")
    annotations: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(annotations_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            annotations.append((path, payload))
    if not annotations:
        raise ValueError(f"no object annotations found under {annotations_dir}")
    return annotations


def _assign_splits(
    annotations: list[tuple[Path, dict[str, Any]]],
    *,
    ratios: tuple[float, float, float],
    seed: int,
) -> list[tuple[Path, dict[str, Any], str]]:
    assigned: list[tuple[Path, dict[str, Any], str]] = []
    unassigned: list[tuple[Path, dict[str, Any]]] = []
    for path, payload in annotations:
        split = str(payload.get("source_split", "")).strip().lower()
        if split in SPLITS:
            assigned.append((path, payload, split))
        else:
            unassigned.append((path, payload))
    if not unassigned:
        return assigned

    random.Random(seed).shuffle(unassigned)
    train_ratio, valid_ratio, test_ratio = ratios
    total_ratio = train_ratio + valid_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("split ratios must sum to a positive value")
    train_count = int(len(unassigned) * train_ratio / total_ratio)
    valid_count = int(len(unassigned) * valid_ratio / total_ratio)
    generated = {
        "train": unassigned[:train_count],
        "valid": unassigned[train_count : train_count + valid_count],
        "test": unassigned[train_count + valid_count :],
    }
    for split, items in generated.items():
        assigned.extend((path, payload, split) for path, payload in items)
    return assigned


def _resolve_data_path(data_root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def _bbox_to_yolo_row(class_id: int, bbox: Any, width: int, height: int) -> str | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height
    return f"{class_id} {cx:.8f} {cy:.8f} {box_w:.8f} {box_h:.8f}"


def _write_data_yaml(path: Path, classes: list[str]) -> None:
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


def _write_readme(path: Path, data_root: Path, classes: list[str], images: dict[str, int], objects: dict[str, int]) -> None:
    lines = [
        "Generated object-detection YOLO dataset.",
        "",
        f"source_data_root: {data_root}",
        f"classes: {classes}",
        "",
    ]
    for split in SPLITS:
        lines.append(f"{split}: images={images[split]}, objects={objects[split]}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_classes(value: str) -> list[str]:
    classes = [item.strip() for item in value.split(",") if item.strip()]
    if not classes:
        raise argparse.ArgumentTypeError("classes must not be empty")
    return classes


def _parse_ratios(value: str) -> tuple[float, float, float]:
    try:
        parts = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratios must use train,valid,test numeric format") from exc
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("ratios must use train,valid,test format")
    return parts[0], parts[1], parts[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export project object JSON annotations to an Ultralytics YOLO dataset.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=Path)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=Path)
    parser.add_argument("--classes", default=DEFAULT_CLASSES, type=_parse_classes, help="Comma-separated YOLO class order.")
    parser.add_argument("--ratios", default=(0.7, 0.2, 0.1), type=_parse_ratios)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        summary = export_objects_yolo_dataset(
            data_root=args.data_root,
            output_root=args.output_root,
            classes=args.classes,
            ratios=args.ratios,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
