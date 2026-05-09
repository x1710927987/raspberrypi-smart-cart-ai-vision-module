from __future__ import annotations

import argparse
import json
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

from yolo_dataset_utils import (
    dataset_split_dirs,
    iter_images,
    parse_class_map,
    parse_yolo_rows,
    read_image,
    read_yolo_config,
    repo_relative,
    row_to_pixel_bbox,
    row_to_pixel_polygon,
    write_image,
)


OBJECT_CLASSES = {
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
}


@dataclass
class ImportSummary:
    task: str
    imported_images: int = 0
    imported_annotations: int = 0
    imported_instances: int = 0
    skipped_unreadable: int = 0
    skipped_missing_label: int = 0
    skipped_empty: int = 0
    skipped_unmapped_rows: int = 0
    invalid_label_rows: int = 0


def import_dataset(
    *,
    dataset_root: Path,
    data_root: Path,
    task: str,
    class_map: dict[str, str],
    prefix: str,
    splits: list[str],
    include_empty: bool = False,
) -> ImportSummary:
    dataset_root = dataset_root.resolve()
    data_root = data_root.resolve()
    config = read_yolo_config(dataset_root / "data.yaml")
    summary = ImportSummary(task=task)
    for split in splits:
        images_dir, labels_dir = dataset_split_dirs(dataset_root, split)
        if not images_dir.exists() or not labels_dir.exists():
            continue
        labels_by_stem = {path.stem: path for path in labels_dir.rglob("*.txt") if path.is_file()}
        for image_path in iter_images(images_dir):
            image = read_image(image_path)
            if image is None or image.size == 0:
                summary.skipped_unreadable += 1
                continue
            label_path = labels_by_stem.get(image_path.stem)
            if label_path is None:
                summary.skipped_missing_label += 1
                continue
            rows, invalids = parse_yolo_rows(label_path, config.names)
            summary.invalid_label_rows += len(invalids)
            if task == "objects":
                imported = _import_object_sample(
                    image=image,
                    image_path=image_path,
                    rows=rows,
                    class_map=class_map,
                    data_root=data_root,
                    dataset_root=dataset_root,
                    prefix=prefix,
                    split=split,
                    include_empty=include_empty,
                )
            elif task == "laneseg":
                imported = _import_laneseg_sample(
                    image=image,
                    image_path=image_path,
                    rows=rows,
                    class_map=class_map,
                    data_root=data_root,
                    dataset_root=dataset_root,
                    prefix=prefix,
                    split=split,
                    include_empty=include_empty,
                )
            else:
                raise ValueError(f"unsupported task: {task}")
            summary.skipped_unmapped_rows += imported["skipped_unmapped_rows"]
            if imported["status"] == "empty":
                summary.skipped_empty += 1
                continue
            summary.imported_images += 1
            summary.imported_annotations += 1
            summary.imported_instances += imported["instances"]
    return summary


def _import_object_sample(
    *,
    image: np.ndarray,
    image_path: Path,
    rows: list[Any],
    class_map: dict[str, str],
    data_root: Path,
    dataset_root: Path,
    prefix: str,
    split: str,
    include_empty: bool,
) -> dict[str, Any]:
    height, width = image.shape[:2]
    objects: list[dict[str, Any]] = []
    skipped_unmapped = 0
    for row in rows:
        target_cls = _mapped_class(row.source_cls, row.class_id, class_map)
        if target_cls is None:
            skipped_unmapped += 1
            continue
        if target_cls not in OBJECT_CLASSES:
            raise ValueError(f"mapped object class is not supported by schema.md: {target_cls!r}")
        objects.append(
            {
                "cls": target_cls,
                "bbox": row_to_pixel_bbox(row, width, height),
                "source_cls": row.source_cls,
                "source_class_id": row.class_id,
            }
        )
    if not objects and not include_empty:
        return {"status": "empty", "instances": 0, "skipped_unmapped_rows": skipped_unmapped}

    output_name = _output_name(prefix, split, image_path)
    image_output = data_root / "raw" / "objects" / "images" / output_name
    annotation_output = data_root / "annotations" / "objects" / f"{Path(output_name).stem}.json"
    image_output.parent.mkdir(parents=True, exist_ok=True)
    annotation_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_output)
    annotation = {
        "image": repo_relative(data_root, image_output),
        "source_image": str(image_path.resolve()),
        "source_dataset": str(dataset_root),
        "source_split": split,
        "width": int(width),
        "height": int(height),
        "objects": objects,
    }
    annotation_output.write_text(json.dumps(annotation, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "imported", "instances": len(objects), "skipped_unmapped_rows": skipped_unmapped}


def _import_laneseg_sample(
    *,
    image: np.ndarray,
    image_path: Path,
    rows: list[Any],
    class_map: dict[str, str],
    data_root: Path,
    dataset_root: Path,
    prefix: str,
    split: str,
    include_empty: bool,
) -> dict[str, Any]:
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons: list[dict[str, Any]] = []
    skipped_unmapped = 0
    for row in rows:
        target_cls = _mapped_class(row.source_cls, row.class_id, class_map)
        if target_cls is None:
            skipped_unmapped += 1
            continue
        points = row_to_pixel_polygon(row, width, height)
        cv2.fillPoly(mask, [points], 255)
        polygons.append(
            {
                "cls": target_cls,
                "source_cls": row.source_cls,
                "source_class_id": row.class_id,
                "points": points.astype(int).tolist(),
            }
        )
    if not polygons and not include_empty:
        return {"status": "empty", "instances": 0, "skipped_unmapped_rows": skipped_unmapped}

    output_name = _output_name(prefix, split, image_path)
    image_output = data_root / "raw" / "sidewalk" / "images" / output_name
    mask_output = data_root / "annotations" / "laneseg" / "masks" / f"{Path(output_name).stem}.png"
    annotation_output = data_root / "annotations" / "laneseg" / f"{Path(output_name).stem}.json"
    image_output.parent.mkdir(parents=True, exist_ok=True)
    mask_output.parent.mkdir(parents=True, exist_ok=True)
    annotation_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_output)
    write_image(mask_output, mask)
    annotation = {
        "image": repo_relative(data_root, image_output),
        "mask": repo_relative(data_root, mask_output),
        "source_image": str(image_path.resolve()),
        "source_dataset": str(dataset_root),
        "source_split": split,
        "width": int(width),
        "height": int(height),
        "laneseg": {
            "classes": sorted({item["cls"] for item in polygons}),
            "positive_pixels": int(np.count_nonzero(mask)),
        },
        "polygons": polygons,
    }
    annotation_output.write_text(json.dumps(annotation, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "imported", "instances": len(polygons), "skipped_unmapped_rows": skipped_unmapped}


def print_summary(summary: ImportSummary, *, as_json: bool = False) -> None:
    payload = {
        "task": summary.task,
        "imported_images": summary.imported_images,
        "imported_annotations": summary.imported_annotations,
        "imported_instances": summary.imported_instances,
        "skipped_unreadable": summary.skipped_unreadable,
        "skipped_missing_label": summary.skipped_missing_label,
        "skipped_empty": summary.skipped_empty,
        "skipped_unmapped_rows": summary.skipped_unmapped_rows,
        "invalid_label_rows": summary.invalid_label_rows,
        "status": "ok" if summary.invalid_label_rows == 0 else "warning",
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for key, value in payload.items():
        print(f"{key}={value}")


def _mapped_class(source_cls: str, class_id: int, class_map: dict[str, str]) -> str | None:
    if not class_map:
        return source_cls
    return class_map.get(source_cls, class_map.get(str(class_id)))


def _output_name(prefix: str, split: str, image_path: Path) -> str:
    safe_prefix = prefix.strip().replace(" ", "_")
    base = f"{safe_prefix}_{split}_{image_path.stem}" if safe_prefix else f"{split}_{image_path.stem}"
    return f"{base}{image_path.suffix.lower()}"


def _parse_splits(value: str) -> list[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in splits if item not in {"train", "valid", "test"}]
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported split(s): {invalid}")
    return splits


def main() -> int:
    parser = argparse.ArgumentParser(description="Import YOLO detection or segmentation data into the project layout.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="YOLO dataset root containing data.yaml.")
    parser.add_argument("--data-root", default=Path("data"), type=Path, help="Project data directory.")
    parser.add_argument("--task", choices=("objects", "laneseg"), required=True)
    parser.add_argument("--class-map", action="append", help="Class mapping in source=target format. Repeat as needed.")
    parser.add_argument("--prefix", required=True, help="Stable prefix for imported images and annotations.")
    parser.add_argument("--splits", default=["train", "valid", "test"], type=_parse_splits)
    parser.add_argument("--include-empty", action="store_true", help="Keep images without mapped labels as negative samples.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    args = parser.parse_args()
    summary = import_dataset(
        dataset_root=args.dataset_root,
        data_root=args.data_root,
        task=args.task,
        class_map=parse_class_map(args.class_map),
        prefix=args.prefix,
        splits=args.splits,
        include_empty=args.include_empty,
    )
    print_summary(summary, as_json=args.json)
    return 0 if summary.invalid_label_rows == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
