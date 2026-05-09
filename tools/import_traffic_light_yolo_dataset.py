from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2

from audit_traffic_light_yolo_dataset import IMAGE_EXTENSIONS, read_image, read_roboflow_yaml, yolo_row_to_bbox


SCHEMA_STATES = {"red", "yellow", "green", "off", "flashing", "unknown"}
DEFAULT_STATE_MAP = {"red": "red", "yellow": "yellow", "green": "green"}


def import_traffic_light_dataset(
    *,
    source_root: Path,
    data_root: Path,
    prefix: str,
    state_map: dict[str, str],
    min_crop_size: int,
) -> tuple[int, int]:
    source_root = source_root.resolve()
    data_root = data_root.resolve()
    config = read_roboflow_yaml(source_root / "data.yaml")
    names = config["names"]
    annotations_dir = data_root / "annotations" / "traffic_light"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    imported_crops = 0
    skipped_rows = 0

    for split in ("train", "valid", "test"):
        images_dir = source_root / split / "images"
        labels_dir = source_root / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue
        labels_by_stem = {path.stem: path for path in labels_dir.rglob("*.txt") if path.is_file()}
        for image_path in sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
            label_path = labels_by_stem.get(image_path.stem)
            if label_path is None:
                continue
            image = read_image(image_path)
            if image is None or image.size == 0:
                skipped_rows += 1
                continue
            height, width = image.shape[:2]
            for row_index, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                line = raw_line.strip()
                if not line:
                    continue
                parsed = _parse_label_row(line, names, state_map, width, height)
                if parsed is None:
                    skipped_rows += 1
                    continue
                state, bbox = parsed
                x1, y1, x2, y2 = [int(round(value)) for value in bbox]
                crop = image[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < min_crop_size or crop.shape[1] < min_crop_size:
                    skipped_rows += 1
                    continue
                safe_stem = _safe_name(image_path.stem)
                output_name = f"{prefix}_{split}_{safe_stem}_{row_index:03d}{image_path.suffix.lower()}"
                output_image = data_root / "raw" / "traffic_light" / state / output_name
                output_image.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output_image), crop):
                    skipped_rows += 1
                    continue
                annotation = {
                    "image": output_image.relative_to(data_root).as_posix(),
                    "traffic_light": {"state": state},
                    "source": {
                        "dataset": source_root.name,
                        "split": split,
                        "image": str(image_path),
                        "label": str(label_path),
                    },
                    "bbox_original": [round(value, 2) for value in bbox],
                    "crop_width": int(crop.shape[1]),
                    "crop_height": int(crop.shape[0]),
                }
                annotation_path = annotations_dir / f"{Path(output_name).stem}.json"
                annotation_path.write_text(json.dumps(annotation, indent=2, ensure_ascii=False), encoding="utf-8")
                imported_crops += 1
    return imported_crops, skipped_rows


def _parse_label_row(line: str, names: list[str], state_map: dict[str, str], width: int, height: int) -> tuple[str, list[float]] | None:
    parts = line.split()
    try:
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return None
    if class_id < 0 or class_id >= len(names):
        return None
    source_state = names[class_id]
    state = state_map.get(source_state)
    if state not in SCHEMA_STATES:
        return None
    bbox = yolo_row_to_bbox(coords)
    if bbox is None:
        return None
    cx, cy, box_w, box_h = bbox
    x1 = max(0.0, (cx - box_w / 2.0) * width)
    y1 = max(0.0, (cy - box_h / 2.0) * height)
    x2 = min(float(width), (cx + box_w / 2.0) * width)
    y2 = min(float(height), (cy + box_h / 2.0) * height)
    if x2 <= x1 or y2 <= y1:
        return None
    return state, [x1, y1, x2, y2]


def _load_state_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return dict(DEFAULT_STATE_MAP)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("state map must be a JSON object")
    mapping = {str(key): str(value) for key, value in payload.items()}
    for value in mapping.values():
        if value not in SCHEMA_STATES:
            raise ValueError(f"unsupported traffic light schema state: {value!r}")
    return mapping


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    normalized = normalized.strip("._-")
    return normalized or "sample"


def main() -> int:
    parser = argparse.ArgumentParser(description="Import YOLO traffic-light labels as cropped state samples and JSON annotations.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--data-root", default=Path("data"), type=Path)
    parser.add_argument("--prefix", default="roboflow_traffic_light_v1")
    parser.add_argument("--state-map", type=Path, help="Optional JSON mapping from source class names to schema states.")
    parser.add_argument("--min-crop-size", default=8, type=int)
    args = parser.parse_args()
    imported, skipped = import_traffic_light_dataset(
        source_root=args.source_root,
        data_root=args.data_root,
        prefix=args.prefix,
        state_map=_load_state_map(args.state_map),
        min_crop_size=args.min_crop_size,
    )
    print(f"imported_crops={imported}")
    print(f"skipped_rows={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
