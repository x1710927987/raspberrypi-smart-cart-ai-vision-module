from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
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


def import_yolo_dataset(
    *,
    images_dir: Path,
    labels_dir: Path,
    data_root: Path,
    class_names: list[str],
    class_map: dict[str, str],
    prefix: str,
    copy_images: bool,
) -> tuple[int, int]:
    image_output_dir = data_root / "raw" / "objects" / "images"
    annotation_output_dir = data_root / "annotations" / "objects"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    annotation_output_dir.mkdir(parents=True, exist_ok=True)

    imported_images = 0
    imported_objects = 0
    for image_path in sorted(_iter_images(images_dir)):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            print(f"WARNING: skipped unreadable image: {image_path}")
            continue
        height, width = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        objects = _read_yolo_label(label_path, class_names, class_map, width, height)
        output_name = _output_image_name(prefix, image_path)
        output_image_path = image_output_dir / output_name
        if copy_images:
            shutil.copy2(image_path, output_image_path)
        else:
            output_image_path = image_path.resolve()

        annotation = {
            "image": _relative_to_data(data_root, output_image_path),
            "source_image": str(image_path.resolve()),
            "width": int(width),
            "height": int(height),
            "objects": objects,
        }
        annotation_path = annotation_output_dir / f"{Path(output_name).stem}.json"
        annotation_path.write_text(json.dumps(annotation, indent=2, ensure_ascii=False), encoding="utf-8")
        imported_images += 1
        imported_objects += len(objects)

    return imported_images, imported_objects


def _iter_images(images_dir: Path):
    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _read_yolo_label(label_path: Path, class_names: list[str], class_map: dict[str, str], width: int, height: int) -> list[dict[str, Any]]:
    if not label_path.exists():
        return []
    objects: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"{label_path}:{line_no}: expected YOLO row: class cx cy w h")
        class_index = int(float(parts[0]))
        if class_index < 0 or class_index >= len(class_names):
            raise ValueError(f"{label_path}:{line_no}: class index out of range: {class_index}")
        source_cls = class_names[class_index]
        target_cls = class_map.get(source_cls, class_map.get(str(class_index), "unknown"))
        if target_cls not in OBJECT_CLASSES:
            raise ValueError(f"{label_path}:{line_no}: mapped class is not supported: {target_cls!r}")
        cx, cy, box_w, box_h = [float(value) for value in parts[1:5]]
        bbox = _yolo_to_pixel_bbox(cx, cy, box_w, box_h, width, height)
        objects.append({"cls": target_cls, "bbox": bbox})
    return objects


def _yolo_to_pixel_bbox(cx: float, cy: float, box_w: float, box_h: float, width: int, height: int) -> list[float]:
    x1 = (cx - box_w / 2.0) * width
    y1 = (cy - box_h / 2.0) * height
    x2 = (cx + box_w / 2.0) * width
    y2 = (cy + box_h / 2.0) * height
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid YOLO bbox after conversion: {[cx, cy, box_w, box_h]}")
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def _output_image_name(prefix: str, image_path: Path) -> str:
    safe_prefix = prefix.strip().replace(" ", "_")
    return f"{safe_prefix}_{image_path.stem}{image_path.suffix.lower()}" if safe_prefix else image_path.name


def _relative_to_data(data_root: Path, path: Path) -> str:
    resolved_data_root = data_root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_data_root).as_posix()
    except ValueError:
        return str(resolved_path)


def _load_class_names(args: argparse.Namespace) -> list[str]:
    if args.class_names:
        return [item.strip() for item in args.class_names.split(",") if item.strip()]
    if args.names_file:
        return [line.strip() for line in args.names_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise ValueError("provide --class-names or --names-file")


def _load_class_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("class map file must contain a JSON object")
    return {str(key): str(value) for key, value in payload.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a YOLO detection dataset into data/raw and data/annotations.")
    parser.add_argument("--images-dir", required=True, type=Path, help="Directory containing YOLO images.")
    parser.add_argument("--labels-dir", required=True, type=Path, help="Directory containing YOLO .txt labels.")
    parser.add_argument("--data-root", default=Path("data"), type=Path, help="Project data directory.")
    parser.add_argument("--class-names", help="Comma-separated source class names in YOLO index order.")
    parser.add_argument("--names-file", type=Path, help="Text file with one source class name per line.")
    parser.add_argument("--class-map", type=Path, help="JSON object mapping source names or indices to schema classes.")
    parser.add_argument("--prefix", default="external", help="Prefix added to imported image and annotation names.")
    parser.add_argument("--link-only", action="store_true", help="Write annotations referencing original images without copying images.")
    args = parser.parse_args()

    class_names = _load_class_names(args)
    class_map = _load_class_map(args.class_map)
    images, objects = import_yolo_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        data_root=args.data_root,
        class_names=class_names,
        class_map=class_map,
        prefix=args.prefix,
        copy_images=not args.link_only,
    )
    print(f"imported_images={images}")
    print(f"imported_objects={objects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
