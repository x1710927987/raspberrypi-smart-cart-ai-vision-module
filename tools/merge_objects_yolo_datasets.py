from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from yolo_dataset_utils import dataset_split_dirs, format_yolo_row, iter_images, parse_yolo_rows, read_image, read_yolo_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "external" / "objects_combined_v2"
DEFAULT_CLASSES = ("pedestrian", "bicycle", "car", "scooter", "roadblock")
DEFAULT_SOURCES = (
    REPO_ROOT / "data" / "processed" / "objects_yolo_v1",
    REPO_ROOT / "data" / "external" / "roboflow_bicycle_v1",
    REPO_ROOT / "data" / "external" / "roboflow_electric-scooter_v1",
    REPO_ROOT / "data" / "external" / "roboflow_roadblock_v1",
)


def merge_objects_yolo_datasets(
    source_roots: list[Path],
    output_root: Path,
    *,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    overwrite: bool = False,
) -> dict[str, int]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_root}")
        shutil.rmtree(output_root)

    output_images, output_labels = dataset_split_dirs(output_root, "train")
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    class_to_id = {name: index for index, name in enumerate(classes)}
    counts = {"images": 0, "labels": 0, "objects": 0}
    class_counts = {name: 0 for name in classes}
    skipped_counts: dict[str, int] = {}

    for source_root in source_roots:
        source_root = source_root.resolve()
        config = read_yolo_config(source_root / "data.yaml")
        source_prefix = source_root.name
        for image_path, label_path in _iter_labeled_images(source_root):
            image = read_image(image_path)
            if image is None or image.size == 0:
                continue
            rows, _invalids = parse_yolo_rows(label_path, config.names)
            remapped_rows = []
            for row in rows:
                target_class = normalize_object_class(row.source_cls, source_root=source_root)
                if target_class not in class_to_id:
                    skipped_counts[target_class] = skipped_counts.get(target_class, 0) + 1
                    continue
                remapped_rows.append(replace(row, class_id=class_to_id[target_class], source_cls=target_class))
                class_counts[target_class] += 1
            if not remapped_rows:
                continue
            output_name = f"{source_prefix}_{image_path.name}"
            shutil.copy2(image_path, output_images / output_name)
            label_text = "\n".join(format_yolo_row(row) for row in remapped_rows) + "\n"
            (output_labels / f"{Path(output_name).stem}.txt").write_text(label_text, encoding="utf-8")
            counts["images"] += 1
            counts["labels"] += 1
            counts["objects"] += len(remapped_rows)

    if counts["images"] == 0:
        raise ValueError("no labeled object samples were merged")
    _write_data_yaml(output_root / "data.yaml", classes)
    _write_readme(output_root / "README.generated.txt", source_roots, classes, counts, class_counts, skipped_counts)
    result = {**counts, **{f"class_{name}": count for name, count in class_counts.items()}}
    result.update({f"skipped_{name}": count for name, count in sorted(skipped_counts.items())})
    return result


def normalize_object_class(label: str, *, source_root: Path | None = None) -> str:
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    source_name = source_root.name.lower().replace("-", "_") if source_root is not None else ""
    if "roadblock" in source_name and normalized in {"cone", "traffic_cone", "roadblock", "barricade"}:
        return "roadblock"
    aliases = {
        "person": "pedestrian",
        "people": "pedestrian",
        "pedestrian": "pedestrian",
        "bike": "bicycle",
        "bicycles": "bicycle",
        "bicycle": "bicycle",
        "car": "car",
        "cars": "car",
        "vehicle": "car",
        "truck": "car",
        "bus": "car",
        "e_scooter": "scooter",
        "electric_scooter": "scooter",
        "escooter": "scooter",
        "scooter": "scooter",
        "motorbike": "scooter",
        "motorcycle": "scooter",
        "road_block": "roadblock",
        "road_barrier": "roadblock",
        "barricade": "roadblock",
        "roadblock": "roadblock",
    }
    return aliases.get(normalized, normalized)


def _iter_labeled_images(source_root: Path):
    for split in ("train", "valid", "test"):
        images_dir, labels_dir = dataset_split_dirs(source_root, split)
        if not images_dir.exists() or not labels_dir.exists():
            continue
        labels_by_stem = {path.stem: path for path in labels_dir.rglob("*.txt") if path.is_file()}
        for image_path in iter_images(images_dir):
            label_path = labels_by_stem.get(image_path.stem)
            if label_path is not None:
                yield image_path, label_path


def _write_data_yaml(path: Path, classes: tuple[str, ...]) -> None:
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


def _write_readme(
    path: Path,
    source_roots: list[Path],
    classes: tuple[str, ...],
    counts: dict[str, int],
    class_counts: dict[str, int],
    skipped_counts: dict[str, int],
) -> None:
    lines = [
        "Generated objects YOLO source pool.",
        "",
        f"classes: {', '.join(classes)}",
        f"images: {counts['images']}",
        f"labels: {counts['labels']}",
        f"objects: {counts['objects']}",
        "",
        "class_counts:",
        *[f"- {name}: {count}" for name, count in class_counts.items()],
        "",
        "skipped_counts:",
        *[f"- {name}: {count}" for name, count in sorted(skipped_counts.items())],
        "",
        "sources:",
        *[f"- {source}" for source in source_roots],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_classes(value: str) -> tuple[str, ...]:
    classes = tuple(item.strip() for item in value.split(",") if item.strip())
    if not classes:
        raise argparse.ArgumentTypeError("classes must not be empty")
    return classes


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge object YOLO datasets into one train-only source pool.")
    parser.add_argument("--source-root", action="append", type=Path, help="Source YOLO dataset root. Can be passed multiple times.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=Path)
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES), type=_parse_classes)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_roots = args.source_root or list(DEFAULT_SOURCES)
    try:
        counts = merge_objects_yolo_datasets(source_roots, args.output_root, classes=args.classes, overwrite=args.overwrite)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for key, value in counts.items():
        print(f"{key}={value}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
