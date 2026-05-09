from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from audit_traffic_light_yolo_dataset import IMAGE_EXTENSIONS, read_image, read_roboflow_yaml, yolo_row_to_bbox


def split_dataset(source_root: Path, output_root: Path, *, ratios: tuple[float, float, float], seed: int) -> dict[str, int]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    config = read_roboflow_yaml(source_root / "data.yaml")
    pairs = _collect_pairs(source_root, config["names"])
    random.Random(seed).shuffle(pairs)

    train_ratio, valid_ratio, test_ratio = ratios
    total_ratio = train_ratio + valid_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("split ratios must sum to a positive value")
    train_count = int(len(pairs) * train_ratio / total_ratio)
    valid_count = int(len(pairs) * valid_ratio / total_ratio)
    splits = {
        "train": pairs[:train_count],
        "valid": pairs[train_count : train_count + valid_count],
        "test": pairs[train_count + valid_count :],
    }

    if output_root.exists():
        raise FileExistsError(f"output directory already exists: {output_root}")
    for split_name, split_pairs in splits.items():
        images_dir = output_root / split_name / "images"
        labels_dir = output_root / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for image_path, label_path in split_pairs:
            shutil.copy2(image_path, images_dir / image_path.name)
            repaired = _repair_label(label_path, config["names"])
            (labels_dir / label_path.name).write_text(repaired, encoding="utf-8")
    _write_data_yaml(output_root / "data.yaml", config["names"])
    readme = source_root / "README.roboflow.txt"
    if readme.exists():
        shutil.copy2(readme, output_root / readme.name)
    return {name: len(items) for name, items in splits.items()}


def _collect_pairs(source_root: Path, names: list[str]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for split in ("train", "valid", "test"):
        images_dir = source_root / split / "images"
        labels_dir = source_root / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue
        labels_by_stem = {path.stem: path for path in labels_dir.rglob("*.txt") if path.is_file()}
        for image_path in sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
            if image_path.resolve() in seen:
                continue
            label_path = labels_by_stem.get(image_path.stem)
            if label_path is None:
                continue
            image = read_image(image_path)
            if image is None or image.size == 0:
                continue
            repaired = _repair_label(label_path, names)
            if not repaired.strip():
                continue
            pairs.append((image_path, label_path))
            seen.add(image_path.resolve())
    if not pairs:
        raise ValueError(f"no image/label pairs found under {source_root}")
    return pairs


def _repair_label(label_path: Path, names: list[str]) -> str:
    rows: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except (IndexError, ValueError):
            continue
        if class_id < 0 or class_id >= len(names):
            continue
        bbox = yolo_row_to_bbox(coords)
        if bbox is None:
            continue
        cx, cy, box_w, box_h = bbox
        rows.append(f"{class_id} {cx:.8f} {cy:.8f} {box_w:.8f} {box_h:.8f}")
    return "\n".join(rows) + ("\n" if rows else "")


def _write_data_yaml(path: Path, names: list[str]) -> None:
    names_text = "[" + ", ".join(f"'{name}'" for name in names) + "]"
    path.write_text(
        "\n".join(
            [
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "",
                f"nc: {len(names)}",
                f"names: {names_text}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _parse_ratios(value: str) -> tuple[float, float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("ratios must use train,valid,test format")
    return parts[0], parts[1], parts[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy a YOLO traffic-light dataset into train/valid/test splits.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--ratios", default=(0.7, 0.2, 0.1), type=_parse_ratios)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    counts = split_dataset(args.source_root, args.output_root, ratios=args.ratios, seed=args.seed)
    print(f"train={counts['train']}")
    print(f"valid={counts['valid']}")
    print(f"test={counts['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
