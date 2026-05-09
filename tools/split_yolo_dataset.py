from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from yolo_dataset_utils import dataset_split_dirs, format_yolo_row, iter_images, parse_yolo_rows, read_image, read_yolo_config


def split_dataset(
    source_root: Path,
    output_root: Path,
    *,
    ratios: tuple[float, float, float],
    seed: int,
    overwrite: bool = False,
) -> dict[str, int]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    config = read_yolo_config(source_root / "data.yaml")
    pairs = _collect_pairs(source_root, config.names)
    random.Random(seed).shuffle(pairs)

    train_ratio, valid_ratio, test_ratio = ratios
    total_ratio = train_ratio + valid_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("split ratios must sum to a positive value")
    train_count = int(len(pairs) * train_ratio / total_ratio)
    valid_count = int(len(pairs) * valid_ratio / total_ratio)
    split_pairs = {
        "train": pairs[:train_count],
        "valid": pairs[train_count : train_count + valid_count],
        "test": pairs[train_count + valid_count :],
    }

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_root}")
        shutil.rmtree(output_root)
    for split_name, pairs_for_split in split_pairs.items():
        images_dir, labels_dir = dataset_split_dirs(output_root, split_name)
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for image_path, label_path, label_text in pairs_for_split:
            shutil.copy2(image_path, images_dir / image_path.name)
            (labels_dir / label_path.name).write_text(label_text, encoding="utf-8")
    _write_data_yaml(output_root / "data.yaml", config.names)
    for extra_name in ("README.dataset.txt", "README.roboflow.txt", "README.md"):
        extra_path = source_root / extra_name
        if extra_path.exists():
            shutil.copy2(extra_path, output_root / extra_path.name)
    return {name: len(items) for name, items in split_pairs.items()}


def _collect_pairs(source_root: Path, names: list[str]) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    seen: set[Path] = set()
    for split in ("train", "valid", "test"):
        images_dir, labels_dir = dataset_split_dirs(source_root, split)
        if not images_dir.exists() or not labels_dir.exists():
            continue
        labels_by_stem = {path.stem: path for path in labels_dir.rglob("*.txt") if path.is_file()}
        for image_path in iter_images(images_dir):
            resolved = image_path.resolve()
            if resolved in seen:
                continue
            label_path = labels_by_stem.get(image_path.stem)
            if label_path is None:
                continue
            image = read_image(image_path)
            if image is None or image.size == 0:
                continue
            label_text = _filtered_label_text(label_path, names)
            if not label_text.strip():
                continue
            pairs.append((image_path, label_path, label_text))
            seen.add(resolved)
    if not pairs:
        raise ValueError(f"no valid image/label pairs found under {source_root}")
    return pairs


def _filtered_label_text(label_path: Path, names: list[str]) -> str:
    rows, _invalids = parse_yolo_rows(label_path, names)
    lines = [format_yolo_row(row) for row in rows]
    return "\n".join(lines) + ("\n" if lines else "")


def _write_data_yaml(path: Path, names: list[str]) -> None:
    names_text = "[" + ", ".join(repr(name) for name in names) + "]"
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
    try:
        parts = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratios must use train,valid,test numeric format") from exc
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("ratios must use train,valid,test format")
    return parts[0], parts[1], parts[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy a YOLO dataset into train/valid/test splits.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--ratios", default=(0.7, 0.2, 0.1), type=_parse_ratios)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--overwrite", action="store_true", help="Replace the output directory if it already exists.")
    args = parser.parse_args()
    counts = split_dataset(args.source_root, args.output_root, ratios=args.ratios, seed=args.seed, overwrite=args.overwrite)
    print(f"train={counts['train']}")
    print(f"valid={counts['valid']}")
    print(f"test={counts['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
