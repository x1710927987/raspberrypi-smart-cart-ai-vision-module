from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from audit_traffic_light_yolo_dataset import IMAGE_EXTENSIONS, read_image, read_roboflow_yaml, yolo_row_to_bbox


CLASS_NAMES = ["green", "red", "yellow"]


@dataclass(frozen=True)
class MergeSource:
    root: Path
    prefix: str


def merge_datasets(sources: list[MergeSource], output_root: str | Path) -> dict[str, int]:
    if not sources:
        raise ValueError("at least one source dataset is required")
    output_path = Path(output_root).resolve()
    if output_path.exists():
        raise FileExistsError(f"output directory already exists: {output_path}")

    images_dir = output_path / "train" / "images"
    labels_dir = output_path / "train" / "labels"
    images_dir.mkdir(parents=True, exist_ok=False)
    labels_dir.mkdir(parents=True, exist_ok=False)

    copied_images = 0
    copied_labels = 0
    skipped_pairs = 0
    label_rows = 0
    for source in sources:
        counts = _copy_source(source, images_dir, labels_dir)
        copied_images += counts["images"]
        copied_labels += counts["labels"]
        skipped_pairs += counts["skipped_pairs"]
        label_rows += counts["label_rows"]

    _write_data_yaml(output_path / "data.yaml")
    return {
        "images": copied_images,
        "labels": copied_labels,
        "label_rows": label_rows,
        "skipped_pairs": skipped_pairs,
    }


def _copy_source(source: MergeSource, output_images_dir: Path, output_labels_dir: Path) -> dict[str, int]:
    root = source.root.resolve()
    config = read_roboflow_yaml(root / "data.yaml")
    if list(config["names"]) != CLASS_NAMES:
        raise ValueError(f"{root}: expected classes {CLASS_NAMES}, got {config['names']}")

    copied_images = 0
    copied_labels = 0
    skipped_pairs = 0
    label_rows = 0
    for split in ("train", "valid", "test"):
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue
        labels_by_stem = {path.stem: path for path in labels_dir.rglob("*.txt") if path.is_file()}
        for image_path in sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
            label_path = labels_by_stem.get(image_path.stem)
            if label_path is None:
                skipped_pairs += 1
                continue
            image = read_image(image_path)
            if image is None or image.size == 0:
                skipped_pairs += 1
                continue
            repaired = _repair_label(label_path)
            if not repaired.strip():
                skipped_pairs += 1
                continue
            output_stem = _unique_stem(source.prefix, split, image_path.stem)
            output_image = output_images_dir / f"{output_stem}{image_path.suffix.lower()}"
            output_label = output_labels_dir / f"{output_stem}.txt"
            shutil.copy2(image_path, output_image)
            output_label.write_text(repaired, encoding="utf-8")
            copied_images += 1
            copied_labels += 1
            label_rows += len([line for line in repaired.splitlines() if line.strip()])
    return {"images": copied_images, "labels": copied_labels, "label_rows": label_rows, "skipped_pairs": skipped_pairs}


def _repair_label(label_path: Path) -> str:
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
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            continue
        bbox = yolo_row_to_bbox(coords)
        if bbox is None:
            continue
        cx, cy, box_w, box_h = bbox
        rows.append(f"{class_id} {cx:.8f} {cy:.8f} {box_w:.8f} {box_h:.8f}")
    return "\n".join(rows) + ("\n" if rows else "")


def _write_data_yaml(path: Path) -> None:
    names_text = "[" + ", ".join(f"'{name}'" for name in CLASS_NAMES) + "]"
    path.write_text(
        "\n".join(
            [
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "",
                f"nc: {len(CLASS_NAMES)}",
                f"names: {names_text}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    for split in ("valid", "test"):
        (path.parent / split / "images").mkdir(parents=True, exist_ok=True)
        (path.parent / split / "labels").mkdir(parents=True, exist_ok=True)


def _unique_stem(prefix: str, split: str, stem: str) -> str:
    safe_prefix = _safe_name(prefix)
    safe_split = _safe_name(split)
    safe_stem = _safe_name(stem)
    return f"{safe_prefix}_{safe_split}_{safe_stem}"


def _safe_name(value: str) -> str:
    chars = [char if char.isalnum() or char in "._-" else "_" for char in value]
    text = "".join(chars).strip("._-")
    return text or "sample"


def _parse_source(value: str) -> MergeSource:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must use prefix=path format")
    prefix, path = value.split("=", 1)
    prefix = prefix.strip()
    if not prefix:
        raise argparse.ArgumentTypeError("source prefix must not be empty")
    return MergeSource(root=Path(path), prefix=prefix)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge traffic-light YOLO datasets into one train-only source pool.")
    parser.add_argument("--source", required=True, action="append", type=_parse_source, help="Source in prefix=path format.")
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        counts = merge_datasets(args.source, args.output_root)
        print(f"output_root={Path(args.output_root).resolve()}")
        for key, value in counts.items():
            print(f"{key}={value}")
        print("status=ok")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
