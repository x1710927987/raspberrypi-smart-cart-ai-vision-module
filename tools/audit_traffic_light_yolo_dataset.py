from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class SplitAudit:
    split: str
    images: int = 0
    labels: int = 0
    label_rows: int = 0
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    unreadable_images: list[str] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)
    missing_split: bool = False
    class_counts: Counter[str] = field(default_factory=Counter)


def audit_dataset(root: Path) -> tuple[dict[str, Any], list[SplitAudit]]:
    dataset_root = root.resolve()
    config = read_roboflow_yaml(dataset_root / "data.yaml")
    names = config.get("names", [])
    split_audits: list[SplitAudit] = []
    for split in ("train", "valid", "test"):
        split_audits.append(audit_split(dataset_root, split, names))
    return config, split_audits


def audit_split(dataset_root: Path, split: str, names: list[str]) -> SplitAudit:
    audit = SplitAudit(split)
    images_dir = dataset_root / split / "images"
    labels_dir = dataset_root / split / "labels"
    if not images_dir.exists() and not labels_dir.exists():
        audit.missing_split = True
        return audit

    image_paths = {path.stem: path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS} if images_dir.exists() else {}
    label_paths = {path.stem: path for path in labels_dir.rglob("*.txt") if path.is_file()} if labels_dir.exists() else {}
    audit.images = len(image_paths)
    audit.labels = len(label_paths)

    for stem, image_path in sorted(image_paths.items()):
        image = read_image(image_path)
        if image is None or image.size == 0:
            audit.unreadable_images.append(str(image_path))
        if stem not in label_paths:
            audit.missing_labels.append(str(image_path))

    for stem, label_path in sorted(label_paths.items()):
        if stem not in image_paths:
            audit.orphan_labels.append(str(label_path))
            continue
        image = read_image(image_paths[stem])
        if image is None or image.size == 0:
            continue
        height, width = image.shape[:2]
        rows, invalids, counts = parse_yolo_label(label_path, names, width, height)
        audit.label_rows += rows
        audit.invalid_labels.extend(invalids)
        audit.class_counts.update(counts)
    return audit


def parse_yolo_label(label_path: Path, names: list[str], width: int, height: int) -> tuple[int, list[str], Counter[str]]:
    rows = 0
    invalids: list[str] = []
    counts: Counter[str] = Counter()
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        rows += 1
        parts = line.split()
        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            invalids.append(f"{label_path}:{line_no}: non-numeric YOLO row")
            continue
        if class_id < 0 or class_id >= len(names):
            invalids.append(f"{label_path}:{line_no}: class index out of range: {class_id}")
            continue
        bbox = yolo_row_to_bbox(coords)
        if bbox is None:
            invalids.append(f"{label_path}:{line_no}: expected bbox or polygon row, got {len(parts)} columns")
            continue
        cx, cy, box_w, box_h = bbox
        x1 = (cx - box_w / 2.0) * width
        y1 = (cy - box_h / 2.0) * height
        x2 = (cx + box_w / 2.0) * width
        y2 = (cy + box_h / 2.0) * height
        if x2 <= x1 or y2 <= y1:
            invalids.append(f"{label_path}:{line_no}: invalid pixel bbox")
            continue
        counts[names[class_id]] += 1
    return rows, invalids, counts


def yolo_row_to_bbox(coords: list[float]) -> tuple[float, float, float, float] | None:
    if len(coords) == 4:
        cx, cy, box_w, box_h = coords
        return (cx, cy, box_w, box_h) if _valid_normalized_box(cx, cy, box_w, box_h) else None
    if len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        if any(value < 0.0 or value > 1.0 for value in xs + ys):
            return None
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0.0 or box_h <= 0.0:
            return None
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, box_w, box_h)
    return None


def read_roboflow_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing data.yaml: {path}")
    config: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("nc:"):
            config["nc"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("names:"):
            config["names"] = _parse_names(line.split(":", 1)[1].strip())
        elif line.startswith(("train:", "val:", "test:")):
            key, value = line.split(":", 1)
            config[key.strip()] = value.strip()
    if "names" not in config:
        raise ValueError(f"{path}: missing names")
    if "nc" in config and config["nc"] != len(config["names"]):
        raise ValueError(f"{path}: nc does not match names length")
    return config


def _parse_names(value: str) -> list[str]:
    text = value.strip()
    if not text.startswith("[") or not text.endswith("]"):
        return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]
    inner = text[1:-1]
    return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]


def _valid_normalized_box(cx: float, cy: float, box_w: float, box_h: float) -> bool:
    return 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < box_w <= 1.0 and 0.0 < box_h <= 1.0


def read_image(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def print_report(config: dict[str, Any], audits: list[SplitAudit]) -> None:
    print(f"classes={config['names']}")
    for audit in audits:
        print(f"[{audit.split}] images={audit.images} labels={audit.labels} label_rows={audit.label_rows}")
        print(f"[{audit.split}] class_counts={dict(sorted(audit.class_counts.items()))}")
        if audit.missing_split:
            print(f"[{audit.split}] missing_split=1")
        if audit.missing_labels:
            print(f"[{audit.split}] missing_labels={len(audit.missing_labels)}")
        if audit.orphan_labels:
            print(f"[{audit.split}] orphan_labels={len(audit.orphan_labels)}")
        if audit.unreadable_images:
            print(f"[{audit.split}] unreadable_images={len(audit.unreadable_images)}")
        if audit.invalid_labels:
            print(f"[{audit.split}] invalid_labels={len(audit.invalid_labels)}")
            for item in audit.invalid_labels[:10]:
                print(f"  {item}")
    has_errors = any(audit.unreadable_images or audit.invalid_labels for audit in audits)
    print("status=failed" if has_errors else "status=ok")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a Roboflow/YOLO traffic-light dataset.")
    parser.add_argument("--root", required=True, type=Path, help="Dataset root containing data.yaml and split folders.")
    args = parser.parse_args()
    config, audits = audit_dataset(args.root)
    print_report(config, audits)
    return 1 if any(audit.unreadable_images or audit.invalid_labels for audit in audits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
