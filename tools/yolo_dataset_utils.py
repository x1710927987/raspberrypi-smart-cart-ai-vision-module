from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class YoloConfig:
    names: list[str]
    nc: int | None = None
    train: str | None = None
    val: str | None = None
    test: str | None = None


@dataclass(frozen=True)
class YoloRow:
    class_id: int
    source_cls: str
    coords: list[float]
    row_type: str


def read_yolo_config(path: Path) -> YoloConfig:
    if not path.exists():
        raise FileNotFoundError(f"missing YOLO data config: {path}")
    raw: dict[str, Any] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "names" and not value:
            names: list[str] = []
            while index < len(lines):
                item = lines[index].strip()
                if not item.startswith("-"):
                    break
                names.append(item[1:].strip().strip("'\""))
                index += 1
            raw[key] = names
        elif key == "names":
            raw[key] = _parse_names(value)
        elif key == "nc":
            raw[key] = int(value)
        elif key in {"train", "val", "test"}:
            raw[key] = value.strip("'\"")
    names = raw.get("names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"{path}: missing names")
    nc = raw.get("nc")
    if nc is not None and int(nc) != len(names):
        raise ValueError(f"{path}: nc does not match names length")
    return YoloConfig(
        names=[str(name) for name in names],
        nc=int(nc) if nc is not None else None,
        train=raw.get("train"),
        val=raw.get("val"),
        test=raw.get("test"),
    )


def parse_class_map(entries: Iterable[str] | None) -> dict[str, str]:
    class_map: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"class map entry must use source=target format: {entry!r}")
        source, target = entry.split("=", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise ValueError(f"class map entry must use source=target format: {entry!r}")
        class_map[source] = target
    return class_map


def iter_images(images_dir: Path) -> Iterable[Path]:
    if not images_dir.exists():
        return
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"failed to encode image: {path}")
    encoded.tofile(str(path))


def parse_yolo_rows(label_path: Path, names: list[str]) -> tuple[list[YoloRow], list[str]]:
    rows: list[YoloRow] = []
    invalids: list[str] = []
    if not label_path.exists():
        return rows, invalids
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except (IndexError, ValueError):
            invalids.append(f"{label_path}:{line_no}: non-numeric YOLO row")
            continue
        if class_id < 0 or class_id >= len(names):
            invalids.append(f"{label_path}:{line_no}: class index out of range: {class_id}")
            continue
        row_type = infer_row_type(coords)
        if row_type is None:
            invalids.append(f"{label_path}:{line_no}: expected bbox or polygon row, got {len(parts)} columns")
            continue
        rows.append(YoloRow(class_id=class_id, source_cls=names[class_id], coords=coords, row_type=row_type))
    return rows, invalids


def infer_row_type(coords: list[float]) -> str | None:
    if len(coords) == 4:
        cx, cy, box_w, box_h = coords
        if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < box_w <= 1.0 and 0.0 < box_h <= 1.0:
            return "bbox"
        return None
    if len(coords) >= 6 and len(coords) % 2 == 0 and all(0.0 <= value <= 1.0 for value in coords):
        xs = coords[0::2]
        ys = coords[1::2]
        return "polygon" if max(xs) > min(xs) and max(ys) > min(ys) else None
    return None


def row_to_pixel_bbox(row: YoloRow, width: int, height: int) -> list[float]:
    if row.row_type == "bbox":
        cx, cy, box_w, box_h = row.coords
        x1 = (cx - box_w / 2.0) * width
        y1 = (cy - box_h / 2.0) * height
        x2 = (cx + box_w / 2.0) * width
        y2 = (cy + box_h / 2.0) * height
    else:
        xs = row.coords[0::2]
        ys = row.coords[1::2]
        x1 = min(xs) * width
        y1 = min(ys) * height
        x2 = max(xs) * width
        y2 = max(ys) * height
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid YOLO bbox after conversion: {row.coords}")
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def row_to_pixel_polygon(row: YoloRow, width: int, height: int) -> np.ndarray:
    if row.row_type == "bbox":
        x1, y1, x2, y2 = row_to_pixel_bbox(row, width, height)
        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    else:
        points = [[row.coords[i] * width, row.coords[i + 1] * height] for i in range(0, len(row.coords), 2)]
    array = np.array(points, dtype=np.float32)
    array[:, 0] = np.clip(array[:, 0], 0, width - 1)
    array[:, 1] = np.clip(array[:, 1], 0, height - 1)
    return np.rint(array).astype(np.int32)


def format_yolo_row(row: YoloRow) -> str:
    coords = " ".join(f"{value:.8f}" for value in row.coords)
    return f"{row.class_id} {coords}"


def dataset_split_dirs(dataset_root: Path, split: str) -> tuple[Path, Path]:
    return dataset_root / split / "images", dataset_root / split / "labels"


def repo_relative(repo_root: Path, path: Path) -> str:
    resolved_root = repo_root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)


def _parse_names(value: str) -> list[str]:
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, list):
            raise ValueError("names must be a list")
        return [str(item) for item in parsed]
    return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]
