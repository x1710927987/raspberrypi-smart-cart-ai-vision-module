from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from yolo_dataset_utils import read_yolo_config


DEFAULT_REVIEW_CSV = REPO_ROOT / "cache" / "evaluation" / "object_detection_scooter_error_review_template.csv"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "external" / "objects_combined_v2_split"
DEFAULT_TASK_CSV = REPO_ROOT / "cache" / "evaluation" / "object_detection_scooter_add_label_tasks.csv"
DEFAULT_TASK_MD = REPO_ROOT / "cache" / "evaluation" / "object_detection_scooter_add_label_tasks.md"
TASK_FIELDS = [
    "task_id",
    "case_id",
    "target_class",
    "class_id",
    "source_image",
    "label_path",
    "gallery_image",
    "confidence",
    "pred_bbox_xyxy",
    "image_width",
    "image_height",
    "yolo_row",
    "label_exists",
    "status",
    "notes",
]


@dataclass(frozen=True)
class LabelFixTask:
    task_id: str
    case_id: str
    target_class: str
    class_id: int
    source_image: str
    label_path: str
    gallery_image: str
    confidence: float | None
    pred_bbox_xyxy: list[float]
    image_width: int
    image_height: int
    yolo_row: str
    label_exists: bool
    status: str = "pending_review"
    notes: str = ""


def build_label_fix_tasks(
    *,
    review_csv: str | Path = DEFAULT_REVIEW_CSV,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    action: str = "add_label",
) -> list[LabelFixTask]:
    review_csv = _resolve_path(Path(review_csv))
    dataset_root = _resolve_path(Path(dataset_root))
    config = read_yolo_config(dataset_root / "data.yaml")
    rows = _read_rows(review_csv)
    tasks: list[LabelFixTask] = []
    for row in rows:
        if row.get("action", "").strip() != action:
            continue
        target_class = row.get("target_class", "").strip()
        if target_class not in config.names:
            raise ValueError(f"{row.get('case_id', '<unknown>')}: target_class not in data.yaml names: {target_class!r}")
        image_path = _resolve_path(Path(row.get("source_image", "").strip()))
        if not image_path.exists():
            raise FileNotFoundError(f"{row.get('case_id', '<unknown>')}: source image does not exist: {image_path}")
        frame = read_image(image_path)
        if frame is None:
            raise ValueError(f"{row.get('case_id', '<unknown>')}: unreadable source image: {image_path}")
        height, width = frame.shape[:2]
        bbox = _parse_bbox(row.get("pred_bbox", ""))
        yolo_row = bbox_xyxy_to_yolo_row(class_id=config.names.index(target_class), bbox=bbox, width=width, height=height)
        label_path = image_to_label_path(image_path)
        task_id = f"add_label_{len(tasks) + 1:04d}"
        tasks.append(
            LabelFixTask(
                task_id=task_id,
                case_id=row.get("case_id", "").strip(),
                target_class=target_class,
                class_id=config.names.index(target_class),
                source_image=_repo_relative_posix(image_path),
                label_path=_repo_relative_posix(label_path),
                gallery_image=row.get("gallery_image", "").strip(),
                confidence=_float_or_none(row.get("confidence", "")),
                pred_bbox_xyxy=bbox,
                image_width=width,
                image_height=height,
                yolo_row=yolo_row,
                label_exists=label_path.exists(),
                notes=row.get("notes", "").strip(),
            )
        )
    return tasks


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        images_index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"cannot derive label path from image path without images directory: {image_path}") from exc
    parts[images_index] = "labels"
    label_path = Path(*parts).with_suffix(".txt")
    return label_path


def bbox_xyxy_to_yolo_row(*, class_id: int, bbox: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid bbox after clipping: {bbox}")
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height
    return f"{class_id} {cx:.8f} {cy:.8f} {box_w:.8f} {box_h:.8f}"


def write_task_csv(path: str | Path, tasks: list[LabelFixTask]) -> None:
    output_path = _resolve_path(Path(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TASK_FIELDS)
        writer.writeheader()
        for task in tasks:
            row = asdict(task)
            row["confidence"] = "" if task.confidence is None else f"{task.confidence:.4f}".rstrip("0").rstrip(".")
            row["pred_bbox_xyxy"] = json.dumps(task.pred_bbox_xyxy, separators=(",", ":"))
            row["label_exists"] = "yes" if task.label_exists else "no"
            writer.writerow(row)


def render_markdown(tasks: list[LabelFixTask]) -> str:
    lines = [
        "# Object Detection Pending Label Fix Tasks",
        "",
        f"- total_tasks: {len(tasks)}",
        f"- label_exists_yes: {sum(1 for task in tasks if task.label_exists)}",
        f"- label_exists_no: {sum(1 for task in tasks if not task.label_exists)}",
        "",
        "## Tasks",
        "",
    ]
    if not tasks:
        lines.append("- none")
        lines.append("")
        return "\n".join(lines)
    for task in tasks:
        lines.extend(
            [
                f"### {task.task_id} | {task.case_id}",
                "",
                f"- source_image: `{task.source_image}`",
                f"- label_path: `{task.label_path}`",
                f"- target_class: `{task.target_class}` (`{task.class_id}`)",
                f"- confidence: `{_format_float(task.confidence)}`",
                f"- pred_bbox_xyxy: `{json.dumps(task.pred_bbox_xyxy, separators=(',', ':'))}`",
                f"- yolo_row_to_append: `{task.yolo_row}`",
                f"- label_exists: `{task.label_exists}`",
                f"- gallery_image: `{task.gallery_image}`",
                f"- notes: {task.notes or 'none'}",
                "",
            ]
        )
    return "\n".join(lines)


def write_task_markdown(path: str | Path, tasks: list[LabelFixTask]) -> None:
    output_path = _resolve_path(Path(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(tasks), encoding="utf-8")


def read_image(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def print_summary(tasks: list[LabelFixTask], *, task_csv: Path, task_md: Path) -> None:
    print(f"tasks={len(tasks)}")
    print(f"label_exists_yes={sum(1 for task in tasks if task.label_exists)}")
    print(f"label_exists_no={sum(1 for task in tasks if not task.label_exists)}")
    print(f"task_csv={_repo_relative_posix(_resolve_path(task_csv))}")
    print(f"task_md={_repo_relative_posix(_resolve_path(task_md))}")
    print("status=ok")


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"review CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _parse_bbox(value: str) -> list[float]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid bbox JSON: {value!r}") from exc
    if not isinstance(payload, list) or len(payload) != 4:
        raise ValueError(f"bbox must be a 4-value JSON list: {value!r}")
    return [round(float(item), 2) for item in payload]


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}".rstrip("0").rstrip(".")


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_relative_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert add_label review rows into pending YOLO label-fix tasks.")
    parser.add_argument("--review-csv", default=DEFAULT_REVIEW_CSV, type=Path)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, type=Path)
    parser.add_argument("--action", default="add_label")
    parser.add_argument("--task-csv", default=DEFAULT_TASK_CSV, type=Path)
    parser.add_argument("--task-md", default=DEFAULT_TASK_MD, type=Path)
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        tasks = build_label_fix_tasks(review_csv=args.review_csv, dataset_root=args.dataset_root, action=args.action)
        write_task_csv(args.task_csv, tasks)
        write_task_markdown(args.task_md, tasks)
        print_summary(tasks, task_csv=args.task_csv, task_md=args.task_md)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
