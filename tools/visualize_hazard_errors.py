from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "cache" / "evaluation" / "hazard_error_gallery"


@dataclass(frozen=True)
class VisualizeSummary:
    input_file: Path
    output_dir: Path
    total_items: int
    rendered: int
    missing_images: int
    unreadable_images: int


def visualize_error_files(
    mistake_files: list[str | Path],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
) -> list[VisualizeSummary]:
    output_root = Path(output_dir)
    summaries: list[VisualizeSummary] = []
    for mistake_file in mistake_files:
        mistake_path = _resolve_path(Path(mistake_file))
        mistakes = load_mistakes(mistake_path)
        if limit is not None:
            mistakes = mistakes[:limit]
        group_output_dir = output_root / _safe_name(mistake_path.stem)
        summaries.append(visualize_mistakes(mistakes, input_file=mistake_path, output_dir=group_output_dir))
    return summaries


def load_mistakes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"mistakes file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"mistakes file must contain a JSON list: {path}")
    mistakes: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{index}: mistake item must be an object")
        if "image" not in item:
            raise ValueError(f"{path}:{index}: missing image field")
        mistakes.append(item)
    return mistakes


def visualize_mistakes(mistakes: list[dict[str, Any]], *, input_file: Path, output_dir: Path) -> VisualizeSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    missing_images = 0
    unreadable_images = 0
    index_rows: list[dict[str, Any]] = []

    for index, mistake in enumerate(mistakes, start=1):
        image_path = _resolve_path(Path(str(mistake["image"])))
        if not image_path.exists():
            missing_images += 1
            index_rows.append(_index_row(index, mistake, image_path, None, "missing_image"))
            continue
        frame = read_image(image_path)
        if frame is None:
            unreadable_images += 1
            index_rows.append(_index_row(index, mistake, image_path, None, "unreadable_image"))
            continue
        annotated = annotate_image(frame, mistake)
        output_path = output_dir / f"{index:04d}_{_safe_name(image_path.stem)}.jpg"
        ok, encoded = cv2.imencode(".jpg", annotated)
        if not ok:
            unreadable_images += 1
            index_rows.append(_index_row(index, mistake, image_path, output_path, "write_failed"))
            continue
        encoded.tofile(str(output_path))
        rendered += 1
        index_rows.append(_index_row(index, mistake, image_path, output_path, "rendered"))

    (output_dir / "index.json").write_text(json.dumps(index_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return VisualizeSummary(input_file=input_file, output_dir=output_dir, total_items=len(mistakes), rendered=rendered, missing_images=missing_images, unreadable_images=unreadable_images)


def annotate_image(frame: np.ndarray, mistake: dict[str, Any]) -> np.ndarray:
    source = frame.copy()
    height, width = source.shape[:2]
    banner_height = max(84, min(140, height // 5 if height >= 420 else 84))
    canvas = np.zeros((height + banner_height, width, 3), dtype=np.uint8)
    canvas[:banner_height, :] = _hazard_color(str(mistake.get("predicted_type", "unknown")))
    canvas[banner_height:, :] = source

    for gt_box in mistake.get("gt_boxes", []):
        if not isinstance(gt_box, dict):
            continue
        _draw_bbox(canvas, gt_box.get("bbox"), banner_height, color=(40, 220, 40), label=f"GT {gt_box.get('type', 'unknown')}")

    predicted_bbox = mistake.get("predicted_bbox")
    if predicted_bbox:
        _draw_bbox(canvas, predicted_bbox, banner_height, color=(40, 40, 230), label=f"PRED {mistake.get('predicted_type', 'unknown')}")

    gt = ",".join(str(item) for item in mistake.get("gt_hazards", [])) or str(mistake.get("primary_gt", "unknown"))
    pred = str(mistake.get("predicted_type", "unknown"))
    conf = float(mistake.get("confidence", 0.0))
    lines = [
        f"GT: {gt}    Pred: {pred}    Conf: {conf:.4f}",
        Path(str(mistake.get("image", ""))).name,
    ]
    _draw_lines(canvas, lines, origin=(12, 28), max_width=width - 24)
    return canvas


def read_image(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def print_summaries(summaries: list[VisualizeSummary], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps([_summary_to_dict(item) for item in summaries], ensure_ascii=False, separators=(",", ":")))
        return
    for summary in summaries:
        print(f"input={summary.input_file}")
        print(f"output_dir={summary.output_dir}")
        print(f"total_items={summary.total_items}")
        print(f"rendered={summary.rendered}")
        print(f"missing_images={summary.missing_images}")
        print(f"unreadable_images={summary.unreadable_images}")
    print("status=ok")


def _draw_bbox(canvas: np.ndarray, bbox: Any, y_offset: int, *, color: tuple[int, int, int], label: str) -> None:
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    y1 += y_offset
    y2 += y_offset
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    cv2.putText(canvas, label, (x1, max(y_offset + 18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def _draw_lines(canvas: np.ndarray, lines: list[str], *, origin: tuple[int, int], max_width: int) -> None:
    x, y = origin
    for line in lines:
        for chunk in _wrap_text(line, max_chars=max(16, max_width // 10)):
            cv2.putText(canvas, chunk, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            y += 26


def _wrap_text(text: str, *, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars)
        if split_at <= 0:
            split_at = max_chars
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _hazard_color(hazard_type: str) -> tuple[int, int, int]:
    normalized = hazard_type.strip().lower()
    if normalized == "pothole":
        return (45, 65, 180)
    if normalized == "curb":
        return (150, 95, 35)
    return (90, 90, 90)


def _index_row(index: int, mistake: dict[str, Any], image_path: Path, output_path: Path | None, status: str) -> dict[str, Any]:
    return {
        "index": index,
        "status": status,
        "image": _repo_relative_posix(image_path),
        "output": _repo_relative_posix(output_path) if output_path is not None else None,
        "gt_hazards": mistake.get("gt_hazards", []),
        "primary_gt": mistake.get("primary_gt"),
        "predicted_type": mistake.get("predicted_type"),
        "confidence": mistake.get("confidence"),
    }


def _summary_to_dict(summary: VisualizeSummary) -> dict[str, Any]:
    return {
        "input_file": _repo_relative_posix(summary.input_file),
        "output_dir": _repo_relative_posix(summary.output_dir),
        "total_items": summary.total_items,
        "rendered": summary.rendered,
        "missing_images": summary.missing_images,
        "unreadable_images": summary.unreadable_images,
    }


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_relative_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    normalized = normalized.strip("._-")
    return normalized or "item"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render hazard mistake JSON files into an annotated image gallery.")
    parser.add_argument("mistakes", nargs="+", type=Path, help="Mistake JSON files produced by tools/evaluate_hazard_model.py.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--limit", type=int, help="Render only the first N mistakes from each file.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        summaries = visualize_error_files(args.mistakes, output_dir=_resolve_path(args.output_dir), limit=args.limit)
        print_summaries(summaries, as_json=args.json)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
