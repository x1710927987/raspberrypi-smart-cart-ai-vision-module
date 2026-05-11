from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "cache" / "evaluation" / "object_detection_v3_test_mistakes.json"
DEFAULT_REPORT = REPO_ROOT / "cache" / "evaluation" / "object_detection_v3_error_report.md"
DEFAULT_REVIEW_CSV = REPO_ROOT / "cache" / "evaluation" / "object_detection_scooter_v3_error_review_template.csv"
DEFAULT_GALLERY_DIR = REPO_ROOT / "cache" / "evaluation" / "object_detection_error_gallery_v3" / "object_detection_v3_test_mistakes"
OBJECT_CLASSES = ("pedestrian", "bicycle", "car", "scooter", "roadblock", "unknown")
REVIEW_FIELDS = [
    "case_id",
    "target_class",
    "error_type",
    "source_image",
    "gallery_image",
    "gt_cls",
    "pred_cls",
    "confidence",
    "iou",
    "gt_bbox",
    "pred_bbox",
    "reason",
    "action",
    "priority",
    "reviewed_by",
    "notes",
]


@dataclass(frozen=True)
class ClassErrorStats:
    false_negative: int = 0
    false_positive: int = 0
    wrong_class_as_gt: int = 0
    wrong_class_as_pred: int = 0


@dataclass(frozen=True)
class ReviewCase:
    case_id: str
    target_class: str
    error_type: str
    source_image: str
    gallery_image: str
    gt_cls: str
    pred_cls: str
    confidence: float | None
    iou: float | None
    gt_bbox: str
    pred_bbox: str
    reason: str = ""
    action: str = ""
    priority: str = ""
    reviewed_by: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ErrorAnalysis:
    input_path: str
    total_images: int
    total_error_events: int
    class_errors: dict[str, ClassErrorStats]
    error_type_counts: dict[str, int]
    confusion_counts: dict[str, int]
    target_class: str
    target_review_cases: list[ReviewCase]


def analyze_mistake_file(
    path: str | Path,
    *,
    target_class: str = "scooter",
    gallery_dir: str | Path | None = DEFAULT_GALLERY_DIR,
) -> ErrorAnalysis:
    input_path = _resolve_path(Path(path))
    mistakes = load_mistakes(input_path)
    gallery_lookup = load_gallery_lookup(gallery_dir) if gallery_dir is not None else {}
    stats = {object_class: Counter() for object_class in OBJECT_CLASSES}
    error_type_counts: Counter[str] = Counter()
    confusion_counts: Counter[str] = Counter()
    review_cases: list[ReviewCase] = []
    target_class = _normalize_class(target_class)

    for image_index, mistake in enumerate(mistakes, start=1):
        source_image = str(mistake.get("image", ""))
        gallery_image = gallery_lookup.get(source_image, "")

        for item_index, gt_box in enumerate(_as_box_list(mistake.get("false_negatives")), start=1):
            cls = _normalize_class(str(gt_box.get("cls", "unknown")))
            stats[cls]["false_negative"] += 1
            error_type_counts["false_negative"] += 1
            confusion_counts[f"{cls}->unknown"] += 1
            if cls == target_class:
                review_cases.append(
                    _review_case(
                        image_index=image_index,
                        item_index=item_index,
                        target_class=target_class,
                        error_type="false_negative",
                        source_image=source_image,
                        gallery_image=gallery_image,
                        gt_box=gt_box,
                        pred_box=None,
                        iou=None,
                    )
                )

        for item_index, pred_box in enumerate(_as_box_list(mistake.get("false_positives")), start=1):
            cls = _normalize_class(str(pred_box.get("cls", "unknown")))
            stats[cls]["false_positive"] += 1
            error_type_counts["false_positive"] += 1
            confusion_counts[f"unknown->{cls}"] += 1
            if cls == target_class:
                review_cases.append(
                    _review_case(
                        image_index=image_index,
                        item_index=item_index,
                        target_class=target_class,
                        error_type="false_positive",
                        source_image=source_image,
                        gallery_image=gallery_image,
                        gt_box=None,
                        pred_box=pred_box,
                        iou=None,
                    )
                )

        for item_index, match in enumerate(_as_box_list(mistake.get("misclassified")), start=1):
            gt_box = match.get("gt") if isinstance(match.get("gt"), dict) else None
            pred_box = match.get("pred") if isinstance(match.get("pred"), dict) else None
            gt_cls = _normalize_class(str((gt_box or {}).get("cls", "unknown")))
            pred_cls = _normalize_class(str((pred_box or {}).get("cls", "unknown")))
            stats[gt_cls]["wrong_class_as_gt"] += 1
            stats[pred_cls]["wrong_class_as_pred"] += 1
            error_type_counts["wrong_class"] += 1
            confusion_counts[f"{gt_cls}->{pred_cls}"] += 1
            if target_class in {gt_cls, pred_cls}:
                review_cases.append(
                    _review_case(
                        image_index=image_index,
                        item_index=item_index,
                        target_class=target_class,
                        error_type="wrong_class",
                        source_image=source_image,
                        gallery_image=gallery_image,
                        gt_box=gt_box,
                        pred_box=pred_box,
                        iou=_float_or_none(match.get("iou")),
                    )
                )

    return ErrorAnalysis(
        input_path=_repo_relative_posix(input_path),
        total_images=len(mistakes),
        total_error_events=sum(error_type_counts.values()),
        class_errors={object_class: ClassErrorStats(**dict(stats[object_class])) for object_class in OBJECT_CLASSES},
        error_type_counts=dict(sorted(error_type_counts.items())),
        confusion_counts=dict(sorted(confusion_counts.items(), key=lambda item: (-item[1], item[0]))),
        target_class=target_class,
        target_review_cases=review_cases,
    )


def load_mistakes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"mistakes file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"mistakes file must contain a JSON list: {path}")
    return [_ensure_mapping(item, index, path) for index, item in enumerate(payload)]


def load_gallery_lookup(gallery_dir: str | Path) -> dict[str, str]:
    index_path = _resolve_path(Path(gallery_dir)) / "index.json"
    if not index_path.exists():
        return {}
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    lookup: dict[str, str] = {}
    if not isinstance(rows, list):
        return lookup
    for row in rows:
        if not isinstance(row, dict):
            continue
        image = str(row.get("image", ""))
        output = str(row.get("output", ""))
        if image and output:
            lookup[image] = output
    return lookup


def write_review_csv(path: str | Path, cases: list[ReviewCase]) -> None:
    output_path = _resolve_path(Path(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for case in cases:
            writer.writerow(_review_case_to_row(case))


def render_markdown(analysis: ErrorAnalysis) -> str:
    lines = [
        "# Object Detection Error Analysis",
        "",
        f"- input: `{analysis.input_path}`",
        f"- total_mistake_images: {analysis.total_images}",
        f"- total_error_events: {analysis.total_error_events}",
        f"- target_class: `{analysis.target_class}`",
        f"- target_review_cases: {len(analysis.target_review_cases)}",
        "",
        "## Error Type Counts",
        "",
        *_counter_lines(analysis.error_type_counts),
        "",
        "## Class Error Counts",
        "",
        "| class | FN | FP | wrong-as-GT | wrong-as-Pred |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for object_class in OBJECT_CLASSES:
        stats = analysis.class_errors[object_class]
        lines.append(
            f"| {object_class} | {stats.false_negative} | {stats.false_positive} | "
            f"{stats.wrong_class_as_gt} | {stats.wrong_class_as_pred} |"
        )
    lines.extend(
        [
            "",
            "## Common Confusions",
            "",
            *_counter_lines(analysis.confusion_counts, limit=20),
            "",
            f"## {analysis.target_class} Review Queue",
            "",
        ]
    )
    if not analysis.target_review_cases:
        lines.append("- none")
    else:
        for case in analysis.target_review_cases[:50]:
            lines.append(
                f"- {case.case_id}: {case.error_type}, gt={case.gt_cls}, pred={case.pred_cls}, "
                f"conf={_format_float(case.confidence)}, iou={_format_float(case.iou)}, image=`{case.source_image}`"
            )
        if len(analysis.target_review_cases) > 50:
            lines.append(f"- ... {len(analysis.target_review_cases) - 50} more cases omitted from markdown preview")
    lines.append("")
    return "\n".join(lines)


def print_summary(analysis: ErrorAnalysis) -> None:
    print(f"input={analysis.input_path}")
    print(f"total_mistake_images={analysis.total_images}")
    print(f"total_error_events={analysis.total_error_events}")
    print("error_type_counts=" + json.dumps(analysis.error_type_counts, ensure_ascii=False, sort_keys=True))
    print("class_errors=" + json.dumps({key: asdict(value) for key, value in analysis.class_errors.items()}, ensure_ascii=False, sort_keys=True))
    print("confusion_counts=" + json.dumps(analysis.confusion_counts, ensure_ascii=False, sort_keys=True))
    print(f"target_class={analysis.target_class}")
    print(f"target_review_cases={len(analysis.target_review_cases)}")
    print("status=ok")


def _review_case(
    *,
    image_index: int,
    item_index: int,
    target_class: str,
    error_type: str,
    source_image: str,
    gallery_image: str,
    gt_box: dict[str, Any] | None,
    pred_box: dict[str, Any] | None,
    iou: float | None,
) -> ReviewCase:
    case_id = f"{target_class}_{image_index:04d}_{item_index:02d}_{error_type}"
    return ReviewCase(
        case_id=case_id,
        target_class=target_class,
        error_type=error_type,
        source_image=source_image,
        gallery_image=gallery_image,
        gt_cls=_normalize_class(str((gt_box or {}).get("cls", ""))) if gt_box else "",
        pred_cls=_normalize_class(str((pred_box or {}).get("cls", ""))) if pred_box else "",
        confidence=_float_or_none((pred_box or {}).get("conf")) if pred_box else None,
        iou=iou,
        gt_bbox=_bbox_to_json((gt_box or {}).get("bbox")) if gt_box else "",
        pred_bbox=_bbox_to_json((pred_box or {}).get("bbox")) if pred_box else "",
    )


def _review_case_to_row(case: ReviewCase) -> dict[str, str]:
    return {
        "case_id": case.case_id,
        "target_class": case.target_class,
        "error_type": case.error_type,
        "source_image": case.source_image,
        "gallery_image": case.gallery_image,
        "gt_cls": case.gt_cls,
        "pred_cls": case.pred_cls,
        "confidence": "" if case.confidence is None else f"{case.confidence:.4f}".rstrip("0").rstrip("."),
        "iou": "" if case.iou is None else f"{case.iou:.4f}".rstrip("0").rstrip("."),
        "gt_bbox": case.gt_bbox,
        "pred_bbox": case.pred_bbox,
        "reason": case.reason,
        "action": case.action,
        "priority": case.priority,
        "reviewed_by": case.reviewed_by,
        "notes": case.notes,
    }


def _as_box_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _ensure_mapping(item: Any, index: int, path: Path) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"{path}:{index}: mistake item must be an object")
    return item


def _normalize_class(label: str) -> str:
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "person": "pedestrian",
        "people": "pedestrian",
        "e_scooter": "scooter",
        "electric_scooter": "scooter",
        "cone": "roadblock",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in OBJECT_CLASSES else "unknown"


def _bbox_to_json(value: Any) -> str:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return ""
    return json.dumps([round(float(item), 2) for item in value], separators=(",", ":"))


def _counter_lines(counter: dict[str, int], *, limit: int | None = None) -> list[str]:
    if not counter:
        return ["- none"]
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        items = items[:limit]
    return [f"- {key}: {value}" for key, value in items]


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


def _json_default(value: Any) -> Any:
    if isinstance(value, ClassErrorStats | ReviewCase):
        return asdict(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze object-detection mistake JSON and generate a target-class review CSV.")
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path, help="Mistake JSON from tools/evaluate_object_detection_model.py.")
    parser.add_argument("--target-class", default="scooter", help="Class to extract into the review CSV.")
    parser.add_argument("--gallery-dir", default=DEFAULT_GALLERY_DIR, type=Path, help="Optional visualized mistake gallery directory containing index.json.")
    parser.add_argument("--review-csv", default=DEFAULT_REVIEW_CSV, type=Path, help="Output review CSV for the target class.")
    parser.add_argument("--report", default=DEFAULT_REPORT, type=Path, help="Output markdown analysis report.")
    parser.add_argument("--json", action="store_true", help="Print JSON analysis instead of a text summary.")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        analysis = analyze_mistake_file(args.input, target_class=args.target_class, gallery_dir=args.gallery_dir)
        write_review_csv(args.review_csv, analysis.target_review_cases)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(analysis), encoding="utf-8")
        if args.json:
            print(json.dumps(asdict(analysis), ensure_ascii=False, indent=2, default=_json_default))
        else:
            print_summary(analysis)
            print(f"review_csv={_repo_relative_posix(_resolve_path(args.review_csv))}")
            print(f"report={_repo_relative_posix(_resolve_path(args.report))}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
