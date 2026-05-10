from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT = Path("cache/evaluation/traffic_light_error_review_template_after_label_fix.csv")
DEFAULT_CSV_OUTPUT = Path("cache/evaluation/traffic_light_add_similar_data_plan.csv")
DEFAULT_MD_OUTPUT = Path("cache/evaluation/traffic_light_add_similar_data_plan.md")

OUTPUT_FIELDS = [
    "case_id",
    "split",
    "primary_gt",
    "predicted_state",
    "reason",
    "priority",
    "suggested_new_images",
    "recommended_focus",
    "source_image",
    "gallery_image",
    "notes",
]


@dataclass(frozen=True)
class AdditionPlan:
    input_path: Path
    output_csv: Path
    output_markdown: Path
    rows: list[dict[str, str]]
    class_counts: dict[str, int]
    reason_counts: dict[str, int]
    confusion_counts: dict[str, int]
    suggested_total_images: int


def create_addition_plan(
    review_csv: str | Path = DEFAULT_INPUT,
    *,
    output_csv: str | Path = DEFAULT_CSV_OUTPUT,
    output_markdown: str | Path = DEFAULT_MD_OUTPUT,
) -> AdditionPlan:
    input_path = Path(review_csv)
    rows = [_plan_row(row) for row in _read_add_similar_rows(input_path)]
    csv_path = Path(output_csv)
    md_path = Path(output_markdown)
    _write_csv(rows, csv_path)
    summary = _summarize(rows)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_render_markdown(input_path, rows, summary), encoding="utf-8")
    return AdditionPlan(
        input_path=input_path,
        output_csv=csv_path,
        output_markdown=md_path,
        rows=rows,
        class_counts=summary["class_counts"],
        reason_counts=summary["reason_counts"],
        confusion_counts=summary["confusion_counts"],
        suggested_total_images=summary["suggested_total_images"],
    )


def print_summary(plan: AdditionPlan) -> None:
    print(f"input={plan.input_path}")
    print(f"output_csv={plan.output_csv}")
    print(f"output_markdown={plan.output_markdown}")
    print(f"rows={len(plan.rows)}")
    print(f"class_counts={plan.class_counts}")
    print(f"reason_counts={plan.reason_counts}")
    print(f"confusion_counts={plan.confusion_counts}")
    print(f"suggested_total_images={plan.suggested_total_images}")
    print("status=ok")


def _read_add_similar_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"review CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return [row for row in rows if row.get("action", "").strip() == "add_similar_data"]


def _plan_row(row: dict[str, str]) -> dict[str, str]:
    primary_gt = row.get("primary_gt", "").strip() or "unknown"
    predicted_state = row.get("predicted_state", "").strip() or "unknown"
    reason = row.get("reason", "").strip() or "unknown"
    notes = row.get("notes", "").strip()
    return {
        "case_id": row.get("case_id", "").strip(),
        "split": row.get("split", "").strip(),
        "primary_gt": primary_gt,
        "predicted_state": predicted_state,
        "reason": reason,
        "priority": _priority_for(reason, primary_gt, predicted_state),
        "suggested_new_images": str(_suggested_new_images(reason)),
        "recommended_focus": _recommended_focus(primary_gt, predicted_state, reason, notes),
        "source_image": row.get("source_image", "").strip(),
        "gallery_image": row.get("gallery_image", "").strip(),
        "notes": notes,
    }


def _priority_for(reason: str, primary_gt: str, predicted_state: str) -> str:
    if predicted_state == "unknown":
        return "high"
    if reason == "wrong_color":
        return "medium"
    if reason == "too_small":
        return "medium"
    if primary_gt == "green":
        return "medium"
    return "normal"


def _suggested_new_images(reason: str) -> int:
    if reason == "missed_detection":
        return 80
    if reason == "too_small":
        return 80
    if reason == "wrong_color":
        return 60
    return 50


def _recommended_focus(primary_gt: str, predicted_state: str, reason: str, notes: str) -> str:
    cues: list[str] = []
    if reason == "missed_detection":
        cues.append(f"{primary_gt} lights that the detector misses")
    elif reason == "too_small":
        cues.append(f"small or distant {primary_gt} lights")
    elif reason == "wrong_color":
        cues.append(f"{primary_gt} lights confused as {predicted_state}")
    else:
        cues.append(f"hard {primary_gt} traffic-light examples")
    lowered = notes.lower()
    if "shell" in lowered:
        cues.append("non-black housings")
    if "perspective" in lowered or "straight" in lowered:
        cues.append("side-view or angled camera perspectives")
    if "snow" in lowered:
        cues.append("snow-covered housings and winter scenes")
    if "between the three lights" in lowered:
        cues.append("middle yellow lamps in vertical three-light heads")
    return "; ".join(cues)


def _write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    class_counts = Counter(row["primary_gt"] for row in rows)
    reason_counts = Counter(row["reason"] for row in rows)
    confusion_counts = Counter(f"{row['primary_gt']}->{row['predicted_state']}" for row in rows)
    total = sum(int(row["suggested_new_images"]) for row in rows)
    return {
        "class_counts": dict(sorted(class_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "confusion_counts": dict(sorted(confusion_counts.items())),
        "suggested_total_images": total,
    }


def _render_markdown(input_path: Path, rows: list[dict[str, str]], summary: dict[str, object]) -> str:
    lines = [
        "# Traffic Light Similar-Data Addition Plan",
        "",
        f"- input: `{input_path}`",
        f"- cases: {len(rows)}",
        f"- suggested_total_images: {summary['suggested_total_images']}",
        "",
        "## Class Targets",
        "",
        *_counter_lines(summary["class_counts"]),
        "",
        "## Reason Targets",
        "",
        *_counter_lines(summary["reason_counts"]),
        "",
        "## Confusion Targets",
        "",
        *_counter_lines(summary["confusion_counts"]),
        "",
        "## Case Plan",
        "",
    ]
    if not rows:
        lines.append("- none")
    for row in rows:
        lines.append(
            f"- {row['case_id']}: add about {row['suggested_new_images']} images for "
            f"{row['primary_gt']}->{row['predicted_state']} ({row['reason']}); "
            f"focus: {row['recommended_focus']}"
        )
    lines.append("")
    return "\n".join(lines)


def _counter_lines(counter: object) -> list[str]:
    if not isinstance(counter, dict) or not counter:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a traffic-light data addition plan from reviewed error cases.")
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path, help="Reviewed traffic-light error CSV.")
    parser.add_argument("--output-csv", default=DEFAULT_CSV_OUTPUT, type=Path)
    parser.add_argument("--output-markdown", default=DEFAULT_MD_OUTPUT, type=Path)
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        plan = create_addition_plan(args.input, output_csv=args.output_csv, output_markdown=args.output_markdown)
        print_summary(plan)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
