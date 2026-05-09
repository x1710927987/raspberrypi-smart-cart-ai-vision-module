from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("cache/evaluation/traffic_light_error_review_template.csv")

BUCKETS = {
    "add_similar_data": "补数据",
    "fix_label": "修标注",
    "adjust_postprocess": "调阈值/后处理",
    "lower_threshold": "调阈值/后处理",
    "raise_threshold": "调阈值/后处理",
    "ignore_sample": "忽略/剔除",
}

BUCKET_ORDER = ["补数据", "修标注", "调阈值/后处理", "忽略/剔除", "待确认"]


@dataclass(frozen=True)
class ReviewCase:
    case_id: str
    split: str
    primary_gt: str
    predicted_state: str
    confidence: float | None
    reason: str
    action: str
    priority: str
    gallery_image: str
    source_image: str
    notes: str


@dataclass(frozen=True)
class ReviewAnalysis:
    input_path: str
    total_cases: int
    reviewed_cases: int
    missing_reason: int
    missing_action: int
    reason_counts: dict[str, int]
    action_counts: dict[str, int]
    split_counts: dict[str, int]
    confusion_counts: dict[str, int]
    buckets: dict[str, list[ReviewCase]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def analyze_review_csv(path: str | Path) -> ReviewAnalysis:
    input_path = Path(path)
    rows = _read_rows(input_path)
    cases = [_case_from_row(row) for row in rows]
    reason_counts = Counter(case.reason or "<empty>" for case in cases)
    action_counts = Counter(case.action or "<empty>" for case in cases)
    split_counts = Counter(case.split or "<empty>" for case in cases)
    confusion_counts = Counter(f"{case.primary_gt}->{case.predicted_state}" for case in cases)

    buckets: dict[str, list[ReviewCase]] = {bucket: [] for bucket in BUCKET_ORDER}
    warnings: list[str] = []
    for case in cases:
        bucket = BUCKETS.get(case.action, "待确认")
        buckets[bucket].append(case)
        if not case.reason:
            warnings.append(f"{case.case_id}: missing reason")
        if not case.action:
            warnings.append(f"{case.case_id}: missing action")
        if case.action == "lower_threshold" and case.confidence == 0.0:
            warnings.append(f"{case.case_id}: lower_threshold with confidence=0.0 may be risky; check whether a low-confidence candidate existed")
        if case.action == "ignore_sample" and case.reason in {"missed_detection", "wrong_color"}:
            warnings.append(f"{case.case_id}: ignore_sample may hide a learnable model failure")

    return ReviewAnalysis(
        input_path=str(input_path),
        total_cases=len(cases),
        reviewed_cases=sum(1 for case in cases if case.reason and case.action),
        missing_reason=sum(1 for case in cases if not case.reason),
        missing_action=sum(1 for case in cases if not case.action),
        reason_counts=dict(sorted(reason_counts.items())),
        action_counts=dict(sorted(action_counts.items())),
        split_counts=dict(sorted(split_counts.items())),
        confusion_counts=dict(sorted(confusion_counts.items())),
        buckets=buckets,
        warnings=warnings,
    )


def render_markdown(analysis: ReviewAnalysis) -> str:
    lines = [
        "# Traffic Light Error Review Decision Report",
        "",
        f"- input: `{analysis.input_path}`",
        f"- total_cases: {analysis.total_cases}",
        f"- reviewed_cases: {analysis.reviewed_cases}",
        f"- missing_reason: {analysis.missing_reason}",
        f"- missing_action: {analysis.missing_action}",
        "",
        "## Reason Counts",
        "",
        *_counter_lines(analysis.reason_counts),
        "",
        "## Action Counts",
        "",
        *_counter_lines(analysis.action_counts),
        "",
        "## Split Counts",
        "",
        *_counter_lines(analysis.split_counts),
        "",
        "## Common Confusions",
        "",
        *_counter_lines(analysis.confusion_counts),
        "",
        "## Decisions",
        "",
    ]
    for bucket in BUCKET_ORDER:
        cases = analysis.buckets.get(bucket, [])
        lines.append(f"### {bucket} ({len(cases)})")
        if not cases:
            lines.append("")
            lines.append("- none")
            lines.append("")
            continue
        for case in cases:
            note = f" | note: {case.notes}" if case.notes else ""
            priority = f" | priority: {case.priority}" if case.priority else ""
            lines.append(
                f"- {case.case_id}: {case.primary_gt}->{case.predicted_state}, "
                f"reason={case.reason}, action={case.action}, conf={_format_conf(case.confidence)}{priority}{note}"
            )
        lines.append("")

    lines.append("## Warnings")
    lines.append("")
    if analysis.warnings:
        lines.extend(f"- {warning}" for warning in analysis.warnings)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def print_summary(analysis: ReviewAnalysis) -> None:
    print(f"input={analysis.input_path}")
    print(f"total_cases={analysis.total_cases}")
    print(f"reviewed_cases={analysis.reviewed_cases}")
    print(f"missing_reason={analysis.missing_reason}")
    print(f"missing_action={analysis.missing_action}")
    for bucket in BUCKET_ORDER:
        print(f"{bucket}={len(analysis.buckets.get(bucket, []))}")
    print(f"warnings={len(analysis.warnings)}")
    print("status=ok" if not analysis.missing_reason and not analysis.missing_action else "status=needs_review")


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"review CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _case_from_row(row: dict[str, str]) -> ReviewCase:
    return ReviewCase(
        case_id=row.get("case_id", "").strip(),
        split=row.get("split", "").strip(),
        primary_gt=row.get("primary_gt", "").strip(),
        predicted_state=row.get("predicted_state", "").strip(),
        confidence=_float_or_none(row.get("confidence", "")),
        reason=row.get("reason", "").strip(),
        action=row.get("action", "").strip(),
        priority=row.get("priority", "").strip(),
        gallery_image=row.get("gallery_image", "").strip(),
        source_image=row.get("source_image", "").strip(),
        notes=row.get("notes", "").strip(),
    )


def _counter_lines(counter: dict[str, int]) -> list[str]:
    if not counter:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]


def _format_conf(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}".rstrip("0").rstrip(".")


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, ReviewCase):
        return asdict(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze traffic-light error review CSV and group actions into decision buckets.")
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path, help="Review CSV path.")
    parser.add_argument("--output", type=Path, help="Optional markdown report output path.")
    parser.add_argument("--json", action="store_true", help="Print JSON analysis instead of a text summary.")
    args = parser.parse_args()

    try:
        analysis = analyze_review_csv(args.input)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(render_markdown(analysis), encoding="utf-8")
        if args.json:
            print(json.dumps(asdict(analysis), ensure_ascii=False, indent=2, default=_json_default))
        else:
            print_summary(analysis)
            if args.output:
                print(f"report={args.output}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
