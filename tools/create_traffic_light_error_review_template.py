from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "cache" / "evaluation" / "traffic_light_error_review_template.csv"
DEFAULT_INDEX_FILES = [
    REPO_ROOT / "cache" / "evaluation" / "traffic_light_error_gallery" / "traffic_light_valid_v1_mistakes" / "index.json",
    REPO_ROOT / "cache" / "evaluation" / "traffic_light_error_gallery" / "traffic_light_test_v1_mistakes" / "index.json",
]
FIELDNAMES = [
    "case_id",
    "split",
    "gallery_image",
    "source_image",
    "gt_states",
    "primary_gt",
    "predicted_state",
    "confidence",
    "status",
    "reason",
    "action",
    "priority",
    "reviewed_by",
    "notes",
]


def create_review_template(
    index_files: list[str | Path],
    output_path: str | Path = DEFAULT_OUTPUT,
    previous_review_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    previous_rows = _load_previous_review(previous_review_path)
    rows: list[dict[str, Any]] = []
    split_counters: dict[str, int] = {}
    for index_file in index_files:
        path = _resolve_path(Path(index_file))
        split = _infer_split(path)
        items = _load_index(path)
        for item in items:
            split_counters[split] = split_counters.get(split, 0) + 1
            row = _row_from_index_item(item, split=split, case_index=split_counters[split])
            _carry_previous_review(row, previous_rows)
            rows.append(row)
    rows.sort(key=lambda row: (row["split"], row["case_id"]))
    write_review_template(rows, _resolve_path(Path(output_path)))
    return rows


def write_review_template(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    split_counts: dict[str, int] = {}
    carried_reviews = 0
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        if row["reason"] or row["action"] or row["notes"]:
            carried_reviews += 1
    print(f"output={output_path}")
    print(f"rows={len(rows)}")
    print(f"split_counts={split_counts}")
    print(f"carried_reviews={carried_reviews}")
    print("status=ok")


def _load_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"index file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"index file must contain a JSON list: {path}")
    items: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{index}: index item must be an object")
        items.append(item)
    return items


def _row_from_index_item(item: dict[str, Any], *, split: str, case_index: int | None = None) -> dict[str, Any]:
    case_id = f"{split}_{int(case_index if case_index is not None else item.get('index', 0)):04d}"
    gt_states = item.get("gt_states", [])
    if isinstance(gt_states, list):
        gt_text = "|".join(str(value) for value in gt_states)
    else:
        gt_text = str(gt_states)
    return {
        "case_id": case_id,
        "split": split,
        "gallery_image": str(item.get("output", "")),
        "source_image": str(item.get("image", "")),
        "gt_states": gt_text,
        "primary_gt": str(item.get("primary_gt", "")),
        "predicted_state": str(item.get("predicted_state", "")),
        "confidence": item.get("confidence", ""),
        "status": str(item.get("status", "")),
        "reason": "",
        "action": "",
        "priority": "",
        "reviewed_by": "",
        "notes": "",
    }


def _load_previous_review(previous_review_path: str | Path | None) -> dict[str, dict[str, str]]:
    if previous_review_path is None:
        return {}
    path = _resolve_path(Path(previous_review_path))
    if not path.exists():
        raise FileNotFoundError(f"previous review CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    keyed_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        source_image = row.get("source_image", "").strip()
        if source_image:
            keyed_rows[source_image] = row
    return keyed_rows


def _carry_previous_review(row: dict[str, Any], previous_rows: dict[str, dict[str, str]]) -> None:
    previous = previous_rows.get(str(row.get("source_image", "")).strip())
    if previous is None:
        return
    for field in ["reason", "action", "priority", "reviewed_by", "notes"]:
        row[field] = previous.get(field, "")


def _infer_split(path: Path) -> str:
    name = path.parent.name.lower()
    if "valid" in name:
        return "valid"
    if "test" in name:
        return "test"
    if "train" in name:
        return "train"
    return path.parent.name


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a CSV template for manual traffic-light error review.")
    parser.add_argument("index_files", nargs="*", type=Path, default=DEFAULT_INDEX_FILES, help="Gallery index.json files.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument(
        "--previous-review",
        type=Path,
        help="Optional previous review CSV; reason/action/notes are carried over by source_image.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        index_files = args.index_files if args.index_files else DEFAULT_INDEX_FILES
        output_path = _resolve_path(args.output)
        rows = create_review_template(index_files, output_path, args.previous_review)
        print_summary(rows, output_path)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
