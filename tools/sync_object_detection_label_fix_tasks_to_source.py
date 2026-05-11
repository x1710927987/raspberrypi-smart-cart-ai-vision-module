from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from apply_object_detection_label_fix_tasks import apply_label_fix_tasks


DEFAULT_TASK_CSV = REPO_ROOT / "cache" / "evaluation" / "object_detection_scooter_add_label_tasks.csv"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "data" / "external" / "objects_combined_v2"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "cache" / "evaluation" / "object_detection_scooter_add_label_tasks_source_pool.csv"
DEFAULT_BACKUP_DIR = REPO_ROOT / "cache" / "label_backups" / "object_detection_scooter_add_label_tasks_source_pool"
TASK_FIELDS = ["task_id", "case_id", "label_path", "yolo_row"]


def build_source_pool_tasks(
    *,
    task_csv: str | Path = DEFAULT_TASK_CSV,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
) -> Path:
    task_csv = _resolve_path(Path(task_csv))
    source_root = _resolve_path(Path(source_root))
    output_csv = _resolve_path(Path(output_csv))
    rows = read_task_rows(task_csv)
    output_rows: list[dict[str, str]] = []
    for row in rows:
        original_label = Path(row["label_path"])
        source_label = source_root / "train" / "labels" / original_label.name
        output_rows.append(
            {
                "task_id": row["task_id"],
                "case_id": row["case_id"],
                "label_path": _repo_relative_posix(source_label),
                "yolo_row": row["yolo_row"],
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TASK_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_csv


def read_task_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"task CSV does not exist: {path}")
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=2):
            missing = [field for field in TASK_FIELDS if not row.get(field)]
            if missing:
                raise ValueError(f"{path}:{index}: missing required fields: {missing}")
            rows.append({field: str(row[field]).strip() for field in TASK_FIELDS})
    return rows


def print_plan(task_csv: Path, summary) -> None:
    print(f"source_task_csv={_repo_relative_posix(task_csv)}")
    print(f"dry_run={str(summary.dry_run).lower()}")
    print(f"backup_dir={summary.backup_dir}")
    print(f"total_tasks={summary.total_tasks}")
    print(f"appended={summary.appended}")
    print(f"skipped_duplicates={summary.skipped_duplicates}")
    print(f"skipped_invalid={summary.skipped_invalid}")
    print(f"missing_labels={summary.missing_labels}")
    print("modified_files=" + json.dumps(summary.modified_files, ensure_ascii=False))
    print("backed_up_files=" + json.dumps(summary.backed_up_files, ensure_ascii=False))
    print("status=ok")


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_relative_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync split label-fix tasks back to the objects source pool.")
    parser.add_argument("--task-csv", default=DEFAULT_TASK_CSV, type=Path)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT, type=Path)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, type=Path)
    parser.add_argument("--backup-dir", default=DEFAULT_BACKUP_DIR, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        source_task_csv = build_source_pool_tasks(task_csv=args.task_csv, source_root=args.source_root, output_csv=args.output_csv)
        summary = apply_label_fix_tasks(task_csv=source_task_csv, backup_dir=args.backup_dir, dry_run=args.dry_run)
        print_plan(source_task_csv, summary)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
