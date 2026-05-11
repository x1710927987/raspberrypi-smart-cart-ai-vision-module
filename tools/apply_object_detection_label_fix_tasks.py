from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_CSV = REPO_ROOT / "cache" / "evaluation" / "object_detection_scooter_add_label_tasks.csv"
DEFAULT_BACKUP_ROOT = REPO_ROOT / "cache" / "label_backups" / "object_detection_scooter_add_label_tasks"


@dataclass(frozen=True)
class LabelFixTask:
    task_id: str
    case_id: str
    label_path: Path
    yolo_row: str


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    case_id: str
    label_path: str
    status: str
    yolo_row: str
    reason: str = ""


@dataclass(frozen=True)
class ApplySummary:
    task_csv: str
    dry_run: bool
    backup_dir: str
    total_tasks: int
    appended: int
    skipped_duplicates: int
    skipped_invalid: int
    missing_labels: int
    modified_files: list[str]
    backed_up_files: list[str]
    outcomes: list[TaskOutcome]


def apply_label_fix_tasks(
    *,
    task_csv: str | Path = DEFAULT_TASK_CSV,
    backup_dir: str | Path | None = None,
    dry_run: bool = False,
) -> ApplySummary:
    task_csv_path = _resolve_path(Path(task_csv))
    backup_dir_path = _resolve_path(Path(backup_dir)) if backup_dir is not None else _default_backup_dir()
    tasks = read_tasks(task_csv_path)
    outcomes: list[TaskOutcome] = []
    modified_files: list[str] = []
    backed_up_files: list[str] = []
    backup_done: set[Path] = set()
    label_cache: dict[Path, list[str]] = {}
    normalized_cache: dict[Path, set[str]] = {}

    for task in tasks:
        label_path = _resolve_path(task.label_path)
        label_text_path = _repo_relative_posix(label_path)
        normalized_row = _normalize_yolo_row(task.yolo_row)
        if normalized_row is None:
            outcomes.append(_outcome(task, label_path, "skipped_invalid", "invalid yolo_row"))
            continue
        if not label_path.exists():
            outcomes.append(_outcome(task, label_path, "missing_label", "label file does not exist"))
            continue

        if label_path not in label_cache:
            lines = _read_label_lines(label_path)
            label_cache[label_path] = lines
            normalized_cache[label_path] = {_normalize_yolo_row(line) or line.strip() for line in lines if line.strip()}

        if normalized_row in normalized_cache[label_path]:
            outcomes.append(_outcome(task, label_path, "skipped_duplicate", "row already exists"))
            continue

        outcomes.append(_outcome(task, label_path, "would_append" if dry_run else "appended"))
        label_cache[label_path].append(normalized_row)
        normalized_cache[label_path].add(normalized_row)
        if label_text_path not in modified_files:
            modified_files.append(label_text_path)

        if dry_run:
            continue
        if label_path not in backup_done:
            backup_path = _backup_label_file(label_path, backup_dir_path)
            backed_up_files.append(_repo_relative_posix(backup_path))
            backup_done.add(label_path)
        _write_label_lines(label_path, label_cache[label_path])

    return ApplySummary(
        task_csv=_repo_relative_posix(task_csv_path),
        dry_run=dry_run,
        backup_dir=_repo_relative_posix(backup_dir_path),
        total_tasks=len(tasks),
        appended=sum(1 for item in outcomes if item.status in {"would_append", "appended"}),
        skipped_duplicates=sum(1 for item in outcomes if item.status == "skipped_duplicate"),
        skipped_invalid=sum(1 for item in outcomes if item.status == "skipped_invalid"),
        missing_labels=sum(1 for item in outcomes if item.status == "missing_label"),
        modified_files=modified_files,
        backed_up_files=backed_up_files,
        outcomes=outcomes,
    )


def read_tasks(path: Path) -> list[LabelFixTask]:
    if not path.exists():
        raise FileNotFoundError(f"task CSV does not exist: {path}")
    tasks: list[LabelFixTask] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=2):
            task_id = str(row.get("task_id", "")).strip()
            case_id = str(row.get("case_id", "")).strip()
            label_path = str(row.get("label_path", "")).strip()
            yolo_row = str(row.get("yolo_row", "")).strip()
            if not task_id or not case_id or not label_path or not yolo_row:
                raise ValueError(f"{path}:{index}: missing required task_id/case_id/label_path/yolo_row")
            tasks.append(LabelFixTask(task_id=task_id, case_id=case_id, label_path=Path(label_path), yolo_row=yolo_row))
    return tasks


def print_summary(summary: ApplySummary, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
        return
    print(f"task_csv={summary.task_csv}")
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


def _outcome(task: LabelFixTask, label_path: Path, status: str, reason: str = "") -> TaskOutcome:
    return TaskOutcome(
        task_id=task.task_id,
        case_id=task.case_id,
        label_path=_repo_relative_posix(label_path),
        status=status,
        yolo_row=_normalize_yolo_row(task.yolo_row) or task.yolo_row.strip(),
        reason=reason,
    )


def _normalize_yolo_row(line: str) -> str | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return None
    if class_id < 0:
        return None
    if any(value < 0.0 or value > 1.0 for value in coords):
        return None
    if coords[2] <= 0.0 or coords[3] <= 0.0:
        return None
    return f"{class_id} " + " ".join(f"{value:.8f}" for value in coords)


def _read_label_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_label_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _backup_label_file(label_path: Path, backup_dir: Path) -> Path:
    relative = label_path.resolve().relative_to(REPO_ROOT.resolve())
    backup_path = backup_dir / relative
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(label_path, backup_path)
    return backup_path


def _default_backup_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_BACKUP_ROOT / stamp


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_relative_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safely append pending YOLO label rows from object-detection label-fix tasks.")
    parser.add_argument("--task-csv", default=DEFAULT_TASK_CSV, type=Path)
    parser.add_argument("--backup-dir", type=Path, help="Backup directory. Defaults to cache/label_backups/.../<timestamp>.")
    parser.add_argument("--dry-run", action="store_true", help="Report planned changes without writing labels or backups.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        summary = apply_label_fix_tasks(task_csv=args.task_csv, backup_dir=args.backup_dir, dry_run=args.dry_run)
        print_summary(summary, as_json=args.json)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
