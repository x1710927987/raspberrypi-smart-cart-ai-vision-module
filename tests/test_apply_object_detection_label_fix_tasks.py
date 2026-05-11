import csv
import importlib.util
import sys
from pathlib import Path
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "apply_object_detection_label_fix_tasks.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("apply_object_detection_label_fix_tasks", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_apply_label_fix_tasks_dry_run_does_not_write_or_backup():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_apply_object_detection_label_fix_tasks" / uuid4().hex
    label_path = workspace / "dataset" / "test" / "labels" / "sample.txt"
    task_csv = workspace / "tasks.csv"
    backup_dir = workspace / "backups"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("3 0.10000000 0.10000000 0.10000000 0.10000000\n", encoding="utf-8")
    _write_task_csv(task_csv, label_path)

    summary = script.apply_label_fix_tasks(task_csv=task_csv, backup_dir=backup_dir, dry_run=True)

    assert summary.dry_run
    assert summary.appended == 1
    assert summary.skipped_duplicates == 2
    assert "3 0.30000000 0.30000000 0.20000000 0.20000000" not in label_path.read_text(encoding="utf-8")
    assert not backup_dir.exists()


def test_apply_label_fix_tasks_appends_once_and_backs_up_original():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_apply_object_detection_label_fix_tasks_apply" / uuid4().hex
    label_path = workspace / "dataset" / "test" / "labels" / "sample.txt"
    task_csv = workspace / "tasks.csv"
    backup_dir = workspace / "backups"
    original = "3 0.10000000 0.10000000 0.10000000 0.10000000\n"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(original, encoding="utf-8")
    _write_task_csv(task_csv, label_path)

    summary = script.apply_label_fix_tasks(task_csv=task_csv, backup_dir=backup_dir, dry_run=False)
    second_summary = script.apply_label_fix_tasks(task_csv=task_csv, backup_dir=backup_dir / "second", dry_run=False)
    label_text = label_path.read_text(encoding="utf-8")
    backup_files = list(backup_dir.rglob("sample.txt"))

    assert summary.appended == 1
    assert summary.skipped_duplicates == 2
    assert len(backup_files) == 1
    assert backup_files[0].read_text(encoding="utf-8") == original
    assert label_text.count("3 0.30000000 0.30000000 0.20000000 0.20000000") == 1
    assert second_summary.appended == 0
    assert second_summary.skipped_duplicates == 3


def test_apply_label_fix_tasks_cli_json(monkeypatch, capsys):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_apply_object_detection_label_fix_tasks_cli" / uuid4().hex
    label_path = workspace / "dataset" / "test" / "labels" / "sample.txt"
    task_csv = workspace / "tasks.csv"
    backup_dir = workspace / "backups"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("", encoding="utf-8")
    _write_task_csv(task_csv, label_path, include_duplicate_existing=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_object_detection_label_fix_tasks.py",
            "--task-csv",
            str(task_csv),
            "--backup-dir",
            str(backup_dir),
            "--dry-run",
            "--json",
        ],
    )

    code = script.main()
    output = capsys.readouterr().out

    assert code == 0
    assert '"dry_run": true' in output
    assert '"appended": 1' in output


def _write_task_csv(task_csv: Path, label_path: Path, *, include_duplicate_existing: bool = True) -> None:
    task_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "task_id": "add_label_0001",
            "case_id": "case_0001",
            "label_path": label_path.resolve().relative_to(REPO_ROOT).as_posix(),
            "yolo_row": "3 0.30000000 0.30000000 0.20000000 0.20000000",
        },
        {
            "task_id": "add_label_0002",
            "case_id": "case_0002",
            "label_path": label_path.resolve().relative_to(REPO_ROOT).as_posix(),
            "yolo_row": "3 0.300000001 0.300000001 0.200000001 0.200000001",
        },
    ]
    if include_duplicate_existing:
        rows.append(
            {
                "task_id": "add_label_0003",
                "case_id": "case_0003",
                "label_path": label_path.resolve().relative_to(REPO_ROOT).as_posix(),
                "yolo_row": "3 0.10000000 0.10000000 0.10000000 0.10000000",
            }
        )
    with task_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
