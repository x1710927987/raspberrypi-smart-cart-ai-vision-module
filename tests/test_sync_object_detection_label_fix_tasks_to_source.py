import csv
import importlib.util
import sys
from pathlib import Path
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "sync_object_detection_label_fix_tasks_to_source.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("sync_object_detection_label_fix_tasks_to_source", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sync_label_fix_tasks_maps_split_labels_to_source_pool_and_applies():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_sync_object_detection_label_fix_tasks_to_source" / uuid4().hex
    split_label = workspace / "split" / "test" / "labels" / "sample.txt"
    source_root = workspace / "source"
    source_label = source_root / "train" / "labels" / "sample.txt"
    task_csv = workspace / "tasks.csv"
    source_task_csv = workspace / "source_tasks.csv"
    split_label.parent.mkdir(parents=True, exist_ok=True)
    source_label.parent.mkdir(parents=True, exist_ok=True)
    split_label.write_text("", encoding="utf-8")
    source_label.write_text("", encoding="utf-8")
    _write_task_csv(task_csv, split_label)

    generated = script.build_source_pool_tasks(task_csv=task_csv, source_root=source_root, output_csv=source_task_csv)
    summary = script.apply_label_fix_tasks(task_csv=generated, backup_dir=workspace / "backup", dry_run=False)
    rows = list(csv.DictReader(source_task_csv.open("r", encoding="utf-8-sig", newline="")))

    assert rows[0]["label_path"].endswith("source/train/labels/sample.txt")
    assert summary.appended == 1
    assert "3 0.50000000 0.50000000 0.20000000 0.20000000" in source_label.read_text(encoding="utf-8")


def test_sync_label_fix_tasks_cli_dry_run(monkeypatch, capsys):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_sync_object_detection_label_fix_tasks_to_source_cli" / uuid4().hex
    split_label = workspace / "split" / "test" / "labels" / "sample.txt"
    source_root = workspace / "source"
    source_label = source_root / "train" / "labels" / "sample.txt"
    task_csv = workspace / "tasks.csv"
    source_task_csv = workspace / "source_tasks.csv"
    split_label.parent.mkdir(parents=True, exist_ok=True)
    source_label.parent.mkdir(parents=True, exist_ok=True)
    split_label.write_text("", encoding="utf-8")
    source_label.write_text("", encoding="utf-8")
    _write_task_csv(task_csv, split_label)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync_object_detection_label_fix_tasks_to_source.py",
            "--task-csv",
            str(task_csv),
            "--source-root",
            str(source_root),
            "--output-csv",
            str(source_task_csv),
            "--backup-dir",
            str(workspace / "backup"),
            "--dry-run",
        ],
    )

    code = script.main()
    output = capsys.readouterr().out

    assert code == 0
    assert "dry_run=true" in output
    assert "appended=1" in output
    assert source_label.read_text(encoding="utf-8") == ""


def _write_task_csv(path: Path, label_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "task_id": "add_label_0001",
        "case_id": "case_0001",
        "label_path": label_path.resolve().relative_to(REPO_ROOT).as_posix(),
        "yolo_row": "3 0.50000000 0.50000000 0.20000000 0.20000000",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
