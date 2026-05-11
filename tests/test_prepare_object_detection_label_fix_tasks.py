import csv
import importlib.util
import json
import sys
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "prepare_object_detection_label_fix_tasks.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_object_detection_label_fix_tasks", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_label_fix_tasks_converts_pred_bbox_to_yolo_row():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_prepare_object_detection_label_fix_tasks" / uuid4().hex
    dataset_root = workspace / "dataset"
    image_path = dataset_root / "test" / "images" / "sample.jpg"
    label_path = dataset_root / "test" / "labels" / "sample.txt"
    review_csv = workspace / "review.csv"
    task_csv = workspace / "tasks.csv"
    task_md = workspace / "tasks.md"
    _write_image(image_path, width=100, height=80)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.50000000 0.50000000 0.10000000 0.10000000\n", encoding="utf-8")
    (dataset_root / "data.yaml").write_text("nc: 5\nnames: ['pedestrian', 'bicycle', 'car', 'scooter', 'roadblock']\n", encoding="utf-8")
    _write_review_csv(review_csv, image_path)

    tasks = script.build_label_fix_tasks(review_csv=review_csv, dataset_root=dataset_root)
    script.write_task_csv(task_csv, tasks)
    script.write_task_markdown(task_md, tasks)
    rows = list(csv.DictReader(task_csv.open("r", encoding="utf-8-sig", newline="")))

    assert len(tasks) == 1
    assert tasks[0].class_id == 3
    assert tasks[0].label_path.endswith("test/labels/sample.txt")
    assert tasks[0].label_exists
    assert tasks[0].yolo_row == "3 0.30000000 0.50000000 0.40000000 0.50000000"
    assert rows[0]["label_exists"] == "yes"
    assert rows[0]["yolo_row"] == "3 0.30000000 0.50000000 0.40000000 0.50000000"
    assert "Object Detection Pending Label Fix Tasks" in task_md.read_text(encoding="utf-8")


def test_prepare_label_fix_tasks_cli_writes_outputs(monkeypatch, capsys):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_prepare_object_detection_label_fix_tasks_cli" / uuid4().hex
    dataset_root = workspace / "dataset"
    image_path = dataset_root / "test" / "images" / "sample.jpg"
    review_csv = workspace / "review.csv"
    task_csv = workspace / "tasks.csv"
    task_md = workspace / "tasks.md"
    _write_image(image_path, width=100, height=80)
    (dataset_root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (dataset_root / "data.yaml").write_text("nc: 5\nnames: ['pedestrian', 'bicycle', 'car', 'scooter', 'roadblock']\n", encoding="utf-8")
    _write_review_csv(review_csv, image_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_object_detection_label_fix_tasks.py",
            "--review-csv",
            str(review_csv),
            "--dataset-root",
            str(dataset_root),
            "--task-csv",
            str(task_csv),
            "--task-md",
            str(task_md),
        ],
    )

    code = script.main()
    output = capsys.readouterr().out

    assert code == 0
    assert "tasks=1" in output
    assert task_csv.exists()
    assert task_md.exists()


def _write_review_csv(path: Path, image_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "case_id": "scooter_0001_01_false_positive",
        "target_class": "scooter",
        "error_type": "false_positive",
        "source_image": image_path.resolve().relative_to(REPO_ROOT).as_posix(),
        "gallery_image": "cache/gallery/sample.jpg",
        "gt_cls": "",
        "pred_cls": "scooter",
        "confidence": "0.72",
        "iou": "",
        "gt_bbox": "",
        "pred_bbox": json.dumps([10, 20, 50, 60], separators=(",", ":")),
        "reason": "label_missing_real_scooter",
        "action": "add_label",
        "priority": "high",
        "reviewed_by": "tester",
        "notes": "unit test",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_image(path: Path, *, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    encoded.tofile(str(path))
