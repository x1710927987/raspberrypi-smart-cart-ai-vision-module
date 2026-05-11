import csv
import importlib.util
import json
import sys
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "augment_object_detection_add_similar_data.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("augment_object_detection_add_similar_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_augment_object_detection_add_similar_data_generates_yolo_source():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_augment_object_detection_add_similar_data" / uuid4().hex
    dataset_root = workspace / "dataset"
    output_root = workspace / "output"
    review_csv = workspace / "review.csv"
    image_path = dataset_root / "test" / "images" / "sample.jpg"
    label_path = dataset_root / "test" / "labels" / "sample.txt"
    _write_image(image_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        "\n".join(
            [
                "3 0.50000000 0.50000000 0.30000000 0.40000000",
                "2 0.20000000 0.20000000 0.10000000 0.10000000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_root / "data.yaml").write_text("nc: 5\nnames: ['pedestrian', 'bicycle', 'car', 'scooter', 'roadblock']\n", encoding="utf-8")
    _write_review_csv(review_csv, image_path)

    samples = script.augment_add_similar_data(
        review_csv=review_csv,
        dataset_root=dataset_root,
        output_root=output_root,
        seed=7,
        max_per_case=2,
    )

    assert len(samples) == 2
    assert (output_root / "data.yaml").exists()
    assert (output_root / "augmentation_manifest.csv").exists()
    assert (output_root / "augmentation_summary.json").exists()
    assert len(list((output_root / "train" / "images").glob("*.jpg"))) == 2
    labels = list((output_root / "train" / "labels").glob("*.txt"))
    assert len(labels) == 2
    for label_file in labels:
        rows = [line.split() for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows
        assert all(len(row) == 5 for row in rows)
        assert all(0.0 <= float(value) <= 1.0 for row in rows for value in row[1:])
    summary = json.loads((output_root / "augmentation_summary.json").read_text(encoding="utf-8"))
    assert summary["generated_samples"] == 2


def test_augment_object_detection_add_similar_data_cli(monkeypatch, capsys):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_augment_object_detection_add_similar_data_cli" / uuid4().hex
    dataset_root = workspace / "dataset"
    output_root = workspace / "output"
    review_csv = workspace / "review.csv"
    image_path = dataset_root / "test" / "images" / "sample.jpg"
    label_path = dataset_root / "test" / "labels" / "sample.txt"
    _write_image(image_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("3 0.50000000 0.50000000 0.30000000 0.40000000\n", encoding="utf-8")
    (dataset_root / "data.yaml").write_text("nc: 5\nnames: ['pedestrian', 'bicycle', 'car', 'scooter', 'roadblock']\n", encoding="utf-8")
    _write_review_csv(review_csv, image_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "augment_object_detection_add_similar_data.py",
            "--review-csv",
            str(review_csv),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--max-per-case",
            "1",
        ],
    )

    code = script.main()
    output = capsys.readouterr().out

    assert code == 0
    assert "generated_samples=1" in output
    assert (output_root / "train" / "images").exists()


def _write_review_csv(path: Path, image_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "case_id": "scooter_0001_01_false_negative",
        "target_class": "scooter",
        "error_type": "false_negative",
        "source_image": image_path.resolve().relative_to(REPO_ROOT).as_posix(),
        "gallery_image": "",
        "gt_cls": "scooter",
        "pred_cls": "",
        "confidence": "",
        "iou": "",
        "gt_bbox": "[10,20,40,60]",
        "pred_bbox": "",
        "reason": "small_distant_crowded_scooter",
        "action": "add_similar_data",
        "priority": "medium",
        "reviewed_by": "tester",
        "notes": "unit test",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[:, :] = (20, 80, 120)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    encoded.tofile(str(path))
