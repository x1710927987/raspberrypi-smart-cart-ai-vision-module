import importlib.util
import json
import sys
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "export_objects_yolo_dataset.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("export_objects_yolo_dataset", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_export_objects_yolo_dataset_writes_split_images_labels_and_yaml():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_export_objects_yolo_dataset" / uuid4().hex
    data_root = workspace / "data"
    output_root = workspace / "objects_yolo_v1"
    _write_image(data_root / "raw" / "objects" / "images" / "sample_train.jpg", size=(100, 200))
    _write_image(data_root / "raw" / "objects" / "images" / "sample_valid.jpg", size=(80, 160))
    _write_annotation(
        data_root / "annotations" / "objects" / "sample_train.json",
        image="raw/objects/images/sample_train.jpg",
        split="train",
        objects=[
            {"cls": "pedestrian", "bbox": [20, 10, 80, 90]},
            {"cls": "unknown", "bbox": [1, 1, 5, 5]},
        ],
    )
    _write_annotation(
        data_root / "annotations" / "objects" / "sample_valid.json",
        image="raw/objects/images/sample_valid.jpg",
        split="valid",
        objects=[{"cls": "car", "bbox": [40, 20, 120, 70]}],
    )

    summary = script.export_objects_yolo_dataset(data_root=data_root, output_root=output_root, classes=["pedestrian", "bicycle", "car"])

    assert summary.images == {"train": 1, "valid": 1, "test": 0}
    assert summary.objects == {"train": 1, "valid": 1, "test": 0}
    assert summary.skipped_objects == 1
    assert (output_root / "train" / "images" / "sample_train.jpg").exists()
    assert (output_root / "valid" / "labels" / "sample_valid.txt").exists()
    assert "names: ['pedestrian', 'bicycle', 'car']" in (output_root / "data.yaml").read_text(encoding="utf-8")
    train_row = (output_root / "train" / "labels" / "sample_train.txt").read_text(encoding="utf-8").strip()
    assert train_row.startswith("0 ")
    assert len(train_row.split()) == 5


def test_export_objects_yolo_dataset_cli(capsys, monkeypatch):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_export_objects_yolo_dataset_cli" / uuid4().hex
    data_root = workspace / "data"
    output_root = workspace / "objects_yolo_v1"
    _write_image(data_root / "raw" / "objects" / "images" / "sample.jpg")
    _write_annotation(
        data_root / "annotations" / "objects" / "sample.json",
        image="raw/objects/images/sample.jpg",
        split="test",
        objects=[{"cls": "bicycle", "bbox": [8, 8, 32, 32]}],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_objects_yolo_dataset.py",
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
            "--classes",
            "pedestrian,bicycle,car",
        ],
    )

    code = script.main()
    output = capsys.readouterr().out

    assert code == 0
    assert "test_images=1" in output
    assert (output_root / "test" / "labels" / "sample.txt").exists()


def _write_annotation(path: Path, *, image: str, split: str, objects: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"image": image, "source_split": split, "width": 64, "height": 64, "objects": objects}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_image(path: Path, *, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (20, 40, 60)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    encoded.tofile(str(path))
