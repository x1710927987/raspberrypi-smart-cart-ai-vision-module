import csv
import importlib.util
import sys
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "augment_traffic_light_add_similar_data.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("augment_traffic_light_add_similar_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_augment_add_similar_data_generates_yolo_images_and_labels():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_augment_traffic_light_add_similar_data" / uuid4().hex
    source_root = workspace / "source"
    output_root = workspace / "traffic_light_add_similar_v1"
    plan_path = workspace / "plan.csv"
    image_path = source_root / "valid" / "images" / "green_example.jpg"
    label_path = source_root / "valid" / "labels" / "green_example.txt"
    _write_source_dataset(source_root, image_path, label_path)
    _write_plan(plan_path, image_path)

    samples = script.augment_add_similar_data(
        plan_csv=plan_path,
        output_root=output_root,
        source_root=source_root,
        seed=7,
        max_per_case=3,
    )

    assert len(samples) == 3
    assert (output_root / "data.yaml").exists()
    assert (output_root / "augmentation_manifest.csv").exists()
    assert (output_root / "augmentation_summary.json").exists()
    output_images = sorted((output_root / "train" / "images").glob("*.jpg"))
    output_labels = sorted((output_root / "train" / "labels").glob("*.txt"))
    assert len(output_images) == 3
    assert len(output_labels) == 3
    for label_file in output_labels:
        rows = label_file.read_text(encoding="utf-8").strip().splitlines()
        assert rows
        for row in rows:
            parts = row.split()
            assert int(parts[0]) == 0
            values = [float(value) for value in parts[1:]]
            assert all(0.0 <= value <= 1.0 for value in values)
            assert values[2] > 0.0
            assert values[3] > 0.0


def test_augment_add_similar_data_cli(capsys, monkeypatch):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_augment_traffic_light_add_similar_data_cli" / uuid4().hex
    source_root = workspace / "source"
    output_root = workspace / "traffic_light_add_similar_v1"
    plan_path = workspace / "plan.csv"
    image_path = source_root / "test" / "images" / "yellow_example.jpg"
    label_path = source_root / "test" / "labels" / "yellow_example.txt"
    _write_source_dataset(source_root, image_path, label_path, class_id=2)
    _write_plan(plan_path, image_path, primary_gt="yellow")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "augment_traffic_light_add_similar_data.py",
            "--plan",
            str(plan_path),
            "--output-root",
            str(output_root),
            "--source-root",
            str(source_root),
            "--max-per-case",
            "2",
        ],
    )

    code = script.main()
    output = capsys.readouterr().out

    assert code == 0
    assert "generated_images=2" in output
    assert "status=ok" in output


def _write_source_dataset(source_root: Path, image_path: Path, label_path: Path, class_id: int = 0) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    image[:, :] = (20, 20, 20)
    cv2.rectangle(image, (56, 30), (72, 58), (0, 255, 0), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    encoded.tofile(str(image_path))
    label_path.write_text(f"{class_id} 0.50000000 0.45833333 0.12500000 0.29166667\n", encoding="utf-8")
    (source_root / "data.yaml").write_text(
        "\n".join(
            [
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "",
                "nc: 3",
                "names: ['green', 'red', 'yellow']",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_plan(plan_path: Path, image_path: Path, primary_gt: str = "green") -> None:
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "valid_0001",
                "split": "valid",
                "primary_gt": primary_gt,
                "predicted_state": "unknown",
                "reason": "missed_detection",
                "priority": "high",
                "suggested_new_images": "4",
                "recommended_focus": f"{primary_gt} lights that the detector misses; side-view or angled camera perspectives",
                "source_image": str(image_path),
                "gallery_image": "",
                "notes": "",
            }
        )
