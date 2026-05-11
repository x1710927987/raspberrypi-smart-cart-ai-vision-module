import importlib.util
import sys
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "merge_objects_yolo_datasets.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("merge_objects_yolo_datasets", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_merge_objects_yolo_datasets_remaps_source_labels():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_merge_objects_yolo_datasets" / uuid4().hex
    pedestrian = _write_dataset(workspace / "objects_yolo_v1", ["pedestrian"], "0 0.5 0.5 0.4 0.4\n")
    scooter = _write_dataset(workspace / "roboflow_electric-scooter_v1", ["e-scooter"], "0 0.5 0.5 0.3 0.3\n")
    roadblock = _write_dataset(workspace / "roboflow_roadblock_v1", ["cone"], "0 0.5 0.5 0.2 0.2\n")
    output_root = workspace / "objects_combined_v2"

    counts = script.merge_objects_yolo_datasets([pedestrian, scooter, roadblock], output_root, overwrite=True)

    assert counts["images"] == 3
    assert counts["class_pedestrian"] == 1
    assert counts["class_scooter"] == 1
    assert counts["class_roadblock"] == 1
    assert "names: ['pedestrian', 'bicycle', 'car', 'scooter', 'roadblock']" in (output_root / "data.yaml").read_text(encoding="utf-8")
    labels = sorted(path.read_text(encoding="utf-8").strip() for path in (output_root / "train" / "labels").glob("*.txt"))
    assert labels == [
        "0 0.50000000 0.50000000 0.40000000 0.40000000",
        "3 0.50000000 0.50000000 0.30000000 0.30000000",
        "4 0.50000000 0.50000000 0.20000000 0.20000000",
    ]


def test_merge_objects_cli_dry_path(monkeypatch, capsys):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_merge_objects_yolo_datasets_cli" / uuid4().hex
    source = _write_dataset(workspace / "roboflow_bicycle_v1", ["bicycle"], "0 0.5 0.5 0.4 0.4\n")
    output_root = workspace / "objects_combined_v2"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_objects_yolo_datasets.py",
            "--source-root",
            str(source),
            "--output-root",
            str(output_root),
            "--overwrite",
        ],
    )

    code = script.main()
    output = capsys.readouterr().out

    assert code == 0
    assert "class_bicycle=1" in output
    assert (output_root / "data.yaml").exists()


def _write_dataset(root: Path, names: list[str], label_text: str) -> Path:
    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_path = images_dir / "sample.jpg"
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    encoded.tofile(str(image_path))
    (labels_dir / "sample.txt").write_text(label_text, encoding="utf-8")
    names_text = "[" + ", ".join(repr(name) for name in names) + "]"
    (root / "data.yaml").write_text(f"train: train/images\nval: valid/images\ntest: test/images\n\nnc: {len(names)}\nnames: {names_text}\n", encoding="utf-8")
    return root
