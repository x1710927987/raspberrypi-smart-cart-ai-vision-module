import importlib.util
import sys
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "merge_hazard_yolo_datasets.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("merge_hazard_yolo_datasets", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_merge_hazard_yolo_datasets_remaps_classes():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_merge_hazard_yolo_datasets" / uuid4().hex
    pothole_root = _make_dataset(workspace / "roboflow_pothole_v1", "pothole", "pothole_001.jpg")
    curb_root = _make_dataset(workspace / "roboflow_curb_v1", "curb", "curb_001.jpg")
    output_root = workspace / "roboflow_hazard_v1"

    counts = script.merge_hazard_yolo_datasets([pothole_root, curb_root], output_root)

    assert counts["images"] == 2
    assert counts["objects"] == 2
    assert counts["class_pothole"] == 1
    assert counts["class_curb"] == 1
    assert (output_root / "data.yaml").read_text(encoding="utf-8").count("pothole") == 1
    label_texts = sorted(path.read_text(encoding="utf-8").strip() for path in (output_root / "train" / "labels").glob("*.txt"))
    assert label_texts == [
        "0 0.50000000 0.50000000 0.25000000 0.25000000",
        "1 0.50000000 0.50000000 0.25000000 0.25000000",
    ]


def test_merge_hazard_cli(capsys, monkeypatch):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_merge_hazard_yolo_datasets_cli" / uuid4().hex
    pothole_root = _make_dataset(workspace / "roboflow_pothole_v1", "pothole", "pothole_001.jpg")
    output_root = workspace / "roboflow_hazard_v1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_hazard_yolo_datasets.py",
            "--source-root",
            str(pothole_root),
            "--output-root",
            str(output_root),
        ],
    )

    code = script.main()
    captured = capsys.readouterr()

    assert code == 0
    assert "images=1" in captured.out
    assert "status=ok" in captured.out


def _make_dataset(root: Path, class_name: str, image_name: str) -> Path:
    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_path = images_dir / image_name
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    encoded.tofile(str(image_path))
    (labels_dir / f"{Path(image_name).stem}.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        "\n".join(
            [
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "",
                "nc: 1",
                f"names: ['{class_name}']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return root
