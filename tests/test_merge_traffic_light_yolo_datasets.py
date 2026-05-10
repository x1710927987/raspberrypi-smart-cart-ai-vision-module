import importlib.util
import sys
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "merge_traffic_light_yolo_datasets.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("merge_traffic_light_yolo_datasets", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_merge_datasets_copies_sources_into_train_pool():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_merge_traffic_light_yolo_datasets" / uuid4().hex
    source_a = workspace / "source_a"
    source_b = workspace / "source_b"
    output_root = workspace / "combined"
    _write_dataset(source_a, "train", "green_a", class_id=0)
    _write_dataset(source_b, "valid", "yellow_b", class_id=2)

    counts = script.merge_datasets(
        [
            script.MergeSource(source_a, "roboflow"),
            script.MergeSource(source_b, "aug"),
        ],
        output_root,
    )

    assert counts["images"] == 2
    assert counts["labels"] == 2
    assert counts["label_rows"] == 2
    assert (output_root / "data.yaml").exists()
    assert (output_root / "train" / "images" / "roboflow_train_green_a.jpg").exists()
    assert (output_root / "train" / "labels" / "aug_valid_yellow_b.txt").exists()
    assert (output_root / "valid" / "images").exists()
    assert (output_root / "test" / "labels").exists()


def test_merge_datasets_cli(capsys, monkeypatch):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_merge_traffic_light_yolo_datasets_cli" / uuid4().hex
    source = workspace / "source"
    output_root = workspace / "combined"
    _write_dataset(source, "test", "red_sample", class_id=1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_traffic_light_yolo_datasets.py",
            "--source",
            f"src={source}",
            "--output-root",
            str(output_root),
        ],
    )

    code = script.main()
    output = capsys.readouterr().out

    assert code == 0
    assert "images=1" in output
    assert "status=ok" in output


def _write_dataset(root: Path, split: str, stem: str, class_id: int) -> None:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[:, :] = (30, 30, 30)
    cv2.circle(image, (50, 40), 8, (0, 255, 0), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    encoded.tofile(str(image_dir / f"{stem}.jpg"))
    (label_dir / f"{stem}.txt").write_text(f"{class_id} 0.50000000 0.50000000 0.16000000 0.20000000\n", encoding="utf-8")
    (root / "data.yaml").write_text(
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
