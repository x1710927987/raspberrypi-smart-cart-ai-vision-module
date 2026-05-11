import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "visualize_object_detection_errors.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("visualize_object_detection_errors", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_visualize_object_detection_errors_renders_gallery_and_index():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_visualize_object_detection_errors"
    image_path = workspace / "images" / "sample.jpg"
    mistakes_path = workspace / "objects_test_mistakes.json"
    output_dir = workspace / "gallery"
    _write_image(image_path)
    mistakes_path.parent.mkdir(parents=True, exist_ok=True)
    mistakes_path.write_text(
        json.dumps(
            [
                {
                    "image": _repo_relative(image_path),
                    "correct": False,
                    "gt_boxes": [{"cls": "roadblock", "bbox": [10, 12, 35, 45], "conf": None}],
                    "predicted_boxes": [{"cls": "scooter", "bbox": [11, 13, 36, 44], "conf": 0.81}],
                    "matches": [],
                    "false_negatives": [],
                    "false_positives": [],
                    "misclassified": [
                        {
                            "gt": {"cls": "roadblock", "bbox": [10, 12, 35, 45], "conf": None},
                            "pred": {"cls": "scooter", "bbox": [11, 13, 36, 44], "conf": 0.81},
                            "iou": 0.88,
                            "status": "misclassified",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    summaries = script.visualize_error_files([mistakes_path], output_dir=output_dir)

    assert summaries[0].total_items == 1
    assert summaries[0].rendered == 1
    group_dir = output_dir / "objects_test_mistakes"
    rendered = list(group_dir.glob("*.jpg"))
    assert len(rendered) == 1
    assert cv2.imread(str(rendered[0])) is not None
    index = json.loads((group_dir / "index.json").read_text(encoding="utf-8"))
    assert index[0]["status"] == "rendered"
    assert index[0]["misclassified"] == 1


def test_visualize_object_detection_errors_tracks_missing_images():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_visualize_object_detection_errors_missing"
    mistakes_path = workspace / "mistakes.json"
    mistakes_path.parent.mkdir(parents=True, exist_ok=True)
    mistakes_path.write_text(
        json.dumps(
            [
                {
                    "image": "cache/pytest/test_visualize_object_detection_errors_missing/nope.jpg",
                    "correct": False,
                    "gt_boxes": [],
                    "predicted_boxes": [],
                    "matches": [],
                    "false_negatives": [],
                    "false_positives": [],
                    "misclassified": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    summaries = script.visualize_error_files([mistakes_path], output_dir=workspace / "gallery")

    assert summaries[0].rendered == 0
    assert summaries[0].missing_images == 1
    index = json.loads((workspace / "gallery" / "mistakes" / "index.json").read_text(encoding="utf-8"))
    assert index[0]["status"] == "missing_image"


def test_visualize_object_detection_cli_json_summary(monkeypatch, capsys):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_visualize_object_detection_errors_cli"
    image_path = workspace / "images" / "sample.jpg"
    mistakes_path = workspace / "mistakes.json"
    _write_image(image_path)
    mistakes_path.parent.mkdir(parents=True, exist_ok=True)
    mistakes_path.write_text(
        json.dumps(
            [
                {
                    "image": _repo_relative(image_path),
                    "correct": False,
                    "gt_boxes": [{"cls": "bicycle", "bbox": [5, 6, 40, 42], "conf": None}],
                    "predicted_boxes": [],
                    "matches": [],
                    "false_negatives": [{"cls": "bicycle", "bbox": [5, 6, 40, 42], "conf": None}],
                    "false_positives": [],
                    "misclassified": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_object_detection_errors.py",
            str(mistakes_path),
            "--output-dir",
            str(workspace / "gallery"),
            "--json",
        ],
    )

    code = script.main()
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload[0]["rendered"] == 1


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    encoded.tofile(str(path))


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()
