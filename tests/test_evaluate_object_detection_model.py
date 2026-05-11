import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from perception.model_inference import FixedPredictionBackend


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "evaluate_object_detection_model.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("evaluate_object_detection_model", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluate_object_detection_model_reports_box_level_metrics_and_mistakes():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_evaluate_object_detection_model"
    dataset_root = workspace / "dataset"
    image_path = dataset_root / "test" / "images" / "sample.jpg"
    label_path = dataset_root / "test" / "labels" / "sample.txt"
    errors_out = workspace / "mistakes.json"
    _write_image(image_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        "\n".join(
            [
                "1 0.250000 0.500000 0.250000 0.500000",
                "4 0.750000 0.500000 0.250000 0.500000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_root / "data.yaml").write_text("nc: 5\nnames: ['pedestrian', 'bicycle', 'car', 'scooter', 'roadblock']\n", encoding="utf-8")

    result = script.evaluate_object_detection_model(
        dataset_root=dataset_root,
        split="test",
        manifest_path=REPO_ROOT / "models" / "model_manifest.objects.example.json",
        backend=FixedPredictionBackend(
            [
                {"label": "bicycle", "bbox": [8, 16, 24, 48], "conf": 0.9},
                {"label": "scooter", "bbox": [56, 16, 72, 48], "conf": 0.8},
                {"label": "car", "bbox": [2, 2, 10, 10], "conf": 0.7},
            ]
        ),
    )
    script.write_mistakes(errors_out, result.mistakes)

    assert result.total_images == 1
    assert result.evaluated_images == 1
    assert result.correct_images == 0
    assert result.true_positives == 1
    assert result.misclassified == 1
    assert result.false_positives == 2
    assert result.false_negatives == 1
    assert result.class_metrics["bicycle"]["precision"] == 1.0
    assert result.class_metrics["bicycle"]["recall"] == 1.0
    assert result.class_metrics["roadblock"]["actual"] == 1
    assert result.class_metrics["scooter"]["predicted"] == 1
    assert result.confusion_matrix["roadblock"]["scooter"] == 1
    assert result.confusion_matrix["unknown"]["car"] == 1
    mistakes = json.loads(errors_out.read_text(encoding="utf-8"))
    assert mistakes[0]["gt_boxes"][0]["cls"] == "bicycle"
    assert mistakes[0]["predicted_boxes"][1]["cls"] == "scooter"
    assert mistakes[0]["misclassified"][0]["gt"]["cls"] == "roadblock"
    assert mistakes[0]["misclassified"][0]["pred"]["cls"] == "scooter"


def test_evaluate_object_detection_cli_json(monkeypatch, capsys):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_evaluate_object_detection_model_cli"
    dataset_root = workspace / "dataset"
    image_path = dataset_root / "test" / "images" / "sample.jpg"
    label_path = dataset_root / "test" / "labels" / "sample.txt"
    _write_image(image_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("3 0.500000 0.500000 0.500000 0.500000\n", encoding="utf-8")
    (dataset_root / "data.yaml").write_text("nc: 5\nnames: ['pedestrian', 'bicycle', 'car', 'scooter', 'roadblock']\n", encoding="utf-8")
    original_load_model_manifest = script.load_model_manifest
    monkeypatch.setattr(
        script,
        "UltralyticsBackend",
        lambda device=None: FixedPredictionBackend([{"label": "scooter", "bbox": [16, 16, 48, 48], "conf": 0.8}]),
    )
    monkeypatch.setattr(script, "load_model_manifest", lambda path, require_artifact=False: original_load_model_manifest(path, require_artifact=False))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_object_detection_model.py",
            "--dataset-root",
            str(dataset_root),
            "--split",
            "test",
            "--manifest",
            str(REPO_ROOT / "models" / "model_manifest.objects.example.json"),
            "--json",
        ],
    )

    code = script.main()
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["accuracy"] == 1.0
    assert payload["class_metrics"]["scooter"]["recall"] == 1.0


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((64, 80, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    encoded.tofile(str(path))
