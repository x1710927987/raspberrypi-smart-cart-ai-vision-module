import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_SCRIPT_PATH = REPO_ROOT / "tools" / "evaluate_traffic_light_model.py"


def _load_evaluate_script():
    spec = importlib.util.spec_from_file_location("evaluate_traffic_light_model", EVALUATE_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluate_traffic_light_model_computes_image_level_metrics():
    script = _load_evaluate_script()
    dataset_root = _write_tiny_yolo_dataset("metrics")
    backend = _SequenceBackend(
        [
            [{"label": "green", "conf": 0.91, "bbox": [1, 2, 3, 4]}],
            [{"label": "red", "conf": 0.82, "bbox": [1, 2, 3, 4]}],
            [],
        ]
    )
    result = script.evaluate_traffic_light_model(
        dataset_root=dataset_root,
        split="valid",
        manifest_path=REPO_ROOT / "models" / "model_manifest.traffic_light.example.json",
        backend=backend,
    )

    assert result.total_images == 3
    assert result.evaluated_images == 3
    assert result.correct_images == 2
    assert result.accuracy == 0.6667
    assert result.no_detection == 1
    assert result.missing_labels == 0
    assert result.unreadable_images == 0
    assert result.confusion_matrix["green"]["green"] == 1
    assert result.confusion_matrix["red"]["red"] == 1
    assert result.confusion_matrix["yellow"]["unknown"] == 1
    assert result.class_metrics["green"]["precision"] == 1.0
    assert result.class_metrics["green"]["recall"] == 1.0
    assert result.class_metrics["red"]["precision"] == 1.0
    assert result.class_metrics["red"]["recall"] == 1.0
    assert result.class_metrics["yellow"]["actual"] == 2
    assert result.class_metrics["yellow"]["recall"] == 0.0
    assert len(result.mistakes) == 1
    assert result.mistakes[0].predicted_state == "unknown"


def test_evaluate_traffic_light_model_limit_and_json_output(capsys):
    script = _load_evaluate_script()
    dataset_root = _write_tiny_yolo_dataset("json_output")
    backend = _SequenceBackend([[{"label": "green", "conf": 0.91, "bbox": [1, 2, 3, 4]}]])
    result = script.evaluate_traffic_light_model(
        dataset_root=dataset_root,
        split="valid",
        manifest_path=REPO_ROOT / "models" / "model_manifest.traffic_light.example.json",
        backend=backend,
        limit=1,
    )
    script.print_evaluation(result, as_json=True)
    payload = json.loads(capsys.readouterr().out)

    assert payload["total_images"] == 1
    assert payload["evaluated_images"] == 1
    assert payload["accuracy"] == 1.0
    assert payload["class_metrics"]["green"]["tp"] == 1


def test_evaluate_traffic_light_model_writes_mistakes():
    script = _load_evaluate_script()
    output_path = REPO_ROOT / "cache" / "pytest" / "test_evaluate_traffic_light_model" / "mistakes.json"
    mistake = script.ImageEvaluation(
        image="sample.jpg",
        gt_states=["yellow"],
        primary_gt="yellow",
        predicted_state="unknown",
        confidence=0.0,
        correct=False,
    )
    script.write_mistakes(output_path, [mistake])
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["image"] == "sample.jpg"
    assert payload[0]["primary_gt"] == "yellow"


class _SequenceBackend:
    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.calls = 0

    def predict(self, frame, preprocess_result, manifest):
        prediction = self.predictions[self.calls]
        self.calls += 1
        return prediction


def _write_tiny_yolo_dataset(name: str) -> Path:
    root = REPO_ROOT / "cache" / "pytest" / "test_evaluate_traffic_light_model" / name
    images_dir = root / "valid" / "images"
    labels_dir = root / "valid" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "\n".join(
            [
                "train: ../train/images",
                "val: ../valid/images",
                "test: ../test/images",
                "",
                "nc: 3",
                "names: ['green', 'red', 'yellow']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    samples = {
        "sample_green": ["0 0.5 0.5 0.2 0.2"],
        "sample_red_yellow": ["1 0.4 0.4 0.2 0.2", "2 0.6 0.6 0.2 0.2"],
        "sample_yellow": ["2 0.5 0.5 0.2 0.2"],
    }
    for stem, rows in samples.items():
        _write_image(images_dir / f"{stem}.jpg")
        (labels_dir / f"{stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def _write_image(path: Path) -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    encoded.tofile(str(path))
