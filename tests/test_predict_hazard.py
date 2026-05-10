import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from perception.model_inference import FixedPredictionBackend
from perception.runtime import Hazard


REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICT_SCRIPT_PATH = REPO_ROOT / "tools" / "predict_hazard.py"


def _load_predict_script():
    spec = importlib.util.spec_from_file_location("predict_hazard", PREDICT_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_predict_hazard_with_injected_backend():
    script = _load_predict_script()
    image_path = _write_test_image("predict_hazard_injected_backend.jpg")
    prediction = script.predict_hazard(
        image_path,
        REPO_ROOT / "models" / "model_manifest.hazard.example.json",
        backend=FixedPredictionBackend([{"label": "pothole", "conf": 0.77}]),
    )
    assert prediction == Hazard("pothole", 0.77)


def test_default_manifest_points_to_registered_hazard_model():
    script = _load_predict_script()
    assert script.DEFAULT_MANIFEST == REPO_ROOT / "models" / "training" / "smartcart_hazard_yolov8n_roboflow_pt_v1.manifest.json"


def test_print_prediction_json(capsys):
    script = _load_predict_script()
    script.print_prediction(
        Path("image.jpg"),
        Path("manifest.json"),
        Hazard("water", 0.66524),
        as_json=True,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "image": "image.jpg",
        "manifest": "manifest.json",
        "detected": True,
        "type": "water",
        "confidence": 0.6652,
    }


def test_print_prediction_no_detection_text(capsys):
    script = _load_predict_script()
    script.print_prediction(Path("image.jpg"), Path("manifest.json"), None, as_json=False)
    output = capsys.readouterr().out
    assert "type=unknown" in output
    assert "confidence=0.0000" in output
    assert "status=no_detection" in output


def test_cli_returns_error_for_missing_image(monkeypatch, capsys):
    script = _load_predict_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict_hazard.py",
            "--image",
            str(REPO_ROOT / "cache" / "pytest" / "missing-hazard-image.jpg"),
        ],
    )
    code = script.main()
    captured = capsys.readouterr()
    assert code == 1
    assert "image does not exist" in captured.err


def _write_test_image(name: str) -> Path:
    output_dir = REPO_ROOT / "cache" / "pytest" / "test_predict_hazard"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / name
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    encoded.tofile(str(image_path))
    return image_path
