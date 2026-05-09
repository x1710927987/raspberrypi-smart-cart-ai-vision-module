import importlib.util
import json
import sys
from pathlib import Path

import pytest

from perception.model_inference import load_model_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPO_ROOT / "tools" / "check_model_manifest.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_model_manifest", CHECKER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_traffic_light_manifest_example_matches_schema():
    manifest = load_model_manifest(REPO_ROOT / "models" / "model_manifest.traffic_light.example.json")
    assert manifest.task == "traffic_light"
    assert manifest.artifact_format == "onnx"
    assert manifest.model_classes == ["green", "red", "yellow"]
    assert manifest.map_label("green") == "green"
    assert manifest.map_label("red") == "red"
    assert manifest.map_label("yellow") == "yellow"


def test_objects_manifest_example_matches_schema():
    manifest = load_model_manifest(REPO_ROOT / "models" / "model_manifest.objects.example.json")
    assert manifest.task == "objects"
    assert manifest.artifact_format == "onnx"
    assert manifest.model_classes == ["pedestrian", "bicycle", "car"]
    assert manifest.map_label("pedestrian") == "pedestrian"
    assert manifest.map_label("bicycle") == "bicycle"
    assert manifest.map_label("car") == "car"


def test_check_model_manifest_accepts_traffic_light_example():
    checker = _load_checker()
    result = checker.check_manifest_file(
        REPO_ROOT / "models" / "model_manifest.traffic_light.example.json",
        expected_task="traffic_light",
    )
    assert result.ok
    assert any("status is example" in warning for warning in result.warnings)


def test_check_model_manifest_accepts_objects_example():
    checker = _load_checker()
    result = checker.check_manifest_file(
        REPO_ROOT / "models" / "model_manifest.objects.example.json",
        expected_task="objects",
    )
    assert result.ok
    assert any("status is example" in warning for warning in result.warnings)


def test_check_model_manifest_rejects_wrong_expected_task():
    checker = _load_checker()
    result = checker.check_manifest_file(
        REPO_ROOT / "models" / "model_manifest.traffic_light.example.json",
        expected_task="objects",
    )
    assert not result.ok
    assert any("does not match expected task" in error for error in result.errors)


def test_check_model_manifest_rejects_missing_class_mapping():
    checker = _load_checker()
    temp_dir = REPO_ROOT / "cache" / "pytest" / "test_model_manifest"
    temp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = temp_dir / "bad_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "model_id": "smartcart_traffic_light_yolov8n_test_onnx_v1",
                "task": "traffic_light",
                "artifact": {
                    "path": "models/weights/smartcart_traffic_light_yolov8n_test_onnx_v1.onnx",
                    "format": "onnx",
                    "architecture": "yolov8n",
                    "version": "v1",
                },
                "preprocessing": {
                    "resize": {"width": 640, "height": 640},
                    "normalize": {"scale": 1.0},
                },
                "onnx": {
                    "input_tensors": [{"name": "images", "shape": [1, 3, 640, 640], "dtype": "float32"}],
                    "output_tensors": [{"name": "output0", "shape": [1, 7, 8400], "dtype": "float32"}],
                },
                "postprocessing": {
                    "confidence_threshold": 0.35,
                    "nms_iou_threshold": 0.45,
                    "max_detections": 20,
                },
                "model_classes": ["green", "red", "yellow"],
                "schema_mapping": {"green": "green", "red": "red"},
            }
        ),
        encoding="utf-8",
    )
    result = checker.check_manifest_file(manifest_path, expected_task="traffic_light")
    assert not result.ok
    assert any("missing model class 'yellow'" in error for error in result.errors)


def test_check_model_manifest_require_artifact_fails_for_example():
    checker = _load_checker()
    result = checker.check_manifest_file(
        REPO_ROOT / "models" / "model_manifest.traffic_light.example.json",
        expected_task="traffic_light",
        require_artifact=True,
    )
    assert not result.ok
    assert any("model artifact does not exist" in error for error in result.errors)


def test_check_model_manifest_cli_prints_status(capsys, monkeypatch):
    checker = _load_checker()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_model_manifest.py",
            str(REPO_ROOT / "models" / "model_manifest.traffic_light.example.json"),
            "--task",
            "traffic_light",
        ],
    )
    code = checker.main()
    captured = capsys.readouterr()
    assert code == 0
    assert "checked=1" in captured.out
    assert "status=ok" in captured.out
