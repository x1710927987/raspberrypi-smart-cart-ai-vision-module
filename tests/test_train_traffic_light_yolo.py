import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT_PATH = REPO_ROOT / "tools" / "train_traffic_light_yolo.py"


def _load_train_script():
    spec = importlib.util.spec_from_file_location("train_traffic_light_yolo", TRAIN_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_default_traffic_light_yolo_config():
    script = _load_train_script()
    config = script.load_train_config(REPO_ROOT / "models" / "training" / "traffic_light_yolo_v1.yaml")
    assert config.data == REPO_ROOT / "data" / "external" / "roboflow_traffic_light_v1_split" / "data.yaml"
    assert config.model == "models/weights/yolov8n.pt"
    assert config.imgsz == 640
    assert config.name == "smartcart_traffic_light_yolov8n_roboflow_v1"
    assert config.export_format == "none"


def test_validate_config_rejects_missing_data():
    script = _load_train_script()
    config = script.TrainConfig(data=REPO_ROOT / "data" / "external" / "missing_dataset" / "data.yaml")
    errors = script.validate_config(config)
    assert any("missing YOLO data config" in error for error in errors)


def test_dry_run_prints_training_plan(capsys):
    script = _load_train_script()
    config = script.load_train_config(REPO_ROOT / "models" / "training" / "traffic_light_yolo_v1.yaml")
    script.train_traffic_light_yolo(config, dry_run=True)
    captured = capsys.readouterr()
    assert "traffic_light_yolo_training_plan" in captured.out
    assert "status=dry_run" in captured.out


def test_cli_dry_run_uses_config(monkeypatch, capsys):
    script = _load_train_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_traffic_light_yolo.py",
            "--config",
            str(REPO_ROOT / "models" / "training" / "traffic_light_yolo_v1.yaml"),
            "--dry-run",
            "--epochs",
            "3",
            "--batch",
            "2",
        ],
    )
    code = script.main()
    captured = capsys.readouterr()
    assert code == 0
    assert "epochs=3" in captured.out
    assert "batch=2" in captured.out


def test_config_rejects_unknown_fields():
    script = _load_train_script()
    temp_dir = REPO_ROOT / "cache" / "pytest" / "test_train_traffic_light_yolo"
    temp_dir.mkdir(parents=True, exist_ok=True)
    config_path = temp_dir / "bad_training_config.yaml"
    config_path.write_text("unknown_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown training config field"):
        script.load_train_config(config_path)
