import importlib.util
import sys
from pathlib import Path
from uuid import uuid4

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "train_laneseg_yolo.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("train_laneseg_yolo", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_default_laneseg_yolo_config():
    script = _load_script()
    config = script.load_train_config(REPO_ROOT / "models" / "training" / "laneseg_yolo_v1.yaml")
    assert config.data == REPO_ROOT / "data" / "external" / "roboflow_sidewalk_v1_split" / "data.yaml"
    assert config.model == "models/weights/yolov8n-seg.pt"
    assert config.imgsz == 640
    assert config.name == "smartcart_laneseg_yolov8n_seg_roboflow_v1"
    assert config.manifest_template == REPO_ROOT / "models" / "model_manifest.laneseg.example.json"


def test_validate_laneseg_config_rejects_missing_data():
    script = _load_script()
    config = script.TrainConfig(data=REPO_ROOT / "data" / "external" / "missing_sidewalk_yolo" / "data.yaml")
    errors = script.validate_config(config)
    assert any("missing YOLO data config" in error for error in errors)


def test_laneseg_dry_run_prints_training_plan(capsys):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_train_laneseg_yolo" / uuid4().hex
    data_yaml = workspace / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text("names: ['sidewalk']\n", encoding="utf-8")

    script.train_laneseg_yolo(script.TrainConfig(data=data_yaml, epochs=3, batch=2), dry_run=True)
    captured = capsys.readouterr()

    assert "laneseg_yolo_training_plan" in captured.out
    assert "epochs=3" in captured.out
    assert "batch=2" in captured.out
    assert "plots=True" in captured.out
    assert "status=dry_run" in captured.out


def test_laneseg_cli_dry_run_uses_config(monkeypatch, capsys):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_train_laneseg_yolo_cli" / uuid4().hex
    data_yaml = workspace / "laneseg" / "data.yaml"
    config_path = workspace / "training.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text("names: ['sidewalk']\n", encoding="utf-8")
    config_path.write_text(f"data: {data_yaml}\nepochs: 5\nbatch: 4\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_laneseg_yolo.py", "--config", str(config_path), "--dry-run", "--no-plots"])

    code = script.main()
    captured = capsys.readouterr()

    assert code == 0
    assert "epochs=5" in captured.out
    assert "batch=4" in captured.out
    assert "plots=False" in captured.out


def test_laneseg_config_rejects_unknown_fields():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_train_laneseg_yolo_bad" / uuid4().hex
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "bad_training_config.yaml"
    config_path.write_text("surprise: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown training config field"):
        script.load_train_config(config_path)
