import importlib.util
import json
import sys
from pathlib import Path
from uuid import uuid4

from perception.model_inference import load_model_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "register_object_detection_model.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("register_object_detection_model", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_register_object_detection_model_copies_artifact_and_writes_manifest():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_register_object_detection_model" / uuid4().hex
    source_model = workspace / "run" / "weights" / "best.pt"
    run_dir = workspace / "run"
    weights_dir = workspace / "weights"
    manifest_dir = workspace / "manifests"
    source_model.parent.mkdir(parents=True, exist_ok=True)
    source_model.write_bytes(b"fake-object-yolo-weights")
    (run_dir / "results.csv").write_text(
        "\n".join(
            [
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)",
                "3,0.61,0.52,0.57,0.31",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = script.RegisterConfig(
        source_model=source_model,
        run_dir=run_dir,
        weights_dir=weights_dir,
        manifest_dir=manifest_dir,
        model_id="smartcart_objects_yolov8n_test_pt_v1",
        license="test-license",
        exported_at="2026-05-09",
        overwrite=True,
    )
    result = script.register_object_detection_model(config)

    assert result.artifact_path.exists()
    assert result.artifact_path.read_bytes() == b"fake-object-yolo-weights"
    assert result.sha256 == script.compute_sha256(source_model)
    manifest = load_model_manifest(result.manifest_path, require_artifact=True)
    assert manifest.model_id == "smartcart_objects_yolov8n_test_pt_v1"
    assert manifest.task == "objects"
    assert manifest.artifact_format == "pt"
    assert manifest.map_label("pedestrian") == "pedestrian"
    assert manifest.map_label("car") == "car"

    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "registered"
    assert payload["source"]["training_platform"] == "ultralytics"
    assert payload["source"]["dataset_manifest"] == "data/processed/objects_yolo_v1/data.yaml"
    assert payload["evaluation"]["metrics"]["map50"] == 0.57
    assert payload["evaluation"]["metrics"]["precision"] == 0.61
    assert payload["evaluation"]["metrics"]["recall"] == 0.52


def test_register_object_detection_model_dry_run_does_not_write_outputs(capsys):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_register_object_detection_model_dry_run" / uuid4().hex
    source_model = workspace / "run" / "weights" / "best.pt"
    weights_dir = workspace / "weights"
    manifest_dir = workspace / "manifests"
    source_model.parent.mkdir(parents=True, exist_ok=True)
    source_model.write_bytes(b"fake-object-yolo-weights")

    config = script.RegisterConfig(
        source_model=source_model,
        run_dir=workspace / "run",
        weights_dir=weights_dir,
        manifest_dir=manifest_dir,
        model_id="smartcart_objects_yolov8n_dryrun_pt_v1",
    )
    result = script.register_object_detection_model(config, dry_run=True)
    captured = capsys.readouterr()

    assert "object_detection_model_registration_plan" in captured.out
    assert "status=dry_run" in captured.out
    assert not result.artifact_path.exists()
    assert not result.manifest_path.exists()


def test_register_object_detection_model_cli_dry_run(monkeypatch, capsys):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_register_object_detection_model_cli" / uuid4().hex
    source_model = workspace / "run" / "weights" / "best.pt"
    source_model.parent.mkdir(parents=True, exist_ok=True)
    source_model.write_bytes(b"fake-object-yolo-weights")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "register_object_detection_model.py",
            "--source-model",
            str(source_model),
            "--run-dir",
            str(workspace / "run"),
            "--weights-dir",
            str(workspace / "weights"),
            "--manifest-dir",
            str(workspace / "manifests"),
            "--model-id",
            "smartcart_objects_yolov8n_cli_pt_v1",
            "--dry-run",
        ],
    )

    code = script.main()
    captured = capsys.readouterr()

    assert code == 0
    assert "status=dry_run" in captured.out
