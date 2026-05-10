import importlib.util
import json
import sys
from pathlib import Path
from uuid import uuid4

from perception.model_inference import load_model_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "register_laneseg_model.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("register_laneseg_model", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_register_laneseg_model_copies_artifact_and_writes_manifest():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_register_laneseg_model" / uuid4().hex
    source_model = workspace / "run" / "weights" / "best.pt"
    run_dir = workspace / "run"
    weights_dir = workspace / "weights"
    manifest_dir = workspace / "manifests"
    source_model.parent.mkdir(parents=True, exist_ok=True)
    source_model.write_bytes(b"fake-laneseg-yolo-weights")
    (run_dir / "results.csv").write_text(
        "\n".join(
            [
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),metrics/precision(M),metrics/recall(M),metrics/mAP50(M),metrics/mAP50-95(M)",
                "3,0.71,0.62,0.67,0.41,0.81,0.72,0.77,0.51",
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
        model_id="smartcart_laneseg_yolov8n_seg_test_pt_v1",
        license="test-license",
        exported_at="2026-05-10",
        overwrite=True,
    )
    result = script.register_laneseg_model(config)

    assert result.artifact_path.exists()
    assert result.artifact_path.read_bytes() == b"fake-laneseg-yolo-weights"
    assert result.sha256 == script.compute_sha256(source_model)
    manifest = load_model_manifest(result.manifest_path, require_artifact=True)
    assert manifest.model_id == "smartcart_laneseg_yolov8n_seg_test_pt_v1"
    assert manifest.task == "laneseg"
    assert manifest.artifact_format == "pt"

    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "registered"
    assert payload["source"]["training_platform"] == "ultralytics"
    assert payload["source"]["dataset_manifest"] == "data/external/roboflow_sidewalk_v1_split/data.yaml"
    assert payload["evaluation"]["metrics"]["mask_map50"] == 0.77
    assert payload["evaluation"]["metrics"]["mask_precision"] == 0.81
    assert payload["evaluation"]["metrics"]["mask_recall"] == 0.72


def test_register_laneseg_model_dry_run_does_not_write_outputs(capsys):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_register_laneseg_model_dry_run" / uuid4().hex
    source_model = workspace / "run" / "weights" / "best.pt"
    weights_dir = workspace / "weights"
    manifest_dir = workspace / "manifests"
    source_model.parent.mkdir(parents=True, exist_ok=True)
    source_model.write_bytes(b"fake-laneseg-yolo-weights")

    config = script.RegisterConfig(
        source_model=source_model,
        run_dir=workspace / "run",
        weights_dir=weights_dir,
        manifest_dir=manifest_dir,
        model_id="smartcart_laneseg_yolov8n_seg_dryrun_pt_v1",
    )
    result = script.register_laneseg_model(config, dry_run=True)
    captured = capsys.readouterr()

    assert "laneseg_model_registration_plan" in captured.out
    assert "status=dry_run" in captured.out
    assert not result.artifact_path.exists()
    assert not result.manifest_path.exists()


def test_register_laneseg_model_cli_dry_run(monkeypatch, capsys):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_register_laneseg_model_cli" / uuid4().hex
    source_model = workspace / "run" / "weights" / "best.pt"
    source_model.parent.mkdir(parents=True, exist_ok=True)
    source_model.write_bytes(b"fake-laneseg-yolo-weights")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "register_laneseg_model.py",
            "--source-model",
            str(source_model),
            "--run-dir",
            str(workspace / "run"),
            "--weights-dir",
            str(workspace / "weights"),
            "--manifest-dir",
            str(workspace / "manifests"),
            "--model-id",
            "smartcart_laneseg_yolov8n_seg_cli_pt_v1",
            "--dry-run",
        ],
    )

    code = script.main()
    captured = capsys.readouterr()

    assert code == 0
    assert "status=dry_run" in captured.out
