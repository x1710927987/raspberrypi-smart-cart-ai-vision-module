from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "models" / "training" / "object_detection_yolo_v1" / "smartcart_objects_yolov8n_roboflow_v1"
DEFAULT_SOURCE_MODEL = DEFAULT_RUN_DIR / "weights" / "best.pt"
DEFAULT_TEMPLATE = REPO_ROOT / "models" / "model_manifest.objects.example.json"
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "models" / "weights"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "models" / "training"
DEFAULT_MODEL_ID = "smartcart_objects_yolov8n_roboflow_pt_v1"


@dataclass(frozen=True)
class RegisterConfig:
    source_model: Path = DEFAULT_SOURCE_MODEL
    run_dir: Path | None = DEFAULT_RUN_DIR
    manifest_template: Path = DEFAULT_TEMPLATE
    weights_dir: Path = DEFAULT_WEIGHTS_DIR
    manifest_dir: Path = DEFAULT_MANIFEST_DIR
    model_id: str = DEFAULT_MODEL_ID
    artifact_format: str | None = None
    architecture: str = "yolov8n"
    version: str = "v1"
    dataset_version: str = "objects_yolo_v1"
    license: str = "unknown"
    exported_at: str = date.today().isoformat()
    overwrite: bool = False


@dataclass(frozen=True)
class RegisterResult:
    artifact_path: Path
    manifest_path: Path
    sha256: str
    size_bytes: int


def register_object_detection_model(config: RegisterConfig, *, dry_run: bool = False) -> RegisterResult:
    source_model = config.source_model.resolve()
    if not source_model.exists():
        raise FileNotFoundError(f"source model does not exist: {source_model}")
    if not config.manifest_template.exists():
        raise FileNotFoundError(f"manifest template does not exist: {config.manifest_template}")

    artifact_format = config.artifact_format or _infer_artifact_format(source_model)
    artifact_path = (config.weights_dir / f"{config.model_id}.{artifact_format}").resolve()
    manifest_path = (config.manifest_dir / f"{config.model_id}.manifest.json").resolve()
    if artifact_path.exists() and not config.overwrite:
        raise FileExistsError(f"artifact already exists: {artifact_path}")
    if manifest_path.exists() and not config.overwrite:
        raise FileExistsError(f"manifest already exists: {manifest_path}")

    sha256 = compute_sha256(source_model)
    size_bytes = source_model.stat().st_size
    manifest = build_manifest(
        template_path=config.manifest_template,
        config=config,
        artifact_format=artifact_format,
        artifact_path=artifact_path,
        sha256=sha256,
        size_bytes=size_bytes,
    )

    if dry_run:
        print_registration_plan(source_model, artifact_path, manifest_path, sha256, size_bytes)
        return RegisterResult(artifact_path=artifact_path, manifest_path=manifest_path, sha256=sha256, size_bytes=size_bytes)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model, artifact_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"artifact={artifact_path}")
    print(f"manifest={manifest_path}")
    print(f"sha256={sha256}")
    print(f"size_bytes={size_bytes}")
    print("status=ok")
    return RegisterResult(artifact_path=artifact_path, manifest_path=manifest_path, sha256=sha256, size_bytes=size_bytes)


def build_manifest(
    *,
    template_path: Path,
    config: RegisterConfig,
    artifact_format: str,
    artifact_path: Path,
    sha256: str,
    size_bytes: int,
) -> dict[str, Any]:
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest template root must be a JSON object")

    payload["model_id"] = config.model_id
    payload["task"] = "objects"
    payload["status"] = "registered"
    payload["description"] = f"Registered object-detection YOLO model artifact from {config.run_dir or config.source_model.parent}."

    artifact = _ensure_object(payload, "artifact")
    artifact["path"] = _repo_relative_posix(artifact_path)
    artifact["format"] = artifact_format
    artifact["architecture"] = config.architecture
    artifact["version"] = config.version
    artifact["sha256"] = sha256
    artifact["size_bytes"] = size_bytes

    source = _ensure_object(payload, "source")
    source["training_platform"] = "ultralytics"
    source["dataset_manifest"] = "data/processed/objects_yolo_v1/data.yaml"
    source["dataset_version"] = config.dataset_version
    source["license"] = config.license
    source["exported_at"] = config.exported_at
    source["training_run"] = _repo_relative_posix(config.run_dir) if config.run_dir else None

    runtime = _ensure_object(payload, "runtime")
    runtime["preferred_backend"] = _preferred_backend(artifact_format)
    runtime["fallback_backend"] = "opencv_dnn" if artifact_format == "onnx" else None

    evaluation = _ensure_object(payload, "evaluation")
    metrics = _ensure_object(evaluation, "metrics")
    metrics.update(_read_final_metrics(config.run_dir) if config.run_dir else {})

    integration = _ensure_object(payload, "integration")
    notes = integration.get("notes", [])
    if not isinstance(notes, list):
        notes = []
    notes = [str(item) for item in notes]
    notes.append("Generated by tools/register_object_detection_model.py.")
    if artifact_format == "pt":
        notes.append("PT artifacts are useful for local validation; export ONNX or TFLite before Raspberry Pi deployment.")
    integration["notes"] = _dedupe(notes)
    return payload


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_registration_plan(source_model: Path, artifact_path: Path, manifest_path: Path, sha256: str, size_bytes: int) -> None:
    print("object_detection_model_registration_plan")
    print(f"source_model={source_model}")
    print(f"artifact={artifact_path}")
    print(f"manifest={manifest_path}")
    print(f"sha256={sha256}")
    print(f"size_bytes={size_bytes}")
    print("status=dry_run")


def _read_final_metrics(run_dir: Path | None) -> dict[str, float | None]:
    if run_dir is None:
        return {}
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return {}
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    final = {key.strip(): value for key, value in rows[-1].items()}
    return {
        "map50": _float_or_none(final.get("metrics/mAP50(B)")),
        "map50_95": _float_or_none(final.get("metrics/mAP50-95(B)")),
        "precision": _float_or_none(final.get("metrics/precision(B)")),
        "recall": _float_or_none(final.get("metrics/recall(B)")),
        "latency_ms": None,
    }


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _infer_artifact_format(source_model: Path) -> str:
    suffix = source_model.suffix.lower().lstrip(".")
    if suffix not in {"pt", "onnx", "tflite"}:
        raise ValueError(f"cannot infer supported artifact format from: {source_model}")
    return suffix


def _preferred_backend(artifact_format: str) -> str:
    if artifact_format == "onnx":
        return "onnxruntime"
    if artifact_format == "tflite":
        return "tflite"
    if artifact_format == "pt":
        return "ultralytics"
    return artifact_format


def _ensure_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        value = {}
        payload[key] = value
    return value


def _repo_relative_posix(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register a trained object-detection YOLO artifact into models/weights and a model manifest.")
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL, help="Trained model artifact, usually a YOLO best.pt file.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Ultralytics run directory containing results.csv.")
    parser.add_argument("--manifest-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--artifact-format", choices=["pt", "onnx", "tflite"], help="Override artifact format inference from source suffix.")
    parser.add_argument("--architecture", default="yolov8n")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--dataset-version", default="objects_yolo_v1")
    parser.add_argument("--license", default="unknown")
    parser.add_argument("--exported-at", default=date.today().isoformat())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_cli_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else REPO_ROOT / path


def _config_from_args(args: argparse.Namespace) -> RegisterConfig:
    return RegisterConfig(
        source_model=_resolve_cli_path(args.source_model),
        run_dir=_resolve_cli_path(args.run_dir),
        manifest_template=_resolve_cli_path(args.manifest_template),
        weights_dir=_resolve_cli_path(args.weights_dir),
        manifest_dir=_resolve_cli_path(args.manifest_dir),
        model_id=args.model_id,
        artifact_format=args.artifact_format,
        architecture=args.architecture,
        version=args.version,
        dataset_version=args.dataset_version,
        license=args.license,
        exported_at=args.exported_at,
        overwrite=args.overwrite,
    )


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        register_object_detection_model(_config_from_args(args), dry_run=args.dry_run)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
