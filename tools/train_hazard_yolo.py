from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "models" / "training" / "hazard_yolo_v1.yaml"
DEFAULT_DATA = REPO_ROOT / "data" / "external" / "roboflow_hazard_v1_split" / "data.yaml"
DEFAULT_PROJECT = REPO_ROOT / "models" / "training" / "hazard_yolo_v1"
DEFAULT_NAME = "smartcart_hazard_yolov8n_roboflow_v1"


@dataclass(frozen=True)
class TrainConfig:
    data: Path = DEFAULT_DATA
    model: str = "models/weights/yolov8n.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str | None = None
    workers: int = 4
    patience: int = 30
    seed: int = 42
    project: Path = DEFAULT_PROJECT
    name: str = DEFAULT_NAME
    exist_ok: bool = False
    plots: bool = True
    export_format: str = "none"
    manifest_template: Path = REPO_ROOT / "models" / "model_manifest.hazard.example.json"
    notes: str = "Hazard YOLO v1 baseline for pothole, curb, steps, speed bump, water, and debris."


def load_train_config(path: str | Path | None = None) -> TrainConfig:
    config = TrainConfig()
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not config_path.exists():
        return config
    values = _read_simple_yaml(config_path)
    return _apply_config_values(config, values)


def validate_config(config: TrainConfig) -> list[str]:
    errors: list[str] = []
    if not config.data.exists():
        errors.append(f"missing YOLO data config: {config.data}")
    if config.epochs <= 0:
        errors.append("epochs must be positive")
    if config.imgsz <= 0:
        errors.append("imgsz must be positive")
    if config.batch == 0:
        errors.append("batch must not be zero")
    if config.workers < 0:
        errors.append("workers must be non-negative")
    if config.patience < 0:
        errors.append("patience must be non-negative")
    if config.export_format not in {"none", "onnx", "tflite"}:
        errors.append("export_format must be one of: none, onnx, tflite")
    return errors


def train_hazard_yolo(config: TrainConfig, *, dry_run: bool = False) -> Path | None:
    errors = validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    if dry_run:
        print_training_plan(config)
        return None

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for training. Install it in the smartcart-ai environment before running this script."
        ) from exc

    model = YOLO(config.model)
    result = model.train(**_ultralytics_train_kwargs(config))
    run_dir = Path(getattr(result, "save_dir", config.project / config.name))
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    trained_model_path = best_pt if best_pt.exists() else last_pt

    if config.export_format != "none":
        if not trained_model_path.exists():
            raise FileNotFoundError(f"trained model checkpoint not found: {trained_model_path}")
        exported_model = YOLO(str(trained_model_path))
        exported_model.export(format=config.export_format, imgsz=config.imgsz)

    print(f"run_dir={run_dir}")
    print(f"trained_model={trained_model_path}")
    print("status=ok")
    return run_dir


def print_training_plan(config: TrainConfig) -> None:
    print("hazard_yolo_training_plan")
    print(f"data={config.data}")
    print(f"model={config.model}")
    print(f"epochs={config.epochs}")
    print(f"imgsz={config.imgsz}")
    print(f"batch={config.batch}")
    print(f"device={config.device or 'auto'}")
    print(f"workers={config.workers}")
    print(f"patience={config.patience}")
    print(f"seed={config.seed}")
    print(f"project={config.project}")
    print(f"name={config.name}")
    print(f"exist_ok={config.exist_ok}")
    print(f"plots={config.plots}")
    print(f"export_format={config.export_format}")
    print(f"manifest_template={config.manifest_template}")
    print("status=dry_run")


def _ultralytics_train_kwargs(config: TrainConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "data": str(config.data),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "workers": config.workers,
        "patience": config.patience,
        "seed": config.seed,
        "project": str(config.project),
        "name": config.name,
        "exist_ok": config.exist_ok,
        "plots": config.plots,
    }
    if config.device:
        kwargs["device"] = config.device
    return kwargs


def _apply_config_values(config: TrainConfig, values: dict[str, Any]) -> TrainConfig:
    path_fields = {"data", "project", "manifest_template"}
    known_fields = set(TrainConfig.__dataclass_fields__)
    updates: dict[str, Any] = {}
    for key, value in values.items():
        if key not in known_fields:
            raise ValueError(f"unknown training config field: {key}")
        if key == "export_format" and value is None:
            updates[key] = "none"
        elif key in path_fields:
            updates[key] = _resolve_repo_path(str(value))
        else:
            updates[key] = value
    return replace(config, **updates)


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_no}: expected key: value")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{line_no}: empty key")
        values[key] = _parse_scalar(raw_value.strip())
    return values


def _parse_scalar(value: str) -> Any:
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the hazard YOLO baseline model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Flat YAML training config.")
    parser.add_argument("--data", type=Path, help="YOLO data.yaml path.")
    parser.add_argument("--model", help="Ultralytics model name or checkpoint path.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", help="Ultralytics device value, for example cpu, 0, or 0,1.")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Disable Ultralytics plot generation for smoke tests.")
    parser.add_argument("--export-format", choices=["none", "onnx", "tflite"])
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the training plan without importing ultralytics.")
    return parser


def _merge_cli_args(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    updates: dict[str, Any] = {}
    for field_name in (
        "data",
        "model",
        "epochs",
        "imgsz",
        "batch",
        "device",
        "workers",
        "patience",
        "seed",
        "project",
        "name",
        "export_format",
    ):
        value = getattr(args, field_name)
        if value is not None:
            updates[field_name] = _resolve_repo_path(str(value)) if field_name in {"data", "project"} else value
    if args.exist_ok:
        updates["exist_ok"] = True
    if args.no_plots:
        updates["plots"] = False
    return replace(config, **updates)


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        config = _merge_cli_args(load_train_config(args.config), args)
        train_hazard_yolo(config, dry_run=args.dry_run)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
