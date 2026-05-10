from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.model_inference import ManifestTrafficLightClassifier, UltralyticsBackend, load_model_manifest
from perception.runtime import TrafficLight


DEFAULT_MANIFEST = REPO_ROOT / "models" / "training" / "smartcart_traffic_light_yolov8n_combined_v2_pt_v1.manifest.json"


def predict_traffic_light(
    image_path: str | Path,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    backend: Any | None = None,
    device: str | None = None,
) -> TrafficLight | None:
    frame = read_image(Path(image_path))
    manifest = load_model_manifest(manifest_path, require_artifact=True)
    predictor = ManifestTrafficLightClassifier(manifest, backend or UltralyticsBackend(device=device))
    return predictor.detect(frame)


def read_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"image does not exist: {path}")
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"failed to read image bytes: {path}") from exc
    if data.size == 0:
        raise ValueError(f"image is empty: {path}")
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError(f"image is not readable by OpenCV: {path}")
    return frame


def print_prediction(image_path: Path, manifest_path: Path, prediction: TrafficLight | None, *, as_json: bool) -> None:
    detected = prediction is not None
    state = prediction.state if prediction is not None else "unknown"
    confidence = float(prediction.conf) if prediction is not None else 0.0
    if as_json:
        print(
            json.dumps(
                {
                    "image": str(image_path),
                    "manifest": str(manifest_path),
                    "detected": detected,
                    "state": state,
                    "confidence": round(confidence, 4),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return
    print(f"image={image_path}")
    print(f"manifest={manifest_path}")
    print(f"state={state}")
    print(f"confidence={confidence:.4f}")
    print("status=ok" if detected else "status=no_detection")


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run traffic-light inference on one image.")
    parser.add_argument("--image", required=True, type=Path, help="Input image path.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path, help="Traffic-light model manifest.")
    parser.add_argument("--device", help="Ultralytics device value, for example cpu, 0, or 0,1.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    image_path = _resolve_path(args.image)
    manifest_path = _resolve_path(args.manifest)
    try:
        prediction = predict_traffic_light(image_path, manifest_path, device=args.device)
        print_prediction(image_path, manifest_path, prediction, as_json=args.json)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
