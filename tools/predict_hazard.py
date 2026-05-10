from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.model_inference import FixedPredictionBackend, ManifestHazardDetector, UltralyticsBackend, load_model_manifest
from perception.runtime import Hazard


DEFAULT_MANIFEST = REPO_ROOT / "models" / "training" / "smartcart_hazard_yolov8n_roboflow_pt_v1.manifest.json"


def predict_hazard(
    image_path: str | Path,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    backend: Any | None = None,
    device: str | None = None,
) -> Optional[Hazard]:
    frame = _read_image(Path(image_path))
    manifest = load_model_manifest(manifest_path)
    detector = ManifestHazardDetector(manifest, backend or UltralyticsBackend(device=device))
    return detector.detect(frame)


def print_prediction(image_path: Path, manifest_path: Path, prediction: Optional[Hazard], *, as_json: bool = False) -> None:
    payload = {
        "image": str(image_path),
        "manifest": str(manifest_path),
        "detected": prediction is not None,
        "type": "unknown" if prediction is None else prediction.type,
        "confidence": 0.0 if prediction is None else round(float(prediction.conf), 4),
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False))
        return
    print(f"type={payload['type']}")
    print(f"confidence={payload['confidence']:.4f}")
    print("status=ok" if prediction is not None else "status=no_detection")


def _read_image(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"image does not exist: {path}")
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"failed to read image: {path}")
    return frame


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hazard detection on one image.")
    parser.add_argument("--image", required=True, type=Path, help="Input image path.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path, help="Model manifest path.")
    parser.add_argument("--device", help="Ultralytics device value, for example cpu, 0, or 0,1.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--mock", action="store_true", help="Use a fixed mock prediction for CLI smoke tests.")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        backend = FixedPredictionBackend({"label": "pothole", "conf": 0.9}) if args.mock else None
        prediction = predict_hazard(args.image, args.manifest, backend=backend, device=args.device)
        print_prediction(args.image, args.manifest, prediction, as_json=args.json)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
