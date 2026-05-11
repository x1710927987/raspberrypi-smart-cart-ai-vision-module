from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from perception import validate_perception_output
from perception.camera_pipeline import (
    PerceptionPipeline,
    PipelineConfig,
    build_default_hazard_provider,
    build_default_laneseg_provider,
    build_default_object_detector,
    build_default_traffic_light_provider,
)
from perception.fusion import FusionConfig
from perception.preprocessing import PreprocessConfig
from perception.runtime import PerceptionOutput

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_OUTPUT_JSON = REPO_ROOT / "cache" / "evaluation" / "unified_pipeline_smoke_test.json"
DEFAULT_OUTPUT_REPORT = REPO_ROOT / "cache" / "evaluation" / "unified_pipeline_smoke_test.md"
DEFAULT_SAMPLE_CANDIDATES = (
    REPO_ROOT / "data" / "external" / "objects_combined_v3_split" / "test" / "images",
    REPO_ROOT / "data" / "external" / "traffic_light_combined_v2_split" / "test" / "images",
    REPO_ROOT / "data" / "external" / "roboflow_sidewalk_v1_split" / "test" / "images",
    REPO_ROOT / "data" / "external" / "roboflow_hazard_v1_split" / "test" / "images",
)


@dataclass(frozen=True)
class SmokeSampleResult:
    sample_index: int
    image: str
    shape_hw: list[int]
    elapsed_ms: float
    object_count: int
    has_laneseg: bool
    has_traffic_light: bool
    has_hazard: bool
    perception_output: dict[str, Any]


@dataclass(frozen=True)
class SmokeResult:
    status: str
    loaded_pipeline_elapsed_ms: float
    sample_count: int
    modules_invoked: list[str]
    samples_with_objects: int
    samples_with_laneseg: int
    samples_with_traffic_light: int
    samples_with_hazard: int
    items: list[SmokeSampleResult]


def run_smoke_test(
    *,
    images: Sequence[str | Path] | None = None,
    limit: int | None = None,
    device: str | None = None,
    target_size: tuple[int, int] = (640, 480),
    max_objects: int = 20,
    detector: Any | None = None,
    traffic_light_provider: Any | None = None,
    laneseg_provider: Any | None = None,
    hazard_provider: Any | None = None,
) -> SmokeResult:
    image_paths = _resolve_images(images, limit=limit)
    if not image_paths:
        raise FileNotFoundError("no sample images were found; pass --image or create default test split images")

    load_start = time.perf_counter()
    pipeline = PerceptionPipeline(
        detector=detector or build_default_object_detector(device=device),
        traffic_light_provider=traffic_light_provider or build_default_traffic_light_provider(device=device),
        laneseg_provider=laneseg_provider or build_default_laneseg_provider(device=device),
        hazard_provider=hazard_provider or build_default_hazard_provider(device=device),
        config=PipelineConfig(
            preprocess=PreprocessConfig(target_size=target_size, color_space="bgr", normalize=False),
            fusion=FusionConfig(max_objects=max_objects),
        ),
    )
    load_elapsed_ms = round((time.perf_counter() - load_start) * 1000.0, 2)

    items: list[SmokeSampleResult] = []
    for sample_index, image_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"unreadable sample image: {image_path}")
        tic = time.perf_counter()
        output = pipeline.process_frame(frame, timestamp=time.time())
        elapsed_ms = round((time.perf_counter() - tic) * 1000.0, 2)
        decoded = PerceptionOutput.from_json(output.to_json())
        validate_perception_output(decoded)
        payload = decoded.to_dict()
        items.append(
            SmokeSampleResult(
                sample_index=sample_index,
                image=_repo_relative_posix(image_path),
                shape_hw=[int(frame.shape[0]), int(frame.shape[1])],
                elapsed_ms=elapsed_ms,
                object_count=len(payload["objects"]),
                has_laneseg=payload["laneseg"] is not None,
                has_traffic_light=payload["traffic_light"] is not None,
                has_hazard=payload["hazard"] is not None,
                perception_output=payload,
            )
        )

    return SmokeResult(
        status="ok",
        loaded_pipeline_elapsed_ms=load_elapsed_ms,
        sample_count=len(items),
        modules_invoked=["objects", "traffic_light", "laneseg", "hazard"],
        samples_with_objects=sum(1 for item in items if item.object_count > 0),
        samples_with_laneseg=sum(1 for item in items if item.has_laneseg),
        samples_with_traffic_light=sum(1 for item in items if item.has_traffic_light),
        samples_with_hazard=sum(1 for item in items if item.has_hazard),
        items=items,
    )


def write_outputs(result: SmokeResult, *, output_json: str | Path | None, output_report: str | Path | None) -> None:
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_to_dict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    if output_report is not None:
        path = Path(output_report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(result), encoding="utf-8")


def render_markdown(result: SmokeResult) -> str:
    lines = [
        "# Unified Perception Pipeline Smoke Test",
        "",
        f"- status: `{result.status}`",
        f"- sample_count: {result.sample_count}",
        f"- modules_invoked: `{', '.join(result.modules_invoked)}`",
        f"- samples_with_objects: {result.samples_with_objects}",
        f"- samples_with_laneseg: {result.samples_with_laneseg}",
        f"- samples_with_traffic_light: {result.samples_with_traffic_light}",
        f"- samples_with_hazard: {result.samples_with_hazard}",
        "",
        "## Samples",
        "",
    ]
    for item in result.items:
        output = item.perception_output
        lines.extend(
            [
                f"### Sample {item.sample_index}",
                "",
                f"- image: `{item.image}`",
                f"- shape_hw: `{item.shape_hw}`",
                f"- elapsed_ms: {item.elapsed_ms}",
                f"- object_count: {item.object_count}",
                f"- laneseg: `{output['laneseg']}`",
                f"- traffic_light: `{output['traffic_light']}`",
                f"- hazard: `{output['hazard']}`",
                f"- objects_preview: `{output['objects'][:5]}`",
                "",
            ]
        )
    return "\n".join(lines)


def _resolve_images(images: Sequence[str | Path] | None, *, limit: int | None) -> list[Path]:
    if images:
        paths = [_resolve_existing_image(Path(image)) for image in images]
    else:
        paths = []
        for directory in DEFAULT_SAMPLE_CANDIDATES:
            first = _first_image(directory)
            if first is not None:
                paths.append(first)
    return paths[:limit] if limit is not None else paths


def _resolve_existing_image(path: Path) -> Path:
    resolved = path if path.is_absolute() else REPO_ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"sample image does not exist: {path}")
    if not resolved.is_file() or resolved.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"sample image must be an image file: {path}")
    return resolved


def _first_image(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    return None


def _to_dict(result: SmokeResult) -> dict[str, Any]:
    return asdict(result)


def _repo_relative_posix(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_target_size(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("target size must be WIDTHxHEIGHT, for example 640x480")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target size must contain integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("target size width and height must be positive")
    return width, height


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real-model smoke test for the unified perception pipeline.")
    parser.add_argument("--image", action="append", default=None, help="Image to process. Can be passed multiple times. Defaults to one image per known test split.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N resolved images.")
    parser.add_argument("--device", default=None, help="Ultralytics device value, for example cpu, 0, or 0,1.")
    parser.add_argument("--target-size", type=_parse_target_size, default=(640, 480), help="Preprocess size as WIDTHxHEIGHT. Default: 640x480.")
    parser.add_argument("--max-objects", type=int, default=20)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    parser.add_argument("--json", action="store_true", help="Print the full JSON result.")
    args = parser.parse_args()

    result = run_smoke_test(
        images=args.image,
        limit=args.limit,
        device=args.device,
        target_size=args.target_size,
        max_objects=args.max_objects,
    )
    write_outputs(result, output_json=args.output_json, output_report=args.output_report)
    payload = _to_dict(result)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"status={result.status}")
        print(f"json={args.output_json}")
        print(f"report={args.output_report}")
        print(f"sample_count={result.sample_count}")
        print(f"samples_with_objects={result.samples_with_objects}")
        print(f"samples_with_laneseg={result.samples_with_laneseg}")
        print(f"samples_with_traffic_light={result.samples_with_traffic_light}")
        print(f"samples_with_hazard={result.samples_with_hazard}")
        for item in result.items:
            output = item.perception_output
            print("---")
            print(f"sample={item.sample_index}")
            print(f"image={item.image}")
            print(f"elapsed_ms={item.elapsed_ms}")
            print(f"objects={item.object_count}")
            print(f"laneseg={output['laneseg']}")
            print(f"traffic_light={output['traffic_light']}")
            print(f"hazard={output['hazard']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
