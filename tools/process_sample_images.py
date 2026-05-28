from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.camera_pipeline import PerceptionPipeline, PipelineConfig
from perception.fusion import FusionConfig
from perception.preprocessing import PreprocessConfig
from perception.runtime import PerceptionOutput
from tools.run_perception_live_view import draw_perception_overlay


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_INPUT_DIR = REPO_ROOT / "demos" / "sample_image_processing" / "original_images"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "demos" / "sample_image_processing" / "processed_images"
SAMPLE_TARGETS: dict[str, tuple[str, str | None]] = {
    "bicycle": ("object", "bicycle"),
    "bike": ("object", "bicycle"),
    "car": ("object", "car"),
    "pedestrian": ("object", "pedestrian"),
    "pedestrain": ("object", "pedestrian"),
    "roadblock": ("object", "roadblock"),
    "scoot": ("object", "scooter"),
    "scooter": ("object", "scooter"),
    "traffic_light": ("traffic_light", None),
    "traffic_lights": ("traffic_light", None),
    "pothole": ("hazard", "pothole"),
    "curb": ("hazard", "curb"),
    "sidewalk": ("laneseg", None),
    "sidewallk": ("laneseg", None),
}


def process_images(
    *,
    input_dir: Path,
    output_dir: Path,
    device: str | None = "cpu",
    target_size: tuple[int, int] = (640, 480),
    max_objects: int = 20,
    overwrite: bool = True,
    show_status_panel: bool = False,
    draw_space: str = "processed",
    filter_mode: str = "auto",
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    image_paths = _find_images(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PerceptionPipeline.with_default_models(
        device=device,
        config=PipelineConfig(
            preprocess=PreprocessConfig(target_size=target_size, color_space="bgr", normalize=False),
            fusion=FusionConfig(max_objects=max_objects),
        ),
    )

    processed: list[dict[str, Any]] = []
    unreadable: list[str] = []
    skipped: list[str] = []

    for image_path in image_paths:
        output_path = output_dir / f"{image_path.stem}_annotated.jpg"
        if output_path.exists() and not overwrite:
            skipped.append(_display_path(image_path))
            continue

        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            unreadable.append(_display_path(image_path))
            continue

        tic = time.perf_counter()
        perception_output = pipeline.process_frame(frame, timestamp=time.time())
        elapsed_ms = (time.perf_counter() - tic) * 1000.0
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
        display_output, display_filter = _display_output_for_image(perception_output, image_path, filter_mode=filter_mode)

        overlay_frame = frame
        if draw_space == "processed":
            if pipeline.last_preprocess_result is None:
                raise RuntimeError("pipeline did not expose a preprocess result")
            overlay_frame = pipeline.last_preprocess_result.image
        annotated = draw_perception_overlay(overlay_frame, display_output, elapsed_ms=elapsed_ms, fps=fps, show_status_panel=show_status_panel)
        ok = cv2.imwrite(str(output_path), annotated)
        if not ok:
            raise RuntimeError(f"failed to write annotated image: {output_path}")

        payload = perception_output.to_dict()
        display_payload = display_output.to_dict()
        processed.append(
            {
                "input": _display_path(image_path),
                "output": _display_path(output_path),
                "infer_ms": round(elapsed_ms, 2),
                "display_filter": display_filter,
                "objects": len(perception_output.objects),
                "displayed_objects": len(display_output.objects),
                "traffic_light": payload["traffic_light"],
                "displayed_traffic_light": display_payload["traffic_light"],
                "laneseg": payload["laneseg"],
                "displayed_laneseg": display_payload["laneseg"],
                "hazard": payload["hazard"],
                "displayed_hazard": display_payload["hazard"],
            }
        )

    summary = {
        "status": "ok",
        "input_dir": _display_path(input_dir),
        "output_dir": _display_path(output_dir),
        "draw_space": draw_space,
        "filter_mode": filter_mode,
        "total_images": len(image_paths),
        "processed_images": len(processed),
        "skipped_images": len(skipped),
        "unreadable_images": len(unreadable),
        "processed": processed,
        "skipped": skipped,
        "unreadable": unreadable,
    }

    summary_path = output_dir / "processing_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _find_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {input_dir}")
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


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


def _parse_draw_space(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"processed", "original"}:
        raise argparse.ArgumentTypeError("draw space must be processed or original")
    return normalized


def _parse_filter_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"auto", "none"}:
        raise argparse.ArgumentTypeError("filter mode must be auto or none")
    return normalized


def _display_output_for_image(output: PerceptionOutput, image_path: Path, *, filter_mode: str) -> tuple[PerceptionOutput, str]:
    if filter_mode == "none":
        return output, "none"

    target = _target_for_image(image_path)
    if target is None:
        return output, "auto:no-match"

    target_type, target_label = target
    if target_type == "object":
        return (
            PerceptionOutput(
                timestamp=output.timestamp,
                laneseg=None,
                objects=[obj for obj in output.objects if obj.cls == target_label],
                traffic_light=None,
                hazard=None,
            ),
            f"auto:object:{target_label}",
        )
    if target_type == "traffic_light":
        return (
            PerceptionOutput(
                timestamp=output.timestamp,
                laneseg=None,
                objects=[],
                traffic_light=output.traffic_light,
                hazard=None,
            ),
            "auto:traffic_light",
        )
    if target_type == "laneseg":
        return (
            PerceptionOutput(
                timestamp=output.timestamp,
                laneseg=output.laneseg,
                objects=[],
                traffic_light=None,
                hazard=None,
            ),
            "auto:laneseg",
        )
    if target_type == "hazard":
        hazard = output.hazard if output.hazard is not None and output.hazard.type == target_label else None
        return (
            PerceptionOutput(
                timestamp=output.timestamp,
                laneseg=None,
                objects=[],
                traffic_light=None,
                hazard=hazard,
            ),
            f"auto:hazard:{target_label}",
        )
    return output, "auto:unknown-target"


def _target_for_image(image_path: Path) -> tuple[str, str | None] | None:
    stem = image_path.stem.lower()
    if stem in SAMPLE_TARGETS:
        return SAMPLE_TARGETS[stem]
    for key, target in SAMPLE_TARGETS.items():
        if key in stem:
            return target
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch annotate sample images with the SmartCart perception pipeline.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help=f"Input image directory. Default: {DEFAULT_INPUT_DIR.relative_to(REPO_ROOT)}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output directory for annotated images. Default: {DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)}")
    parser.add_argument("--device", default="cpu", help="Ultralytics device value, for example cpu or 0. Default: cpu.")
    parser.add_argument("--target-size", type=_parse_target_size, default=(640, 480), help="Preprocess size as WIDTHxHEIGHT. Default: 640x480.")
    parser.add_argument("--max-objects", type=int, default=20, help="Maximum objects kept after fusion. Default: 20.")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip output files that already exist.")
    parser.add_argument("--show-status-panel", action="store_true", help="Draw the live-view status panel in addition to boxes and labels.")
    parser.add_argument(
        "--draw-space",
        type=_parse_draw_space,
        default="processed",
        help="Image coordinate space used for drawing boxes. Use processed to match pipeline coordinates. Default: processed.",
    )
    parser.add_argument(
        "--filter-mode",
        type=_parse_filter_mode,
        default="auto",
        help="Display filtering mode. auto keeps only the intended sample target inferred from the filename; none draws all detections. Default: auto.",
    )
    args = parser.parse_args()

    summary = process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device,
        target_size=args.target_size,
        max_objects=args.max_objects,
        overwrite=not args.no_overwrite,
        show_status_panel=args.show_status_panel,
        draw_space=args.draw_space,
        filter_mode=args.filter_mode,
    )
    print(json.dumps({key: summary[key] for key in ("status", "total_images", "processed_images", "skipped_images", "unreadable_images", "output_dir")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
