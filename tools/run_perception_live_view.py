from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.camera_pipeline import PerceptionPipeline, PipelineConfig
from perception.fusion import FusionConfig
from perception.preprocessing import PreprocessConfig
from perception.runtime import PerceptionOutput
from io_camera.camera import CameraSource, create_camera_source


CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "pedestrian": (72, 220, 72),
    "bicycle": (0, 180, 255),
    "car": (255, 160, 64),
    "scooter": (64, 220, 255),
    "roadblock": (64, 64, 255),
    "obstacle": (64, 64, 255),
    "unknown": (180, 180, 180),
}
TEXT_COLOR = (245, 245, 245)
PANEL_BG = (32, 32, 32)
DEFAULT_WINDOW_NAME = "SmartCart Perception Live View"


def draw_perception_overlay(
    frame: np.ndarray,
    output: PerceptionOutput,
    *,
    elapsed_ms: float | None = None,
    fps: float | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    for obj in output.objects:
        x1, y1, x2, y2 = _clamp_bbox(obj.bbox, width, height)
        color = CLASS_COLORS.get(obj.cls, CLASS_COLORS["unknown"])
        label = f"{obj.cls} {obj.conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        _draw_label(annotated, label, x1, y1, color)

    status_lines = _status_lines(output, elapsed_ms=elapsed_ms, fps=fps)
    _draw_status_panel(annotated, status_lines)
    return annotated


def run_live_view(
    *,
    camera_index: int = 0,
    camera_backend: str = "auto",
    device: str | None = "cpu",
    target_size: tuple[int, int] = (640, 480),
    camera_width: int | None = 640,
    camera_height: int | None = 480,
    camera_fps: float | None = None,
    pixel_format: str = "BGR888",
    fps_limit: float = 5.0,
    max_objects: int = 20,
    max_frames: int | None = None,
    window_name: str = DEFAULT_WINDOW_NAME,
    display_scale: float = 1.0,
    show_window: bool = True,
    print_json_every: int = 0,
    save_dir: Path | None = None,
    save_every: int = 0,
    pipeline: PerceptionPipeline | None = None,
    camera_source: CameraSource | None = None,
) -> int:
    active_camera = camera_source or create_camera_source(
        backend=camera_backend,
        index=camera_index,
        width=camera_width,
        height=camera_height,
        fps=camera_fps,
        pixel_format=pixel_format,
    )
    active_camera.start()

    active_pipeline = pipeline or PerceptionPipeline.with_default_models(
        device=device,
        config=PipelineConfig(
            preprocess=PreprocessConfig(target_size=target_size, color_space="bgr", normalize=False),
            fusion=FusionConfig(max_objects=max_objects),
        ),
    )

    frame_count = 0
    last_frame_start = 0.0
    min_frame_interval = 1.0 / fps_limit if fps_limit > 0 else 0.0
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            if min_frame_interval > 0:
                wait_time = min_frame_interval - (time.perf_counter() - last_frame_start)
                if wait_time > 0:
                    time.sleep(wait_time)
            last_frame_start = time.perf_counter()

            frame = active_camera.read()

            tic = time.perf_counter()
            output = active_pipeline.process_frame(frame, timestamp=time.time())
            elapsed_ms = (time.perf_counter() - tic) * 1000.0
            frame_count += 1
            fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

            if print_json_every > 0 and frame_count % print_json_every == 0:
                print(
                    json.dumps(
                        {
                            "frame": frame_count,
                            "infer_ms": round(elapsed_ms, 2),
                            "output": output.to_dict(),
                        },
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            should_save = save_dir is not None and save_every > 0 and frame_count % save_every == 0
            annotated = None
            if show_window or should_save:
                annotated = draw_perception_overlay(frame, output, elapsed_ms=elapsed_ms, fps=fps)

            if should_save and annotated is not None:
                cv2.imwrite(str(save_dir / f"frame_{frame_count:04d}.jpg"), annotated)

            if show_window and annotated is not None:
                if display_scale != 1.0:
                    annotated = cv2.resize(annotated, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_LINEAR)
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        active_camera.release()
        if show_window:
            cv2.destroyWindow(window_name)
    return frame_count


def _status_lines(output: PerceptionOutput, *, elapsed_ms: float | None, fps: float | None) -> list[str]:
    traffic = "none"
    if output.traffic_light is not None:
        traffic = f"{output.traffic_light.state} {output.traffic_light.conf:.2f}"

    laneseg = "none"
    if output.laneseg is not None:
        laneseg = f"mask={output.laneseg.mask_id} {output.laneseg.conf:.2f}"

    hazard = "none"
    if output.hazard is not None:
        hazard = f"{output.hazard.type} {output.hazard.conf:.2f}"

    timing = []
    if elapsed_ms is not None:
        timing.append(f"{elapsed_ms:.1f} ms")
    if fps is not None:
        timing.append(f"{fps:.2f} FPS")

    return [
        f"objects: {len(output.objects)}",
        f"traffic_light: {traffic}",
        f"laneseg: {laneseg}",
        f"hazard: {hazard}",
        f"inference: {' | '.join(timing) if timing else 'n/a'}",
        "press q or Esc to exit",
    ]


def _draw_status_panel(frame: np.ndarray, lines: list[str]) -> None:
    if not lines:
        return
    x, y = 10, 10
    line_height = 22
    panel_width = max(260, max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for line in lines) + 18)
    panel_height = line_height * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (x + 9, y + 22 + index * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    label_y = max(0, y - text_height - baseline - 6)
    cv2.rectangle(frame, (x, label_y), (x + text_width + 8, label_y + text_height + baseline + 6), color, -1)
    cv2.putText(frame, text, (x + 4, label_y + text_height + 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _clamp_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain four coordinates, got {bbox!r}")
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


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


def _optional_positive_int(value: str) -> int | None:
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


def _optional_positive_float(value: str) -> float | None:
    parsed = float(value)
    if parsed <= 0:
        return None
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Show live camera frames annotated with SmartCart perception outputs.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for OpenCV backend. Default: 0.")
    parser.add_argument(
        "--camera-backend",
        choices=("opencv", "picamera2", "auto"),
        default="auto",
        help="Camera backend. Use picamera2 for Raspberry Pi CSI camera, opencv for USB/V4L2. Default: auto.",
    )
    parser.add_argument("--device", default="cpu", help="Ultralytics device value, for example cpu or 0. Default: cpu.")
    parser.add_argument("--target-size", type=_parse_target_size, default=(640, 480), help="Preprocess size as WIDTHxHEIGHT. Default: 640x480.")
    parser.add_argument("--camera-width", type=_optional_positive_int, default=640, help="Requested camera width. Use 0 to leave unchanged.")
    parser.add_argument("--camera-height", type=_optional_positive_int, default=480, help="Requested camera height. Use 0 to leave unchanged.")
    parser.add_argument("--camera-fps", type=_optional_positive_float, default=None, help="Requested camera capture FPS. Use 0 to let Picamera2 choose. Default: auto.")
    parser.add_argument("--pixel-format", default="BGR888", help="Picamera2 pixel format. Default: BGR888.")
    parser.add_argument("--fps", type=float, default=5.0, help="Maximum processing FPS. Use 0 for no cap. Default: 5.")
    parser.add_argument("--max-objects", type=int, default=20, help="Maximum objects kept after fusion. Default: 20.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames, useful for checks.")
    parser.add_argument("--window-name", default=DEFAULT_WINDOW_NAME)
    parser.add_argument("--display-scale", type=float, default=1.0, help="Resize display window content, for example 0.75.")
    parser.add_argument("--no-window", action="store_true", help="Run without cv2.imshow; useful for SSH smoke checks.")
    parser.add_argument("--print-json-every", type=int, default=0, help="Print frame metadata and PerceptionOutput JSON every N frames.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory for annotated frame snapshots.")
    parser.add_argument("--save-every", type=int, default=0, help="Save an annotated frame every N frames when --save-dir is set.")
    args = parser.parse_args()

    frames = run_live_view(
        camera_index=args.camera,
        camera_backend=args.camera_backend,
        device=args.device,
        target_size=args.target_size,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        pixel_format=args.pixel_format,
        fps_limit=args.fps,
        max_objects=args.max_objects,
        max_frames=args.max_frames,
        window_name=args.window_name,
        display_scale=args.display_scale,
        show_window=not args.no_window,
        print_json_every=args.print_json_every,
        save_dir=args.save_dir,
        save_every=args.save_every,
    )
    print(json.dumps({"status": "ok", "frames": frames}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
