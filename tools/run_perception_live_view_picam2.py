#!/usr/bin/env python3
import argparse
import json
import time
from dataclasses import asdict, is_dataclass

import cv2

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from picamera2 import Picamera2

from perception.camera_pipeline import (
    PerceptionPipeline,
    build_default_object_detector,
    build_default_traffic_light_provider,
    build_default_laneseg_provider,
    build_default_hazard_provider,
)


def build_pipeline(device: str = "cpu") -> PerceptionPipeline:
    return PerceptionPipeline(
        detector=build_default_object_detector(device=device),
        traffic_light_provider=build_default_traffic_light_provider(device=device),
        laneseg_provider=build_default_laneseg_provider(device=device),
        hazard_provider=build_default_hazard_provider(device=device),
    )


def to_jsonable(obj):
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


def get_field(obj, *names, default=None):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def draw_objects(frame_bgr, output):
    objects = get_field(output, "objects", default=[]) or []

    for obj in objects:
        bbox = get_field(obj, "bbox", "xyxy", "box", "bbox_xyxy")
        if bbox is None or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        label = get_field(obj, "class_name", "label", "name", default="object")
        conf = get_field(obj, "conf", "confidence", "score", default=None)

        if conf is None:
            text = str(label)
        else:
            try:
                text = f"{label} {float(conf):.2f}"
            except Exception:
                text = f"{label} {conf}"

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_bgr,
            text,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


def run_live_view(
    device="cpu",
    fps=3,
    show_window=True,
    max_frames=None,
    print_json_every=None,
    width=640,
    height=480,
):
    print(f"[info] building perception pipeline on device={device}")
    pipeline = build_pipeline(device=device)

    print("[info] starting Picamera2")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": (width, height)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    frame_interval = 1.0 / max(fps, 1)
    frame_count = 0
    window_name = "Perception Live View"

    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while max_frames is None or frame_count < max_frames:
            loop_start = time.time()

            frame_rgb = picam2.capture_array()
            if frame_rgb is None:
                raise RuntimeError("failed to capture frame from Picamera2")

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            infer_start = time.time()
            output = pipeline.process_frame(frame_bgr)
            infer_ms = (time.time() - infer_start) * 1000.0

            frame_count += 1

            if print_json_every and frame_count % print_json_every == 0:
                print(
                    json.dumps(
                        {
                            "frame": frame_count,
                            "infer_ms": round(infer_ms, 2),
                            "output": to_jsonable(output),
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                )

            if show_window:
                draw_objects(frame_bgr, output)
                cv2.putText(
                    frame_bgr,
                    f"frame={frame_count} infer={infer_ms:.1f} ms",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow(window_name, frame_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        print("[info] stopping Picamera2")
        picam2.stop()
        if show_window:
            cv2.destroyAllWindows()

    return frame_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Kept for compatibility; ignored by Picamera2.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--print-json-every", type=int, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    if args.camera != 0:
        print(f"[warn] --camera {args.camera} is ignored because this script uses Picamera2.")

    frames = run_live_view(
        device=args.device,
        fps=args.fps,
        show_window=not args.no_window,
        max_frames=args.max_frames,
        print_json_every=args.print_json_every,
        width=args.width,
        height=args.height,
    )

    print(f"[info] processed_frames={frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())