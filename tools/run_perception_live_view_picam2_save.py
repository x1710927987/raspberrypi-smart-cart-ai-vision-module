#!/usr/bin/env python3
import argparse
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from picamera2 import Picamera2

from perception.camera_pipeline import (
    PerceptionPipeline,
    build_default_object_detector,
    build_default_traffic_light_provider,
    build_default_laneseg_provider,
    build_default_hazard_provider,
)


def build_pipeline(device="cpu"):
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


def draw_output(frame_bgr, output, frame_count, infer_ms):
    objects = get_field(output, "objects", default=[]) or []

    for obj in objects:
        bbox = get_field(obj, "bbox", "xyxy", "box", "bbox_xyxy")
        if bbox is None or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        label = get_field(obj, "cls", "class_name", "label", "name", default="object")
        conf = get_field(obj, "conf", "confidence", "score", default=None)

        if conf is not None:
            text = f"{label} {float(conf):.2f}"
        else:
            text = str(label)

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

    traffic_light = get_field(output, "traffic_light", default=None)
    if traffic_light is not None:
        state = get_field(traffic_light, "state", default="unknown")
        conf = get_field(traffic_light, "conf", "confidence", default=None)
        if conf is not None:
            tl_text = f"traffic_light: {state} {float(conf):.2f}"
        else:
            tl_text = f"traffic_light: {state}"
        cv2.putText(
            frame_bgr,
            tl_text,
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        frame_bgr,
        f"frame={frame_count} infer={infer_ms:.1f} ms",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    return frame_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Ignored; Picamera2 is used.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--print-json-every", type=int, default=1)
    parser.add_argument("--save-dir", default="cache/live_view_frames")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] building perception pipeline on device={args.device}")
    pipeline = build_pipeline(device=args.device)

    print("[info] starting Picamera2")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": (args.width, args.height)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    frame_interval = 1.0 / max(args.fps, 1)
    frame_count = 0

    try:
        while frame_count < args.max_frames:
            loop_start = time.time()

            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            infer_start = time.time()
            output = pipeline.process_frame(frame_bgr)
            infer_ms = (time.time() - infer_start) * 1000.0

            frame_count += 1

            if args.print_json_every and frame_count % args.print_json_every == 0:
                print(json.dumps(
                    {
                        "frame": frame_count,
                        "infer_ms": round(infer_ms, 2),
                        "output": to_jsonable(output),
                    },
                    ensure_ascii=False,
                    default=str,
                ))

            if args.save_every and frame_count % args.save_every == 0:
                vis = draw_output(frame_bgr.copy(), output, frame_count, infer_ms)
                out_path = save_dir / f"frame_{frame_count:04d}.jpg"
                cv2.imwrite(str(out_path), vis)
                print(f"[info] saved {out_path}")

            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        print("[info] stopping Picamera2")
        picam2.stop()

    print(f"[info] processed_frames={frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
