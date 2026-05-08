from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: Path, output_dir: Path, *, every_n_frames: int, prefix: str, max_frames: int | None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    saved = 0
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % every_n_frames == 0:
            output_path = output_dir / f"{prefix}_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(output_path), frame):
                raise ValueError(f"could not write frame: {output_path}")
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        frame_index += 1
    capture.release()
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract image frames from a local video for dataset collection.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--every-n-frames", default=15, type=int)
    parser.add_argument("--prefix", default="video_frame")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.every_n_frames < 1:
        raise ValueError("--every-n-frames must be positive")
    saved = extract_frames(
        args.video,
        args.output_dir,
        every_n_frames=args.every_n_frames,
        prefix=args.prefix,
        max_frames=args.max_frames,
    )
    print(f"frames_saved={saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
