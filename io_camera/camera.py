from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


CAMERA_BACKENDS = {"opencv", "picamera2", "auto"}


class CameraSource(Protocol):
    def start(self) -> None:
        ...

    def read(self) -> np.ndarray:
        ...

    def release(self) -> None:
        ...


@dataclass(frozen=True)
class CameraConfig:
    backend: str = "auto"
    index: int = 0
    width: int | None = 640
    height: int | None = 480
    fps: float | None = 30.0
    pixel_format: str = "BGR888"
    warmup_seconds: float = 0.1


class OpenCVCameraSource:
    def __init__(self, config: CameraConfig, *, cv2_module: Any | None = None):
        self.config = config
        self._cv2 = cv2_module
        self._capture: Any | None = None

    def start(self) -> None:
        cv2 = self._cv2 or _import_cv2()
        capture = cv2.VideoCapture(self.config.index)
        if self.config.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps is not None:
            capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"cannot open OpenCV camera index {self.config.index}")
        self._capture = capture

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("OpenCV camera source is not started")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("failed to read frame from OpenCV camera")
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class PiCamera2Source:
    def __init__(self, config: CameraConfig, *, picamera2_class: Any | None = None):
        self.config = config
        self._picamera2_class = picamera2_class
        self._camera: Any | None = None

    def start(self) -> None:
        picamera2_class = self._picamera2_class or _import_picamera2()
        camera = picamera2_class()
        main_config: dict[str, Any] = {"format": self.config.pixel_format}
        if self.config.width is not None and self.config.height is not None:
            main_config["size"] = (int(self.config.width), int(self.config.height))
        controls = {"FrameRate": float(self.config.fps)} if self.config.fps is not None else None
        video_config = camera.create_video_configuration(main=main_config, controls=controls)
        camera.configure(video_config)
        camera.start()
        if self.config.warmup_seconds > 0:
            time.sleep(self.config.warmup_seconds)
        self._camera = camera

    def read(self) -> np.ndarray:
        if self._camera is None:
            raise RuntimeError("Picamera2 source is not started")
        frame = self._camera.capture_array()
        if frame is None:
            raise RuntimeError("failed to read frame from Picamera2")
        return _ensure_bgr_frame(np.asarray(frame), pixel_format=self.config.pixel_format)

    def release(self) -> None:
        if self._camera is not None:
            stop = getattr(self._camera, "stop", None)
            close = getattr(self._camera, "close", None)
            if callable(stop):
                stop()
            if callable(close):
                close()
            self._camera = None


class AutoCameraSource:
    def __init__(self, config: CameraConfig):
        self.config = config
        self._active: CameraSource | None = None

    def start(self) -> None:
        errors: list[str] = []
        for backend in ("picamera2", "opencv"):
            source = _build_camera_source(self.config, backend=backend)
            try:
                source.start()
            except Exception as exc:
                source.release()
                errors.append(f"{backend}: {exc}")
                continue
            self._active = source
            return
        raise RuntimeError("failed to open camera with auto backend: " + "; ".join(errors))

    def read(self) -> np.ndarray:
        if self._active is None:
            raise RuntimeError("auto camera source is not started")
        return self._active.read()

    def release(self) -> None:
        if self._active is not None:
            self._active.release()
            self._active = None


def create_camera_source(
    *,
    backend: str = "auto",
    index: int = 0,
    width: int | None = 640,
    height: int | None = 480,
    fps: float | None = 30.0,
    pixel_format: str = "BGR888",
    warmup_seconds: float = 0.1,
) -> CameraSource:
    normalized_backend = backend.strip().lower()
    if normalized_backend not in CAMERA_BACKENDS:
        raise ValueError(f"camera backend must be one of {sorted(CAMERA_BACKENDS)}, got {backend!r}")
    config = CameraConfig(
        backend=normalized_backend,
        index=index,
        width=width,
        height=height,
        fps=fps,
        pixel_format=pixel_format,
        warmup_seconds=warmup_seconds,
    )
    return _build_camera_source(config, backend=normalized_backend)


def _build_camera_source(config: CameraConfig, *, backend: str) -> CameraSource:
    if backend == "opencv":
        return OpenCVCameraSource(config)
    if backend == "picamera2":
        return PiCamera2Source(config)
    if backend == "auto":
        return AutoCameraSource(config)
    raise ValueError(f"unsupported camera backend: {backend!r}")


def _ensure_bgr_frame(frame: np.ndarray, *, pixel_format: str = "BGR888") -> np.ndarray:
    if frame.ndim != 3:
        raise RuntimeError(f"camera frame must be HxWxC, got shape {frame.shape}")
    normalized_format = pixel_format.strip().upper()
    if frame.shape[2] == 3:
        if normalized_format in {"RGB", "RGB888"}:
            return frame[:, :, ::-1].copy()
        return frame
    if frame.shape[2] == 4:
        rgb_oriented = normalized_format.startswith("RGB") or normalized_format.startswith("RGBA")
        frame3 = frame[:, :, :3]
        return frame3[:, :, ::-1].copy() if rgb_oriented else frame3
    raise RuntimeError(f"camera frame must have 3 or 4 channels, got shape {frame.shape}")


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv camera backend is not available. Install opencv-python or python3-opencv.") from exc
    return cv2


def _import_picamera2() -> Any:
    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise RuntimeError("picamera2 camera backend is not available. Install it with: sudo apt install python3-picamera2") from exc
    return Picamera2
