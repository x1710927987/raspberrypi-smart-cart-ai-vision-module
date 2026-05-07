from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

ColorSpace = str


@dataclass(frozen=True)
class PreprocessConfig:
    target_size: Tuple[int, int] = (640, 480)
    color_space: ColorSpace = "bgr"
    normalize: bool = False


@dataclass(frozen=True)
class PreprocessResult:
    image: np.ndarray
    original_size: Tuple[int, int]
    target_size: Tuple[int, int]
    scale_x: float
    scale_y: float
    color_space: ColorSpace
    normalized: bool

    def bbox_to_original(self, bbox: Sequence[float]) -> List[float]:
        _validate_bbox_values(bbox)
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return [x1 / self.scale_x, y1 / self.scale_y, x2 / self.scale_x, y2 / self.scale_y]

    def bbox_to_processed(self, bbox: Sequence[float]) -> List[float]:
        _validate_bbox_values(bbox)
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return [x1 * self.scale_x, y1 * self.scale_y, x2 * self.scale_x, y2 * self.scale_y]


def preprocess_frame(frame: np.ndarray, config: PreprocessConfig | None = None) -> PreprocessResult:
    cfg = config or PreprocessConfig()
    width, height = _validate_target_size(cfg.target_size)
    source = validate_frame(frame)
    original_height, original_width = source.shape[:2]
    resized = cv2.resize(source, (width, height), interpolation=cv2.INTER_LINEAR)
    converted = convert_color(resized, cfg.color_space)
    image = converted.astype(np.float32) / 255.0 if cfg.normalize else converted.copy()
    return PreprocessResult(
        image=image,
        original_size=(original_width, original_height),
        target_size=(width, height),
        scale_x=width / float(original_width),
        scale_y=height / float(original_height),
        color_space=cfg.color_space.lower(),
        normalized=cfg.normalize,
    )


def validate_frame(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy.ndarray")
    if frame.size == 0:
        raise ValueError("frame must not be empty")
    if frame.ndim not in (2, 3):
        raise ValueError("frame must be a grayscale or color image")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ValueError("frame height and width must be positive")
    if frame.ndim == 3 and frame.shape[2] not in (1, 3, 4):
        raise ValueError("color frame must have 1, 3, or 4 channels")
    return frame


def convert_color(frame: np.ndarray, color_space: ColorSpace) -> np.ndarray:
    source = validate_frame(frame)
    target = color_space.lower()
    if target not in {"bgr", "rgb", "gray"}:
        raise ValueError("color_space must be one of: bgr, rgb, gray")
    channels = 1 if source.ndim == 2 else source.shape[2]
    if target == "gray":
        if source.ndim == 2:
            return source.copy()
        if channels == 1:
            return source[:, :, 0].copy()
        return cv2.cvtColor(source, cv2.COLOR_BGR2GRAY if channels == 3 else cv2.COLOR_BGRA2GRAY)
    if target == "bgr":
        if source.ndim == 2 or channels == 1:
            return cv2.cvtColor(source if source.ndim == 2 else source[:, :, 0], cv2.COLOR_GRAY2BGR)
        return source.copy() if channels == 3 else cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
    if source.ndim == 2 or channels == 1:
        return cv2.cvtColor(source if source.ndim == 2 else source[:, :, 0], cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(source, cv2.COLOR_BGR2RGB if channels == 3 else cv2.COLOR_BGRA2RGB)


def _validate_target_size(target_size: Tuple[int, int]) -> Tuple[int, int]:
    if len(target_size) != 2:
        raise ValueError("target_size must be a (width, height) tuple")
    width, height = [int(v) for v in target_size]
    if width <= 0 or height <= 0:
        raise ValueError("target_size width and height must be positive")
    return width, height


def _validate_bbox_values(bbox: Sequence[float]) -> None:
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly 4 values")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x2 > x1 and y2 > y1")
