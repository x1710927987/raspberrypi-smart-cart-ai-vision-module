from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from perception.preprocessing import PreprocessResult, validate_frame
from perception.runtime import LaneSeg


@dataclass(frozen=True)
class LaneSegConfig:
    mask_id: int = 1
    input_color_space: str = "bgr"
    roi_top_ratio: float = 0.45
    min_drivable_ratio: float = 0.08
    min_value: int = 35
    max_saturation: int = 110
    morphology_kernel_size: int = 5


@dataclass(frozen=True)
class LaneSegMask:
    mask_id: int
    mask: np.ndarray
    conf: float
    drivable_ratio: float

    def to_schema(self) -> LaneSeg:
        return LaneSeg(mask_id=int(self.mask_id), conf=float(self.conf))


class RuleBasedLaneSegmenter:
    """Lightweight drivable-area baseline for early Raspberry Pi integration tests."""

    def __init__(self, config: Optional[LaneSegConfig] = None) -> None:
        self.config = config or LaneSegConfig()
        _validate_config(self.config)

    def segment_mask(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> Optional[LaneSegMask]:
        source = validate_frame(frame)
        hsv = _to_hsv(source, preprocess_result.color_space if preprocess_result is not None else self.config.input_color_space)
        height, width = hsv.shape[:2]
        roi_top = int(round(height * self.config.roi_top_ratio))
        roi = hsv[roi_top:, :]
        if roi.size == 0:
            return None

        mask_roi = cv2.inRange(
            roi,
            np.array([0, 0, self.config.min_value], dtype=np.uint8),
            np.array([180, self.config.max_saturation, 255], dtype=np.uint8),
        )
        mask_roi = _morphology(mask_roi, self.config.morphology_kernel_size)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[roi_top:, :] = mask_roi

        drivable_ratio = float(cv2.countNonZero(mask_roi)) / float(mask_roi.size)
        if drivable_ratio < self.config.min_drivable_ratio:
            return None

        conf = _confidence(drivable_ratio, self.config.min_drivable_ratio)
        return LaneSegMask(
            mask_id=self.config.mask_id,
            mask=mask,
            conf=round(conf, 4),
            drivable_ratio=round(drivable_ratio, 4),
        )

    def segment(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> Optional[LaneSeg]:
        result = self.segment_mask(frame, preprocess_result)
        return None if result is None else result.to_schema()

    def predict(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> Optional[LaneSeg]:
        return self.segment(frame, preprocess_result)


def _to_hsv(frame: np.ndarray, color_space: str) -> np.ndarray:
    source_space = color_space.strip().lower()
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
        gray = frame if frame.ndim == 2 else frame[:, :, 0]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2HSV)
    if source_space == "bgr":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if source_space == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    if source_space == "bgra":
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2HSV)
    if source_space == "rgba":
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2HSV)
    raise ValueError("lane segmenter input color space must be bgr, rgb, bgra, rgba, or gray")


def _morphology(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)


def _confidence(drivable_ratio: float, min_drivable_ratio: float) -> float:
    if drivable_ratio <= 0.0:
        return 0.0
    normalized = drivable_ratio / max(min_drivable_ratio, 1e-6)
    return min(1.0, 0.30 + 0.18 * normalized)


def _validate_config(config: LaneSegConfig) -> None:
    if config.mask_id < 0:
        raise ValueError("mask_id must be non-negative")
    if not 0.0 <= config.roi_top_ratio < 1.0:
        raise ValueError("roi_top_ratio must be in [0.0, 1.0)")
    if not 0.0 < config.min_drivable_ratio <= 1.0:
        raise ValueError("min_drivable_ratio must be in (0.0, 1.0]")
    if not 0 <= config.min_value <= 255:
        raise ValueError("min_value must be in [0, 255]")
    if not 0 <= config.max_saturation <= 255:
        raise ValueError("max_saturation must be in [0, 255]")
    if config.morphology_kernel_size < 1 or config.morphology_kernel_size % 2 == 0:
        raise ValueError("morphology_kernel_size must be positive and odd")
