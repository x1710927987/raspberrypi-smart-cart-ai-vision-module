from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from perception.preprocessing import PreprocessResult, validate_frame
from perception.runtime import Hazard


@dataclass(frozen=True)
class HazardDetectionConfig:
    input_color_space: str = "bgr"
    roi_top_ratio: float = 0.55
    hazard_type: str = "pothole"
    dark_value_threshold: int = 65
    min_area_pixels: int = 25
    min_area_ratio: float = 0.001
    morphology_kernel_size: int = 5
    unknown_conf: float = 0.0


@dataclass(frozen=True)
class HazardCandidate:
    type: str
    bbox: list[float]
    area: int
    conf: float

    def to_schema(self) -> Hazard:
        return Hazard(type=str(self.type), conf=float(self.conf))


class RuleBasedHazardDetector:
    """Dark-region road hazard baseline for pothole/step-like obstacle smoke tests."""

    def __init__(self, config: Optional[HazardDetectionConfig] = None) -> None:
        self.config = config or HazardDetectionConfig()
        _validate_config(self.config)

    def detect_candidate(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> Optional[HazardCandidate]:
        source = validate_frame(frame)
        hsv = _to_hsv(source, preprocess_result.color_space if preprocess_result is not None else self.config.input_color_space)
        height, width = hsv.shape[:2]
        roi_top = int(round(height * self.config.roi_top_ratio))
        roi = hsv[roi_top:, :]
        if roi.size == 0:
            return None

        value = roi[:, :, 2]
        mask_roi = cv2.inRange(value, 0, self.config.dark_value_threshold)
        mask_roi = _morphology(mask_roi, self.config.morphology_kernel_size)
        min_area = max(int(self.config.min_area_pixels), int(round(mask_roi.size * self.config.min_area_ratio)))
        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[HazardCandidate] = []
        for contour in contours:
            area = int(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [float(x), float(y + roi_top), float(x + w), float(y + h + roi_top)]
            conf = _confidence(area, mask_roi.size, min_area)
            candidates.append(HazardCandidate(self.config.hazard_type, bbox, area, round(conf, 4)))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item.conf, item.area), reverse=True)
        return candidates[0]

    def detect(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> Optional[Hazard]:
        candidate = self.detect_candidate(frame, preprocess_result)
        return None if candidate is None else candidate.to_schema()

    def classify(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> Optional[Hazard]:
        return self.detect(frame, preprocess_result)

    def predict(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> Optional[Hazard]:
        return self.detect(frame, preprocess_result)


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
    raise ValueError("hazard detector input color space must be bgr, rgb, bgra, rgba, or gray")


def _morphology(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def _confidence(area: int, roi_area: int, min_area: int) -> float:
    area_ratio = float(area) / max(float(roi_area), 1.0)
    threshold_ratio = float(min_area) / max(float(roi_area), 1.0)
    return min(1.0, 0.35 + area_ratio / max(threshold_ratio * 8.0, 1e-6))


def _validate_config(config: HazardDetectionConfig) -> None:
    if not 0.0 <= config.roi_top_ratio < 1.0:
        raise ValueError("roi_top_ratio must be in [0.0, 1.0)")
    if config.hazard_type not in {"pothole", "curb", "step_up", "step_down", "speed_bump", "water", "debris", "unknown"}:
        raise ValueError("hazard_type is not supported by schema.md")
    if not 0 <= config.dark_value_threshold <= 255:
        raise ValueError("dark_value_threshold must be in [0, 255]")
    if config.min_area_pixels < 1:
        raise ValueError("min_area_pixels must be positive")
    if not 0.0 <= config.min_area_ratio <= 1.0:
        raise ValueError("min_area_ratio must be in [0.0, 1.0]")
    if config.morphology_kernel_size < 1 or config.morphology_kernel_size % 2 == 0:
        raise ValueError("morphology_kernel_size must be positive and odd")
    if not 0.0 <= config.unknown_conf <= 1.0:
        raise ValueError("unknown_conf must be in [0.0, 1.0]")
