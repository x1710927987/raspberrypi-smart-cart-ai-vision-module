from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from perception.preprocessing import PreprocessResult, validate_frame
from perception.runtime import TrafficLight


@dataclass(frozen=True)
class TrafficLightDetectionConfig:
    input_color_space: str = "bgr"
    min_area_pixels: int = 20
    min_area_ratio: float = 0.0002
    min_dominance_ratio: float = 1.15
    morphology_kernel_size: int = 3
    unknown_conf: float = 0.0


class ColorTrafficLightDetector:
    def __init__(self, config: Optional[TrafficLightDetectionConfig] = None) -> None:
        self.config = config or TrafficLightDetectionConfig()
        _validate_config(self.config)

    def detect(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> TrafficLight:
        source = validate_frame(frame)
        if source.ndim == 2 or (source.ndim == 3 and source.shape[2] == 1):
            return TrafficLight("unknown", self.config.unknown_conf)
        color_space = preprocess_result.color_space if preprocess_result is not None else self.config.input_color_space
        hsv = _to_hsv(source, color_space)
        masks = _build_color_masks(hsv)
        areas = {state: _mask_area(mask, self.config.morphology_kernel_size) for state, mask in masks.items()}
        image_area = float(source.shape[0] * source.shape[1])
        min_area = max(float(self.config.min_area_pixels), image_area * self.config.min_area_ratio)
        ranked = sorted(areas.items(), key=lambda item: item[1], reverse=True)
        best_state, best_area = ranked[0]
        second_area = ranked[1][1] if len(ranked) > 1 else 0
        if best_area < min_area:
            return TrafficLight("unknown", self.config.unknown_conf)
        conf = _confidence(best_area, image_area, min_area)
        if second_area > 0 and best_area / second_area < self.config.min_dominance_ratio:
            return TrafficLight("unknown", round(conf * 0.5, 4))
        return TrafficLight(best_state, round(conf, 4))

    def classify(self, frame: np.ndarray, preprocess_result: Optional[PreprocessResult] = None) -> TrafficLight:
        return self.detect(frame, preprocess_result)


def _to_hsv(frame: np.ndarray, color_space: str) -> np.ndarray:
    source_space = color_space.strip().lower()
    if source_space == "bgr":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if source_space == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    if source_space == "bgra":
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2HSV)
    if source_space == "rgba":
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2HSV)
    raise ValueError("traffic light detector input color space must be bgr, rgb, bgra, or rgba")


def _build_color_masks(hsv: np.ndarray) -> Dict[str, np.ndarray]:
    red_low = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    red_high = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    return {
        "red": cv2.bitwise_or(red_low, red_high),
        "yellow": cv2.inRange(hsv, np.array([18, 80, 80]), np.array([36, 255, 255])),
        "green": cv2.inRange(hsv, np.array([40, 60, 60]), np.array([90, 255, 255])),
    }


def _mask_area(mask: np.ndarray, kernel_size: int) -> int:
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return int(cv2.countNonZero(mask))


def _confidence(area: float, image_area: float, min_area: float) -> float:
    area_ratio = area / max(image_area, 1.0)
    threshold_ratio = min_area / max(image_area, 1.0)
    return min(1.0, 0.35 + (area_ratio / max(threshold_ratio * 12.0, 1e-6)))


def _validate_config(config: TrafficLightDetectionConfig) -> None:
    if config.min_area_pixels < 1:
        raise ValueError("min_area_pixels must be positive")
    if not 0.0 <= config.min_area_ratio <= 1.0:
        raise ValueError("min_area_ratio must be in [0.0, 1.0]")
    if config.min_dominance_ratio < 1.0:
        raise ValueError("min_dominance_ratio must be >= 1.0")
    if config.morphology_kernel_size < 1 or config.morphology_kernel_size % 2 == 0:
        raise ValueError("morphology_kernel_size must be positive and odd")
    if not 0.0 <= config.unknown_conf <= 1.0:
        raise ValueError("unknown_conf must be in [0.0, 1.0]")
