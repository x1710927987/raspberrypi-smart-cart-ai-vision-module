from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np

from perception.detection import EmptyObjectDetector, ObjectDetector
from perception.fusion import FusionConfig, PerceptionFusion
from perception.preprocessing import PreprocessConfig, PreprocessResult, preprocess_frame
from perception.runtime import PerceptionOutput


class FrameProvider(Protocol):
    def __call__(self, frame: np.ndarray, preprocess_result: PreprocessResult) -> Any:
        ...


@dataclass(frozen=True)
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)


class PerceptionPipeline:
    def __init__(
        self,
        *,
        detector: Optional[ObjectDetector] = None,
        laneseg_provider: Optional[Any] = None,
        traffic_light_provider: Optional[Any] = None,
        hazard_provider: Optional[Any] = None,
        config: Optional[PipelineConfig] = None,
        fusion: Optional[PerceptionFusion] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.detector = detector or EmptyObjectDetector()
        self.laneseg_provider = laneseg_provider
        self.traffic_light_provider = traffic_light_provider
        self.hazard_provider = hazard_provider
        self.fusion = fusion or PerceptionFusion(self.config.fusion)
        self.last_preprocess_result: Optional[PreprocessResult] = None

    def process_frame(self, frame: np.ndarray, *, timestamp: Optional[float] = None, base: Optional[PerceptionOutput] = None) -> PerceptionOutput:
        preprocess_result = preprocess_frame(frame, self.config.preprocess)
        self.last_preprocess_result = preprocess_result
        fusion_kwargs: dict[str, Any] = {
            "base": base,
            "timestamp": timestamp,
            "objects": self.detector.detect(preprocess_result.image, preprocess_result),
        }
        if self.laneseg_provider is not None:
            fusion_kwargs["laneseg"] = _run_provider(self.laneseg_provider, ("segment", "detect", "predict"), preprocess_result.image, preprocess_result, "laneseg_provider")
        if self.traffic_light_provider is not None:
            fusion_kwargs["traffic_light"] = _run_provider(self.traffic_light_provider, ("detect", "classify", "predict"), preprocess_result.image, preprocess_result, "traffic_light_provider")
        if self.hazard_provider is not None:
            fusion_kwargs["hazard"] = _run_provider(self.hazard_provider, ("detect", "classify", "predict"), preprocess_result.image, preprocess_result, "hazard_provider")
        return self.fusion.fuse(**fusion_kwargs)


def _run_provider(provider: Any, method_names: tuple[str, ...], frame: np.ndarray, preprocess_result: PreprocessResult, provider_name: str) -> Any:
    if callable(provider):
        return provider(frame, preprocess_result)
    for method_name in method_names:
        method = getattr(provider, method_name, None)
        if isinstance(method, Callable):
            return method(frame, preprocess_result)
    raise TypeError(f"{provider_name} must be callable or provide one of these methods: {', '.join(method_names)}")
