from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

from perception.detection import EmptyObjectDetector, ObjectDetector
from perception.fusion import FusionConfig, PerceptionFusion
from perception.preprocessing import PreprocessConfig, PreprocessResult, preprocess_frame
from perception.runtime import PerceptionOutput


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECTS_MANIFEST = REPO_ROOT / "models" / "training" / "smartcart_objects_yolov8n_combined_v3_pt_v1.manifest.json"
DEFAULT_TRAFFIC_LIGHT_MANIFEST = REPO_ROOT / "models" / "training" / "smartcart_traffic_light_yolov8n_combined_v2_pt_v1.manifest.json"
DEFAULT_LANESEG_MANIFEST = REPO_ROOT / "models" / "training" / "smartcart_laneseg_yolov8n_seg_roboflow_pt_v1.manifest.json"
DEFAULT_HAZARD_MANIFEST = REPO_ROOT / "models" / "training" / "smartcart_hazard_yolov8n_roboflow_pt_v1.manifest.json"


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

    @classmethod
    def with_default_objects(
        cls,
        *,
        manifest_path: str | Path = DEFAULT_OBJECTS_MANIFEST,
        backend: Any | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> "PerceptionPipeline":
        return cls(detector=build_default_object_detector(manifest_path=manifest_path, backend=backend, device=device), **kwargs)

    @classmethod
    def with_default_traffic_light(
        cls,
        *,
        manifest_path: str | Path = DEFAULT_TRAFFIC_LIGHT_MANIFEST,
        backend: Any | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> "PerceptionPipeline":
        return cls(traffic_light_provider=build_default_traffic_light_provider(manifest_path=manifest_path, backend=backend, device=device), **kwargs)

    @classmethod
    def with_default_laneseg(
        cls,
        *,
        manifest_path: str | Path = DEFAULT_LANESEG_MANIFEST,
        backend: Any | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> "PerceptionPipeline":
        return cls(laneseg_provider=build_default_laneseg_provider(manifest_path=manifest_path, backend=backend, device=device), **kwargs)

    @classmethod
    def with_default_hazard(
        cls,
        *,
        manifest_path: str | Path = DEFAULT_HAZARD_MANIFEST,
        backend: Any | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> "PerceptionPipeline":
        return cls(hazard_provider=build_default_hazard_provider(manifest_path=manifest_path, backend=backend, device=device), **kwargs)

    @classmethod
    def with_default_models(
        cls,
        *,
        objects_manifest_path: str | Path = DEFAULT_OBJECTS_MANIFEST,
        traffic_light_manifest_path: str | Path = DEFAULT_TRAFFIC_LIGHT_MANIFEST,
        laneseg_manifest_path: str | Path = DEFAULT_LANESEG_MANIFEST,
        hazard_manifest_path: str | Path = DEFAULT_HAZARD_MANIFEST,
        backend: Any | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> "PerceptionPipeline":
        return cls(
            detector=build_default_object_detector(manifest_path=objects_manifest_path, backend=backend, device=device),
            traffic_light_provider=build_default_traffic_light_provider(manifest_path=traffic_light_manifest_path, backend=backend, device=device),
            laneseg_provider=build_default_laneseg_provider(manifest_path=laneseg_manifest_path, backend=backend, device=device),
            hazard_provider=build_default_hazard_provider(manifest_path=hazard_manifest_path, backend=backend, device=device),
            **kwargs,
        )

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


def build_default_object_detector(
    *,
    manifest_path: str | Path = DEFAULT_OBJECTS_MANIFEST,
    backend: Any | None = None,
    device: str | None = None,
) -> Any:
    from perception.model_inference import ManifestObjectDetector, UltralyticsBackend, load_model_manifest

    manifest = load_model_manifest(manifest_path, require_artifact=True)
    return ManifestObjectDetector(manifest, backend or UltralyticsBackend(device=device))


def build_default_traffic_light_provider(
    *,
    manifest_path: str | Path = DEFAULT_TRAFFIC_LIGHT_MANIFEST,
    backend: Any | None = None,
    device: str | None = None,
) -> Any:
    from perception.model_inference import ManifestTrafficLightClassifier, UltralyticsBackend, load_model_manifest

    manifest = load_model_manifest(manifest_path, require_artifact=True)
    return ManifestTrafficLightClassifier(manifest, backend or UltralyticsBackend(device=device))


def build_default_laneseg_provider(
    *,
    manifest_path: str | Path = DEFAULT_LANESEG_MANIFEST,
    backend: Any | None = None,
    device: str | None = None,
) -> Any:
    from perception.model_inference import ManifestLaneSegmenter, UltralyticsBackend, load_model_manifest

    manifest = load_model_manifest(manifest_path, require_artifact=True)
    return ManifestLaneSegmenter(manifest, backend or UltralyticsBackend(device=device))


def build_default_hazard_provider(
    *,
    manifest_path: str | Path = DEFAULT_HAZARD_MANIFEST,
    backend: Any | None = None,
    device: str | None = None,
) -> Any:
    from perception.model_inference import ManifestHazardDetector, UltralyticsBackend, load_model_manifest

    manifest = load_model_manifest(manifest_path, require_artifact=True)
    return ManifestHazardDetector(manifest, backend or UltralyticsBackend(device=device))
