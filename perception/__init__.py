from perception.runtime import Hazard, LaneSeg, ObjectBBox, PerceptionOutput, TrafficLight

__all__ = [
    "ColorTrafficLightDetector",
    "DEFAULT_SCENARIOS",
    "DetectionConfig",
    "DummyObjectDetector",
    "EmptyObjectDetector",
    "FrameProvider",
    "DEFAULT_OBJECTS_MANIFEST",
    "DEFAULT_LANESEG_MANIFEST",
    "DEFAULT_TRAFFIC_LIGHT_MANIFEST",
    "FusionConfig",
    "HAZARD_TYPES",
    "FixedPredictionBackend",
    "Hazard",
    "HazardCandidate",
    "HazardDetectionConfig",
    "LaneSeg",
    "LaneSegConfig",
    "LaneSegMask",
    "ManifestHazardDetector",
    "ManifestLaneSegmenter",
    "ManifestObjectDetector",
    "ManifestTrafficLightClassifier",
    "MockPerceptionRuntime",
    "ModelManifest",
    "ModelDetection",
    "OBJECT_CLASSES",
    "ObjectBBox",
    "PerceptionFusion",
    "PerceptionOutput",
    "PerceptionPipeline",
    "PipelineConfig",
    "PreprocessConfig",
    "PreprocessResult",
    "RuleBasedHazardDetector",
    "RuleBasedLaneSegmenter",
    "TRAFFIC_LIGHT_STATES",
    "TrafficLight",
    "TrafficLightDetectionConfig",
    "build_default_laneseg_provider",
    "build_default_object_detector",
    "build_default_traffic_light_provider",
    "build_provider_from_manifest",
    "fuse_perception",
    "list_mock_scenarios",
    "make_mock_perception",
    "map_model_class",
    "load_model_manifest",
    "postprocess_detections",
    "preprocess_frame",
    "validate_perception_output",
]


def __getattr__(name: str):
    from importlib import import_module

    if name in {
        "DEFAULT_SCENARIOS",
        "HAZARD_TYPES",
        "OBJECT_CLASSES",
        "TRAFFIC_LIGHT_STATES",
        "MockPerceptionRuntime",
        "list_mock_scenarios",
        "make_mock_perception",
        "validate_perception_output",
    }:
        return getattr(import_module("perception.mock_perception"), name)
    if name in {"PreprocessConfig", "PreprocessResult", "preprocess_frame"}:
        return getattr(import_module("perception.preprocessing"), name)
    if name in {
        "DetectionConfig",
        "DummyObjectDetector",
        "EmptyObjectDetector",
        "ModelDetection",
        "map_model_class",
        "postprocess_detections",
    }:
        return getattr(import_module("perception.detection"), name)
    if name in {"FusionConfig", "PerceptionFusion", "fuse_perception"}:
        return getattr(import_module("perception.fusion"), name)
    if name in {
        "DEFAULT_LANESEG_MANIFEST",
        "DEFAULT_OBJECTS_MANIFEST",
        "DEFAULT_TRAFFIC_LIGHT_MANIFEST",
        "FrameProvider",
        "PerceptionPipeline",
        "PipelineConfig",
        "build_default_laneseg_provider",
        "build_default_object_detector",
        "build_default_traffic_light_provider",
    }:
        return getattr(import_module("perception.camera_pipeline"), name)
    if name in {"ColorTrafficLightDetector", "TrafficLightDetectionConfig"}:
        return getattr(import_module("perception.traffic_light"), name)
    if name in {"LaneSegConfig", "LaneSegMask", "RuleBasedLaneSegmenter"}:
        return getattr(import_module("perception.laneseg"), name)
    if name in {"HazardCandidate", "HazardDetectionConfig", "RuleBasedHazardDetector"}:
        return getattr(import_module("perception.hazard"), name)
    if name in {
        "FixedPredictionBackend",
        "ManifestHazardDetector",
        "ManifestLaneSegmenter",
        "ManifestObjectDetector",
        "ManifestTrafficLightClassifier",
        "ModelManifest",
        "build_provider_from_manifest",
        "load_model_manifest",
    }:
        return getattr(import_module("perception.model_inference"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
