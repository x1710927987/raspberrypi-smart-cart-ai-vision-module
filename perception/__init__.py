from perception.runtime import Hazard, LaneSeg, ObjectBBox, PerceptionOutput, TrafficLight

__all__ = [
    "ColorTrafficLightDetector",
    "DEFAULT_SCENARIOS",
    "DetectionConfig",
    "DummyObjectDetector",
    "EmptyObjectDetector",
    "FrameProvider",
    "FusionConfig",
    "HAZARD_TYPES",
    "Hazard",
    "HazardCandidate",
    "HazardDetectionConfig",
    "LaneSeg",
    "LaneSegConfig",
    "LaneSegMask",
    "MockPerceptionRuntime",
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
    "fuse_perception",
    "list_mock_scenarios",
    "make_mock_perception",
    "map_model_class",
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
    if name in {"FrameProvider", "PerceptionPipeline", "PipelineConfig"}:
        return getattr(import_module("perception.camera_pipeline"), name)
    if name in {"ColorTrafficLightDetector", "TrafficLightDetectionConfig"}:
        return getattr(import_module("perception.traffic_light"), name)
    if name in {"LaneSegConfig", "LaneSegMask", "RuleBasedLaneSegmenter"}:
        return getattr(import_module("perception.laneseg"), name)
    if name in {"HazardCandidate", "HazardDetectionConfig", "RuleBasedHazardDetector"}:
        return getattr(import_module("perception.hazard"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
