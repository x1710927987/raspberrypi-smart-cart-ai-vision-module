import cv2
import numpy as np
import pytest

from perception import make_mock_perception, validate_perception_output
from perception.camera_pipeline import (
    DEFAULT_HAZARD_MANIFEST,
    DEFAULT_LANESEG_MANIFEST,
    DEFAULT_OBJECTS_MANIFEST,
    DEFAULT_TRAFFIC_LIGHT_MANIFEST,
    PerceptionPipeline,
    PipelineConfig,
    build_default_object_detector,
    build_default_hazard_provider,
    build_default_laneseg_provider,
    build_default_traffic_light_provider,
)
from perception.detection import DetectionConfig, DummyObjectDetector, ModelDetection, postprocess_detections
from perception.fusion import FusionConfig, PerceptionFusion, fuse_perception
from perception.hazard import HazardDetectionConfig, RuleBasedHazardDetector
from perception.laneseg import LaneSegConfig, RuleBasedLaneSegmenter
from perception.model_inference import FixedPredictionBackend
from perception.preprocessing import PreprocessConfig, preprocess_frame
from perception.runtime import Hazard, LaneSeg, ObjectBBox, PerceptionOutput, TrafficLight
from perception.traffic_light import ColorTrafficLightDetector, TrafficLightDetectionConfig


def test_mock_perception_json_round_trip_and_validation():
    output = make_mock_perception("mixed_risk", timestamp=1720000123.456)
    decoded = PerceptionOutput.from_json(output.to_json())
    assert decoded.to_dict() == output.to_dict()
    validate_perception_output(decoded)


def test_mock_perception_rejects_unknown_scene():
    with pytest.raises(ValueError, match="unknown mock perception scenario"):
        make_mock_perception("missing_scene")


def test_preprocess_frame_resizes_converts_and_maps_bbox():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = [10, 20, 30]
    result = preprocess_frame(frame, PreprocessConfig(target_size=(320, 240), color_space="rgb"))
    assert result.image.shape == (240, 320, 3)
    assert result.image[0, 0].tolist() == [30, 20, 10]
    assert result.original_size == (640, 480)
    assert result.bbox_to_processed([100, 120, 300, 360]) == [50.0, 60.0, 150.0, 180.0]
    assert result.bbox_to_original([50, 60, 150, 180]) == [100.0, 120.0, 300.0, 360.0]


def test_preprocess_frame_normalizes_and_rejects_bad_inputs():
    result = preprocess_frame(np.full((4, 4, 3), 255, dtype=np.uint8), PreprocessConfig(target_size=(2, 2), normalize=True))
    assert result.image.dtype == np.float32
    assert float(result.image.max()) == 1.0
    with pytest.raises(TypeError, match="numpy.ndarray"):
        preprocess_frame("not an image")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="color_space"):
        preprocess_frame(np.zeros((10, 10, 3), dtype=np.uint8), PreprocessConfig(color_space="hsv"))


def test_detection_postprocess_maps_labels_filters_and_unknowns():
    detections = [
        ModelDetection("person", [10, 20, 110, 220], 0.91),
        ModelDetection("traffic cone", [200, 210, 250, 300], 0.72),
        ModelDetection("roadblock", [300, 210, 360, 310], 0.82),
        ModelDetection("person", [1, 2, 3, 4], 0.10),
        ModelDetection("banana", [20, 20, 40, 40], 0.99),
    ]
    objects = postprocess_detections(detections, config=DetectionConfig(conf_threshold=0.35))
    assert objects == [
        ObjectBBox("pedestrian", [10.0, 20.0, 110.0, 220.0], 0.91),
        ObjectBBox("obstacle", [200.0, 210.0, 250.0, 300.0], 0.72),
        ObjectBBox("roadblock", [300.0, 210.0, 360.0, 310.0], 0.82),
    ]
    unknown = postprocess_detections([detections[-1]], config=DetectionConfig(keep_unknown=True))
    assert unknown == [ObjectBBox("unknown", [20.0, 20.0, 40.0, 40.0], 0.99)]


def test_detection_maps_processed_bbox_to_original_coordinates():
    preprocess_result = preprocess_frame(np.zeros((480, 640, 3), dtype=np.uint8), PreprocessConfig(target_size=(320, 240)))
    objects = postprocess_detections(
        [ModelDetection("person", [50, 60, 150, 180], 0.9)],
        preprocess_result=preprocess_result,
        bbox_space="processed",
    )
    assert objects == [ObjectBBox("pedestrian", [100.0, 120.0, 300.0, 360.0], 0.9)]


def test_fusion_keeps_base_overrides_and_clears_fields():
    base = make_mock_perception("mixed_risk", timestamp=1720000123.456)
    assert fuse_perception(base=base).to_dict() == base.to_dict()
    fused = fuse_perception(
        base=base,
        objects=[ObjectBBox("pedestrian", [10, 20, 110, 220], 0.91)],
        laneseg={"mask_id": 2, "conf": 0.81},
        traffic_light={"state": "red", "conf": 0.95},
        hazard=Hazard("pothole", 0.83),
    )
    assert fused.laneseg == LaneSeg(2, 0.81)
    assert fused.objects == [ObjectBBox("pedestrian", [10.0, 20.0, 110.0, 220.0], 0.91)]
    assert fused.traffic_light == TrafficLight("red", 0.95)
    assert fused.hazard == Hazard("pothole", 0.83)
    cleared = fuse_perception(base=base, objects=None, laneseg=None, traffic_light=None, hazard=None)
    assert cleared.objects == []
    assert cleared.laneseg is None
    assert cleared.traffic_light is None
    assert cleared.hazard is None


def test_fusion_filters_by_confidence_and_limits_objects():
    fusion = PerceptionFusion(FusionConfig(min_object_conf=0.75, min_laneseg_conf=0.9, max_objects=1))
    fused = fusion.fuse(
        timestamp=1720000123.456,
        objects=[
            ObjectBBox("obstacle", [1, 2, 3, 4], 0.76),
            ObjectBBox("pedestrian", [5, 6, 7, 8], 0.9),
        ],
        laneseg=LaneSeg(1, 0.5),
    )
    assert fused.objects == [ObjectBBox("pedestrian", [5.0, 6.0, 7.0, 8.0], 0.9)]
    assert fused.laneseg is None


def test_pipeline_processes_frame_and_provider_outputs():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def laneseg_provider(processed_frame, preprocess_result):
        assert processed_frame.shape == (240, 320, 3)
        return {"mask_id": 4, "conf": 0.87}

    pipeline = PerceptionPipeline(
        detector=DummyObjectDetector([ModelDetection("traffic cone", [10, 20, 60, 90], 0.78)]),
        laneseg_provider=laneseg_provider,
        traffic_light_provider=lambda frame, result: TrafficLight("red", 0.93),
        hazard_provider=lambda frame, result: Hazard("pothole", 0.82),
        config=PipelineConfig(preprocess=PreprocessConfig(target_size=(320, 240))),
    )
    output = pipeline.process_frame(frame, timestamp=1720000123.456)
    assert output.laneseg == LaneSeg(4, 0.87)
    assert output.objects == [ObjectBBox("obstacle", [10.0, 20.0, 60.0, 90.0], 0.78)]
    assert output.traffic_light == TrafficLight("red", 0.93)
    assert output.hazard == Hazard("pothole", 0.82)


def test_pipeline_supports_provider_objects_and_rejects_bad_provider():
    class TrafficProvider:
        def classify(self, frame, preprocess_result):
            return {"state": "green", "conf": 0.84}

    output = PerceptionPipeline(traffic_light_provider=TrafficProvider()).process_frame(
        np.zeros((32, 32, 3), dtype=np.uint8),
        timestamp=1720000123.456,
    )
    assert output.traffic_light == TrafficLight("green", 0.84)

    class BadProvider:
        pass

    with pytest.raises(TypeError, match="laneseg_provider"):
        PerceptionPipeline(laneseg_provider=BadProvider()).process_frame(np.zeros((32, 32, 3), dtype=np.uint8))


def test_pipeline_default_object_detector_uses_v3_manifest():
    assert DEFAULT_OBJECTS_MANIFEST.name == "smartcart_objects_yolov8n_combined_v3_pt_v1.manifest.json"
    detector = build_default_object_detector(
        backend=FixedPredictionBackend([{"label": "scooter", "bbox": [1, 2, 20, 30], "conf": 0.88}])
    )
    output = PerceptionPipeline(detector=detector).process_frame(np.zeros((32, 32, 3), dtype=np.uint8))
    assert output.objects == [ObjectBBox("scooter", [1.0, 2.0, 20.0, 30.0], 0.88)]


def test_pipeline_default_traffic_light_provider_uses_v2_manifest():
    assert DEFAULT_TRAFFIC_LIGHT_MANIFEST.name == "smartcart_traffic_light_yolov8n_combined_v2_pt_v1.manifest.json"
    provider = build_default_traffic_light_provider(backend=FixedPredictionBackend([{"label": "yellow", "conf": 0.88}]))
    output = PerceptionPipeline(traffic_light_provider=provider).process_frame(np.zeros((32, 32, 3), dtype=np.uint8))
    assert output.traffic_light == TrafficLight("yellow", 0.88)


def test_pipeline_default_laneseg_provider_uses_registered_manifest():
    assert DEFAULT_LANESEG_MANIFEST.name == "smartcart_laneseg_yolov8n_seg_roboflow_pt_v1.manifest.json"
    provider = build_default_laneseg_provider(backend=FixedPredictionBackend({"mask_id": 1, "conf": 0.88}))
    output = PerceptionPipeline(laneseg_provider=provider).process_frame(np.zeros((32, 32, 3), dtype=np.uint8))
    assert output.laneseg == LaneSeg(1, 0.88)


def test_pipeline_default_hazard_provider_uses_registered_manifest():
    assert DEFAULT_HAZARD_MANIFEST.name == "smartcart_hazard_yolov8n_roboflow_pt_v1.manifest.json"
    provider = build_default_hazard_provider(backend=FixedPredictionBackend({"label": "curb", "conf": 0.82}))
    output = PerceptionPipeline(hazard_provider=provider).process_frame(np.zeros((32, 32, 3), dtype=np.uint8))
    assert output.hazard == Hazard("curb", 0.82)


def test_rule_based_lane_segmenter_detects_synthetic_sidewalk_roi():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[70:118, 20:140] = (120, 120, 120)
    segmenter = RuleBasedLaneSegmenter(LaneSegConfig(mask_id=7, roi_top_ratio=0.5))
    mask = segmenter.segment_mask(frame)
    assert mask is not None
    assert mask.mask_id == 7
    assert mask.mask.shape == (120, 160)
    assert mask.drivable_ratio > 0.5
    assert mask.to_schema() == LaneSeg(7, mask.conf)
    assert segmenter.segment(frame) == LaneSeg(7, mask.conf)


def test_rule_based_lane_segmenter_rejects_empty_and_bad_config():
    segmenter = RuleBasedLaneSegmenter(LaneSegConfig(min_drivable_ratio=0.5))
    assert segmenter.segment(np.zeros((80, 80, 3), dtype=np.uint8)) is None
    with pytest.raises(ValueError, match="roi_top_ratio"):
        RuleBasedLaneSegmenter(LaneSegConfig(roi_top_ratio=1.0))


def test_rule_based_hazard_detector_detects_synthetic_pothole():
    frame = np.full((120, 160, 3), 150, dtype=np.uint8)
    cv2.ellipse(frame, (80, 92), (24, 10), 0, 0, 360, (18, 18, 18), -1)
    detector = RuleBasedHazardDetector(HazardDetectionConfig(roi_top_ratio=0.5, min_area_pixels=20))
    candidate = detector.detect_candidate(frame)
    assert candidate is not None
    assert candidate.type == "pothole"
    assert candidate.area >= 20
    assert candidate.conf > 0.4
    assert candidate.bbox[1] >= 60
    assert candidate.to_schema() == Hazard("pothole", candidate.conf)
    assert detector.detect(frame) == Hazard("pothole", candidate.conf)


def test_rule_based_hazard_detector_rejects_clean_frame_and_bad_config():
    detector = RuleBasedHazardDetector(HazardDetectionConfig(roi_top_ratio=0.5))
    assert detector.detect(np.full((80, 80, 3), 150, dtype=np.uint8)) is None
    assert RuleBasedHazardDetector(HazardDetectionConfig(hazard_type="curb")).config.hazard_type == "curb"
    with pytest.raises(ValueError, match="hazard_type"):
        RuleBasedHazardDetector(HazardDetectionConfig(hazard_type="crack"))


def test_pipeline_accepts_rule_based_laneseg_and_hazard_detectors():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[70:118, 20:140] = (120, 120, 120)
    cv2.ellipse(frame, (80, 92), (18, 8), 0, 0, 360, (15, 15, 15), -1)
    pipeline = PerceptionPipeline(
        laneseg_provider=RuleBasedLaneSegmenter(LaneSegConfig(mask_id=3, roi_top_ratio=0.5)),
        hazard_provider=RuleBasedHazardDetector(HazardDetectionConfig(roi_top_ratio=0.5, min_area_pixels=15)),
        config=PipelineConfig(preprocess=PreprocessConfig(target_size=(160, 120))),
    )
    output = pipeline.process_frame(frame, timestamp=1720000123.456)
    assert output.laneseg == LaneSeg(3, output.laneseg.conf)
    assert output.hazard == Hazard("pothole", output.hazard.conf)


@pytest.mark.parametrize(
    ("state", "bgr"),
    [("red", (0, 0, 255)), ("yellow", (0, 255, 255)), ("green", (0, 255, 0))],
)
def test_color_traffic_light_detector_classifies_synthetic_lights(state, bgr):
    result = ColorTrafficLightDetector().detect(_traffic_light_frame(bgr))
    assert result.state == state
    assert result.conf > 0.5


def test_color_traffic_light_detector_negative_rgb_and_config():
    assert ColorTrafficLightDetector().detect(np.zeros((100, 100, 3), dtype=np.uint8)) == TrafficLight("unknown", 0.0)
    bgr_frame = _traffic_light_frame((0, 0, 255))
    preprocess_result = preprocess_frame(bgr_frame, PreprocessConfig(target_size=(100, 100), color_space="rgb"))
    assert ColorTrafficLightDetector().detect(preprocess_result.image, preprocess_result).state == "red"
    with pytest.raises(ValueError, match="min_area_pixels"):
        ColorTrafficLightDetector(TrafficLightDetectionConfig(min_area_pixels=0))


def _traffic_light_frame(bgr, width=100, height=100):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(frame, (width // 2, height // 2), 12, bgr, -1)
    return frame
