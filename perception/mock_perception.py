from __future__ import annotations

import argparse
import time
from typing import Iterable, List, Optional, Sequence

from perception.runtime import Hazard, LaneSeg, ObjectBBox, PerceptionOutput, TrafficLight


OBJECT_CLASSES = {
    "pedestrian",
    "obstacle",
    "roadblock",
    "bicycle",
    "car",
    "animal",
    "stroller",
    "wheelchair",
    "bollard",
    "scooter",
    "unknown",
}
TRAFFIC_LIGHT_STATES = {"red", "yellow", "green", "off", "flashing", "unknown"}
HAZARD_TYPES = {"pothole", "curb", "step_up", "step_down", "speed_bump", "water", "debris", "unknown"}
DEFAULT_SCENARIOS = (
    "clear_path",
    "pedestrian_ahead",
    "obstacle_ahead",
    "red_light",
    "road_hazard",
    "mixed_risk",
)


def list_mock_scenarios() -> List[str]:
    return list(DEFAULT_SCENARIOS)


def make_mock_perception(
    scenario: str = "clear_path",
    *,
    timestamp: Optional[float] = None,
    image_width: int = 640,
    image_height: int = 480,
) -> PerceptionOutput:
    ts = time.time() if timestamp is None else float(timestamp)
    scenario_key = scenario.strip().lower()

    if scenario_key == "clear_path":
        output = PerceptionOutput(ts, LaneSeg(1, 0.94), [], TrafficLight("green", 0.86), None)
    elif scenario_key == "pedestrian_ahead":
        output = PerceptionOutput(
            ts,
            LaneSeg(1, 0.90),
            [ObjectBBox("pedestrian", _scaled_bbox([270, 150, 365, 420], image_width, image_height), 0.88)],
            TrafficLight("green", 0.82),
            None,
        )
    elif scenario_key == "obstacle_ahead":
        output = PerceptionOutput(
            ts,
            LaneSeg(1, 0.89),
            [ObjectBBox("obstacle", _scaled_bbox([285, 260, 390, 350], image_width, image_height), 0.80)],
            TrafficLight("green", 0.80),
            None,
        )
    elif scenario_key == "red_light":
        output = PerceptionOutput(ts, LaneSeg(1, 0.91), [], TrafficLight("red", 0.90), None)
    elif scenario_key == "road_hazard":
        output = PerceptionOutput(ts, LaneSeg(1, 0.86), [], TrafficLight("green", 0.76), Hazard("pothole", 0.83))
    elif scenario_key == "mixed_risk":
        output = PerceptionOutput(
            ts,
            LaneSeg(1, 0.84),
            [
                ObjectBBox("pedestrian", _scaled_bbox([105, 160, 205, 430], image_width, image_height), 0.85),
                ObjectBBox("bollard", _scaled_bbox([385, 235, 430, 370], image_width, image_height), 0.78),
            ],
            TrafficLight("yellow", 0.79),
            Hazard("debris", 0.72),
        )
    else:
        raise ValueError(f"unknown mock perception scenario: {scenario!r}. Valid scenarios: {', '.join(DEFAULT_SCENARIOS)}")

    validate_perception_output(output)
    return output


class MockPerceptionRuntime:
    def __init__(self, scenarios: Sequence[str] = DEFAULT_SCENARIOS, *, image_width: int = 640, image_height: int = 480):
        if not scenarios:
            raise ValueError("scenarios must not be empty")
        self._scenarios = list(scenarios)
        self._image_width = int(image_width)
        self._image_height = int(image_height)
        self._index = 0

    def next_frame(self) -> PerceptionOutput:
        scenario = self._scenarios[self._index % len(self._scenarios)]
        self._index += 1
        return make_mock_perception(scenario, image_width=self._image_width, image_height=self._image_height)


def validate_perception_output(output: PerceptionOutput) -> None:
    if output.timestamp < 0.001:
        raise ValueError("timestamp must be a UNIX epoch timestamp in seconds")
    if output.laneseg is not None:
        _validate_conf(output.laneseg.conf, "laneseg.conf")
        if output.laneseg.mask_id < 0:
            raise ValueError("laneseg.mask_id must be non-negative")
        if output.laneseg.bbox is not None:
            _validate_bbox(output.laneseg.bbox, "laneseg.bbox")
    for index, obj in enumerate(output.objects):
        if obj.cls not in OBJECT_CLASSES:
            raise ValueError(f"objects[{index}].cls is not supported: {obj.cls!r}")
        _validate_bbox(obj.bbox, f"objects[{index}].bbox")
        _validate_conf(obj.conf, f"objects[{index}].conf")
    if output.traffic_light is not None:
        if output.traffic_light.state not in TRAFFIC_LIGHT_STATES:
            raise ValueError(f"unsupported traffic light state: {output.traffic_light.state!r}")
        _validate_conf(output.traffic_light.conf, "traffic_light.conf")
        if output.traffic_light.bbox is not None:
            _validate_bbox(output.traffic_light.bbox, "traffic_light.bbox")
    if output.hazard is not None:
        if output.hazard.type not in HAZARD_TYPES:
            raise ValueError(f"unsupported hazard type: {output.hazard.type!r}")
        _validate_conf(output.hazard.conf, "hazard.conf")
        if output.hazard.bbox is not None:
            _validate_bbox(output.hazard.bbox, "hazard.bbox")


def _scaled_bbox(bbox: Iterable[float], image_width: int, image_height: int) -> List[float]:
    x_scale = float(image_width) / 640.0
    y_scale = float(image_height) / 480.0
    x1, y1, x2, y2 = bbox
    return [round(x1 * x_scale, 1), round(y1 * y_scale, 1), round(x2 * x_scale, 1), round(y2 * y_scale, 1)]


def _validate_conf(value: float, field_name: str) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field_name} must be in [0.0, 1.0]")


def _validate_bbox(bbox: Sequence[float], field_name: str) -> None:
    if len(bbox) != 4:
        raise ValueError(f"{field_name} must contain exactly 4 values")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{field_name} must satisfy x2 > x1 and y2 > y1")
    if min(x1, y1, x2, y2) < 0.0:
        raise ValueError(f"{field_name} must use non-negative pixel coordinates")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Emit one mock PerceptionOutput JSON frame.")
    parser.add_argument("scenario", nargs="?", default="clear_path", choices=DEFAULT_SCENARIOS)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    print(make_mock_perception(args.scenario, image_width=args.width, image_height=args.height).to_json())


if __name__ == "__main__":
    _main()
