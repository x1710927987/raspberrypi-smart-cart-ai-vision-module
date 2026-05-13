import time

from perception.mock_perception import validate_perception_output
from perception.runtime import Hazard, LaneSeg, ObjectBBox, PerceptionOutput, TrafficLight


def test_perception_output_has_control_contract_fields():
    output = PerceptionOutput(
        timestamp=time.time(),
        laneseg=LaneSeg(mask_id=1, conf=0.91),
        objects=[ObjectBBox(cls="pedestrian", bbox=[10.0, 20.0, 100.0, 220.0], conf=0.87)],
        traffic_light=TrafficLight(state="red", conf=0.96),
        hazard=Hazard(type="curb", conf=0.74),
    )

    validate_perception_output(output)
    payload = output.to_dict()

    assert set(payload) == {"timestamp", "laneseg", "objects", "traffic_light", "hazard"}
    assert payload["objects"][0]["cls"] == "pedestrian"
    assert payload["traffic_light"]["state"] == "red"
    assert payload["laneseg"]["mask_id"] == 1
    assert payload["hazard"]["type"] == "curb"


def test_perception_output_round_trips_through_json():
    output = PerceptionOutput(
        timestamp=123.456,
        laneseg=None,
        objects=[ObjectBBox(cls="roadblock", bbox=[0.0, 1.0, 2.0, 3.0], conf=0.8)],
        traffic_light=None,
        hazard=None,
    )

    decoded = PerceptionOutput.from_json(output.to_json())

    assert decoded == output
    validate_perception_output(decoded)
