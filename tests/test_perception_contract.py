import pytest
from perception.runtime import PerceptionOutput

def test_perception_output_has_required_fields():
    """Ensure PerceptionOutput contains fields needed by control logic."""
    data = PerceptionOutput(
        has_obstacle=True,
        obstacle_distance=1.5,
        traffic_light="red",
        is_on_sidewalk=True
    )
    assert isinstance(data.has_obstacle, bool)
    assert isinstance(data.obstacle_distance, float)
    assert data.traffic_light in ["red", "green", "yellow", "none"]
    assert isinstance(data.is_on_sidewalk, bool)