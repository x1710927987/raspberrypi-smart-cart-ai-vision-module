import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

from perception.runtime import Hazard, LaneSeg, ObjectBBox, PerceptionOutput, TrafficLight


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "run_perception_live_view.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_perception_live_view", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_draw_perception_overlay_draws_objects_and_status_panel():
    script = _load_script()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    output = _sample_output()

    annotated = script.draw_perception_overlay(frame, output, elapsed_ms=42.5, fps=23.5)

    assert annotated.shape == frame.shape
    assert int(annotated.sum()) > int(frame.sum())
    assert not np.array_equal(annotated[20, 20], frame[20, 20])


def test_run_live_view_without_window_processes_limited_frames(monkeypatch):
    script = _load_script()
    frames = [np.zeros((80, 100, 3), dtype=np.uint8) for _ in range(3)]
    fake_capture = _FakeCapture(frames)
    monkeypatch.setattr(script.cv2, "VideoCapture", lambda camera_index: fake_capture)

    count = script.run_live_view(
        camera_index=0,
        show_window=False,
        max_frames=2,
        fps_limit=0,
        pipeline=_FakePipeline(),
    )

    assert count == 2
    assert fake_capture.released is True


def test_clamp_bbox_rejects_invalid_bbox_length():
    script = _load_script()

    try:
        script._clamp_bbox([1.0, 2.0, 3.0], 100, 100)
    except ValueError as exc:
        assert "four coordinates" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def _sample_output() -> PerceptionOutput:
    return PerceptionOutput(
        timestamp=time.time(),
        laneseg=LaneSeg(mask_id=1, conf=0.88),
        objects=[ObjectBBox(cls="pedestrian", bbox=[12.0, 18.0, 80.0, 100.0], conf=0.91)],
        traffic_light=TrafficLight(state="green", conf=0.93),
        hazard=Hazard(type="curb", conf=0.72),
    )


class _FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


class _FakePipeline:
    def process_frame(self, frame, *, timestamp=None, base=None):
        return _sample_output()
