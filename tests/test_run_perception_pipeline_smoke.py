import importlib.util
import json
import sys
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from perception.runtime import Hazard, LaneSeg, ObjectBBox, TrafficLight


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "run_perception_pipeline_smoke.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_perception_pipeline_smoke", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_perception_pipeline_smoke_writes_valid_outputs():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_run_perception_pipeline_smoke" / uuid4().hex
    image_path = workspace / "sample.jpg"
    output_json = workspace / "smoke.json"
    output_report = workspace / "smoke.md"
    _write_image(image_path)

    result = script.run_smoke_test(
        images=[image_path],
        detector=_Detector([ObjectBBox("scooter", [1, 2, 20, 30], 0.88)]),
        traffic_light_provider=_TrafficLightProvider(TrafficLight("green", 0.91)),
        laneseg_provider=_LaneSegProvider(LaneSeg(1, 0.86)),
        hazard_provider=_HazardProvider(Hazard("curb", 0.82)),
    )
    script.write_outputs(result, output_json=output_json, output_report=output_report)

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["sample_count"] == 1
    assert payload["samples_with_objects"] == 1
    assert payload["samples_with_laneseg"] == 1
    assert payload["samples_with_traffic_light"] == 1
    assert payload["samples_with_hazard"] == 1
    output = payload["items"][0]["perception_output"]
    assert output["objects"] == [{"cls": "scooter", "bbox": [1.0, 2.0, 20.0, 30.0], "conf": 0.88}]
    assert output["traffic_light"] == {"state": "green", "conf": 0.91}
    assert output["laneseg"] == {"mask_id": 1, "conf": 0.86}
    assert output["hazard"] == {"type": "curb", "conf": 0.82}
    assert "Unified Perception Pipeline Smoke Test" in output_report.read_text(encoding="utf-8")


def test_run_perception_pipeline_smoke_cli_json(monkeypatch, capsys):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_run_perception_pipeline_smoke_cli" / uuid4().hex
    image_path = workspace / "sample.jpg"
    output_json = workspace / "smoke.json"
    output_report = workspace / "smoke.md"
    _write_image(image_path)
    monkeypatch.setattr(script, "build_default_object_detector", lambda device=None: _Detector([]))
    monkeypatch.setattr(script, "build_default_traffic_light_provider", lambda device=None: _TrafficLightProvider(TrafficLight("red", 0.77)))
    monkeypatch.setattr(script, "build_default_laneseg_provider", lambda device=None: _LaneSegProvider(LaneSeg(2, 0.73)))
    monkeypatch.setattr(script, "build_default_hazard_provider", lambda device=None: _HazardProvider(None))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_perception_pipeline_smoke.py",
            "--image",
            str(image_path),
            "--output-json",
            str(output_json),
            "--output-report",
            str(output_report),
            "--json",
        ],
    )

    code = script.main()
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["status"] == "ok"
    assert payload["sample_count"] == 1
    assert payload["samples_with_traffic_light"] == 1
    assert output_json.exists()
    assert output_report.exists()


def test_run_perception_pipeline_smoke_rejects_missing_image():
    script = _load_script()
    missing = REPO_ROOT / "cache" / "pytest" / "missing_smoke_image.jpg"

    try:
        script.run_smoke_test(images=[missing])
    except FileNotFoundError as exc:
        assert "sample image does not exist" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((64, 80, 3), dtype=np.uint8)
    frame[:, :] = [20, 40, 60]
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    encoded.tofile(str(path))


class _Detector:
    def __init__(self, objects):
        self.objects = list(objects)

    def detect(self, frame, preprocess_result):
        return self.objects


class _TrafficLightProvider:
    def __init__(self, traffic_light):
        self.traffic_light = traffic_light

    def detect(self, frame, preprocess_result):
        return self.traffic_light


class _LaneSegProvider:
    def __init__(self, laneseg):
        self.laneseg = laneseg

    def segment(self, frame, preprocess_result):
        return self.laneseg


class _HazardProvider:
    def __init__(self, hazard):
        self.hazard = hazard

    def detect(self, frame, preprocess_result):
        return self.hazard
