import sys
import pathlib
import time

import numpy as np


# Ensure project root is importable and takes precedence over stdlib modules
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from perception.runtime import ControlCommand
from io_camera.camera import CameraConfig, OpenCVCameraSource, PiCamera2Source, create_camera_source
from io_camera.protocol import (
    encode_status,
    decode_status,
    verify,
    encode_command,
    ProtocolError,
)


def test_encode_decode_status_roundtrip():
    ts = 1719999999.123
    frame = encode_status(timestamp=ts, voltage=12.34, temp=36.5, err=0)
    assert frame.endswith("\n")
    assert verify(frame)
    parsed = decode_status(frame)
    assert abs(parsed["timestamp"] - ts) < 1e-3
    assert abs(parsed["voltage"] - 12.34) < 1e-2
    assert abs(parsed["temp"] - 36.5) < 1e-1
    assert parsed["err"] == 0


def test_decode_status_crc_mismatch_raises():
    # Create a valid frame, then tamper one character
    frame = encode_status(timestamp=1.0, voltage=10.00, temp=30.0, err=2)
    assert verify(frame)
    tampered = frame.replace("10.00", "10.01")
    assert not verify(tampered)
    try:
        decode_status(tampered)
        assert False, "Expected ProtocolError"
    except ProtocolError:
        pass


def test_encode_command_format():
    cmd = ControlCommand(
        mode="auto",
        v=0.5,
        steer=5.0,
        brake=False,
        reason="rule_0",
        timestamp=time.time(),
    )
    line = encode_command(cmd)
    assert line.startswith("CMD,")
    assert line.endswith("\n")
    assert verify(line)


def test_opencv_camera_source_reads_frame_and_releases():
    frame = np.ones((8, 12, 3), dtype=np.uint8)
    fake_cv2 = _FakeCV2(frame)
    source = OpenCVCameraSource(CameraConfig(backend="opencv", index=0, width=12, height=8, fps=30), cv2_module=fake_cv2)

    source.start()
    read_frame = source.read()
    source.release()

    assert np.array_equal(read_frame, frame)
    assert fake_cv2.capture.released is True
    assert fake_cv2.capture.set_calls == [
        (fake_cv2.CAP_PROP_FRAME_WIDTH, 12),
        (fake_cv2.CAP_PROP_FRAME_HEIGHT, 8),
        (fake_cv2.CAP_PROP_FPS, 30),
    ]


def test_picamera2_source_reads_bgr_frame_and_closes():
    frame = np.ones((8, 12, 4), dtype=np.uint8)
    fake_camera_class = _FakePicamera2Factory(frame)
    source = PiCamera2Source(CameraConfig(backend="picamera2", width=12, height=8, warmup_seconds=0), picamera2_class=fake_camera_class)

    source.start()
    read_frame = source.read()
    source.release()

    assert read_frame.shape == (8, 12, 3)
    assert fake_camera_class.instance.started is True
    assert fake_camera_class.instance.stopped is True
    assert fake_camera_class.instance.closed is True
    assert fake_camera_class.instance.main_config == {"format": "BGR888", "size": (12, 8)}
    assert fake_camera_class.instance.controls is None


def test_picamera2_source_raises_on_read_timeout():
    fake_camera_class = _FakePicamera2Factory(np.ones((1, 1, 3), dtype=np.uint8), hanging_job=True)
    source = PiCamera2Source(
        CameraConfig(backend="picamera2", width=1, height=1, read_timeout_seconds=0.01, warmup_seconds=0),
        picamera2_class=fake_camera_class,
    )

    source.start()
    try:
        source.read()
    except RuntimeError as exc:
        assert "timed out reading frame" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    finally:
        source.release()


def test_picamera2_source_applies_explicit_frame_rate_control():
    frame = np.ones((8, 12, 3), dtype=np.uint8)
    fake_camera_class = _FakePicamera2Factory(frame)
    source = PiCamera2Source(
        CameraConfig(backend="picamera2", width=12, height=8, fps=15, warmup_seconds=0),
        picamera2_class=fake_camera_class,
    )

    source.start()
    source.release()

    assert fake_camera_class.instance.controls == {"FrameRate": 15.0}


def test_picamera2_source_converts_rgb888_to_bgr():
    frame = np.array([[[10, 20, 30]]], dtype=np.uint8)
    fake_camera_class = _FakePicamera2Factory(frame)
    source = PiCamera2Source(
        CameraConfig(backend="picamera2", width=1, height=1, pixel_format="RGB888", warmup_seconds=0),
        picamera2_class=fake_camera_class,
    )

    source.start()
    read_frame = source.read()
    source.release()

    assert read_frame.tolist() == [[[30, 20, 10]]]


def test_create_camera_source_rejects_unknown_backend():
    try:
        create_camera_source(backend="unknown")
    except ValueError as exc:
        assert "camera backend" in str(exc)
    else:
        raise AssertionError("expected ValueError")


class _FakeCapture:
    def __init__(self, frame):
        self.frame = frame
        self.released = False
        self.set_calls = []

    def set(self, prop, value):
        self.set_calls.append((prop, value))

    def isOpened(self):
        return True

    def read(self):
        return True, self.frame

    def release(self):
        self.released = True


class _FakeCV2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5

    def __init__(self, frame):
        self.capture = _FakeCapture(frame)

    def VideoCapture(self, index):
        return self.capture


class _FakePicamera2Factory:
    def __init__(self, frame, *, hanging_job=False):
        self.frame = frame
        self.hanging_job = hanging_job
        self.instance = None

    def __call__(self):
        self.instance = _FakePicamera2(self.frame, hanging_job=self.hanging_job)
        return self.instance


class _FakePicamera2:
    def __init__(self, frame, *, hanging_job=False):
        self.frame = frame
        self.hanging_job = hanging_job
        self.started = False
        self.stopped = False
        self.closed = False
        self.main_config = None
        self.controls = None

    def create_video_configuration(self, *, main, controls=None):
        self.main_config = main
        self.controls = controls
        return {"main": main}

    def configure(self, config):
        self.config = config

    def start(self):
        self.started = True

    def capture_array(self, wait=True):
        if not wait:
            return _FakeCaptureJob(self.frame, hanging=self.hanging_job)
        return self.frame

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class _FakeCaptureJob:
    def __init__(self, frame, *, hanging=False):
        self.frame = frame
        self.hanging = hanging

    def get_result(self, timeout=None):
        if self.hanging:
            from concurrent.futures import TimeoutError as FutureTimeoutError

            raise FutureTimeoutError()
        return self.frame
