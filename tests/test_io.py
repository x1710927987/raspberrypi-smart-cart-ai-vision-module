import sys
import pathlib
import time


# Ensure project root is importable and takes precedence over stdlib modules
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from perception.runtime import ControlCommand
from io.protocol import (
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
