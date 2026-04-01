from typing import List, Tuple, Dict, Any
from perception.runtime import ControlCommand


class ProtocolError(Exception):
    pass


def _crc8_ascii(s: str) -> int:
    acc = 0
    for ch in s.encode("ascii", errors="ignore"):
        acc = (acc + ch) & 0xFF
    return acc


def _format_crc(v: int) -> str:
    return f"{v:02X}"


def _make_frame(prefix: str, fields: List[str]) -> str:
    payload = ",".join([prefix] + fields)
    crc = _format_crc(_crc8_ascii(payload))
    return f"{payload},{crc}\n"


def parse(line: str) -> Tuple[str, List[str], str]:
    s = line.strip("\r\n")
    parts = s.split(",")
    if len(parts) < 2:
        raise ProtocolError("invalid frame")
    prefix = parts[0]
    crc = parts[-1]
    fields = parts[1:-1]
    return prefix, fields, crc


def verify(line: str) -> bool:
    s = line.strip("\r\n")
    if "," not in s:
        return False
    payload, _, crc = s.rpartition(",")
    try:
        return _format_crc(_crc8_ascii(payload)) == crc.upper()
    except Exception:
        return False


def encode_command(cmd: ControlCommand) -> str:
    ts = f"{cmd.timestamp:.3f}"
    v = f"{cmd.v:.3f}"
    steer = f"{cmd.steer:.1f}"
    brake = "1" if cmd.brake else "0"
    mode = str(cmd.mode)
    return _make_frame("CMD", [ts, v, steer, brake, mode])


def decode_status(line: str) -> Dict[str, Any]:
    if not verify(line):
        raise ProtocolError("crc mismatch")
    prefix, fields, _ = parse(line)
    if prefix != "STAT" or len(fields) != 4:
        raise ProtocolError("invalid STAT frame")
    ts_s, volt_s, temp_s, err_s = fields
    return {
        "timestamp": float(ts_s),
        "voltage": float(volt_s),
        "temp": float(temp_s),
        "err": int(err_s),
    }


def encode_status(timestamp: float, voltage: float, temp: float, err: int) -> str:
    return _make_frame(
        "STAT",
        [f"{timestamp:.3f}", f"{voltage:.2f}", f"{temp:.1f}", str(int(err))],
    )

