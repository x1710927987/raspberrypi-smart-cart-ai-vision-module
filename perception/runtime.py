from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import json
import time


@dataclass
class ObjectBBox:
    cls: str
    bbox: List[float]
    conf: float


@dataclass
class TrafficLight:
    state: str
    conf: float


@dataclass
class Hazard:
    type: str
    conf: float


@dataclass
class LaneSeg:
    mask_id: int
    conf: float


@dataclass
class PerceptionOutput:
    timestamp: float
    laneseg: Optional[LaneSeg]
    objects: List[ObjectBBox]
    traffic_light: Optional[TrafficLight]
    hazard: Optional[Hazard]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "laneseg": asdict(self.laneseg) if self.laneseg is not None else None,
            "objects": [asdict(o) for o in self.objects],
            "traffic_light": asdict(self.traffic_light) if self.traffic_light is not None else None,
            "hazard": asdict(self.hazard) if self.hazard is not None else None,
        }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PerceptionOutput":
        laneseg = LaneSeg(**d["laneseg"]) if d.get("laneseg") else None
        objects = [ObjectBBox(**o) for o in d.get("objects", [])]
        traffic_light = TrafficLight(**d["traffic_light"]) if d.get("traffic_light") else None
        hazard = Hazard(**d["hazard"]) if d.get("hazard") else None
        return PerceptionOutput(
            timestamp=float(d.get("timestamp", time.time())),
            laneseg=laneseg,
            objects=objects,
            traffic_light=traffic_light,
            hazard=hazard,
        )

    @staticmethod
    def from_json(s: str) -> "PerceptionOutput":
        return PerceptionOutput.from_dict(json.loads(s))


@dataclass
class ControlCommand:
    mode: str
    v: float
    steer: float
    brake: bool
    reason: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "v": float(self.v),
            "steer": float(self.steer),
            "brake": bool(self.brake),
            "reason": self.reason,
            "timestamp": float(self.timestamp),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ControlCommand":
        return ControlCommand(
            mode=str(d["mode"]),
            v=float(d["v"]),
            steer=float(d["steer"]),
            brake=bool(d["brake"]),
            reason=str(d.get("reason", "")),
            timestamp=float(d.get("timestamp", time.time())),
        )

    @staticmethod
    def from_json(s: str) -> "ControlCommand":
        return ControlCommand.from_dict(json.loads(s))

