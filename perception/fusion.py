from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from perception.mock_perception import validate_perception_output
from perception.runtime import Hazard, LaneSeg, ObjectBBox, PerceptionOutput, TrafficLight

_UNSET = object()


@dataclass(frozen=True)
class FusionConfig:
    min_object_conf: float = 0.0
    min_laneseg_conf: float = 0.0
    min_traffic_light_conf: float = 0.0
    min_hazard_conf: float = 0.0
    max_objects: int = 50
    sort_objects_by_conf: bool = True


class PerceptionFusion:
    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self.config = config or FusionConfig()
        _validate_config(self.config)

    def fuse(
        self,
        *,
        base: Optional[PerceptionOutput] = None,
        timestamp: Optional[float] = None,
        laneseg: Any = _UNSET,
        objects: Any = _UNSET,
        traffic_light: Any = _UNSET,
        hazard: Any = _UNSET,
    ) -> PerceptionOutput:
        return fuse_perception(
            base=base,
            timestamp=timestamp,
            laneseg=laneseg,
            objects=objects,
            traffic_light=traffic_light,
            hazard=hazard,
            config=self.config,
        )


def fuse_perception(
    *,
    base: Optional[PerceptionOutput] = None,
    timestamp: Optional[float] = None,
    laneseg: Any = _UNSET,
    objects: Any = _UNSET,
    traffic_light: Any = _UNSET,
    hazard: Any = _UNSET,
    config: Optional[FusionConfig] = None,
) -> PerceptionOutput:
    cfg = config or FusionConfig()
    _validate_config(cfg)
    output = PerceptionOutput(
        timestamp=float(timestamp if timestamp is not None else (base.timestamp if base else time.time())),
        laneseg=_resolve_laneseg(laneseg, base, cfg),
        objects=_resolve_objects(objects, base, cfg),
        traffic_light=_resolve_traffic_light(traffic_light, base, cfg),
        hazard=_resolve_hazard(hazard, base, cfg),
    )
    validate_perception_output(output)
    return output


def _resolve_laneseg(value: Any, base: Optional[PerceptionOutput], config: FusionConfig) -> Optional[LaneSeg]:
    if value is _UNSET:
        value = base.laneseg if base is not None else None
    laneseg = _coerce_laneseg(value)
    return None if laneseg is not None and laneseg.conf < config.min_laneseg_conf else laneseg


def _resolve_objects(value: Any, base: Optional[PerceptionOutput], config: FusionConfig) -> List[ObjectBBox]:
    if value is _UNSET:
        value = base.objects if base is not None else []
    objects = [_coerce_object(obj) for obj in _ensure_iterable(value, "objects")]
    objects = [obj for obj in objects if obj.conf >= config.min_object_conf]
    if config.sort_objects_by_conf:
        objects.sort(key=lambda obj: obj.conf, reverse=True)
    return objects[: config.max_objects]


def _resolve_traffic_light(value: Any, base: Optional[PerceptionOutput], config: FusionConfig) -> Optional[TrafficLight]:
    if value is _UNSET:
        value = base.traffic_light if base is not None else None
    traffic_light = _coerce_traffic_light(value)
    return None if traffic_light is not None and traffic_light.conf < config.min_traffic_light_conf else traffic_light


def _resolve_hazard(value: Any, base: Optional[PerceptionOutput], config: FusionConfig) -> Optional[Hazard]:
    if value is _UNSET:
        value = base.hazard if base is not None else None
    hazard = _coerce_hazard(value)
    return None if hazard is not None and hazard.conf < config.min_hazard_conf else hazard


def _coerce_laneseg(value: Any) -> Optional[LaneSeg]:
    if value is None:
        return None
    if isinstance(value, LaneSeg):
        return LaneSeg(int(value.mask_id), float(value.conf))
    if isinstance(value, Mapping):
        return LaneSeg(int(_required(value, "mask_id")), float(_required(value, "conf")))
    raise TypeError("laneseg must be LaneSeg, mapping, None, or omitted")


def _coerce_object(value: Any) -> ObjectBBox:
    if isinstance(value, ObjectBBox):
        return ObjectBBox(str(value.cls), [float(v) for v in value.bbox], float(value.conf))
    if isinstance(value, Mapping):
        return ObjectBBox(str(_required(value, "cls")), [float(v) for v in _required(value, "bbox")], float(_required(value, "conf")))
    raise TypeError("objects entries must be ObjectBBox or mapping")


def _coerce_traffic_light(value: Any) -> Optional[TrafficLight]:
    if value is None:
        return None
    if isinstance(value, TrafficLight):
        return TrafficLight(str(value.state), float(value.conf))
    if isinstance(value, Mapping):
        return TrafficLight(str(_required(value, "state")), float(_required(value, "conf")))
    raise TypeError("traffic_light must be TrafficLight, mapping, None, or omitted")


def _coerce_hazard(value: Any) -> Optional[Hazard]:
    if value is None:
        return None
    if isinstance(value, Hazard):
        return Hazard(str(value.type), float(value.conf))
    if isinstance(value, Mapping):
        return Hazard(str(_required(value, "type")), float(_required(value, "conf")))
    raise TypeError("hazard must be Hazard, mapping, None, or omitted")


def _ensure_iterable(value: Any, field_name: str) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of detection objects, not text")
    return value


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required field: {key}")
    return mapping[key]


def _validate_config(config: FusionConfig) -> None:
    for field_name, value in (
        ("min_object_conf", config.min_object_conf),
        ("min_laneseg_conf", config.min_laneseg_conf),
        ("min_traffic_light_conf", config.min_traffic_light_conf),
        ("min_hazard_conf", config.min_hazard_conf),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{field_name} must be in [0.0, 1.0]")
    if config.max_objects < 0:
        raise ValueError("max_objects must be non-negative")
