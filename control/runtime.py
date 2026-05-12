"""
Runtime data structures for control module.

Defines modes, reasons, configurations, and command serialization.
"""

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, Any, Optional
import json
import time


class ControlMode(Enum):
    """Vehicle operating mode."""
    AUTO = "auto"
    MANUAL = "manual"
    EMERGENCY_BRAKE = "emergency_brake"


class DecisionReason(Enum):
    """Why the decision was made."""
    CLEAR_PATH = "clear_path"
    RED_LIGHT = "red_light"
    PEDESTRIAN_DETECTED = "pedestrian_detected"
    OBSTACLE_DETECTED = "obstacle_detected"
    HAZARD_DETECTED = "hazard_detected"
    LANE_LOST = "lane_lost"
    EMERGENCY = "emergency"
    MANUAL_OVERRIDE = "manual_override"


@dataclass(frozen=True)
class ControlConfig:
    """Tunable parameters for control behavior."""
    # Speed control (m/s)
    max_speed: float = 0.8
    safe_speed: float = 0.3
    cruise_speed: float = 0.5
    
    # Distance thresholds (pixels, for 640x480 camera)
    dangerous_dist_threshold: float = 100.0
    caution_dist_threshold: float = 200.0
    
    # Steering
    max_steer_angle: float = 30.0
    neutral_steer: float = 0.0
    
    # Confidence thresholds
    min_object_conf: float = 0.6
    min_traffic_light_conf: float = 0.8
    min_hazard_conf: float = 0.7
    
    # Lane detection
    min_laneseg_conf: float = 0.6
    
    # Timing
    heartbeat_interval_ms: int = 200  # 5 Hz minimum


@dataclass
class ControlState:
    """Current control state for diagnostics."""
    frames_processed: int = 0
    commands_sent: int = 0
    frame_times_ms: list = field(default_factory=list)
    last_reason: DecisionReason = DecisionReason.CLEAR_PATH
    current_mode: ControlMode = ControlMode.AUTO
    error_count: int = 0
    last_error: Optional[str] = None
    board_voltage: Optional[float] = None
    board_temperature: Optional[float] = None
    board_error_code: int = 0
    
    def get_avg_frame_time_ms(self) -> float:
        """Average processing time per frame."""
        if not self.frame_times_ms:
            return 0.0
        return sum(self.frame_times_ms[-100:]) / len(self.frame_times_ms[-100:])


@dataclass(frozen=True)
class ControlDecision:
    """Intermediate decision before conversion to ControlCommand."""
    reason: DecisionReason
    target_speed: float
    target_steer: float
    should_brake: bool
    confidence: float = 1.0
    
    def to_control_command(self, mode: ControlMode = ControlMode.AUTO, config: Optional[ControlConfig] = None) -> "ControlCommand":
        """Convert decision to final command."""
        cfg = config or ControlConfig()
        
        # Apply constraints
        v = max(0.0, min(self.target_speed, cfg.max_speed))
        steer = max(-cfg.max_steer_angle, min(self.target_steer, cfg.max_steer_angle))
        
        # Brake overrides speed
        if self.should_brake:
            v = 0.0
        
        return ControlCommand(
            mode=mode.value,
            v=v,
            steer=steer,
            brake=self.should_brake,
            reason=self.reason.value,
            timestamp=time.time(),
        )


@dataclass(frozen=True)
class ControlCommand:
    """Final command to be sent to control board.
    
    Matches schema.md § 3.
    """
    mode: str  # "auto" or "manual"
    v: float  # m/s, [0, 1.2]
    steer: float  # degrees, [-30, 30], left-positive
    brake: bool
    reason: str
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "mode": self.mode,
            "v": float(self.v),
            "steer": float(self.steer),
            "brake": bool(self.brake),
            "reason": self.reason,
            "timestamp": float(self.timestamp),
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ControlCommand":
        """Create from dict."""
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
        """Create from JSON string."""
        return ControlCommand.from_dict(json.loads(s))
    
    def to_serial_frame(self) -> str:
        """Convert to serial protocol frame.
        
        Format: CMD,<TS>,<v>,<steer>,<brake>,<mode>,<CRC>\n
        Matches schema.md § 4.1.
        """
        # Quantize v to 0.05 m/s step
        v_quantized = round(self.v / 0.05) * 0.05
        
        # Format according to spec
        payload = f"CMD,{self.timestamp:.3f},{v_quantized:.3f},{self.steer:.1f},{1 if self.brake else 0},{self.mode}"
        
        # Calculate CRC
        crc = sum(ord(c) for c in payload) % 256
        
        frame = f"{payload},{crc:02X}\n"
        return frame
    
    @staticmethod
    def from_serial_frame(frame: str) -> Optional["ControlCommand"]:
        """Parse from serial frame (for echo validation)."""
        frame = frame.strip()
        if not frame.startswith("CMD,"):
            return None
        
        try:
            parts = frame.split(",")
            if len(parts) < 7:
                return None
            
            # Verify CRC
            payload = ",".join(parts[:-1])
            received_crc = parts[-1]
            expected_crc = f"{sum(ord(c) for c in payload) % 256:02X}"
            if received_crc != expected_crc:
                return None
            
            return ControlCommand(
                mode=parts[5],
                v=float(parts[2]),
                steer=float(parts[3]),
                brake=bool(int(parts[4])),
                reason="",
                timestamp=float(parts[1]),
            )
        except (ValueError, IndexError):
            return None


@dataclass(frozen=True)
class BoardStatus:
    """Status received from control board.
    
    Matches schema.md § 4.2.
    """
    timestamp: float
    voltage: float  # volts
    temperature: float  # Celsius
    error_code: int  # bitmask
    
    def is_ok(self) -> bool:
        """No errors?"""
        return self.error_code == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return asdict(self)
    
    @staticmethod
    def from_serial_frame(frame: str) -> Optional["BoardStatus"]:
        """Parse STAT frame.
        
        Format: STAT,<TS>,<voltage>,<temp>,<err>,<CRC>\n
        Example: STAT,1720000123.700,12.10,36.5,0,24\n
        Matches schema.md § 4.2.
        """
        frame = frame.strip()
        if not frame.startswith("STAT,"):
            return None
        
        try:
            parts = frame.split(",")
            if len(parts) < 6:
                return None
            
            # Verify CRC
            payload = ",".join(parts[:-1])
            received_crc = parts[-1]
            expected_crc = f"{sum(ord(c) for c in payload) % 256:02X}"
            if received_crc != expected_crc:
                return None
            
            return BoardStatus(
                timestamp=float(parts[1]),
                voltage=float(parts[2]),
                temperature=float(parts[3]),
                error_code=int(parts[4]),
            )
        except (ValueError, IndexError):
            return None
