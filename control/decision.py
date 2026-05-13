"""
Behavior decision engine with mode management.

Coordinates rule engine output and applies mode constraints.
"""

from dataclasses import dataclass
from typing import Optional
from perception.runtime import PerceptionOutput
from control.runtime import (
    ControlMode,
    DecisionReason,
    ControlConfig,
    ControlState,
    ControlCommand,
    ControlDecision,
)
from control.rules import TrafficRulesEngine
import time


@dataclass(frozen=True)
class BehaviorConfig:
    """Configuration for behavior engine."""
    pass


class BehaviorDecisionEngine:
    """Coordinates traffic rules and mode management.
    
    Decides what the car should do based on:
    1. Current mode (auto/manual/emergency)
    2. Perception input
    3. Traffic rules
    """
    
    def __init__(self, config: Optional[ControlConfig] = None):
        self.config = config or ControlConfig()
        self.rules_engine = TrafficRulesEngine(self.config)
        self.current_mode = ControlMode.AUTO
        self.state = ControlState()
    
    def decide(self, perception: PerceptionOutput) -> ControlCommand:
        """Make final decision and return command.
        
        Args:
            perception: PerceptionOutput from vision module
        
        Returns:
            ControlCommand ready to send to board
        """
        start_time = time.time()
        
        # Apply rules to get decision
        decision = self.rules_engine.decide(perception)
        
        # Convert to command
        cmd = decision.to_control_command(self.current_mode, self.config)
        
        # Update state
        frame_time = (time.time() - start_time) * 1000  # ms
        self.state.frames_processed += 1
        self.state.frame_times_ms.append(frame_time)
        self.state.last_reason = decision.reason
        
        return cmd
    
    def set_mode(self, mode: ControlMode) -> None:
        """Switch operating mode."""
        self.current_mode = mode
    
    def emergency_brake(self, reason: str = "emergency") -> ControlCommand:
        """Trigger emergency brake."""
        self.current_mode = ControlMode.EMERGENCY_BRAKE
        self.state.last_reason = DecisionReason.EMERGENCY
        
        return ControlCommand(
            mode=ControlMode.EMERGENCY_BRAKE.value,
            v=0.0,
            steer=0.0,
            brake=True,
            reason=reason,
            timestamp=time.time(),
        )
    
    def get_state(self) -> ControlState:
        """Get diagnostic state."""
        return self.state

# Backward compatibility alias
DecisionEngine = BehaviorDecisionEngine
