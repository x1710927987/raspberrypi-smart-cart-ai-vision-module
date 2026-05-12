"""
Main application coordinator.

Ties together perception, decision, and serial communication.
"""

from typing import Optional, Dict, Any
from perception.runtime import PerceptionOutput
from control.decision import BehaviorDecisionEngine
from control.serial_comm import SerialCommandSender, MockSerialSender
from control.runtime import ControlConfig, ControlMode, ControlState
import time
import logging

logger = logging.getLogger(__name__)


class SmartCartController:
    """Main controller: perception → decision → action.
    
    Coordinates the entire control pipeline.
    """
    
    def __init__(self, config: Optional[ControlConfig] = None, use_mock_serial: bool = False):
        self.config = config or ControlConfig()
        self.behavior = BehaviorDecisionEngine(self.config)
        
        if use_mock_serial:
            self.serial = MockSerialSender()
        else:
            self.serial = SerialCommandSender()
        
        self.state = ControlState()
        self.last_frame_time = time.time()
    
    def connect(self) -> bool:
        """Initialize serial connection."""
        return self.serial.connect()
    
    def disconnect(self) -> None:
        """Close connections."""
        self.serial.disconnect()
    
    def process_frame(self, perception: PerceptionOutput) -> bool:
        """Process one frame: perception → decision → command → send.
        
        Args:
            perception: PerceptionOutput from vision module
        
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            # Step 1: Make decision
            cmd = self.behavior.decide(perception)
            
            # Step 2: Send command
            success = self.serial.send_command(cmd)
            if not success:
                self.state.error_count += 1
                return False
            
            # Step 3: Update state
            self.state.frames_processed += 1
            self.state.commands_sent += 1
            self.state.last_reason = self.behavior.state.last_reason
            self.state.current_mode = self.behavior.current_mode
            
            # Step 4: Try to read status
            status = self.serial.read_status()
            if status:
                self.state.board_voltage = status.voltage
                self.state.board_temperature = status.temperature
                self.state.board_error_code = status.error_code
            
            frame_time = (time.time() - start_time) * 1000  # ms
            self.state.frame_times_ms.append(frame_time)
            
            return True
        
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self.state.error_count += 1
            self.state.last_error = str(e)
            return False
    
    def set_mode(self, mode: ControlMode) -> None:
        """Switch operating mode."""
        self.behavior.set_mode(mode)
        self.state.current_mode = mode
    
    def emergency_brake(self, reason: str = "manual_trigger") -> bool:
        """Trigger emergency brake."""
        cmd = self.behavior.emergency_brake(reason)
        success = self.serial.send_command(cmd)
        
        if success:
            self.state.commands_sent += 1
        else:
            self.state.error_count += 1
        
        return success
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        return {
            "frames_processed": self.state.frames_processed,
            "commands_sent": self.state.commands_sent,
            "error_count": self.state.error_count,
            "avg_frame_time_ms": self.state.get_avg_frame_time_ms(),
            "current_mode": self.state.current_mode.value,
            "current_reason": self.state.last_reason.value,
            "last_error": self.state.last_error,
            "board": {
                "voltage": self.state.board_voltage,
                "temperature": self.state.board_temperature,
                "error_code": self.state.board_error_code,
            },
        }
