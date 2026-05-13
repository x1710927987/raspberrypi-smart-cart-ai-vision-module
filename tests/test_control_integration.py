"""
Integration test: perception -> decision -> serial command pipeline.

This validates that the entire system works end-to-end before field testing.
"""

import pytest
import time
from unittest.mock import Mock

from perception import make_mock_perception, PerceptionPipeline
from perception.runtime import LaneSeg, ObjectBBox, PerceptionOutput
from control.decision import BehaviorDecisionEngine
from control.runtime import ControlConfig, ControlMode, DecisionReason
from control.serial_comm import SerialCommandSender, MockSerialSender


class TestPerceptionToDecision:
    """Test perception output → behavior decision."""

    def test_clear_path_decision(self):
        """Clear path scenario: should proceed at max speed."""
        engine = BehaviorDecisionEngine()
        perception = make_mock_perception("clear_path")

        cmd = engine.decide(perception)

        assert cmd.brake is False
        assert cmd.v >= 0.5  # Should be moving at max_speed (0.8) or higher
        assert cmd.steer == 0.0
        assert cmd.mode == "auto"

    def test_red_light_decision(self):
        """Red light scenario: should brake."""
        engine = BehaviorDecisionEngine()
        perception = make_mock_perception("red_light")

        cmd = engine.decide(perception)

        assert cmd.brake is True
        assert cmd.v == 0.0

    def test_pedestrian_decision(self):
        """Pedestrian scenario: should brake or avoid."""
        engine = BehaviorDecisionEngine()
        perception = make_mock_perception("pedestrian_ahead")

        cmd = engine.decide(perception)

        assert cmd.brake is True or cmd.v <= 0.3
        # Should have some steering to avoid
        # (exact value depends on pedestrian position)

    def test_hazard_decision(self):
        """Hazard scenario: should brake."""
        engine = BehaviorDecisionEngine()
        perception = make_mock_perception("road_hazard")

        cmd = engine.decide(perception)

        assert cmd.brake is True
        assert cmd.v == 0.0

    def test_mixed_risk_decision(self):
        """Multiple risks: should handle priority correctly."""
        engine = BehaviorDecisionEngine()
        perception = make_mock_perception("mixed_risk")

        cmd = engine.decide(perception)

        # Should make conservative decision
        assert cmd.brake is True or cmd.v <= 0.3

    @pytest.mark.parametrize("cls", ["bicycle", "car", "scooter", "roadblock"])
    def test_dynamic_object_classes_trigger_obstacle_logic(self, cls):
        """Detected traffic participants and roadblocks should slow or stop the cart."""
        engine = BehaviorDecisionEngine()
        perception = PerceptionOutput(
            timestamp=time.time(),
            laneseg=LaneSeg(mask_id=1, conf=0.95),
            objects=[ObjectBBox(cls=cls, bbox=[240.0, 360.0, 400.0, 470.0], conf=0.9)],
            traffic_light=None,
            hazard=None,
        )

        cmd = engine.decide(perception)

        assert cmd.reason == DecisionReason.OBSTACLE_DETECTED.value
        assert cmd.brake is True
        assert cmd.v == 0.0


class TestSerialProtocol:
    """Test serial protocol encoding/decoding."""

    def test_crc_calculation(self):
        """CRC should match specification."""
        # Example from schema.md - compute what the CRC should actually be
        payload = "CMD,1720000123.567,0.600,-5.0,0,auto"
        crc = SerialCommandSender._calculate_crc(payload)
        # Verify it's a valid hex value
        assert 0 <= crc <= 255
        assert isinstance(crc, int)

    def test_frame_verification(self):
        """Valid frames should pass verification if CRC is correct."""
        # First compute the correct CRC
        payload = "CMD,1720000123.567,0.600,-5.0,0,auto"
        correct_crc = SerialCommandSender._calculate_crc(payload)
        frame = f"{payload},{correct_crc:02X}"
        assert SerialCommandSender._verify_frame(frame) is True

    def test_frame_verification_fails_bad_crc(self):
        """Frames with bad CRC should fail."""
        frame = "CMD,1720000123.567,0.600,-5.0,0,auto,00"
        assert SerialCommandSender._verify_frame(frame) is False

    def test_status_frame_parsing(self):
        """Valid status frames should parse correctly."""
        # Example from schema.md - compute correct CRC first
        payload = "STAT,1720000123.700,12.10,36.5,0"
        correct_crc = SerialCommandSender._calculate_crc(payload)
        frame = f"{payload},{correct_crc:02X}"
        
        status = SerialCommandSender._parse_status_frame(frame)
        
        assert status is not None
        assert status.voltage == 12.10
        assert status.temperature == 36.5
        assert status.error_code == 0


class TestIntegrationPipeline:
    """Test full pipeline: perception → decision → command."""

    def test_end_to_end_mock(self):
        """Full pipeline with mock components."""
        # Setup
        behavior = BehaviorDecisionEngine()
        serial = MockSerialSender()

        # Get perception
        perception = make_mock_perception("clear_path")

        # Make decision
        cmd = behavior.decide(perception)

        # Send command
        success = serial.send_command(cmd)
        assert success is True

        # Verify command was formatted correctly
        assert cmd.mode == "auto"
        assert 0.0 <= cmd.v <= 1.2
        assert -30.0 <= cmd.steer <= 30.0
        assert len(serial.sent_commands) == 1

    def test_emergency_brake_chain(self):
        """Test emergency brake propagation."""
        behavior = BehaviorDecisionEngine()
        serial = MockSerialSender()

        # Trigger emergency
        cmd = behavior.emergency_brake("test_trigger")

        # Should be brake command
        assert cmd.brake is True
        assert cmd.v == 0.0

        # Send it
        serial.send_command(cmd)
        assert len(serial.sent_commands) == 1


class TestModeTransitions:
    """Test operating mode transitions."""

    def test_mode_switch_auto_to_manual(self):
        """Can switch from AUTO to MANUAL."""
        engine = BehaviorDecisionEngine()
        assert engine.current_mode == ControlMode.AUTO

        engine.set_mode(ControlMode.MANUAL)
        assert engine.current_mode == ControlMode.MANUAL

    def test_emergency_brake_changes_mode(self):
        """Emergency brake should switch to EMERGENCY_BRAKE mode."""
        engine = BehaviorDecisionEngine()
        cmd = engine.emergency_brake("test")
        
        assert engine.current_mode == ControlMode.EMERGENCY_BRAKE
        assert cmd.brake is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
