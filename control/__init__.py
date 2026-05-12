"""
Control module: vehicle behavior decision and serial communication.

This module handles:
1. Traffic rules engine (rule-based decision making)
2. Behavior decision engine (converts perception to commands)
3. Serial communication with control board
4. Mode management (auto/manual/emergency)
"""

from control.runtime import (
    ControlMode,
    DecisionReason,
    ControlConfig,
    ControlState,
    ControlDecision,
    ControlCommand,
    BoardStatus,
)

from control.rules import TrafficRulesEngine, RulesConfig

from control.decision import BehaviorDecisionEngine, DecisionConfig

from control.serial_comm import (
    SerialCommandSender,
    MockSerialSender,
    BoardStatusListener,
)

from control.app import SmartCartController, ControllerConfig

__all__ = [
    # Runtime types
    "ControlMode",
    "DecisionReason",
    "ControlConfig",
    "ControlState",
    "ControlDecision",
    "ControlCommand",
    "BoardStatus",
    # Rules
    "TrafficRulesEngine",
    "RulesConfig",
    # Decision
    "BehaviorDecisionEngine",
    "DecisionConfig",
    # Serial
    "SerialCommandSender",
    "MockSerialSender",
    "BoardStatusListener",
    # App
    "SmartCartController",
    "ControllerConfig",
]


def create_test_controller(use_mock: bool = True):
    """Factory function for testing."""
    return SmartCartController(use_mock_serial=use_mock)
