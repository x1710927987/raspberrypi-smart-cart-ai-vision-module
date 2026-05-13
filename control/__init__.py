"""
Smart Cart Control Module

Coordinates behavior decision, traffic rules, and serial communication.
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
from control.rules import TrafficRulesEngine, TrafficRulesConfig
from control.decision import BehaviorDecisionEngine, BehaviorConfig
from control.serial_comm import SerialCommandSender, MockSerialSender, BoardStatus
# from control.app import SmartCartController

__all__ = [
    "ControlMode",
    "DecisionReason",
    "ControlConfig",
    "ControlState",
    "ControlDecision",
    "ControlCommand",
    "BoardStatus",
    "TrafficRulesEngine",
    "TrafficRulesConfig",
    "BehaviorDecisionEngine",
    "BehaviorConfig",
    "SerialCommandSender",
    "MockSerialSender",
    "SmartCartController",
]


def create_test_controller(use_mock=True):
    """Quick factory for testing."""
    return SmartCartController(use_mock_serial=use_mock)
