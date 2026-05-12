"""
Serial communication with control board.

Handles encoding/decoding of commands and status frames.
"""

import serial
import time
from typing import Optional, List
from control.runtime import ControlCommand, BoardStatus
import logging

logger = logging.getLogger(__name__)


class SerialCommandSender:
    """Real serial communication with control board."""
    
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.sent_count = 0
        self.recv_count = 0
        self.error_count = 0
    
    def connect(self) -> bool:
        """Open serial connection."""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )
            logger.info(f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            logger.error(f"Failed to connect: {e}")
            self.error_count += 1
            return False
    
    def disconnect(self) -> None:
        """Close serial connection."""
        if self.ser:
            self.ser.close()
            logger.info("Disconnected")
    
    def send_command(self, cmd: ControlCommand) -> bool:
        """Send command to board.
        
        Encodes to serial frame and sends.
        Matches schema.md § 4.1.
        """
        if not self.ser or not self.ser.is_open:
            logger.warning("Serial not connected")
            self.error_count += 1
            return False
        
        try:
            frame = cmd.to_serial_frame()
            self.ser.write(frame.encode('ascii'))
            self.sent_count += 1
            logger.debug(f"Sent: {frame.strip()}")
            return True
        except Exception as e:
            logger.error(f"Failed to send: {e}")
            self.error_count += 1
            return False
    
    def read_status(self) -> Optional[BoardStatus]:
        """Read status from board.
        
        Reads line from serial, parses STAT frame.
        Matches schema.md § 4.2.
        """
        if not self.ser or not self.ser.is_open:
            return None
        
        try:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('ascii', errors='ignore')
                if line:
                    status = BoardStatus.from_serial_frame(line)
                    if status:
                        self.recv_count += 1
                        logger.debug(f"Received: {line.strip()}")
                        return status
                    else:
                        logger.warning(f"Invalid frame: {line.strip()}")
                        self.error_count += 1
        except Exception as e:
            logger.error(f"Failed to read: {e}")
            self.error_count += 1
        
        return None
    
    @staticmethod
    def _calculate_crc(payload: str) -> int:
        """Calculate CRC checksum.
        
        CRC = (sum of ASCII bytes) % 256
        
        Matches schema.md § 4.
        """
        return sum(ord(c) for c in payload) % 256
    
    @staticmethod
    def _verify_frame(frame: str) -> bool:
        """Verify frame CRC."""
        frame = frame.strip()
        if "," not in frame:
            return False
        
        try:
            parts = frame.split(",")
            if len(parts) < 2:
                return False
            
            payload = ",".join(parts[:-1])
            received_crc = parts[-1].strip()
            expected_crc = f"{SerialCommandSender._calculate_crc(payload):02X}"
            
            return received_crc == expected_crc
        except Exception:
            return False
    
    @staticmethod
    def _parse_status_frame(frame: str) -> Optional[BoardStatus]:
        """Parse status frame (alias for BoardStatus.from_serial_frame)."""
        return BoardStatus.from_serial_frame(frame)


class MockSerialSender:
    """Mock serial sender for testing (no hardware needed)."""
    
    def __init__(self):
        self.sent_commands: List[ControlCommand] = []
        self.statuses: List[BoardStatus] = []
        self.sent_count = 0
        self.error_count = 0
    
    def connect(self) -> bool:
        """Pretend to connect."""
        return True
    
    def disconnect(self) -> None:
        """Pretend to disconnect."""
        pass
    
    def send_command(self, cmd: ControlCommand) -> bool:
        """Mock send: just store the command."""
        self.sent_commands.append(cmd)
        self.sent_count += 1
        return True
    
    def read_status(self) -> Optional[BoardStatus]:
        """Mock read: return simulated status."""
        # Simulate board response
        return BoardStatus(
            timestamp=time.time(),
            voltage=12.0,
            temperature=35.0,
            error_code=0,
        )
