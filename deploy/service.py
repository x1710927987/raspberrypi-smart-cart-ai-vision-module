from __future__ import annotations

import logging
import logging.handlers
import os
import time
from pathlib import Path
from typing import Any

import yaml

from control.decision import DecisionEngine
from control.runtime import ControlCommand
from control.serial_comm import MockSerialSender, SerialCommandSender
from io_camera.camera import CameraSource, create_camera_source
from perception.camera_pipeline import PerceptionPipeline
from perception.mock_perception import make_mock_perception
from perception.runtime import PerceptionOutput


class SmartCartService:
    """Deployment service that wires camera, perception, decision, and serial output."""

    def __init__(self, config_path: str | os.PathLike[str]):
        self.running = False
        self.config = self._load_config(config_path)
        self._setup_logging()

        self.logger = logging.getLogger("SmartCart.Service")
        self.mock_mode = bool(self.config.get("runtime", {}).get("mock_mode", False))
        self.target_loop_time = 1.0 / float(self.config.get("runtime", {}).get("target_fps_min", 10.0))

        self.perception: PerceptionPipeline | None = None
        self.control: DecisionEngine | None = None
        self.serial_io: SerialCommandSender | MockSerialSender | None = None
        self.camera_source: CameraSource | None = None

        self.logger.info("System initializing...")
        self._initialize_components()

    def _initialize_components(self) -> None:
        perception_cfg = self.config.get("perception", {})
        serial_cfg = self.config.get("serial", {})

        self.control = DecisionEngine()
        self.serial_io = (
            MockSerialSender()
            if self.mock_mode
            else SerialCommandSender(
                port=str(serial_cfg.get("port", "/dev/ttyUSB0")),
                baudrate=int(serial_cfg.get("baud", serial_cfg.get("baudrate", 115200))),
            )
        )
        if not self.serial_io.connect():
            raise RuntimeError("serial initialization failed")

        if self.mock_mode:
            self.perception = None
            self.camera_source = None
            self.logger.info("Mock mode enabled; camera and real perception are skipped.")
            return

        self.perception = PerceptionPipeline.with_default_models(device=perception_cfg.get("device"))
        self.camera_source = create_camera_source(
            backend=str(perception_cfg.get("camera_backend", "auto")),
            index=int(perception_cfg.get("camera_index", 0)),
            width=int(perception_cfg.get("camera_width", 640)),
            height=int(perception_cfg.get("camera_height", 480)),
            fps=float(perception_cfg.get("camera_fps", 30.0)),
            pixel_format=str(perception_cfg.get("pixel_format", "BGR888")),
        )
        self.camera_source.start()

    @staticmethod
    def _load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"configuration file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _setup_logging(self) -> None:
        log_cfg = self.config.get("logging", {})
        log_dir = Path(log_cfg.get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("SmartCart")
        logger.handlers.clear()
        logger.setLevel(getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO))

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_dir / "cart_system.log",
            when="midnight",
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def _send_emergency_brake(self) -> None:
        if self.serial_io is None:
            self.logger.warning("Emergency brake requested, but serial output is unavailable.")
            return
        command = ControlCommand(
            mode="emergency_brake",
            v=0.0,
            steer=0.0,
            brake=True,
            reason="emergency_stop",
            timestamp=time.time(),
        )
        self.serial_io.send_command(command)
        self.logger.info("Emergency brake command sent.")

    def stop(self) -> None:
        self.logger.warning("Stopping system; dispatching emergency brake command.")
        self.running = False
        self._send_emergency_brake()
        if self.camera_source is not None:
            self.camera_source.release()
            self.camera_source = None
        if self.serial_io is not None:
            self.serial_io.disconnect()
        self.logger.info("System safely stopped.")

    def start(self) -> None:
        if self.control is None or self.serial_io is None:
            raise RuntimeError("service components are not initialized")

        self.running = True
        self.logger.info("Main loop started.")

        while self.running:
            loop_start = time.time()
            try:
                perception_result = self._read_perception(loop_start)
                control_cmd = self.control.decide(perception_result)
                self.serial_io.send_command(control_cmd)
                self.logger.debug("Perception: %s | Control: %s", perception_result.to_dict(), control_cmd.to_dict())
            except Exception as exc:
                self.logger.error("Exception in main loop: %s. Safety protection triggered.", exc, exc_info=True)
                self._send_emergency_brake()
                time.sleep(0.5)

            sleep_time = self.target_loop_time - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _read_perception(self, timestamp: float) -> PerceptionOutput:
        if self.mock_mode:
            return make_mock_perception("clear_path")
        if self.camera_source is None or self.perception is None:
            raise RuntimeError("camera or perception pipeline is not initialized")
        frame = self.camera_source.read()
        return self.perception.process_frame(frame, timestamp=timestamp)
