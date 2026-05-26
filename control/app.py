#!/usr/bin/env python3
"""SmartCart control loop entrypoint."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

from control.decision import DecisionEngine
from control.runtime import ControlCommand
from control.serial_comm import MockSerialSender, SerialCommandSender
from io_camera.camera import CameraSource, create_camera_source
from perception.camera_pipeline import PerceptionPipeline
from perception.runtime import PerceptionOutput


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SmartCartApplication:
    """Wire camera frames through perception, decision rules, and serial output."""

    def __init__(
        self,
        *,
        camera_device: int = 0,
        camera_backend: str = "auto",
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: float = 30.0,
        serial_port: str = "/dev/ttyUSB0",
        serial_baudrate: int = 115200,
        use_mock_serial: bool = False,
        target_fps: int = 10,
    ):
        self.camera_device = camera_device
        self.camera_backend = camera_backend
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.serial_port = serial_port
        self.serial_baudrate = serial_baudrate
        self.use_mock_serial = use_mock_serial
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps if target_fps > 0 else 0.0

        self.camera_source: Optional[CameraSource] = None
        self.perception_pipeline: Optional[PerceptionPipeline] = None
        self.decision_engine: Optional[DecisionEngine] = None
        self.serial_comm: Optional[SerialCommandSender | MockSerialSender] = None

        self.frame_count = 0
        self.start_time = time.time()
        self.last_perception_time = 0.0
        self.last_decision_time = 0.0
        self.last_serial_time = 0.0

        logger.info("SmartCartApplication initializing...")
        logger.info(
            "  camera: backend=%s index=%s size=%sx%s fps=%s",
            camera_backend,
            camera_device,
            camera_width,
            camera_height,
            camera_fps,
        )
        logger.info("  serial: %s @ %s baud", serial_port, serial_baudrate)
        logger.info("  mode: %s", "mock serial" if use_mock_serial else "real serial")
        logger.info("  target loop rate: %s FPS", target_fps)

    def initialize(self) -> bool:
        try:
            logger.info("Initializing camera...")
            self.camera_source = create_camera_source(
                backend=self.camera_backend,
                index=self.camera_device,
                width=self.camera_width,
                height=self.camera_height,
                fps=self.camera_fps,
            )
            self.camera_source.start()
            logger.info("Camera ready.")

            logger.info("Initializing perception pipeline...")
            self.perception_pipeline = PerceptionPipeline.with_default_models()
            logger.info("Perception pipeline ready.")

            logger.info("Initializing decision engine...")
            self.decision_engine = DecisionEngine()
            logger.info("Decision engine ready.")

            logger.info("Initializing serial communication...")
            if self.use_mock_serial:
                self.serial_comm = MockSerialSender()
            else:
                self.serial_comm = SerialCommandSender(
                    port=self.serial_port,
                    baudrate=self.serial_baudrate,
                )
            if not self.serial_comm.connect():
                logger.error("Serial connection failed.")
                return False
            logger.info("Serial communication ready.")

            logger.info("All components initialized.")
            return True
        except Exception as exc:
            logger.error("Initialization failed: %s", exc, exc_info=True)
            return False

    def process_frame(self) -> bool:
        try:
            if self.camera_source is None or self.perception_pipeline is None or self.decision_engine is None or self.serial_comm is None:
                raise RuntimeError("application components are not initialized")

            frame_start = time.time()

            frame = self.camera_source.read()

            perception_start = time.time()
            perception_output: PerceptionOutput = self.perception_pipeline.process_frame(frame)
            self.last_perception_time = time.time() - perception_start

            decision_start = time.time()
            control_cmd: ControlCommand = self.decision_engine.decide(perception_output)
            self.last_decision_time = time.time() - decision_start

            serial_start = time.time()
            success = self.serial_comm.send_command(control_cmd)
            self.last_serial_time = time.time() - serial_start
            if not success:
                logger.warning("Serial send failed.")
                return False

            self.frame_count += 1
            if self.frame_count % 10 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "[Frame %s] FPS=%.1f | perception=%.1fms | decision=%.1fms | serial=%.1fms | reason=%s | v=%.2fm/s steer=%.1fdeg",
                    self.frame_count,
                    fps,
                    self.last_perception_time * 1000.0,
                    self.last_decision_time * 1000.0,
                    self.last_serial_time * 1000.0,
                    control_cmd.reason,
                    control_cmd.v,
                    control_cmd.steer,
                )

            elapsed_time = time.time() - frame_start
            if self.frame_time > 0 and elapsed_time < self.frame_time:
                time.sleep(self.frame_time - elapsed_time)
            return True
        except Exception as exc:
            logger.error("Frame processing failed: %s", exc, exc_info=True)
            return False

    def run(self) -> int:
        if not self.initialize():
            logger.error("Initialization failed; exiting.")
            return 1

        logger.info("Main loop started. Press Ctrl+C to exit.")
        try:
            while True:
                if not self.process_frame():
                    logger.warning("Frame processing stopped the main loop.")
                    break
        except KeyboardInterrupt:
            logger.info("Stop signal received.")
        except Exception as exc:
            logger.error("Runtime error: %s", exc, exc_info=True)
            return 1
        finally:
            self.cleanup()

        logger.info("Application exited normally.")
        return 0

    def cleanup(self) -> None:
        logger.info("Cleaning up resources...")

        if self.camera_source is not None:
            self.camera_source.release()
            self.camera_source = None
            logger.info("Camera closed.")

        if self.serial_comm is not None:
            self.serial_comm.disconnect()
            self.serial_comm = None
            logger.info("Serial connection closed.")

        if self.frame_count > 0:
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0.0
            logger.info("Run stats: frames=%s elapsed=%.1fs avg_fps=%.1f", self.frame_count, elapsed, avg_fps)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smart Cart AI Control Application")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for OpenCV backend. Default: 0.")
    parser.add_argument(
        "--camera-backend",
        choices=("opencv", "picamera2", "auto"),
        default="auto",
        help="Camera backend. Use picamera2 for Raspberry Pi CSI camera, opencv for USB/V4L2. Default: auto.",
    )
    parser.add_argument("--camera-width", type=int, default=640, help="Requested camera width. Default: 640.")
    parser.add_argument("--camera-height", type=int, default=480, help="Requested camera height. Default: 480.")
    parser.add_argument("--camera-fps", type=float, default=30.0, help="Requested camera FPS. Default: 30.")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port. Default: /dev/ttyUSB0.")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate. Default: 115200.")
    parser.add_argument("--mock-serial", action="store_true", help="Use mock serial sender for local tests.")
    parser.add_argument("--fps", type=int, default=10, help="Target loop FPS. Default: 10.")
    args = parser.parse_args()

    app = SmartCartApplication(
        camera_device=args.camera,
        camera_backend=args.camera_backend,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        serial_port=args.port,
        serial_baudrate=args.baudrate,
        use_mock_serial=args.mock_serial,
        target_fps=args.fps,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
