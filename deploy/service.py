import yaml
import time
import logging
import logging.handlers
import os
import cv2

# Team A & B interfaces (assumed implemented)
from perception.runtime import PerceptionPipeline
from control.engine import ControlEngine
from io_camera.protocol import SerialCommunicator


class SmartCartService:
    def __init__(self, config_path: str):
        self.running = False
        self.config = self._load_config(config_path)
        self._setup_logging()

        self.logger = logging.getLogger("SmartCart.Service")
        self.logger.info("System initializing...")

        # Mock mode flag (for testing without real modules)
        self.mock_mode = self.config.get('runtime', {}).get('mock_mode', False)

        if not self.mock_mode:
            # 1. Perception module (Team A)
            # It may require camera index / model path; we pass its config subsection.
            self.perception = PerceptionPipeline(self.config.get('perception', {}))

            # 2. Control module (Team B)
            self.control = ControlEngine(self.config.get('control', {}))

            # 3. Serial communication module (Team B)
            self.serial_io = SerialCommunicator(self.config.get('serial', {}))

            # Optional: camera capture if A does not handle it internally
            # Some PerceptionPipeline may have its own capture; but we assume we
            # need to read frames and pass to perception.process(frame).
            # We'll create a VideoCapture instance for that purpose.
            camera_cfg = self.config.get('perception', {})
            camera_idx = camera_cfg.get('camera_index', 0)
            self.cap = cv2.VideoCapture(camera_idx)
            if not self.cap.isOpened():
                self.logger.error(f"Cannot open camera index {camera_idx}")
                raise RuntimeError("Camera initialization failed")
        else:
            # Dummy placeholders for mock mode (avoid attribute errors)
            self.perception = None
            self.control = None
            self.serial_io = None
            self.cap = None

        # Framerate control (Schema requires >=5Hz, we target >=10Hz)
        self.target_loop_time = 1.0 / self.config.get('runtime', {}).get('target_fps_min', 10.0)

    def _load_config(self, path: str) -> dict:
        """Load YAML configuration file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Configure robust logging system (console and file outputs)."""
        log_cfg = self.config.get('logging', {})
        log_dir = log_cfg.get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        logger = logging.getLogger("SmartCart")
        level_str = log_cfg.get('level', 'INFO').upper()
        logger.setLevel(getattr(logging, level_str, logging.INFO))

        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.handlers.TimedRotatingFileHandler(
            os.path.join(log_dir, 'cart_system.log'),
            when='midnight', backupCount=7, encoding='utf-8'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def _send_emergency_brake(self):
        """Send emergency brake command to the underlying board (if serial available)."""
        if self.mock_mode or not self.serial_io:
            self.logger.warning("Emergency brake requested, but serial_io unavailable (mock mode).")
            return
        emergency_cmd = {
            "mode": "auto",
            "v": 0.0,
            "steer": 0.0,
            "brake": True,
            "reason": "emergency_stop"
        }
        try:
            self.serial_io.send_command(emergency_cmd)
            self.logger.info("Emergency brake command sent.")
        except Exception as e:
            self.logger.error(f"Failed to send emergency brake: {e}")

    def stop(self):
        """Safely stop the entire system."""
        self.logger.warning("Stopping system, dispatching emergency brake command!")
        self.running = False
        self._send_emergency_brake()
        if self.cap is not None:
            self.cap.release()
        self.logger.info("System safely stopped.")

    def start(self):
        """Main execution loop."""
        self.running = True
        self.logger.info("Main loop started, beginning processing...")

        while self.running:
            loop_start = time.time()

            try:
                if self.mock_mode:
                    # Use mock data when no real modules are available
                    perception_result = {
                        "timestamp": loop_start,
                        "objects": [],
                        "hazard": None
                    }
                    # Mock control command
                    control_cmd = {"v": 0.5, "steer": 0.0, "brake": False, "mode": "auto"}
                else:
                    # --- Step 1: Capture frame and run perception ---
                    ret, frame = self.cap.read()
                    if not ret:
                        self.logger.warning("Failed to grab frame, skipping cycle")
                        time.sleep(0.05)
                        continue
                    perception_result = self.perception.process(frame)
                    # perception_result should be a PerceptionOutput object
                    # (convert to dict for logging if needed)
                    if hasattr(perception_result, '__dict__'):
                        perception_dict = perception_result.__dict__
                    else:
                        perception_dict = perception_result

                    # --- Step 2: Control decision ---
                    control_cmd = self.control.decide(perception_result)
                    # control_cmd may be a ControlCommand object or dict; we log as dict
                    if hasattr(control_cmd, '__dict__'):
                        cmd_dict = control_cmd.__dict__
                    else:
                        cmd_dict = control_cmd

                    # --- Step 3: Send command via serial ---
                    self.serial_io.send_command(control_cmd)

                    # Logging at DEBUG level
                    self.logger.debug(f"Perception: {perception_dict} | Control: {cmd_dict}")

            except Exception as e:
                self.logger.error(f"Exception in main loop: {e}. Safety protection triggered!", exc_info=True)
                self._send_emergency_brake()
                time.sleep(0.5)

            # --- Framerate control ---
            loop_duration = time.time() - loop_start
            sleep_time = self.target_loop_time - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif loop_duration > 0.3:
                self.logger.warning(
                    f"System lagging significantly! Loop took {loop_duration*1000:.1f}ms "
                    f"(May trigger board timeout protection)"
                )