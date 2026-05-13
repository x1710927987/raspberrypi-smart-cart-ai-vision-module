#!/usr/bin/env python3
"""
Smart Cart Control Application
智能小车控制应用 - 集成YOLO感知模块

运行位置：Raspberry Pi 或 本地开发电脑
用途：主程序，持续运行，协调感知->决策->控制循环
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from control.runtime import ControlCommand
from control.decision import DecisionEngine
from control.serial_comm import MockSerialSender, SerialCommandSender

from perception.camera_pipeline import PerceptionPipeline
from perception.runtime import PerceptionOutput

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class SmartCartApplication:
    """
    主应用：感知-决策-控制完整循环
    
    架构：
    ┌─────────────────────────────────────────────┐
    │      Camera Frame Input                     │
    └────────────────┬────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────┐
    │  YOLO Perception Pipeline                  │
    │  (同学的模块)                                │
    │  ├─ Object Detection                       │
    │  ├─ Traffic Light Detection                │
    │  ├─ Lane Segmentation                      │
    │  └─ Hazard Detection                       │
    └────────────────┬────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────┐
    │  PerceptionOutput (JSON Schema)            │
    │  {timestamp, objects[], laneseg, ...}     │
    └────────────────┬────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────┐
    │  Decision Engine (7-Layer Rules)           │
    │  ├─ Layer 1: Red Light Stop                │
    │  ├─ Layer 2: Emergency Brake               │
    │  ├─ Layer 3: Obstacle Avoidance            │
    │  ├─ Layer 4: Lane Keeping                  │
    │  ├─ Layer 5: Speed Control                 │
    │  ├─ Layer 6: Steering Smoothing            │
    │  └─ Layer 7: Manual Override               │
    └────────────────┬────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────┐
    │  ControlCommand (JSON Schema)              │
    │  {mode, v, steer, brake, reason}           │
    └────────────────┬────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────┐
    │  Serial Communicator                       │
    │  - CRC Checksum                            │
    │  - 115200 baud                             │
    │  - Heartbeat (5Hz)                         │
    └────────────────┬────────────────────────────┘
                     │
    ┌────────────────▼────────────────────────────┐
    │  Control Board (Motor, Steering, Brake)    │
    └─────────────────────────────────────────────┘
    """

    def __init__(
        self,
        *,
        camera_device: int = 0,
        serial_port: str = "/dev/ttyUSB0",  # Raspberry Pi
        serial_baudrate: int = 115200,
        use_mock_serial: bool = False,
        target_fps: int = 10,
    ):
        """
        初始化应用
        
        Args:
            camera_device: 摄像头设备ID (0 = /dev/video0)
            serial_port: 串口 (/dev/ttyUSB0 on RPi, COM3 on Windows)
            serial_baudrate: 波特率
            use_mock_serial: 是否使用模拟串口（测试）
            target_fps: 目标帧率
        """
        self.camera_device = camera_device
        self.serial_port = serial_port
        self.serial_baudrate = serial_baudrate
        self.use_mock_serial = use_mock_serial
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps

        # 组件初始化
        self.cap: Optional[cv2.VideoCapture] = None
        self.perception_pipeline: Optional[PerceptionPipeline] = None
        self.decision_engine: Optional[DecisionEngine] = None
        self.serial_comm: Optional[SerialCommandSender | MockSerialSender] = None

        # 统计信息
        self.frame_count = 0
        self.start_time = time.time()
        self.last_perception_time = 0.0
        self.last_decision_time = 0.0
        self.last_serial_time = 0.0

        logger.info(f"SmartCartApplication 初始化中...")
        logger.info(f"  摄像头: device={camera_device}")
        logger.info(f"  串口: {serial_port} @ {serial_baudrate} baud")
        logger.info(f"  模式: {'模拟串口' if use_mock_serial else '真实串口'}")
        logger.info(f"  目标帧率: {target_fps} FPS")

    def initialize(self) -> bool:
        """初始化所有组件"""
        try:
            # 1. 初始化摄像头
            logger.info("初始化摄像头...")
            self.cap = cv2.VideoCapture(self.camera_device)
            if not self.cap.isOpened():
                logger.error(f"无法打开摄像头 {self.camera_device}")
                return False
            
            # 设置摄像头分辨率和帧率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            logger.info("✅ 摄像头就绪")

            # 2. 初始化感知管道（YOLO）
            logger.info("初始化YOLO感知管道...")
            self.perception_pipeline = PerceptionPipeline.with_default_models()
            logger.info("✅ YOLO感知管道就绪")

            # 3. 初始化决策引擎
            logger.info("初始化决策引擎...")
            self.decision_engine = DecisionEngine()
            logger.info("✅ 决策引擎就绪")

            # 4. 初始化串口通信
            logger.info(f"初始化串口通信...")
            if self.use_mock_serial:
                self.serial_comm = MockSerialSender()
            else:
                self.serial_comm = SerialCommandSender(
                    port=self.serial_port,
                    baudrate=self.serial_baudrate,
                )
            if not self.serial_comm.connect():
                logger.error("无法连接串口")
                return False
            logger.info("✅ 串口通信就绪")

            logger.info("=" * 60)
            logger.info("✅ 所有组件初始化成功！")
            logger.info("=" * 60)
            return True

        except Exception as e:
            logger.error(f"初始化失败: {e}", exc_info=True)
            return False

    def process_frame(self) -> bool:
        """
        处理单帧图像
        
        返回：成功处理返回 True，失败或应该退出返回 False
        """
        try:
            frame_start = time.time()

            # 1. 读取摄像头帧
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("无法读取摄像头帧")
                return False

            # 2. 感知处理（YOLO）
            perception_start = time.time()
            perception_output: PerceptionOutput = self.perception_pipeline.process_frame(frame)
            self.last_perception_time = time.time() - perception_start

            # 3. 决策处理
            decision_start = time.time()
            control_cmd: ControlCommand = self.decision_engine.decide(perception_output)
            self.last_decision_time = time.time() - decision_start

            # 4. 串口通信
            serial_start = time.time()
            success = self.serial_comm.send_command(control_cmd)
            self.last_serial_time = time.time() - serial_start

            if not success:
                logger.warning("串口发送失败")
                return False

            # 5. 统计和日志
            self.frame_count += 1
            frame_time = time.time() - frame_start

            # 每 10 帧打印一次诊断
            if self.frame_count % 10 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed
                logger.info(
                    f"[Frame {self.frame_count}] "
                    f"FPS={fps:.1f} | "
                    f"perception={self.last_perception_time*1000:.1f}ms | "
                    f"decision={self.last_decision_time*1000:.1f}ms | "
                    f"serial={self.last_serial_time*1000:.1f}ms | "
                    f"reason={control_cmd.reason} | "
                    f"v={control_cmd.v:.2f}m/s steer={control_cmd.steer:.1f}°"
                )

            # 6. 控制帧率
            elapsed_time = time.time() - frame_start
            if elapsed_time < self.frame_time:
                time.sleep(self.frame_time - elapsed_time)

            return True

        except Exception as e:
            logger.error(f"处理帧失败: {e}", exc_info=True)
            return False

    def run(self) -> int:
        """
        主程序循环
        
        返回：退出代码
        """
        if not self.initialize():
            logger.error("初始化失败，退出")
            return 1

        logger.info("开始主程序循环... (Ctrl+C 退出)")

        try:
            while True:
                if not self.process_frame():
                    logger.warning("帧处理失败")
                    break

        except KeyboardInterrupt:
            logger.info("\n收到停止信号")

        except Exception as e:
            logger.error(f"运行时错误: {e}", exc_info=True)
            return 1

        finally:
            self.cleanup()

        logger.info("程序正常退出")
        return 0

    def cleanup(self):
        """清理资源"""
        logger.info("清理资源...")

        if self.cap is not None:
            self.cap.release()
            logger.info("✅ 摄像头已关闭")

        if self.serial_comm is not None:
            self.serial_comm.disconnect()
            logger.info("✅ 串口已关闭")

        # 打印最终统计
        if self.frame_count > 0:
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed
            logger.info("=" * 60)
            logger.info(f"运行统计:")
            logger.info(f"  总帧数: {self.frame_count}")
            logger.info(f"  运行时间: {elapsed:.1f}s")
            logger.info(f"  平均FPS: {avg_fps:.1f}")
            logger.info(f"  平均感知时间: {self.last_perception_time*1000:.1f}ms")
            logger.info(f"  平均决策时间: {self.last_decision_time*1000:.1f}ms")
            logger.info(f"  平均串口时间: {self.last_serial_time*1000:.1f}ms")
            logger.info("=" * 60)


def main():
    """
    程序入口
    
    使用方法：
    
    1. 本地开发（模拟串口）：
       python control/app.py --mock-serial
    
    2. Raspberry Pi（真实硬件）：
       python control/app.py --port /dev/ttyUSB0
    
    3. Windows 开发：
       python control/app.py --port COM3 --mock-serial
    """
    import argparse

    parser = argparse.ArgumentParser(description="Smart Cart AI Control Application")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="摄像头设备ID (默认: 0)",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyUSB0",
        help="串口端口 (默认: /dev/ttyUSB0 on RPi, COM3 on Windows)",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="串口波特率 (默认: 115200)",
    )
    parser.add_argument(
        "--mock-serial",
        action="store_true",
        help="使用模拟串口（用于测试）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="目标帧率 (默认: 10 FPS)",
    )

    args = parser.parse_args()

    # 创建应用
    app = SmartCartApplication(
        camera_device=args.camera,
        serial_port=args.port,
        serial_baudrate=args.baudrate,
        use_mock_serial=args.mock_serial,
        target_fps=args.fps,
    )

    # 运行
    exit_code = app.run()
    return exit_code


if __name__ == "__main__":
    import sys
    sys.exit(main())

