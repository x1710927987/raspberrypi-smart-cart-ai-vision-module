#!/usr/bin/env python3
"""
Quick integration test - YOLO感知 + 控制系统端到端测试

这个脚本直接演示如何集成同学的YOLO模型和你的控制系统
运行位置：项目根目录（本地或Raspberry Pi）
"""

import logging
import time
import cv2
import numpy as np
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_perception_only():
    """测试 1：仅感知模块（YOLO）"""
    print("\n" + "=" * 70)
    print("测试 1：YOLO 感知模块")
    print("=" * 70)

    try:
        from perception import PerceptionPipeline, make_mock_perception

        # 加载 YOLO 模型
        logger.info("加载YOLO物体检测模型...")
        perception = PerceptionPipeline.with_default_objects()
        logger.info("✅ YOLO模型已加载")

        # 用模拟数据测试
        for scenario in ["clear_path", "pedestrian_ahead", "red_light"]:
            logger.info(f"\n测试场景: {scenario}")
            perception_output = make_mock_perception(scenario)
            
            print(f"  检测到的物体: {len(perception_output.objects)} 个")
            for obj in perception_output.objects:
                print(f"    - {obj.cls} (confidence={obj.conf:.2f})")
            
            if perception_output.traffic_light:
                print(f"  交通灯: {perception_output.traffic_light.state} "
                      f"(confidence={perception_output.traffic_light.conf:.2f})")
            
            if perception_output.laneseg:
                print(f"  车道分割: mask_id={perception_output.laneseg.mask_id} "
                      f"(confidence={perception_output.laneseg.conf:.2f})")
            
            if perception_output.hazard:
                print(f"  危险检测: {perception_output.hazard.type} "
                      f"(confidence={perception_output.hazard.conf:.2f})")

        print("\n✅ 感知模块测试完成！")
        return True

    except Exception as e:
        logger.error(f"感知测试失败: {e}", exc_info=True)
        return False


def test_decision_only():
    """测试 2：仅决策模块"""
    print("\n" + "=" * 70)
    print("测试 2：决策引擎")
    print("=" * 70)

    try:
        from control.decision import DecisionEngine
        from perception import make_mock_perception

        logger.info("初始化决策引擎...")
        engine = DecisionEngine()
        logger.info("✅ 决策引擎已初始化")

        # 用各种场景测试决策
        scenarios = [
            ("clear_path", "绿灯，路线清晰"),
            ("red_light", "红灯停止"),
            ("pedestrian_ahead", "前方有行人"),
            ("obstacle_ahead", "前方有障碍"),
            ("road_hazard", "路上有坑洞"),
            ("mixed_risk", "多重风险"),
        ]

        for scenario, description in scenarios:
            logger.info(f"\n场景: {scenario} - {description}")
            perception = make_mock_perception(scenario)
            
            cmd = engine.decide(perception)
            
            print(f"  决策理由: {cmd.reason}")
            print(f"  速度: {cmd.v:.2f} m/s (最高 1.2)")
            print(f"  转向: {cmd.steer:.1f}° (范围 -30 ~ 30)")
            print(f"  制动: {'是' if cmd.brake else '否'}")
            print(f"  模式: {cmd.mode}")

        print("\n✅ 决策引擎测试完成！")
        return True

    except Exception as e:
        logger.error(f"决策测试失败: {e}", exc_info=True)
        return False


def test_perception_to_decision():
    """测试 3：感知 -> 决策（完整集成）"""
    print("\n" + "=" * 70)
    print("测试 3：感知 + 决策完整流程")
    print("=" * 70)

    try:
        from perception import PerceptionPipeline, make_mock_perception
        from control.decision import DecisionEngine

        logger.info("初始化模块...")
        perception_pipeline = PerceptionPipeline.with_default_objects()
        decision_engine = DecisionEngine()
        logger.info("✅ 所有模块已初始化")

        # 模拟10帧处理
        logger.info("\n处理10帧模拟数据...")
        for i in range(10):
            # 轮流使用不同场景
            scenario = ["clear_path", "pedestrian_ahead", "red_light"][i % 3]
            
            # 1. 感知（这里用模拟，实际应该用摄像头）
            perception_output = make_mock_perception(scenario)
            
            # 2. 决策
            control_cmd = decision_engine.decide(perception_output)
            
            # 3. 显示结果
            print(f"[Frame {i+1}] {scenario:20s} -> "
                  f"理由={control_cmd.reason:20s} "
                  f"v={control_cmd.v:.2f}m/s "
                  f"brake={'✓' if control_cmd.brake else '✗'}")
            
            time.sleep(0.1)  # 模拟处理时间

        print("\n✅ 完整流程测试完成！")
        return True

    except Exception as e:
        logger.error(f"完整流程测试失败: {e}", exc_info=True)
        return False


def test_with_real_camera():
    """测试 4：真实摄像头 + YOLO 感知（如果有摄像头）"""
    print("\n" + "=" * 70)
    print("测试 4：真实摄像头 + YOLO 感知")
    print("=" * 70)

    try:
        import cv2
        from perception import PerceptionPipeline
        from control.decision import DecisionEngine

        logger.info("尝试打开摄像头...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.warning("⚠️  摄像头不可用，跳过此测试")
            return True

        logger.info("✅ 摄像头已打开")
        logger.info("初始化YOLO模型（第一次加载可能需要 30-60 秒）...")
        
        perception_pipeline = PerceptionPipeline.with_default_objects()
        decision_engine = DecisionEngine()
        
        logger.info("✅ 模型加载完成")
        logger.info("处理5帧真实视频... (按 Ctrl+C 停止)")

        for i in range(5):
            ret, frame = cap.read()
            if not ret:
                break

            # 感知处理
            start_time = time.time()
            perception_output = perception_pipeline.process_frame(frame)
            perception_time = time.time() - start_time

            # 决策
            start_time = time.time()
            control_cmd = decision_engine.decide(perception_output)
            decision_time = time.time() - start_time

            # 显示结果
            print(f"[Frame {i+1}] "
                  f"perception={perception_time*1000:.1f}ms "
                  f"decision={decision_time*1000:.1f}ms "
                  f"objects={len(perception_output.objects)} "
                  f"reason={control_cmd.reason}")

            time.sleep(0.5)  # 模拟处理

        cap.release()
        print("\n✅ 真实摄像头测试完成！")
        return True

    except Exception as e:
        logger.error(f"真实摄像头测试失败: {e}", exc_info=True)
        return False


def main():
    """运行所有集成测试"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  YOLO感知 + 控制系统 集成测试".center(68) + "║")
    print("║" + "=" * 68 + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  运行位置：项目根目录".ljust(68) + "║")
    print("║" + "  命令：python tests/integration_test.py".ljust(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    results = []

    # 运行各个测试
    tests = [
        ("感知模块（YOLO）", test_perception_only),
        ("决策引擎", test_decision_only),
        ("感知+决策集成", test_perception_to_decision),
        ("真实摄像头", test_with_real_camera),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "✅ 通过" if result else "❌ 失败"))
        except Exception as e:
            logger.error(f"测试异常: {e}")
            results.append((test_name, "❌ 异常"))

    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    for test_name, result in results:
        print(f"  {test_name:30s} {result}")

    print("=" * 70)

    # 所有测试通过
    if all("✅" in result for _, result in results):
        print("\n🎉 所有测试通过！")
        print("\n现在可以运行主程序：")
        print("  python control/app.py [选项]")
        print("\n选项：")
        print("  --camera 0          摄像头设备ID (默认: 0)")
        print("  --port /dev/ttyUSB0 串口端口")
        print("  --mock-serial       使用模拟串口（测试）")
        print("  --fps 10            目标帧率 (默认: 10)")
        return 0
    else:
        print("\n⚠️  某些测试失败，请检查错误日志")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
