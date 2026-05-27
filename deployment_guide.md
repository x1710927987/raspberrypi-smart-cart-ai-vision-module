# Raspberry Pi 5 AI Vision Deployment Guide

本指南用于在 Raspberry Pi 5 上部署本项目的 AI 视觉识别模型和感知控制集成入口。目标是先完成“软部署验证”：模型能加载、图片能推理、摄像头能打开、控制主循环能在 mock 串口下运行。确认这些都正常后，再接入真实控制板串口。

远程仓库：

```text
https://github.com/x1710927987/raspberrypi-smart-cart-ai-vision-module.git
```

## 1. 部署前准备

### 1.1 推荐硬件和系统

推荐配置：

```text
Raspberry Pi 5
Raspberry Pi OS 64-bit
8GB RAM 更稳，4GB 也可以测试
microSD 卡建议 64GB 或以上
USB 摄像头或 Raspberry Pi Camera
控制板串口连接线，例如 USB-TTL
稳定 5V/5A 电源
```

建议使用 64-bit 系统。AI 模型依赖 PyTorch 和 Ultralytics，32-bit 系统更容易遇到 wheel 不兼容或性能不足问题。

### 1.2 更新系统包

```bash
sudo apt update
sudo apt upgrade -y
```

安装基础工具：

```bash
sudo apt install -y \
  git \
  git-lfs \
  python3 \
  python3-venv \
  python3-pip \
  python3-picamera2 \
  rpicam-apps \
  libgl1 \
  libglib2.0-0 \
  v4l-utils \
  htop
```

说明：

```text
git / git-lfs: 拉取代码和模型权重
python3-venv / pip: 创建 Python 运行环境
libgl1 / libglib2.0-0: OpenCV 常见运行依赖
v4l-utils: 检查 USB 摄像头设备
htop: 观察 CPU / 内存占用
```

## 2. 拉取项目代码和模型权重

### 2.1 克隆仓库

```bash
cd ~
git lfs install
git clone https://github.com/x1710927987/raspberrypi-smart-cart-ai-vision-module.git
cd raspberrypi-smart-cart-ai-vision-module
```

### 2.2 拉取 Git LFS 模型文件

本项目的 `.pt` 模型权重通过 Git LFS 管理。克隆后必须执行：

```bash
git lfs pull
```

检查 LFS 文件：

```bash
git lfs ls-files
```

检查模型权重是否真实存在：

```bash
ls -lh models/weights/
```

至少应该看到以下四个部署用模型：

```text
models/weights/smartcart_objects_yolov8n_combined_v3_pt_v1.pt
models/weights/smartcart_traffic_light_yolov8n_combined_v2_pt_v1.pt
models/weights/smartcart_laneseg_yolov8n_seg_roboflow_pt_v1.pt
models/weights/smartcart_hazard_yolov8n_roboflow_pt_v1.pt
```

如果看到的 `.pt` 文件只有几十或几百字节，通常说明拉到的是 LFS 指针文件，不是真实模型。重新执行：

```bash
git lfs install
git lfs pull
```

## 3. 创建 Python 运行环境

树莓派上建议使用 `venv`，不建议使用 conda。PyTorch 对 Python 版本比较敏感，建议使用系统自带 Python 3.11 或 Python 3.10。

创建虚拟环境：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

这里使用 `--system-site-packages` 是为了让虚拟环境能直接访问 Raspberry Pi OS 通过 apt 安装的 `picamera2`。

确认 Python 版本：

```bash
python --version
```

建议输出类似：

```text
Python 3.11.x
```

如果是过新的 Python 版本导致 `torch` 无法安装，建议换回 Raspberry Pi OS 默认 Python 版本，或使用官方支持的 Python 3.10/3.11 环境。

## 4. 安装运行依赖

### 4.1 安装基础依赖

```bash
python -m pip install \
  numpy \
  pyyaml \
  pyserial \
  opencv-python-headless
```

说明：

```text
numpy: 图像数组处理
pyyaml: 读取 deploy/config.yaml
pyserial: 串口通信
opencv-python-headless: 摄像头和图片读取，不安装 GUI 组件
```

### 4.2 安装 PyTorch 和 Ultralytics

先尝试普通安装：

```bash
python -m pip install torch torchvision ultralytics
```

验证：

```bash
python -c "import torch; print(torch.__version__)"
python -c "import ultralytics; print(ultralytics.__version__)"
```

如果成功，可以继续下一步。

如果 `torch` 安装失败，请优先查：

```text
Python 版本是否过新
系统是否为 64-bit
pip 是否已经更新
网络是否能访问 PyPI
```

可以用下面命令确认系统架构：

```bash
uname -m
getconf LONG_BIT
```

期望：

```text
aarch64
64
```

如果 `torchvision` 安装失败，但 `torch` 和 `ultralytics` 能正常导入，可以先继续做 smoke test；部分 YOLO 推理场景未必立即用到 `torchvision`。但正式部署前建议补齐。

## 5. 检查模型 manifest 和权重

本项目默认通过 manifest 加载模型，而不是在代码里写死 `.pt` 路径。

四个默认 manifest：

```text
models/training/smartcart_objects_yolov8n_combined_v3_pt_v1.manifest.json
models/training/smartcart_traffic_light_yolov8n_combined_v2_pt_v1.manifest.json
models/training/smartcart_laneseg_yolov8n_seg_roboflow_pt_v1.manifest.json
models/training/smartcart_hazard_yolov8n_roboflow_pt_v1.manifest.json
```

检查 manifest 和权重文件：

```bash
python tools/check_model_manifest.py \
  models/training/smartcart_objects_yolov8n_combined_v3_pt_v1.manifest.json \
  --task objects \
  --require-artifact

python tools/check_model_manifest.py \
  models/training/smartcart_traffic_light_yolov8n_combined_v2_pt_v1.manifest.json \
  --task traffic_light \
  --require-artifact

python tools/check_model_manifest.py \
  models/training/smartcart_laneseg_yolov8n_seg_roboflow_pt_v1.manifest.json \
  --task laneseg \
  --require-artifact

python tools/check_model_manifest.py \
  models/training/smartcart_hazard_yolov8n_roboflow_pt_v1.manifest.json \
  --task hazard \
  --require-artifact
```

每条命令应正常退出，不应出现文件不存在、sha256 不匹配、task 不匹配等错误。

## 6. 运行统一感知 smoke test

这是部署前最关键的一步。它会加载四个真实模型，并生成统一的 `PerceptionOutput`。

```bash
python tools/run_perception_pipeline_smoke.py --device cpu --limit 2
```

期望输出类似：

```text
status=ok
json=cache/evaluation/unified_pipeline_smoke_test.json
report=cache/evaluation/unified_pipeline_smoke_test.md
sample_count=2
samples_with_objects=...
samples_with_laneseg=...
samples_with_traffic_light=...
samples_with_hazard=...
```

如果本地没有测试图片，可以手动指定图片：

```bash
python tools/run_perception_pipeline_smoke.py \
  --device cpu \
  --image path/to/road_scene.jpg
```

smoke test 成功标准：

```text
打印 status=ok
没有 ImportError / RuntimeError
cache/evaluation/unified_pipeline_smoke_test.json 被写出
PerceptionOutput 能通过 runtime 校验
```

如果这一步失败，不要继续接控制板。

## 7. 检查摄像头

### 7.1 USB 摄像头

查看设备：

```bash
ls /dev/video*
v4l2-ctl --list-devices
```

通常 USB 摄像头是：

```text
/dev/video0
```

CSI 摄像头默认用 Picamera2 测试：

```bash
python -c "from picamera2 import Picamera2; cam=Picamera2(); cam.configure(cam.create_video_configuration(main={'format':'RGB888','size':(640,480)})); cam.start(); frame=cam.capture_array(); print(frame.shape); cam.stop(); cam.close()"
```

期望输出：

```text
True
```

### 7.2 Raspberry Pi Camera

如果使用 Raspberry Pi Camera，先测试系统是否能识别：

```bash
rpicam-hello --timeout 3000
```

如果 `rpicam-hello` 可用，但 OpenCV 的 `VideoCapture(0)` 不可用，说明 CSI 相机走的是 libcamera 栈。当前项目已经支持 Picamera2，CSI 摄像头直接使用 `--camera-backend picamera2`：

```text
1. CSI 排线摄像头：使用 --camera-backend picamera2。
2. USB 摄像头：使用 --camera-backend opencv --camera <index>。
3. 不确定时：使用 --camera-backend auto，程序会先尝试 Picamera2，再回退到 OpenCV。
```

阶段验收建议在 Raspberry Pi 5 上优先使用 Picamera2 跑 CSI 摄像头。

### 7.3 在 VNC 上查看实时识别画面

如果已经通过 VNC 打开 Raspberry Pi 桌面，可以运行实时可视化工具：

```bash
python tools/run_perception_live_view.py --camera-backend picamera2 --device cpu --fps 3
```

这个窗口会显示：

```text
摄像头实时画面
objects 检测框、类别标签、置信度
traffic_light / laneseg / hazard 当前状态文本；如果模型输出 bbox，也会显示对应框和标签
推理耗时和 FPS
```

退出方式：

```text
按 q
或按 Esc
```

如果摄像头不是 `0`，换成实际编号：

```bash
python tools/run_perception_live_view.py --camera-backend opencv --camera 1 --device cpu --fps 3
```

如果通过 SSH 远程检查、暂时没有 VNC 图形窗口，可以用无窗口模式跑几帧：

```bash
python tools/run_perception_live_view.py \
  --camera-backend picamera2 \
  --camera 0 \
  --device cpu \
  --no-window \
  --max-frames 5 \
  --print-json-every 1 \
  --save-dir cache/live_view_frames \
  --save-every 1
```

如果 VNC 中无法弹出窗口，请确认：

```text
当前是在 Raspberry Pi 桌面终端中运行，而不是纯 SSH 终端
VNC 已经连接到同一个桌面会话
OpenCV 使用的是 opencv-python-headless 时，部分系统可能不支持 imshow
```

如果 `cv2.imshow` 因为 headless OpenCV 不可用，可以先安装带 GUI 支持的 OpenCV，或继续使用 `--no-window` 做命令行验证：

```bash
python -m pip uninstall -y opencv-python-headless
python -m pip install opencv-python
```

## 8. 检查串口和权限

插上控制板后查看串口：

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

常见结果：

```text
/dev/ttyUSB0
/dev/ttyACM0
```

如果没有权限访问串口，把当前用户加入 `dialout`：

```bash
sudo usermod -aG dialout $USER
```

然后重启或重新登录：

```bash
sudo reboot
```

重启后确认：

```bash
groups
```

应该能看到：

```text
dialout
```

## 9. 先运行 mock 模式主服务

部署配置文件：

```text
deploy/config.yaml
```

初次部署时保持：

```yaml
runtime:
  mock_mode: true

perception:
  camera_backend: "picamera2"
  camera_index: 0
  camera_width: 640
  camera_height: 480
  camera_fps:
  camera_warmup_seconds: 1.0
  pixel_format: "RGB888"
  camera_color_order: "bgr"
  device: "cpu"

serial:
  port: "/dev/ttyUSB0"
  baud: 115200
```

运行部署服务：

```bash
python deploy/run.py --config deploy/config.yaml
```

mock 模式下：

```text
不会打开真实摄像头
不会向真实控制板发送串口命令
会使用 mock perception 和 mock serial 验证主循环
```

如果这个模式都无法启动，先修 Python 环境或入口代码，不要进入真实运行。

## 10. 使用真实摄像头和 mock 串口测试

如果想测试真实摄像头和真实四模型 pipeline，但暂时不接控制板，可以暂时使用 `control/app.py` 的 mock 串口入口：

```bash
python control/app.py --camera-backend picamera2 --mock-serial --fps 5
```

如果需要观察实时识别效果，优先使用可视化工具：

```bash
python tools/run_perception_live_view.py --camera-backend picamera2 --device cpu --fps 3
```

注意：首次加载四个 `.pt` 模型会比较慢。第一帧推理通常明显慢于后续帧，这是正常现象。

如果摄像头不是 `0`，改成实际编号：

```bash
python control/app.py --camera-backend opencv --camera 1 --mock-serial --fps 5
```

## 11. 切换到真实串口运行

确认以下条件都满足后，才能接真实控制板：

```text
统一感知 smoke test 通过
CSI 摄像头 Picamera2 测试通过
mock 主服务能启动
串口设备存在
当前用户有 dialout 权限
车辆处于安全状态
```

修改 `deploy/config.yaml`：

```yaml
runtime:
  mock_mode: false

perception:
  camera_backend: "picamera2"
  camera_index: 0
  camera_width: 640
  camera_height: 480
  camera_fps:
  camera_warmup_seconds: 1.0
  pixel_format: "RGB888"
  camera_color_order: "bgr"
  device: "cpu"

serial:
  port: "/dev/ttyUSB0"
  baud: 115200
```

如果你的串口是 `/dev/ttyACM0`，则改为：

```yaml
serial:
  port: "/dev/ttyACM0"
  baud: 115200
```

真实运行：

```bash
python deploy/run.py --config deploy/config.yaml
```

第一次真实运行建议：

```text
车轮离地
或断开电机动力
只观察串口命令和控制板响应
旁边保留人工断电手段
先低 FPS，例如 3 到 5 FPS
确认刹车命令逻辑正确后再逐步提高
```

## 12. 开机自启动可选配置

如果后续需要开机自动运行，可以创建 systemd 服务。

创建服务文件：

```bash
sudo nano /etc/systemd/system/smartcart.service
```

填入内容，注意把 `pi` 和项目路径改成实际用户和路径：

```ini
[Unit]
Description=Smart Cart AI Vision Control Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/raspberrypi-smart-cart-ai-vision-module
ExecStart=/home/pi/raspberrypi-smart-cart-ai-vision-module/.venv/bin/python deploy/run.py --config deploy/config.yaml
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

启用服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable smartcart.service
sudo systemctl start smartcart.service
```

查看日志：

```bash
sudo systemctl status smartcart.service
journalctl -u smartcart.service -f
```

调试阶段不建议一开始就启用自启动。先手动运行稳定后再配置。

## 13. 常见问题和处理方法

### 13.1 `git lfs pull` 后模型仍然很小

现象：

```text
.pt 文件只有几百字节
运行模型时报 invalid load key 或模型加载失败
```

处理：

```bash
git lfs install
git lfs pull
ls -lh models/weights/
```

如果仍然失败，检查当前仓库是否真的上传了 LFS 对象：

```bash
git lfs ls-files
```

### 13.2 `ModuleNotFoundError: No module named 'ultralytics'`

说明虚拟环境没激活或依赖没装。

处理：

```bash
source .venv/bin/activate
python -m pip install ultralytics
python -c "import ultralytics; print(ultralytics.__version__)"
```

### 13.3 `ModuleNotFoundError: No module named 'cv2'`

处理：

```bash
source .venv/bin/activate
python -m pip install opencv-python-headless
python -c "import cv2; print(cv2.__version__)"
```

如果 OpenCV 仍然报系统库问题：

```bash
sudo apt install -y libgl1 libglib2.0-0
```

### 13.4 `torch` 安装失败

优先检查：

```bash
python --version
uname -m
getconf LONG_BIT
```

建议：

```text
使用 64-bit Raspberry Pi OS
使用 Python 3.10 或 3.11
先升级 pip setuptools wheel
先安装 torch，再安装 ultralytics
```

命令：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision
python -m pip install ultralytics
```

如果仍然失败，需要根据当前 Raspberry Pi OS 和 Python 版本选择对应的 PyTorch wheel。不要在项目代码里绕过这个问题；先保证：

```bash
python -c "import torch; print(torch.__version__)"
```

能正常运行。

### 13.5 摄像头打不开

现象：

```text
Picamera2 无法启动，或 USB 摄像头的 cv2.VideoCapture(0).isOpened() 返回 False
```

检查设备：

```bash
ls /dev/video*
v4l2-ctl --list-devices
```

尝试换编号：

```bash
python tools/run_perception_live_view.py --camera-backend opencv --camera 1 --device cpu --no-window --max-frames 5
```

如果是 Raspberry Pi Camera：

```bash
rpicam-hello --timeout 3000
```

如果 `rpicam-hello` 正常但 OpenCV 不正常，请直接使用 `--camera-backend picamera2`；如果 USB 摄像头正常但 CSI 不正常，请改用 `--camera-backend opencv --camera <index>` 做临时验收。

### 13.6 Picamera2 抓帧超时

现象：
```text
Camera frontend has timed out
timed out reading frame from Picamera2
```

这通常是 libcamera 没有从 CSI 传感器拿到帧，不是 YOLO 模型或控制逻辑问题。先检查：
```bash
sudo fuser -v /dev/video* /dev/media*
rpicam-hello --timeout 5000
python -c "from picamera2 import Picamera2; cam=Picamera2(); cam.configure(cam.create_video_configuration(main={'format':'RGB888','size':(640,480)})); cam.start(); print(cam.capture_array().shape); cam.stop(); cam.close()"
```

处理建议：
```text
1. 确认没有其他 rpicam / python / VNC 摄像头进程占用相机。
2. 断电后重新插拔 CSI 排线，确认蓝色面和接口方向正确，卡扣压紧。
3. 优先用 rpicam-hello 验证硬件；如果 rpicam-hello 也超时，基本就是排线、接口或摄像头模块问题。
4. 如果只在本项目中超时，可临时增大 --camera-read-timeout 或降低分辨率，例如 --camera-width 320 --camera-height 240。
```

### 13.7 Picamera2 画面颜色异常

现象：
```text
rpicam-hello 预览颜色正常，但 run_perception_live_view.py 保存或显示的画面红蓝通道反了，例如人脸偏蓝。
```

处理建议：
```bash
python tools/run_perception_live_view.py \
  --camera-backend picamera2 \
  --camera-color-order bgr \
  --pixel-format RGB888 \
  --device cpu \
  --fps 3 \
  --max-frames 10 \
  --no-window \
  --save-dir cache/live_view_frames \
  --save-every 1
```

说明：
```text
项目内部的 OpenCV、YOLO 和可视化统一使用 BGR 图像。
Picamera2 请求格式和 capture_array 返回数组的实际通道顺序在不同系统配置下可能不完全符合直觉。
当前项目默认使用 pixel_format=RGB888、camera_color_order=bgr，也就是把 Picamera2 返回数组直接按 OpenCV BGR 处理。
如果仍然出现蓝脸、红蓝互换，请尝试把 --camera-color-order 改成 rgb。
```

### 13.8 串口权限不足

现象：

```text
Permission denied: '/dev/ttyUSB0'
```

处理：

```bash
sudo usermod -aG dialout $USER
sudo reboot
```

重启后：

```bash
groups
```

确认包含 `dialout`。

### 13.9 找不到 `/dev/ttyUSB0`

检查：

```bash
ls /dev/ttyUSB* /dev/ttyACM*
dmesg | tail -40
```

可能情况：

```text
设备实际是 /dev/ttyACM0
USB 线只能供电不能传数据
控制板没有上电
驱动没有识别 USB-TTL 芯片
```

如果实际是 `/dev/ttyACM0`，修改 `deploy/config.yaml`：

```yaml
serial:
  port: "/dev/ttyACM0"
  baud: 115200
```

### 13.10 推理很慢，FPS 不够

`.pt` 模型在 Raspberry Pi CPU 上可能较慢。先记录实际耗时：

```bash
python tools/run_perception_pipeline_smoke.py --device cpu --limit 2
```

观察输出中的：

```text
elapsed_ms
```

优化方向：

```text
1. 降低输入分辨率
2. 降低运行 FPS
3. 只在必要帧调用部分模型，例如红绿灯不必每帧跑
4. 导出 ONNX / NCNN / TFLite 等边缘端格式
5. 后续在 manifest 中登记优化后的模型
```

Ultralytics 在 Raspberry Pi 上常见优化方向是导出更适合边缘设备的格式，例如 NCNN：

```bash
python -m pip install "ultralytics[export]"
yolo export model=models/weights/smartcart_objects_yolov8n_combined_v3_pt_v1.pt format=ncnn
```

注意：导出后的模型还不能自动被当前 manifest 使用，需要后续补充 backend 或登记新的 manifest。当前阶段先用 `.pt` 跑通部署链路。

### 13.11 运行中温度过高

查看温度：

```bash
vcgencmd measure_temp
```

持续高负载建议：

```text
安装主动散热
降低 FPS
减少同时运行的模型数量
避免长时间满载无散热测试
```

### 13.12 `deploy/run.py` 无法导入项目模块

确保在项目根目录运行：

```bash
cd ~/raspberrypi-smart-cart-ai-vision-module
source .venv/bin/activate
python deploy/run.py --config deploy/config.yaml
```

不要在 `deploy/` 子目录中直接运行：

```bash
cd deploy
python run.py
```

这样可能导致 Python 模块搜索路径不对。

## 14. 部署验收清单

完成部署前逐项确认：

```text
[ ] Raspberry Pi OS 是 64-bit
[ ] 仓库已 clone 到树莓派
[ ] git lfs pull 已执行
[ ] 四个部署用 .pt 权重真实存在
[ ] Python venv 已创建并激活
[ ] cv2 / torch / ultralytics / yaml / serial 都能 import
[ ] check_model_manifest.py 四个 manifest 都通过
[ ] run_perception_pipeline_smoke.py 输出 status=ok
[ ] CSI 摄像头 Picamera2 测试通过，或 USB 摄像头 OpenCV 测试通过
[ ] run_perception_live_view.py 能在 VNC 中显示实时识别画面，或无窗口模式能输出 JSON
[ ] mock_mode=true 的 deploy/run.py 能启动
[ ] 串口设备存在
[ ] 用户有 dialout 权限
[ ] 实车运行前车轮离地或电机动力断开
```

## 15. 推荐部署顺序

严格按这个顺序执行，最稳：

```text
1. 克隆仓库和 LFS 权重
2. 创建 venv
3. 安装依赖
4. 验证 Python import
5. 验证 manifest 和权重
6. 跑统一感知 smoke test
7. 测试摄像头
8. 跑 VNC 实时可视化或无窗口 live view
9. 跑 mock 主服务
10. 检查串口
11. 车轮离地，切换真实串口
12. 小 FPS 实车联调
13. 记录 FPS、延迟、温度和控制表现
```

## 16. 参考链接

```text
PyTorch official site:
https://pytorch.org/

Ultralytics documentation:
https://docs.ultralytics.com/

Ultralytics Raspberry Pi guide:
https://docs.ultralytics.com/guides/raspberry-pi/
```
