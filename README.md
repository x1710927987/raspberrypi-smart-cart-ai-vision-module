# AI Vision Control Module for Elderly & Children Sidewalk Smart Cart

This repository contains the development work of the AI vision control module for a single-person low-speed three-wheeled/four-wheeled electric smart cart, designed for daily sidewalk travel of the elderly and children. It is the core supporting module of the underlying switch control board, and does not involve hardware development of the bottom control board.

## Project Overview

The project is developed based on Raspberry Pi 5 and Python. The current stage uses a Raspberry Pi Camera Rev 1.3 CSI camera through Picamera2 for live image capture, runs four YOLO-based perception models, and outputs a unified `PerceptionOutput` for downstream control logic.

The cart is limited to low-speed driving on sidewalks, with the core goal of realizing safe AI-assisted obstacle avoidance driving, while retaining full manual takeover authority to ensure the safety of elderly and child users throughout the use.

This project is a 8-week (2-month) development program for sophomore interns majoring in Computer Science, Automation, Artificial Intelligence and related fields, focusing on AI algorithm deployment and vehicle control logic implementation.

## Core Features

- Real-time sidewalk area recognition and lane keeping control
- Static/dynamic obstacle (pedestrians, roadblocks, etc.) detection and active steering obstacle avoidance
- Traffic light recognition and automatic start/stop response
- Sudden road condition (steps, potholes, etc.) prediction and emergency braking
- Seamless switching between AI-assisted driving and manual takeover
- Abnormal protection: automatic parking when recognition fails or communication is interrupted
- Real-time image preprocessing and multi-target simultaneous detection

## Tech Stack

| Category                  | Details                                                               |
| ------------------------- | --------------------------------------------------------------------- |
| Core Hardware             | Raspberry Pi 5, Raspberry Pi Camera Rev 1.3 CSI camera                |
| Programming Language      | Python 3.11 on Raspberry Pi; Python 3.14 is used in local training    |
| Computer Vision Library   | OpenCV, Picamera2                                                     |
| AI Inference Frameworks   | PyTorch / Ultralytics YOLO; ONNX/TFLite can be added later            |
| Lightweight Vision Models | YOLOv8n detection and segmentation models                             |
| Communication             | Serial communication with underlying control board                    |

## Development Cycle & Phased Plan (8 Weeks)

The whole development is divided into 4 phases, with clear task nodes and deliverables for each phase:

### Phase 1: Project Preparation & Technical Learning (Weeks 1-2)

- Complete project requirement disassembly and technical disclosure, clarify the communication rules between the AI module and the underlying control board
- Confirm the technical implementation route and division of labor for 2 interns (model deployment & control logic development)
- Complete Raspberry Pi development environment setup, including Python, OpenCV, AI inference framework installation
- Complete AI vision model selection and dataset collection plan formulation
- **Deliverables**: Requirement analysis document, division plan, environment configuration document, model selection report

### Phase 2: Environment Optimization & Data Collection & Preprocessing (Weeks 3-4)

- Complete hardware connection between Raspberry Pi, camera, power supply and communication module, test real-time image acquisition
- Optimize the development environment and AI inference acceleration to meet the real-time response requirements of cart driving
- Build a stable communication channel with the underlying control board, and define a unified instruction format
- Collect and preprocess sidewalk scene image dataset, complete data annotation, cleaning and enhancement
- **Deliverables**: Optimized environment configuration document, dataset file, camera acquisition & basic communication code, hardware test report

### Phase 3: AI Vision Recognition Model Deployment & Debugging (Weeks 5-6)

- Train and optimize lightweight target detection and semantic segmentation models based on the preprocessed dataset
- Transplant the trained model to Raspberry Pi, complete deployment and inference test
- Debug and optimize each recognition module (obstacle, sidewalk, traffic light, sudden road condition)
- Improve recognition accuracy and response speed, adapt to complex sidewalk environments
- **Deliverables**: Full AI vision recognition Python code (with detailed comments), trained model files, recognition accuracy test report, debugging logs

### Phase 4: Control Logic Development & Whole Cart Joint Debugging (Weeks 7-8)

- Develop comprehensive road condition analysis logic and safe driving rule judgment program
- Develop vehicle control instruction output program, realize speed regulation, steering and start/stop control
- Complete manual takeover program development, realize seamless mode switching
- Add exception protection logic, carry out whole cart real vehicle joint debugging and multi-scenario test
- Sort out full project documents and complete final delivery
- **Deliverables**: Complete AI control program, model files, joint debugging test video, full project documents, internship summary report

## Repository Structure

```text
control/          Decision logic and control application entry point
deploy/           Deployment service entry point and runtime config
io_camera/        Camera backend abstraction for Picamera2 and OpenCV
perception/       Perception runtime, fusion, preprocessing, and model providers
models/           Model manifests, training configs, and Git LFS weights
tools/            Dataset, training, evaluation, smoke-test, and live-view tools
tests/            Unit and integration tests for perception, control, and tools
data/             Local datasets; ignored by Git except placeholders/docs
cache/            Local evaluation, smoke-test, and temporary outputs
docs/             Deployment guides, command notes, and requirement documents
demos/            Demo/acceptance reports and handoff summaries
```

## Current Default Models

The current integrated perception chain uses four default manifests:

```text
models/training/smartcart_objects_yolov8n_combined_v3_pt_v1.manifest.json
models/training/smartcart_traffic_light_yolov8n_combined_v2_pt_v1.manifest.json
models/training/smartcart_laneseg_yolov8n_seg_roboflow_pt_v1.manifest.json
models/training/smartcart_hazard_yolov8n_roboflow_pt_v1.manifest.json
```

Hazard recognition keeps the original two-class scheme:

```text
pothole, curb
```

In this project, water-filled potholes or similar ground pits are handled by the
`pothole` hazard class. A separate `water` model class is not used in the current
stage.

## Model Change Acceptance Workflow

After registering or switching any perception model, run the unified perception
pipeline smoke test before treating the model as ready for integration. This
checks that the default manifests, model artifacts, backend loading,
postprocessing, fusion, JSON serialization, and `PerceptionOutput` validation
all work together.

```powershell
conda activate smartcart-ai
python tools\run_perception_pipeline_smoke.py --device 0
```

For CPU-only validation, use:

```powershell
conda activate smartcart-ai
python tools\run_perception_pipeline_smoke.py --device cpu
```

The command writes:

```text
cache/evaluation/unified_pipeline_smoke_test.json
cache/evaluation/unified_pipeline_smoke_test.md
```

The model change is acceptable when the command reports `status=ok` and every
saved `PerceptionOutput` can be read back and validated.

For Raspberry Pi or VNC visual validation, run the live camera viewer after the
smoke test passes. On the current Raspberry Pi 5 + Raspberry Pi Camera Rev 1.3
CSI setup, use Picamera2:

```bash
python tools/run_perception_live_view.py --camera-backend picamera2 --device cpu --fps 3
```

It draws object boxes and labels on live camera frames. It also draws
traffic-light, laneseg, and hazard boxes when the active model output includes
`bbox`, and shows FPS and inference latency status.
Use `--camera-backend opencv --camera 0` for a USB camera.
Add `--save-dir cache/live_view_frames --save-every 1` to save annotated
frames during an SSH or VNC check.

## Deployment Documents

Use these files for the current Raspberry Pi 5 handoff:

```text
docs/deployment_guide.md           Step-by-step Raspberry Pi 5 deployment guide
docs/commands.md                   Common commands for setup, smoke tests, live view, and control checks
docs/development_requirements.docx Original requirement/development planning document
demos/perception_delivery_report.md Perception model inventory, metrics, limitations, and acceptance checklist
schema.md                         PerceptionOutput and serial/control interface contract
```

`docs/deployment_guide.md` is the main document for deploying on Raspberry Pi 5
with Raspberry Pi Camera Rev 1.3. `docs/commands.md` is the shorter command
cheat sheet. `demos/perception_delivery_report.md` records the current model
inventory, evaluation metrics, known limitations, and stage-acceptance evidence.
