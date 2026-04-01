# AI Vision Control Module for Elderly & Children Sidewalk Smart Cart
This repository contains the development work of the AI vision control module for a single-person low-speed three-wheeled/four-wheeled electric smart cart, designed for daily sidewalk travel of the elderly and children. It is the core supporting module of the underlying switch control board, and does not involve hardware development of the bottom control board.

## Project Overview
The project is developed based on a Raspberry Pi (or other development boards with AI inference capability) and Python programming language. It realizes real-time road condition image collection via the cart's camera, intelligent analysis of the sidewalk environment, and vehicle driving instruction output.

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
| Category | Details |
|----------|---------|
| Core Hardware | Raspberry Pi (with AI inference capability), compatible camera module |
| Programming Language | Python 3.x |
| Computer Vision Library | OpenCV |
| AI Inference Frameworks | PyTorch / TensorFlow Lite |
| Lightweight Vision Models | YOLOv8-tiny, MobileNet |
| Communication | Serial communication with underlying control board |

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
