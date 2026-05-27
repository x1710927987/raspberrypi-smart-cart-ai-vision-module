# Commands
> This file contains all the commands used to run the project

## Environment setup
> Run these commands each time you open a new terminal
``` bash
cd ~/raspberrypi-smart-cart-ai-vision-module
source .venv/bin/activate
```

## Capture images
> Run this command to capture 10 images
``` bash
python tools/run_perception_live_view.py \
  --camera-backend picamera2 \
  --device cpu \
  --fps 3 \
  --max-frames 10 \
  --no-window \
  --print-json-every 1 \
  --save-dir cache/live_view_frames \
  --save-every 1
```

## Real-time image recognition
> Run this command to start the real-time image recognition
``` bash
python tools/run_perception_live_view.py \
  --camera-backend picamera2 \
  --device cpu \
  --fps 3 \
  --save-dir cache/live_view_frames \
  --save-every 30
```