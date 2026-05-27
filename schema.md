# AI Vision Module Schema & Interface Contract (V0)

Version: V0.1 (frozen)

Scope: Defines the data schemas and wire protocol between the AI vision module
and the underlying control board, plus the in-process data structures shared
across perception and control.

Current target hardware:

- Raspberry Pi 5.
- Raspberry Pi Camera Rev 1.3 connected through the CSI ribbon interface.
- Picamera2 is the default camera backend for the CSI camera.
- OpenCV BGR frames are the in-process image convention.

## 1. Common Conventions

- **Time unit**: seconds (float). `timestamp` is UNIX epoch time with
  millisecond precision.
- **Confidence**: range `[0.0, 1.0]`.
- **Coordinates (image)**: origin at the top-left. `x` increases rightward,
  `y` increases downward. Unless noted, bbox values are in camera pixel
  coordinates.
- **Modes**: `auto` or `manual`.
- **Safety**: if a valid command is not received for more than `300 ms`, the
  underlying board must brake to a safe stop.

## 2. Perception Output (JSON, In-Process)

Produced each frame by the perception runtime and consumed by control/decision.
All top-level keys are present; nullable blocks use `null` when unavailable.

```json
{
  "timestamp": 1719999999.123,
  "laneseg": {"mask_id": 3, "conf": 0.92, "bbox": [10.0, 20.0, 300.0, 420.0]},
  "objects": [
    {"cls": "pedestrian", "bbox": [120.0, 180.0, 200.0, 360.0], "conf": 0.86}
  ],
  "traffic_light": {"state": "red", "conf": 0.88, "bbox": [500.0, 40.0, 540.0, 120.0]},
  "hazard": {"type": "pothole", "conf": 0.81, "bbox": [260.0, 360.0, 400.0, 430.0]}
}
```

Field definitions:

- `laneseg.mask_id` (int): identifier of the selected sidewalk/drivable-area
  segmentation region. `bbox` is optional and used mainly for live-view
  visualization.
- `objects[]`:
  - `cls` (str): object class. V0.1 accepted classes are `pedestrian`,
    `obstacle`, `roadblock`, `bicycle`, `car`, `animal`, `stroller`,
    `wheelchair`, `bollard`, `scooter`, and `unknown`.
  - `bbox` (float[4]): `[x1, y1, x2, y2]` in pixels. Require `x2 > x1` and
    `y2 > y1`.
  - `conf` (float): detection confidence.
- `traffic_light.state` (enum): `red`, `yellow`, `green`, `off`, `flashing`,
  or `unknown`. `bbox` is optional but present when the traffic-light detector
  returns a matched box.
- `hazard.type` (enum): `pothole`, `curb`, `step_up`, `step_down`,
  `speed_bump`, `water`, `debris`, or `unknown`. `bbox` is optional but present
  when the hazard detector returns a matched box.

Current model coverage:

- The default objects model currently emits `pedestrian`, `bicycle`, `car`,
  `scooter`, and `roadblock`.
- The default traffic-light model currently emits `red`, `yellow`, and `green`.
- The default laneseg model currently emits the selected sidewalk/drivable-area
  region as `LaneSeg(mask_id=1, conf=...)`.
- The default hazard model currently emits `pothole` and `curb`. Water-filled
  potholes or similar ground pits are treated as `pothole`; there is no
  separate `water` class in the current default model.

Constraints and notes:

- All confidences must be in `[0.0, 1.0]`.
- Empty arrays are allowed.
- Nullable blocks may be `null`.
- Additional fields may be added; consumers must ignore unknown fields for
  forward compatibility.

### Example

```json
{
  "timestamp": 1720000123.456,
  "laneseg": {"mask_id": 1, "conf": 0.90},
  "objects": [
    {"cls": "pedestrian", "bbox": [120.0, 180.0, 200.0, 360.0], "conf": 0.84},
    {"cls": "roadblock", "bbox": [320.0, 220.0, 360.0, 280.0], "conf": 0.77}
  ],
  "traffic_light": {"state": "green", "conf": 0.85, "bbox": [500.0, 40.0, 540.0, 120.0]},
  "hazard": null
}
```

## 3. Control Command (JSON, In-Process)

Produced by control/decision and consumed by the I/O layer to transmit or
actuate.

```json
{
  "mode": "auto",
  "v": 0.60,
  "steer": -5.0,
  "brake": false,
  "reason": "rule_avoid",
  "timestamp": 1720000123.567
}
```

Constraints and notes:

- Velocity `v` is non-negative.
- Steering `steer` is left-positive; see Section 7.
- If `brake == true`, the underlying board must stop regardless of `v`.
- Additional keys are allowed; consumers ignore unknown keys.
- The I/O encoding layer must apply the following before transmission:
  - `v`: clip to `[0.0, 1.2] m/s`, quantize to `0.05 m/s`, and format with
    3 decimals.
  - `steer`: left-positive, clip to `[-30.0, 30.0] deg`, and format with
    1 decimal.

## 4. Serial Protocol (AI <-> Control Board)

ASCII CSV frames are terminated by `\n`, with CRC appended as a two-digit
uppercase hex value. CRC is the ASCII byte sum modulo 256 over the payload
(`CMD`/`STAT` plus fields), excluding the trailing CRC and newline.

Grammar:

```text
<frame> := <payload> "," <CRC> "\n"
<payload> := ("CMD" | "STAT") ("," <field>)*
<CRC> := two uppercase hex digits of (sum of ASCII bytes of <payload>) mod 256
```

### 4.1 Downlink (AI -> Board)

```text
CMD,<TS>,<v>,<steer>,<brake>,<mode>,<CRC>\n
```

- `<TS>`: seconds with 3 decimals, for example `1720000123.567`.
- `<v>`: m/s with 3 decimals.
- `<steer>`: degrees with 1 decimal.
- `<brake>`: `1` or `0`.
- `<mode>`: `auto` or `manual`.

Example:

```text
CMD,1720000123.567,0.600,-5.0,0,auto,EA\n
```

### 4.2 Uplink (Board -> AI)

```text
STAT,<TS>,<voltage>,<temp>,<err>,<CRC>\n
```

- `<TS>`: seconds with 3 decimals.
- `<voltage>`: volts with 2 decimals.
- `<temp>`: degrees Celsius with 1 decimal.
- `<err>`: integer error code; see Section 5.

Example:

```text
STAT,1720000123.700,12.10,36.5,0,24\n
```

### 4.3 Validation

- `verify(line)` must succeed before accepting a frame. Invalid frames are
  dropped.
- Heartbeat: AI sends `CMD` at least `5 Hz`.
- If the board misses valid `CMD` frames for more than `300 ms`, it brakes to a
  safe stop.
- The board may send `STAT` at least `5 Hz` for diagnostics.

## 5. Error Codes (Uplink `<err>`)

Proposed as a bitmask and still to be confirmed with the lower-level control
board owner.

- `0`: OK
- `1`: COMM_LOSS
- `2`: CAMERA_FAIL
- `4`: OVER_TEMP
- `8`: LOW_VOLTAGE
- `16`: MOTOR_FAULT

## 6. Configuration Keys (`deploy/config.yaml`)

Current V0.1 keys:

- `runtime.target_fps_min`: minimum expected control-loop FPS.
- `runtime.safety_brake_on_lost_ms`: fail-safe timeout in milliseconds.
- `runtime.mock_mode`: whether to use mock serial behavior in deployment tests.
- `perception.camera_backend`: `picamera2`, `opencv`, or `auto`. Use
  `picamera2` for Raspberry Pi Camera Rev 1.3 on Raspberry Pi 5.
- `perception.camera_index`: camera index, mainly used by OpenCV/USB cameras.
- `perception.camera_width`, `perception.camera_height`: requested capture
  resolution.
- `perception.camera_fps`: requested capture FPS; `null` lets Picamera2 choose.
- `perception.camera_warmup_seconds`: delay after camera start before first
  capture.
- `perception.camera_read_timeout_seconds`: frame-read timeout.
- `perception.camera_stop_timeout_seconds`: camera stop/close timeout.
- `perception.pixel_format`: Picamera2 pixel format. Current default:
  `RGB888`.
- `perception.camera_color_order`: channel order returned by camera source
  before OpenCV processing. Current default: `bgr`.
- `perception.device`: model inference device, usually `cpu` on Raspberry Pi.
- `control.serial_port`, `control.baud_rate`: serial output settings used by
  the control app.
- `serial.port`, `serial.baud`: serial settings used by deployment service
  compatibility code.
- `logging.level`, `logging.log_dir`: logging configuration.

Recommended Raspberry Pi 5 camera block:

```yaml
perception:
  camera_backend: "picamera2"
  camera_index: 0
  camera_width: 640
  camera_height: 480
  camera_fps:
  camera_warmup_seconds: 1.0
  camera_read_timeout_seconds: 2.0
  camera_stop_timeout_seconds: 2.0
  pixel_format: "RGB888"
  camera_color_order: "bgr"
  device: "cpu"
```

## 7. Frozen Conventions (V0.1)

- **BBox coordinates**: pixel coordinates with top-left origin, not normalized.
- **Steer sign and range**: left-positive; clip to `[-30.0 deg, 30.0 deg]`;
  1 decimal precision on wire.
- **Speed bounds**: clip to `[0.0, 1.2] m/s`; quantize to `0.05 m/s`; V0.1
  does not constrain acceleration or jerk.
- **Object classes**: `pedestrian`, `obstacle`, `roadblock`, `bicycle`, `car`,
  `animal`, `stroller`, `wheelchair`, `bollard`, `scooter`, `unknown`.
- **Traffic-light states**: `red`, `yellow`, `green`, `off`, `flashing`,
  `unknown`.
- **Hazard types**: `pothole`, `curb`, `step_up`, `step_down`, `speed_bump`,
  `water`, `debris`, `unknown`. Current default model only emits `pothole` and
  `curb`.
- **Serial details**: `\n` line ending; ASCII encoding; 115200 baud; CRC =
  ASCII sum modulo 256, encoded as two uppercase hex digits.
- **Heartbeat and fail-safe**: AI downlink at least `5 Hz`; if more than
  `300 ms` passes without valid `CMD`, the board brakes to a safe stop. The
  board is the fail-safe authority.
- **Voltage/temp warning ranges**: approximately `9-16 V` and `0-80 deg C`,
  to be finalized with hardware owners.
- **Mode values**: only `auto` and `manual` in V0.1.
- **Forward compatibility**: consumers must ignore unknown JSON keys.

## 8. Acceptance Criteria (DoD) for V0

- Perception and control can exchange the JSON payloads without schema errors.
- Raspberry Pi 5 can capture frames from Raspberry Pi Camera Rev 1.3 through
  Picamera2.
- The live-view tool can display or save annotated frames with detection boxes
  and labels.
- Board accepts `CMD` frames and applies speed/steer/brake correctly.
- AI validates `STAT` frames and logs anomalies.
- Heartbeat and fail-safe behavior are verified in bench tests.

V0.1 is frozen for current-stage integration. Future schema changes should be
versioned and accompanied by tests.
