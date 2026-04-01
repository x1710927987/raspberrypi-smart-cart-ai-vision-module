# AI Vision Module Schema & Interface Contract (V0)

Version: V0.1 (frozen)

Scope: Defines the data schemas and wire protocol between the AI vision module (running on Raspberry Pi) and the underlying control board, plus the in-process data structures shared across perception and control.

## 1. Common Conventions
- **Time unit**: seconds (float). `timestamp` is UNIX epoch with millisecond precision (>= 0.001s).
- **Confidence**: range [0.0, 1.0].
- **Coordinates (image)**: origin at the top-left. x increases rightward, y increases downward. Unless noted, bbox values are in camera pixel coordinates.
- **Modes**: `auto` or `manual`.
- **Safety**: If a valid command is not received for > 300 ms, underlying board must brake to safe stop.

## 2. Perception Output (JSON, in-process)
Produced each frame by perception runtime; consumed by control/decision.

Shape (all keys present; nullable where noted):

```json
{
  "timestamp": 1719999999.123,
  "laneseg": {"mask_id": 3, "conf": 0.92},             // nullable
  "objects": [
    {"cls": "pedestrian", "bbox": [x1, y1, x2, y2], "conf": 0.86}
  ],
  "traffic_light": {"state": "red", "conf": 0.88},  // nullable
  "hazard": {"type": "pothole", "conf": 0.81}       // nullable
}
```

Field definitions:
- `laneseg.mask_id` (int): Identifier of the drivable-area mask (implementation-defined indexing). `conf` confidence.
- `objects[]`:
  - `cls` (str): Object class. Baseline classes: `pedestrian`, `obstacle`, `bicycle`, `car`, `animal`, `stroller`, `wheelchair`, `bollard`, `scooter`, `unknown`.
  - `bbox` (float[4]): `[x1, y1, x2, y2]` in pixels. Require `x2 > x1`, `y2 > y1`.
  - `conf` (float): detection confidence.
- `traffic_light.state` (enum): `red` | `yellow` | `green` | `off` | `flashing` | `unknown`. `conf` confidence.
- `hazard.type` (enum): baseline `pothole` | `step_up` | `step_down` | `speed_bump` | `water` | `debris` | `unknown`. `conf` confidence.

Constraints and notes:
- All confidences in [0, 1]. Empty arrays allowed. Nullable blocks may be `null` when not available.
- Additional fields MAY be added; consumers MUST ignore unknown fields for forward compatibility.

### Example

```json
{
  "timestamp": 1720000123.456,
  "laneseg": {"mask_id": 2, "conf": 0.90},
  "objects": [
    {"cls": "pedestrian", "bbox": [120.0, 180.0, 200.0, 360.0], "conf": 0.84},
    {"cls": "obstacle", "bbox": [320.0, 220.0, 360.0, 280.0], "conf": 0.77}
  ],
  "traffic_light": {"state": "green", "conf": 0.85},
  "hazard": null
}
```

## 3. Control Command (JSON, in-process)
Produced by control/decision; consumed by I/O layer to transmit or actuate.

Shape:

```json
{
  "mode": "auto",          // "auto" | "manual"
  "v": 0.60,                // m/s
  "steer": -5.0,            // degrees
  "brake": false,           // true => immediate braking
  "reason": "rule_avoid",
  "timestamp": 1720000123.567
}
```

Constraints and notes:

- Velocity `v` ≥ 0. Steering `steer` range and sign convention see §7.
- If `brake == true`, underlying board must stop regardless of `v`.
- Additional keys allowed; consumers ignore unknowns.
- IO/encoding layer MUST apply the following prior to transmission:
  - `v`: clip to [0.0, 1.2] m/s, quantize to 0.05 m/s step, format with 3 decimals.
  - `steer`: left-positive, clip to [-30.0, 30.0] deg, format with 1 decimal.

## 4. Serial Protocol (AI <-> Control Board)
ASCII CSV frames terminated by `\n`, with CRC appended as 2-digit uppercase hex of ASCII-sum modulo 256 over the payload (prefix+fields, excluding the trailing CRC and `\n`).

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
- `<TS>`: seconds with 3 decimals (e.g., `1720000123.567`)
- `<v>`: m/s with 3 decimals
- `<steer>`: degrees with 1 decimal
- `<brake>`: `1` or `0`
- `<mode>`: `auto` or `manual`

Example:

```text
CMD,1720000123.567,0.600,-5.0,0,auto,EA\n
```

### 4.2 Uplink (Board -> AI)

```text
STAT,<TS>,<voltage>,<temp>,<err>,<CRC>\n
```
- `<TS>`: seconds with 3 decimals
- `<voltage>`: volts with 2 decimals
- `<temp>`: degrees Celsius with 1 decimal
- `<err>`: integer error code (see §5)

Example:

```text
STAT,1720000123.700,12.10,36.5,0,24\n
```

### 4.3 Validation
- `verify(line)` must succeed before accepting a frame. Invalid frames are dropped.
- Heartbeat: AI sends `CMD` ≥ 5 Hz. If the board misses valid `CMD` for > 300 ms, it brakes to safe stop.
- Board may send `STAT` ≥ 5 Hz for diagnostics.

## 5. Error Codes (uplink `<err>`)
Proposed as a bitmask (combine by sum). To be confirmed.
- `0`: OK
- `1`: COMM_LOSS
- `2`: CAMERA_FAIL
- `4`: OVER_TEMP
- `8`: LOW_VOLTAGE
- `16`: MOTOR_FAULT

## 6. Configuration Keys (deploy/config.yaml)
- `camera.device_index`, `camera.width`, `camera.height`, `camera.fps`
- `serial.port`, `serial.baudrate`, `serial.timeout_sec`, `serial.heartbeat_hz`
- `runtime.target_fps_min`, `runtime.safety_brake_on_lost_ms`
- `logging.level`, `logging.save_frames`, `logging.log_dir`

## 7. Frozen Conventions (V0.1)

- **BBox coordinates**: pixel coordinates (top-left origin), not normalized.
- **Steer sign & range**: left-positive; clip to [-30.0°, 30.0°]; 1 decimal precision on wire.
- **Speed bounds**: clip to [0.0, 1.2] m/s; quantize to 0.05 m/s; V0 does not constrain acceleration/jerk.
- **Object classes**: `pedestrian`, `obstacle`, `bicycle`, `car`, `animal`, `stroller`, `wheelchair`, `bollard`, `scooter`, `unknown`.
- **Traffic-light states**: `red`, `yellow`, `green`, `off`, `flashing`, `unknown`.
- **Hazard types**: `pothole`, `step_up`, `step_down`, `speed_bump`, `water`, `debris`, `unknown`.
- **Serial details**: `\n` line ending; ASCII encoding; 115200 baud; CRC = ASCII sum mod 256 (2-digit uppercase hex).
- **Heartbeat & fail-safe**: AI downlink ≥ 5 Hz; if > 300 ms without valid `CMD`, board brakes to safe stop; board is fail-safe authority.
- **Voltage/temp ranges**: 9–16 V; 0–80 °C for warnings.
- **Mode values**: only `auto` | `manual` in V0.1.
- **Forward-compatibility**: consumers MUST ignore unknown JSON keys.

## 8. Acceptance Criteria (DoD) for V0
- Perception and control can exchange the JSON payloads without schema errors.
- Board accepts `CMD` frames and applies speed/steer/brake correctly.
- AI validates `STAT` frames and logs anomalies.
- Heartbeat and fail-safe behavior verified in bench tests.

---
If你确认以上不确定项，我会据此冻结 V0.1 并更新样例与单测。需要我顺带提供正式的 JSON Schema（Draft 07）文件吗？

