# Native 50Hz G1 deployment

This path uses the same `PlanTimeline` as simulation. Startup history contains
10 measured 50Hz qpos36 samples. Configure `--switch-seconds` on **both** the
GPU server and the Orin bridge; the default is 1.0s. Valid values are 0.5–2.0s
in 20ms increments. Protocol metadata carries the period, and mismatches are
rejected before inference, including initialization.

For period P frames, request at P−25 with raw history [P−10:P], and switch at P.
The reply must be accepted before consuming phase P−11: ten preview frames plus
one derivative sample. The total inference/transport budget stays 280ms.

| Switch period | Request frame | History frames | Switch frame | Preview deadline |
|---|---:|---|---:|---:|
| 0.5s | 0 | 15–24 | 25 | 14 |
| 0.76s | 13 | 28–37 | 38 | 27 |
| 1.0s (default) | 25 | 40–49 | 50 | 39 |
| 2.0s | 75 | 90–99 | 100 | 89 |

The Python API uses the same immutable `Schedule(switch_seconds=...)` in
`run`, `PlanTimeline`, `RemotePlanner`, `PlannerService`, and `DeploymentBridge`.
Prediction remains 100 frames and history remains 10 frames regardless of period.

## GPU planner

```bash
PYTHONPATH=src python -m omg.cli.realtime.planner_server \
  --config /path/to/resolved_training_config.yaml \
  --checkpoint /path/to/native50.ckpt --bind tcp://0.0.0.0:5571 --switch-seconds 1.0
```

Strict checkpoint loading, architecture contract validation, history10,
horizon100, 50 DDIM steps, CFG2.5. Native checkpoints carry their representation;
Use the matching 38D checkpoint and statistics. No DiT skip/cache approximation is enabled.

## Orin bridge

```bash
PYTHONPATH=src python -m omg.cli.realtime.holomotion_real_bridge \
  --connect tcp://GPU_HOST:5571 --text 'walk forward' \
  --publish-bind ipc:///tmp/omg-preview-v2 --switch-seconds 1.0
```

The bridge waits for measured history from an **already armed** HoloMotion
consumer. Initial inference happens outside the active tracking clock. During
tracking, generation runs in a worker and the 50Hz clock never waits for it.
A late plan or missed control tick terminates the stream; the consumer must
revoke tracking on timeout. No hold-last-reference recovery is implemented.

## HoloMotion consumer integration

Import `deployment/holomotion_hook.py` into the existing Orin controller and call
`run_armed_tracker` after its existing remote-B gate. Supply the initialized
HoloMotion session/metadata/handles and these concrete controller callbacks:

- `read_state()`: fresh `joint_pos[29]`, `joint_vel[29]` in G1 order,
  pelvis `root_quat_wxyz[4]`, body `gyro[3]`; reject stale feedback.
- `apply_action(action)`: existing ONNX-order action-to-PD/motor path.
- `stop_tracking()`: existing disarm/stop path, always called on exit.
- `measured_history[10,36]`: native50 startup history from the state estimator.
  Unitree LowState alone has no root translation. Do not substitute nominal
  height or planned root position and label it measured history.

The adapter emits the same clipped policy action as the simulator (clip10,
matching HoloMotion's configured action limit). It does not publish LowCmd or
replace the robot controller's gains, joint mapping, arming or emergency stop.
This hook must be integrated in that external controller; stock HoloMotion's
legacy `obs65` subscriber is **not** compatible.

Each Orin-local v2 packet contains the preceding sample, current sample, ten
future samples and one derivative sample. The consumer computes the exact
reference features used in simulation and builds observations from live state.
Packets are ordered and age-checked against the same-host monotonic clock.
No future frames are synthesized by repeating the current reference.

## Verification boundary

`tests/pipeline/test_native_schedule.py` checks scheduling, missed deadlines,
preview derivative support and simulated30/50/70ms latency equivalence.
`scripts/validation/native50_sim.py` exercises the deployed observation/action
adapter with the same MuJoCo dynamics as the normal tracker. Actual network
latency, Orin throughput, controller integration and physical acceptance still
require deployment-side verification. Do not interpret simulated latency as
measured inference speed.

The concrete ROS/Orin adapters and pinned external-controller stop patch are
in [`deployment/README.md`](../deployment/README.md). That document separates
completed network/isolated checks from the still-pending shadow and physical
acceptance gates. The native policy adapter is not yet hardware-qualified.

Configurable-period validation (2026-10-21): 219 tests passed (one existing
PyTorch scheduler deprecation warning). Schedule and ZMQ transport tests cover
0.5, 0.76, 1.0 and 2.0 seconds; deployment preview parity covers simulated
30/50/70/210ms inference delays. This validates scheduling and protocol behavior,
not new motion-quality benchmarks or physical deployment acceptance.

