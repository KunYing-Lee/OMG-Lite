# Native 50Hz G1 deployment

This path uses the same `PlanTimeline` as simulation. Startup history contains
10 measured 50Hz qpos36 samples. Each subsequent request uses raw plan 15–24,
starts at phase0, and replaces the reference suffix at phase25. The reply
must be available before phase14: ten preview frames plus one derivative sample.
The complete transport/inference budget is 280ms; switching happens every 500ms.

## GPU planner

```bash
PYTHONPATH=src python -m omg.cli.realtime.planner_server \
  --config /path/to/resolved_training_config.yaml \
  --checkpoint /path/to/native50.ckpt --bind tcp://0.0.0.0:5571
```

Strict checkpoint loading, architecture contract validation, history10,
horizon100, 50 DDIM steps, CFG2.5. Native checkpoints carry their representation;
38D and 125D are not interchangeable. No DiT skip/cache approximation is enabled.

## Orin bridge

```bash
PYTHONPATH=src python -m omg.cli.realtime.holomotion_real_bridge \
  --connect tcp://GPU_HOST:5571 --text 'walk forward' \
  --publish-bind ipc:///tmp/omg-preview-v2
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
