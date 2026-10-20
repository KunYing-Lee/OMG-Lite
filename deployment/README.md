# G1 native 50 Hz deployment — pending hardware acceptance

These adapters target the HoloMotion v1.4.1 ROS 2 Humble container on G1.
They are development code, not a completed physical acceptance release.
Merging this directory does not install, start, or arm any robot process.

## Components

- `g1_native50.py`: native LowState/odometry readers and CUDA tracker loader.
  Startup history uses ten actual odometry samples paired to LowState within
  2 ms. Root height comes from the native estimator's `body_height`; its
  physical accuracy on a suspended robot has not been established.
- `shadow_native50.py`: live sensors → remote planner → tracker, with no motor
  publisher. Records candidate actions, measured state, timing and failures.
- `native50_policy_node.py`: replacement policy node retaining the external
  controller's arming, remote controls, gains and joint mapping. Initial plans
  run asynchronously; cancellation detaches the pending episode. B requests
  motion, Y cancels/returns to velocity, and native emergency controls remain.
  A bounded episode defaults to 250 frames; completion requests native stop.
- `holomotion_stop.patch`: explicit stop topic, nonfinite-action rejection and
  40 ms action watchdog in the external C++ controller. It invokes that
  controller's existing emergency-stop procedure, not a reference hold.
- `test_native_stop.py`: isolated controller test, restricted to localhost
  ROS domain 213 with remapped topics. Never run it in the robot's live domain.

The native policy node **requires the patched controller**. Subscriber count
checks alone do not prove that a controller binary contains this patch.
Verify the source hash and deployed binary/build provenance before activation.

## Build and checks on Orin

Use an isolated copy of the external controller source, including its
`launch_profiles` directory. From that copy's root:

```bash
sha256sum -c /path/to/OMG-Lite/deployment/holomotion_controller_source.sha256
patch -p1 < /path/to/OMG-Lite/deployment/holomotion_stop.patch
# Source ROS Humble and the existing Unitree message package installation first.
colcon build --base-paths src --packages-select humanoid_control \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Source ROS and the existing HoloMotion installation before running Python;
preserve their `PYTHONPATH` when prepending `OMG-Lite/src`. Use the Orin
HoloMotion Python environment. The tracker ONNX file also requires its external
weights file next to it; copying the graph alone is insufficient.

The isolated stop-test script currently expects the integration workspace at
`/workspace/native50` and the external image's configuration under
`/opt/holomotion/deployment/unitree_g1_ros2_29dof`. Inspect these explicit paths
before use. Run it only with `ROS_DOMAIN_ID=213 ROS_LOCALHOST_ONLY=1`.

For a live-sensor test with no motor output, in the robot's normal ROS domain:

```bash
PYTHONPATH=src:$PYTHONPATH python deployment/shadow_native50.py \
  --tracker /path/to/tracker.onnx --connect tcp://GPU_HOST:15571 \
  --text 'raise hands' --frames 500 --output /path/to/new-shadow-report
```

The planner retains 50 DDIM steps. These robot adapters use the shared default
one-second switch period; the server must use the same period. Requests start
at phase 25, use raw frames
40–49, and switch at phase 50. Tracker preview/derivative support imposes a
280 ms reply deadline. A missed control period or planner deadline is fatal;
there is no interpolation, smoothing, fabricated history or hold-last recovery.

## Verification record and remaining gates

Observed during the 2026-10-20 session (remote logs were not retrieved before
network loss; these are session observations, not bundled reproducible reports):

- Wired G1 ↔ Mars5090: 2 initial requests and 30 replans passed; maximum formal
  request latency 206.819 ms. Cold initialization was outside the active clock.
- Patched native controller compiled. Isolated ROS tests triggered stop for
  explicit stop (~0.420 ms), nonfinite action (~0.449 ms), and missing actions
  (~41.964 ms after entering POLICY). These measure state transitions, not
  physical stopping distance or torque removal.
- Python DDS decoding was too slow. The replacement compiled ROS message
  reader has **not yet passed** the complete 500-frame shadow run.
- The final policy-node adapter has **not been loaded/tested on Orin**.
- No native50 motor-control trial was performed. Suspended and ground trials,
  actual tracking error and operator emergency-stop acceptance remain open.

Before any physical activation: pass the shadow run and isolated node load
check (`--check`, localhost domain 213), verify exact model/gain/mapping and
controller provenance, confirm exclusive publishers and fresh sensor timing,
and establish current on-site supervision. Launching/arming remains a manual
operator operation; this repository provides no automatic live launcher.
