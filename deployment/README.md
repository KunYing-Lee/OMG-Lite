# G1 Controller Adapters

See [G1 Deployment](../docs/realtime_g1.md) for the pipeline, planner and shadow
commands. These adapters target the external HoloMotion v1.4.1 ROS 2 controller.
**Full shadow and physical acceptance are pending.**

| File | Purpose |
|---|---|
| `g1_native50.py` | Native sensor reads and CUDA tracker |
| `shadow_native50.py` | Live-sensor validation without motor output |
| `native50_policy_node.py` | Native remote-gated policy integration |
| `holomotion_hook.py` | Alternative preview-stream consumer hook |
| `holomotion_stop.patch` | Explicit stop, nonfinite-action rejection, 40 ms watchdog |
| `test_native_stop.py` | Isolated controller-stop tests |

## Build the controller

Copy the external controller source, including `launch_profiles`, into a
separate build directory. Set `OMG_LITE_ROOT` to this repository's absolute path.
From the copied controller root, after sourcing ROS and Unitree messages:

```bash
sha256sum -c "$OMG_LITE_ROOT/deployment/holomotion_controller_source.sha256"
patch -p1 < "$OMG_LITE_ROOT/deployment/holomotion_stop.patch"
colcon build --base-paths src --packages-select humanoid_control \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

The patch invokes the existing emergency-stop procedure. Verify the patched
binary is deployed; a stop-topic subscriber alone does not prove this.

## Validate

Use the Orin HoloMotion Python environment. Source ROS and HoloMotion first,
then prepend `$OMG_LITE_ROOT/src` to `PYTHONPATH`. Place ONNX external weights
beside the model graph.

The isolated stop tests take explicit paths:

```bash
ROS_DOMAIN_ID=213 ROS_LOCALHOST_ONLY=1 \
python "$OMG_LITE_ROOT/deployment/test_native_stop.py" \
  --controller /path/to/install/humanoid_control/lib/humanoid_control/humanoid_control \
  --config /path/to/g1_29dof_holomotion.yaml --output /path/to/new-stop-report
```

Run the policy node's `--check` in the same isolated localhost domain, supplying
`--tracker`, `--connect` and the external controller's ROS parameter file.
It checks disabled startup, model mapping and gains. Then run the live-sensor
shadow command in the robot's normal ROS domain.

Startup history pairs native odometry to LowState within 2 ms and rejects
stale data. `body_height` is the native estimator output; its accuracy while
suspended remains unverified. Physical trials require fresh on-site supervision
and the native remote/emergency-stop controls.
