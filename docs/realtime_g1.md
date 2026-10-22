# G1 Deployment

Generation runs on the GPU host; HoloMotion tracking runs on G1 Orin.
Use a matching **38D / 50 Hz** checkpoint and statistics.
The concrete ROS adapters target HoloMotion v1.4.1 / ROS 2 Humble.

**Status:** wired planner and isolated controller-stop tests passed during
integration. The final sensor shadow run, Orin policy-node load and physical
motion acceptance remain pending.

## Pipeline

Generate 100 frames from 10 history frames with 50-step DDIM and CFG 2.5.
Startup history is measured; subsequent history comes from the raw plan.

| Default schedule | Frame |
|---|---:|
| Request next plan | 25 |
| Raw history | 40–49 |
| Preview deadline | 39 |
| Switch to new plan frame 0 | 50 |

Ten preview frames plus one derivative sample leave **280 ms** for inference
and transport. Missed deadlines stop tracking. There is no interpolation,
smoothing or hold-last recovery; planned history alone does not correct drift.

The server and generic bridge support `--switch-seconds` from 0.5 to 2.0 in
20 ms increments and must match. For P frames, request at P−25, use history
[P−10:P], and switch at P. The concrete ROS adapters currently use **1.0s**.

## 1. GPU planner

From the OMG-Lite checkout, with the downloaded inference bundle:

```bash
export OMG38_ARTIFACT_DIR="$PWD/hf-artifacts/checkpoints/updated/38D"
PYTHONPATH=src python -m omg.cli.realtime.planner_server \
  --config "$OMG38_ARTIFACT_DIR/config.yaml" \
  --checkpoint "$OMG38_ARTIFACT_DIR/sstep=075500.ckpt" \
  --bind tcp://0.0.0.0:5571 --switch-seconds 1.0
```

## 2. Orin validation

Follow [controller setup](../deployment/README.md). Source ROS and the external
HoloMotion installation, preserving their Python paths. From OMG-Lite:

```bash
PYTHONPATH=src:$PYTHONPATH python deployment/shadow_native50.py \
  --tracker /path/to/model_16200_dynamic_batch.onnx \
  --connect tcp://GPU_HOST:5571 --text 'raise hands' \
  --frames 500 --output outputs/shadow
```

This reads live sensors and records candidate actions without publishing motor
commands. The output directory must be new. Pass this check and the isolated
policy load/stop checks before a supervised physical trial.

## 3. Controller integration

`deployment/native50_policy_node.py` replaces the external policy node and
requires the patched native controller. It retains native gains, mapping and
remote arming: B requests motion; Y cancels or returns to velocity. The default
250-frame episode requests native stop on completion. There is no automatic
live launcher.

For a separate preview-stream consumer, run:

```bash
PYTHONPATH=src python -m omg.cli.realtime.holomotion_real_bridge \
  --connect tcp://GPU_HOST:5571 --text 'raise hands' \
  --publish-bind ipc:///tmp/omg-preview-v2 --switch-seconds 1.0
```

Integrate [`run_armed_tracker`](../deployment/holomotion_hook.py) after the
controller's existing arming gate. Supply fresh G1 joint positions/velocities,
pelvis quaternion, body gyro, measured startup `qpos36[10]`, and native
apply-action/stop callbacks. LowState alone does not provide root translation.
This is an alternative integration to the concrete policy node, not a second
motor publisher. Legacy `obs65` consumers do not support the preview protocol.
