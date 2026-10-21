# OMG-Lite

Native **50Hz**, text-conditioned motion generation and HoloMotion tracking.
Training and inference use only the **38D** pose representation at **50Hz**. See [pose representation](docs/pose38.md).

## One pipeline

- Generate 100 frames (2 seconds), conditioned on 10 history frames.
- Configure `--switch-seconds` (default **1.0**, minimum **0.5**, maximum **2.0**).
  Values must be exact 20ms frame increments; invalid values are rejected.
- For a period of P frames, request at P−25 using **raw frames P−10 through P−1**;
  new plan frame 0 starts at P. Generation always starts 0.5s before switching.
- Publish the new future reference as soon as ready. The tracker previews ten
  frames and velocity needs one more sample: accept before consuming frame P−11.
  This leaves 280ms for inference and transport at every supported period.
- `sync` pauses simulation during inference. `async` continues tracking with
  wall-clock deadlines. They share the same raw-plan timeline.
- No interpolation, alignment, smoothing, skipped generated frames, or late
  plan substitution. Unsupported frame rates and missed deadlines are errors.

History after startup is **planned**, not measured: this schedule alone does
not correct tracking drift or guarantee continuity at a generated boundary.

## Install and artifacts

```bash
make venv
source .venv/bin/activate
make install
export PYTHONPATH=src
```

## Released checkpoint: generate a motion

Download the [38D inference bundle](https://huggingface.co/THU-MARS/OMG/tree/main/checkpoints/updated/38D):

```bash
python - <<'PYCODE'
from huggingface_hub import snapshot_download
snapshot_download("THU-MARS/OMG", allow_patterns=["checkpoints/updated/38D/*"], local_dir="hf-artifacts")
PYCODE
python hf-artifacts/checkpoints/updated/38D/infer.py \
  --text 'walk forward' --output outputs/walk_38d.npz
```

The bundle contains step75500, portable config, matching stats, startup history,
robot kinematics and an inference script. First use downloads T5-base; use
`--t5 /absolute/path/to/t5-base` for offline inference. Output is 100 frames of
raw `qpos_36` at 50Hz. No training dataset or tracker is needed for this command.
See [artifacts](docs/artifacts.md) for validation and dependencies.

Training uses a native50Hz dataset/cache. Set `OMG_DATA_ROOT`,
`OMG_MATERIALIZED_ROOT`, and `OMG_MODELS_ROOT`; see [data](docs/data.md).
Download HoloMotion v1.4.1 separately for tracking.

## Train

```bash
python -m omg.cli.generation.train exp=base
```

`base` uses 100 prediction frames, history10, sinusoidal history encoding,
Q/K norm, global batch6144 on eight GPUs, and three epochs. Model-size presets
inherit this protocol. See [training](docs/training_50hz.md).

## Simulate

```bash
python -m omg.cli.pipeline.main \
  --config /path/to/resolved_training_config.yaml \
  --checkpoint /path/to/sstep=075500.ckpt \
  --tracker /path/to/model_16200_dynamic_batch.onnx \
  --seed-motion /path/to/native_50hz_seed.npz \
  --text 'walk forward' --mode sync --frames 500 --output outputs/sync_walk
```

Use `--switch-seconds 0.5` for half-second switching, or omit it for one second.
Use `--mode async` for real wall-clock execution with a persistent, independent
planner process so inference does not block the tracker process. Both keep 50 DDIM steps and
CFG2.5. The seed NPZ must contain `qpos_36` and scalar `fps=50`; at least ten
valid history frames are required. Outputs retain raw plans and executed motion.

## Robot deployment

[Deployment](docs/realtime_g1.md) describes the GPU planner, Orin bridge and
HoloMotion consumer hook. The old current-frame `obs65` bridge is removed: it
cannot express the required preview. Physical arming, state estimation, PD and
emergency stop remain owned by the robot controller. Simulation validation
is not physical-robot acceptance.

## Benchmark and development

Generate native50Hz manifests for the active dataset. See [benchmark](docs/benchmark.md). Historical OMG pipelines remain in the OMG
repository, not in this runtime.

[MIT License](LICENSE)
