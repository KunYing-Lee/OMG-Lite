# OMG-Lite

Native **50Hz**, text-conditioned motion generation and HoloMotion tracking.
Training defaults to a 50M **38D** pose model; matching 50Hz **125D** checkpoints
are also supported. See [pose representation](docs/pose38.md).

## One pipeline

- Generate 100 frames (2 seconds), conditioned on 10 history frames.
- At local frame 25, request the next plan using **raw plan frames 40–49**.
- Execute the old plan through frame 49; new plan frame 0 starts at frame 50.
- Publish the new future reference as soon as it is ready. HoloMotion previews
  10 frames and its velocity features read one additional sample, so the plan
  must arrive **before the frame-39 control tick** (280ms after request).
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

Use a native 50Hz dataset/cache and matching stats; the original 30Hz OMG
release is not a runtime input. Set `OMG_DATA_ROOT`, `OMG_MATERIALIZED_ROOT`,
and `OMG_MODELS_ROOT` for your storage. See [data](docs/data.md).

The [50Hz epoch3 125D checkpoint](https://huggingface.co/THU-MARS/OMG/blob/main/checkpoints/updated/50Hz/sstep=049431.ckpt)
is step 49431. It is **not** a 38D checkpoint. Pair it with its saved resolved
training config and `assets/stats/g1_125d_50hz_hist10_stats.json`.
Download HoloMotion v1.4.1 separately from its official release.

## Train

```bash
python -m omg.cli.generation.train exp=base
# Explicit 125D model, same native 50Hz protocol:
python -m omg.cli.generation.train exp=base representation=125d
```

`base` uses 100 prediction frames, history10, sinusoidal history encoding,
Q/K norm, global batch6144 on eight GPUs, and three epochs. Model-size presets
inherit this protocol. See [training](docs/training_50hz.md).

## Simulate

```bash
python -m omg.cli.pipeline.main \
  --config /path/to/resolved_training_config.yaml \
  --checkpoint /path/to/sstep=049431.ckpt \
  --tracker /path/to/model_16200_dynamic_batch.onnx \
  --seed-motion /path/to/native_50hz_seed.npz \
  --text 'walk forward' --mode sync --frames 500 --output outputs/sync_walk
```

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

Generate new native-50Hz manifests for the active dataset; do not reuse 30Hz
window indices. See [benchmark](docs/benchmark.md) and
[validation results](docs/native50_validation.md). Historical OMG pipelines remain in the OMG
repository, not in this runtime.

[MIT License](LICENSE)
