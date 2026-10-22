# Training

Prepare the [50 Hz data/cache](data.md) and T5-base under
`$OMG_MODELS_ROOT/t5-base-local`. The default is a 38D text-conditioned model:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m omg.cli.generation.train exp=base
```

See [training protocol](training_50hz.md) for batch size, learning rate and
fixed evaluation. Inspect the resolved configuration with `--cfg job --resolve`.

## Resume or initialize

Add `ckpt_path=/path/to/last.ckpt` for a full resume, including optimizer and
scheduler. Use `init_weights_only_ckpt=/path/to/model.ckpt` for weights only.

## Overrides

Hydra configs live under `configs/generation`. Use `logger=none` to disable
logging, or `logger=wandb` for W&B. Model-size presets include `50m`, `100m`,
`300m`, `500m` and `1b`; inspect their resolved settings before use.

Changing data, batch size or GPU count requires recalculating epoch lengths,
learning-rate milestones and fixed evaluations. Training outputs include
resolved configs, metrics and checkpoints under the configured output directory.
