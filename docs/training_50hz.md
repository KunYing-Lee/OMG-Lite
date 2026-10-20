# 50Hz text-only pretraining

`exp=base` and its alias `exp=50m_50hz_text_3epochs` select the same protocol.
The explicit older size presets (`50m`, `100m`, etc.) retain their existing behavior.

## Inputs and launch

Use an already prepared `omg.episode_cache.g1_motion.v2` cache named
`omg_50hz_text_rot6d_seq100_hist10_k1` under `OMG_MATERIALIZED_ROOT`, with train,
val and test splits. Its identity is pinned in `data/omg_data_50hz_cached.yaml`
to `local/OMG-Data-50Hz`, revision
`sha256:ff4243bdc4dd1fb242ca9419cd99080329300de55bb7b70a417ca5055e8a843e`.
The cache must describe 50Hz, 100 future frames, 10 history frames, canonical
anchor 9 and rot6d. The reference training split contains 101,240,125 windows.
Supply the matching window-weighted 125D statistics separately (reference file
SHA-256 `e54506da9f218ea8f259fd18f7f8de3d9c577a53e34d95cb53e2a7d972846c8f`).
This change does not include resampling, statistics computation, cache building,
data artifacts or machine-specific paths.

```bash
export OMG_MATERIALIZED_ROOT=/path/to/prepared/cache
export OMG50_STATS_PATH=/path/to/g1_125d_50hz_hist10_stats.json
export OMG_MODELS_ROOT=/path/to/models  # contains t5-base-local
PYTHONPATH=src python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m omg.cli.generation.train exp=base
```

Use `--cfg job --resolve` to inspect the resolved Hydra configuration before
launching. Hydra writes configuration/overrides under `${output_dir}/hydra/.hydra`;
training logs the fully resolved configuration and full checkpoints retain model
hyperparameters plus optimizer/scheduler state. CSV metrics are under
`${output_dir}/metrics`.

## Protocol

- Fresh motion-model initialization, seed 1234; frozen pretrained T5-base.
- 125D rot6d, 100 predicted frames (2s), 10 history frames (0.2s).
- 10 layers, width 512, 8 heads, MLP ratio 4, dropout 0.1. Sinusoidal history
  positions, RoPE motion attention, self/cross QK normalization.
- Text only; text/history condition dropout 0.3/0.1. Diffusion-only loss;
  no velocity, contact, FK or seam auxiliary losses.
- Pred-x0 cosine diffusion with 1000 training timesteps. Evaluation uses
  deterministic DDIM, 50 steps, CFG 2.5.
- Eight ranks, batch 768/rank (6144 global), accumulation 1; four workers/rank,
  pinned memory, prefetch 2; bf16 mixed precision and gradient norm clipping 1.
- AdamW peak LR 1e-4, weight decay 0.01. Linear warmup for 250 steps from 1e-8,
  followed by cosine decay to 1e-6 at step 49,431.
- Three full loader epochs, 16,477 steps each. With full-batch dropping,
  101,234,688 windows are processed per epoch. Validation/checkpoints every
  250 steps; validation limited to 200 batches.
- Fixed evaluation at steps 16,477 / 32,954 / 49,431: 4096 loss windows,
  64 generation windows, seed 314159, shared fixed sample manifest and noise.
  Evaluation restores training RNG and module modes. Regular validation uses
  seed 271828 and restores RNG. Paired generation MSE is not a physical score.

The fixed-evaluation callback rejects an unexpected loader epoch length. If
changing the data snapshot, batch size or GPU count, explicitly recalculate the
schedule and milestones rather than retaining the reference epoch labels.

## Training runtime changes

Training scalar metrics are packed by dtype/device into one rank-mean reduction
per group. The differentiable local loss is unchanged; validation keeps Lightning's
batch-weighted epoch reduction. Diagnostic reads are packed by dtype/device before
GPU-to-CPU transfer, preserving values, shapes and integer precision.

`balanced_shards` partitions the shuffled shard stream by equal window quotas,
splitting only boundary shards. Ranks have equal epoch lengths and disjoint
samples, while retaining shard-local shuffle/interleaving. This is a training
sampler change, not a dataset preprocessing change. The `auto` sampler behavior
remains available to other experiments.

The migration comes from training commits `4376b23` and `59b4e99`, plus the
training-facing configuration from `83adf16`. It excludes deployment/RTC changes.
