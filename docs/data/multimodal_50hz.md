# 50 Hz multimodal data

The public dataset is [THU-MARS/OMG-Lite-Data](https://huggingface.co/datasets/THU-MARS/OMG-Lite-Data).
It derives from OMG-Data revision `6e0dfbc1c5298bff14d4e2b1459ad678af0a38e7`:
798,181 episodes and 201,153,658 frames. Original train/val/test membership,
text, modality masks and source metadata are retained. This is a time resampling,
not new measured motion and not a deduplicated split. Use the OMG-Lite reader to
apply `meta/info.json` episode ranges: parquet file boundaries can cross splits.
An automatically inferred Hugging Face `train` split is not the official training split.

## Download and compact cache

Install `pip install -e '.[data]'`. The new data configuration pins the HF commit
and manifest digest. If `OMG_LITE_DATA_ROOT` is unset, the reader downloads that
exact snapshot; set it to an existing snapshot directory to work offline.

```bash
export OMG_LITE_DATA_ROOT=/path/to/OMG-Lite-Data
export OMG_MATERIALIZED_ROOT=/path/to/cache
bash scripts/materialize_omg_lite_multimodal.sh --splits train val test --device cpu
python -m omg.cli.data.validate_episode_cache \
  "$OMG_MATERIALIZED_ROOT/omg_lite_multimodal_50hz_v3_seq100_hist10_k1" \
  --splits train val test
```

Cache v3 stores qpos, audio/human conditions, their masks, captions and episode
provenance. It does not store action, duplicate per-frame indices, body positions,
body quaternions, or precomputed 38D canonical windows. FK is derived on episode
load and reused in the bounded in-memory episode cache. This saves storage and
I/O, but introduces CPU FK work on episode cache misses; it is not a claim of
higher training throughput. Old v2 caches must be rebuilt; old run artifacts are
not modified. The historical text-release configs remain available for reproducing
existing experiments, with cache rebuilding required for this code version.

## New training

The new preset includes separate exact statistics over all 102,352,253 training
windows (10,230,609,902 valid frame observations). The matching file is also published
under `normalization/` in the data repository. The existing text checkpoint retains
its original stats. To recompute the new statistics:

```bash
export OMG_LITE_MULTIMODAL_STATS=/path/to/g1_38d_multimodal_50hz_stats.json
python -m omg.cli.generation.compute_stats \
  --data-config configs/generation/data/omg_lite_multimodal_50hz_cached.yaml \
  --representation-config configs/generation/representation/38d.yaml \
  --paths-config configs/generation/paths/default.yaml \
  --split train --device cuda --batch-size 1024 \
  --output "$OMG_LITE_MULTIMODAL_STATS"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m omg.cli.generation.train exp=multimodal_50hz
```

The new preset enables audio and human reference, retains 38D/50Hz and diffusion-only
loss, and uses 16,659 steps/epoch at global batch 6,144 (102,352,253 training windows).
Its schedule is separate from the historical text run. Enabling multimodal modules
changes model parameters: the published text-only checkpoint is not a trained
multimodal checkpoint. `loss=fk` remains an explicit opt-in.

## Resampling and conditions

- Target grid: `j/50 <= (N-1)/30`; no extrapolation or time stretching.
- Root XYZ and bounded joint angles: piecewise linear; root quaternion: shortest-arc SLERP.
- Action: next target qpos, terminal repetition.
- Human reference: 22x3 Cartesian joint positions, source coordinate convention retained.
- Audio: continuous coordinates use linear interpolation. AIST++/FineDance channels
  33 and 34 are binary events and are placed at the nearest in-domain target timestamp.
  The other four audio families use continuous 35D features. The per-episode
  `omg/audio_feature_type` distinguishes these definitions; they are not one uniform
  audio representation. These released features are resampled, not re-extracted from WAV.
- Availability masks are verified constant per source episode and copied exactly.
- Source frame offsets remain in source 30Hz units; episode lengths and dataset
  offsets are updated to the 50Hz grid.

Rebuild using `python -m omg.cli.data.resample_release --source ... --dest ...
--source-hashes ...`, with `provenance/source_file_sha256.json` from the public snapshot.
Then run `python -m omg.cli.data.validate_resampled_release --source ... --target ...`.
The validator checks every frame, episode boundary, quaternion interpolation,
action, condition, mask, timestamp and retained metadata. Resume verifies completed
shard hashes; never resume a conversion after changing its resampling algorithm.

Keep existing benchmark manifests/checkpoint stats for historical comparisons.
New benchmark manifests must pin this release, source identities and time grid.
Official split preservation does not establish recording-level leakage freedom;
the upstream overlap audit remains relevant.
