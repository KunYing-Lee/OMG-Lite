# Native 50Hz data

For the current public multimodal release, use the [multimodal 50 Hz guide](data/multimodal_50hz.md). This page documents the historical text-only data and training protocol.

Use the prepared `local/OMG-Data-50Hz` snapshot pinned by
`configs/generation/data/omg_data_50hz.yaml`, or the matching episode cache in
`omg_data_50hz_cached.yaml`. These configs pin source revision and manifest
identity. Changing `fps` on the original OMG dataset does not convert it.

Set `OMG_DATA_ROOT` to the native50 source and `OMG_MATERIALIZED_ROOT` to its
cache parent. `scripts/materialize_omg_data.sh` creates the seq100/history10
cache. Validate it with `omg.cli.data.validate_episode_cache`.

Use `assets/stats/g1_38d_50hz_hist10_stats.json`. Recompute stats when the
source or window distribution changes. Both dataset readers enforce native50Hz.
The runtime does not interpolate motion.
