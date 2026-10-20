# Native 50Hz data

Use the prepared `local/OMG-Data-50Hz` snapshot pinned by
`configs/generation/data/omg_data_50hz.yaml`, or the matching episode cache in
`omg_data_50hz_cached.yaml`. These configs pin source revision and manifest
identity. Changing `fps` on the original OMG dataset does not convert it.

Set `OMG_DATA_ROOT` to the native50 source and `OMG_MATERIALIZED_ROOT` to its
cache parent. `scripts/materialize_omg_data.sh` creates the seq100/history10
cache. Validate it with `omg.cli.data.validate_episode_cache`.

125D statistics are `assets/stats/g1_125d_50hz_hist10_stats.json`. The matching
38D statistics are an explicit projection of the unchanged first38 channels;
the projection source hash is recorded in the JSON. Recompute stats when the
underlying source, window distribution or representation changes.

Dataset conversion and historical 30Hz workflows belong to OMG. The Lite
runtime rejects other rates; it does not interpolate motion.
