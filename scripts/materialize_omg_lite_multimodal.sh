#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="src:${PYTHONPATH:-}"
"${PYTHON:-python3}" -m omg.cli.data.materialize_episode_cache \
  --data-config configs/generation/data/omg_lite_multimodal_50hz.yaml \
  --representation-config configs/generation/representation/38d.yaml \
  --paths-config configs/generation/paths/default.yaml \
  --output-root "${OMG_MATERIALIZED_ROOT:?set OMG_MATERIALIZED_ROOT}/omg_lite_multimodal_50hz_v3_seq100_hist10_k1" \
  "$@"
