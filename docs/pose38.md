# 38D pose representation

Each motion frame contains root translation (3), root rotation in rot6d (6),
and 29 G1 joint angles. The feature codec has exactly 38 channels. Body positions
are derived by differentiable forward kinematics, including during decoding.
Raw input/output remains `qpos_36`: root XYZ, quaternion WXYZ, 29 joint angles.

The native50Hz protocol uses ten history frames, canonical anchor9 and 100
future frames. Both direct and cached dataset readers construct the same pose
features from raw qpos; caches can also retain body states for auxiliary losses.
Only rot6d pose features are supported. `include_body_positions: false` is
accepted in saved configs; setting it true raises an error.

Training defaults to diffusion loss only and does not decode FK in that loss
path. Enable `loss=fk` to supervise FK body positions/rotations. Gradients flow
through joint rotations and root pose; no generated body-position channels or
body-position consistency branch are used.

Use `assets/stats/g1_38d_50hz_hist10_stats.json` for the pinned training data.
Recompute normalization with `omg.cli.generation.compute_stats` when the source
or window distribution changes. The committed stats preserve their source
content hash and channel provenance. Checkpoints must match the model config
and normalization; dimensions are checked at load time.
