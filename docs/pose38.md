# 38D pose representation

The default training configuration generates 38 channels: canonical root position
(3), root rotation in 6D (6), and G1 joint angles (29). Target and history use the
same layout. The 29 link positions are derived by differentiable FK when decoding;
they are not independent diffusion variables. Rotation decoding uses ordinary
Gram–Schmidt autograd (`rot6d_gradient_mode: vanilla`).

Default training still uses diffusion loss only. The generator skips MotionLoss
entirely when all auxiliary weights are zero. To enable only the additional FK
position objective:

```bash
python -m omg.cli.generation.train loss=fk
# Optional weight override:
python -m omg.cli.generation.train loss=fk loss.fk_body_pos=0.5
```

The total objective is diffusion loss plus `fk_body_pos` times the mean squared
error of FK link positions in the canonical frame, averaged over valid scalar
coordinates per sample. Gradients propagate through root translation, root
rotation and joint angles. No independently predicted link positions or position
consistency penalty are needed. The older full auxiliary-loss preset remains
available; its position consistency term is zero for the compact representation.

## Dataset and statistics

Original LeRobot qpos data and v2 episode caches can be reused unchanged. The
cache stores `qpos_36`, `body_pos_w`, and `body_quat_w`, not fixed 125D targets.
Both dataset readers accept `include_body_positions: false`; all generation data
configs take that setting from the representation. Window sampling, valid masks,
canonical anchors and captions are unchanged. Targets, history and statistics
batches are assembled as 38D. Body geometry remains available for evaluation.

Use 38D normalization statistics from exactly the same sampling protocol. Because
these are precisely the first 38 coordinates of 125D, their marginal mean/std can
be projected exactly without another corpus scan:

```bash
python -m omg.cli.generation.project_pose_stats \
  assets/stats/g1_125d_50hz_hist10_stats.json \
  assets/stats/g1_38d_50hz_hist10_stats.json
```

The converter validates the source layout, preserves protocol metadata and default
pose, records source SHA-256, and refuses to overwrite output files. Training never
silently truncates a mismatched statistics file. Existing `compute_stats` also
supports the compact representation, with `38d.yaml` as its default.

The committed `g1_38d_50hz_hist10_stats.json` is projected from the committed native50Hz 100-frame 125D
statistics. It is **not** the 100-frame/50Hz statistics required by the default
experiment. That experiment expects `g1_38d_50hz_hist10_stats.json`, or the path
supplied through `OMG50_STATS_PATH`; convert the matching 50Hz source on the
training host before launching.

## Compatibility

Use `representation=125d` for the 125D feature layout. Existing 125D generation
checkpoints are not 38D checkpoints: input/output and history projections change
shape. Train a new 38D model; do not resume a 125D optimizer/checkpoint as 38D.
The decoded world-space controller interface remains `qpos_36`.
