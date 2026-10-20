> Historical 1-second schedule validation at `aa5acd2`. For the current 0.5-second schedule, see [half_second_validation.md](half_second_validation.md).

# Native50 pipeline validation — 2026-10-20

Implemented in branch `codex/native-50hz-pipeline`.

## Mars5090 acceptance

RTX5090, PyTorch2.7.0/cu128, ORT1.23.2, MuJoCo3.7.0. The complete epoch3
checkpoint SHA256 is
`b66c884c991e53129a8a9d03310a47194a77604372a3b9c62f6814ab63d9100e`;
tracker SHA256 is
`15f7914f6460a1c2e71bc06191874146e7b4d1669dded095e072cf29aac7c056`.
Full tests:202 passed. Graph replay matches ordinary50-step output exactly;
one warmed inference measured199.3ms (cold ordinary call is not a speed baseline).

The original same-process async implementation missed control tick28 for all
three commands despite adequate total inference throughput. An independent
planner service passed; default local async now uses one persistent spawned
planner process, keeping CUDA inference outside the control process.
No sampling, deadlines, histories or output motions were changed.

| Command | Frames | Replans | Median inference | Maximum inference | Root RMSE |
|---|---:|---:|---:|---:|---:|
| walk forward |500|10|208.8ms|210.7ms|76.7cm|
| squat down and stand up |500|10|208.3ms|209.0ms|12.8cm|
| raise hands |500|10|209.1ms|209.9ms|4.9cm|

All30 plans were accepted at local frame36, before deadline39. Raw plans,
references and executed qpos matched sync **exactly** for each command.
Timing includes actual50-step inference; this is not simulated latency.
The separate70ms wall-clock fixed-plan test also passed450 frames per command.

**Realtime scheduling passes these finite trials; motion quality is not fully
accepted.** Walking has substantial absolute root drift. Maximum raw seam
joint increments were0.176/0.235/0.422rad for walk/squat/hands. These are
one-frame increments, not errors relative to a continuous ground truth.
No smoothing, alignment or interpolation was applied. Following cameras are
independent, so assess absolute root error from metrics, not screen centering.
The renderer now frames the body for following views rather than zooming out
to contain the full travelled path.

Evidence on Mars:
`/home/leeky/runs/omg_lite_native50_acceptance/`
(`summary.json`, `isolated_async_*`, `sync_*`, original failing `async_*`,
`checkpoint_replay.json`, `parity`, test logs and scripts).
No robot was armed. This does not establish Orin/network/physical acceptance.

## Verified

- Full repository test suite: **202 passed** (one dependency warning).
- Strict epoch3 checkpoint load and synchronous MuJoCo rollout with the real
  50Hz125D step49431 checkpoint.
- Source dataset `local/OMG-Data-50Hz` checksum-index identity verified; actual
  sample shapes: qpos `[100,36]`, pose features `[100,38]`.
- CUDA Graph denoiser replay versus ordinary50-step inference: maximum qpos
  absolute difference **0.0**. No step skipping or approximate cache.
- Three fixed commands (walk, squat, hands),150 control frames each. The actual
  deployment observation/action adapter was driven by native preview packets
  under logical30/50/70ms generation delays. Against synchronous tracking,
  every executed qpos value matched exactly (**maximum difference0.0**).
- Request25, raw history40–49, switch50; late plans and stale/reordered packets
  are rejected. IPC planner request/response round trips are tested.

## Deadline and physical boundary

A ten-frame preview contains velocity features. The feature at frame49 reads
frame50, so the first affected control tick is39. Deadline is therefore280ms
from the frame25 request. A500ms inference bound is insufficient.

On the currently shared A800 GPU, prewarmed50-step graph inference measured
**0.718664s**. The actual async run failed closed at control frame26 after
starting its replan; it did not pass realtime acceptance. No training process
was stopped and no deadline was relaxed. Simulated latency parity does not
establish hardware throughput.

The Orin bridge and consumer hook are implemented, but the user's active
external HoloMotion controller checkout has not yet been provided. Connecting
its state estimator, remote-B gate, PD writer and stop callback is outstanding.
No robot was armed or driven. LowState alone lacks measured root translation;
the new integration explicitly requires the controller's root estimate instead
of substituting planned state.

Runtime motion interpolation and legacy30Hz presets, statistics, benchmark
manifests and old pipeline/obs65 bridge are removed. Native frame-rate validation
replaces resampling. Hardware sensor-state estimation remains external.
