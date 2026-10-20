# Native50 pipeline validation — 2026-10-20

Implemented in branch `codex/native-50hz-pipeline`.

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
