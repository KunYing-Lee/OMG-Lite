# Inference contract

Checkpoint inference uses the saved training config and strict checkpoint
loading. The recorded architecture contract must match history positional
encoding, self/cross QK normalization and feature conditioning. Tensor shapes
alone do not establish semantic compatibility. Models are native 50Hz,
38D, prediction100/history10 and text-only.

The deployment planner server defaults to **TensorRT FP16, CFG batch2,
DDIM20 + DiT cache**. The cache reuses the previous guided x0 prediction when
successive full predictions have cosine similarity at least 0.995, after
four full forwards, for at most two consecutive steps. All DDIM updates still
execute. This is an approximation, explicitly evaluated separately from the
uncached baseline. No continuation, interpolation, blending or motion alignment
is applied. Each plan starts with a fresh cache and NumPy RNG seeded by
seed + plan ID. Frozen T5 prompt features are cached exactly.

The checkpoint-based sync/async pipeline and explicit `--backend pytorch`
server retain their 50-step baseline. Training and benchmark defaults are not
changed by the deployment default. Their PyTorch RNG stream differs from the
NumPy deployment stream; matching integer seeds alone does not imply matching
noise across backends.

ONNX export defaults to 20 steps and batch2 and retains numerical parity
checks. Deployment requires the corresponding complete sampling metadata,
38D/50Hz statistics, kinematics and T5 assets. It rejects incompatible metadata
or unavailable TensorRT providers. See [G1 deployment](realtime_g1.md).
Production deadline compliance must be measured on the selected device,
including cold starts, transport and tracker contention.
