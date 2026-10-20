# Inference contract

Native inference uses the saved training config and strict checkpoint loading.
The recorded architecture contract must match history positional encoding,
self/cross QK normalization and feature conditioning. Tensor shapes alone do
not establish semantic compatibility. The runtime accepts native50Hz,
prediction100/history10 text-only future models with50 DDIM steps.

No DiT approximation, diffusion continuation, interpolation or motion alignment
is used. Prompt encoding can use the model's exact frozen T5 feature cache.
The same seed plus plan ID gives the same initial noise in sync and async.

ONNX export is retained as an independent numerical-parity-checked artifact
tool. The legacy ONNX multi-mode planner is removed. Production deadline
compliance must be measured on the selected inference backend and device.
