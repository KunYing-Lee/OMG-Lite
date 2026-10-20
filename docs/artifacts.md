# Artifacts

The supported public125D50Hz checkpoint is
[step49431](https://huggingface.co/THU-MARS/OMG/blob/main/checkpoints/updated/50Hz/sstep=049431.ckpt),
SHA256 `b66c884c991e53129a8a9d03310a47194a77604372a3b9c62f6814ab63d9100e`.
Use its resolved training configuration and native125D50Hz statistics.
The default38D training architecture requires its own checkpoint.

Download T5-base locally and set `OMG_MODELS_ROOT`. HoloMotion artifacts must
be acquired separately from their official distribution; validation uses
v1.4.1 `model_16200_dynamic_batch.onnx` with context1 and preview10.

The original OMG release evaluator remains a separate benchmark artifact.
Do not compare different sampling rates without an explicitly defined evaluator
protocol; runtime frame-rate conversion is intentionally absent in Lite.
