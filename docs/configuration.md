# Configuration

`exp=base` is native50Hz text-only38D50M, prediction100, history10, sinusoidal
history encoding and self/cross QK norm. Size presets inherit this protocol.
Use `representation=125d` only with matching125D statistics/checkpoints.

`data=omg_data_50hz_cached` is the training default;
`data=omg_data_50hz` reads the pinned native source directly.
The resolved training config is required for checkpoint inference.

The execution schedule is defined once in `omg.pipeline.schedule`.
It is not controlled by independent CLI delay, skip or remaining-buffer knobs.
