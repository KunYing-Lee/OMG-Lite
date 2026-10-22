# Generation and simulation

Use `python -m omg.cli.pipeline.main --help`. The only integrated execution
modes are `sync` and `async`; both use native50Hz, prediction100, history10,
a configurable switch period (default: request25, raw history40–49,
switch50 and a phase39 preview deadline).

Pass the saved resolved training config with its checkpoint. Representation,
normalization and architecture must match. No automatic frame-rate conversion
or checkpoint downgrade is performed. The simulator records generated plans,
reference motion and executed motion. Sync pauses the simulation clock while
planning. Async errors on missed wall-clock deadlines.

Standalone generation remains available through `omg.cli.generation.generate`.
ONNX export remains an artifact tool, but the retired multi-mode ONNX pipeline,
DiT skipping and diffusion continuation scheduler are not deployment paths.

Videos use `holomotion` for generated/raw reference motion and the tracker’s
original `tracker_native` scene for execution. `--video` records both side by side.
