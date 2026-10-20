# Benchmark

Prepare fixed sample manifests from the active native50Hz dataset:

```bash
PYTHONPATH=src python -m omg.cli.evaluation.prepare_samples \
  --data omg_data_50hz --num_frames 100 --output_dir outputs/benchmark_samples
```

Use `python -m omg.cli.generation.benchmark --help` for evaluator runs.
Match representation, statistics, test manifests and frame rate explicitly.
Historical30Hz sample indices and published scores are not interchangeable
with the native50Hz cohort. Cross-rate comparisons belong to the separate OMG
benchmark workflow, where resampling is explicit and documented.

For deployment parity, run `scripts/validation/native50_sim.py` with fixed
raw plans, initial history and the HoloMotion model. It compares synchronous
tracking against the deployment observation/action path at logical30/50/70ms
latency. This does not measure inference wall time or physical robot quality.
