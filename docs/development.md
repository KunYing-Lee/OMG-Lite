# Development

Run `PYTHONPATH=src python -m pytest tests -q` in the full project environment.
Native pipeline tests cover exact raw history, immutable committed prefixes,
preview and derivative deadlines, late/stale messages and simulated-latency
parity. Hardware-facing code does not send motor commands in unit tests.

For real-model validation, run the sync CLI and
`scripts/validation/native50_sim.py`. Retain raw plans, reference/execution
arrays and checkpoint provenance. Simulator success is not hardware acceptance.
