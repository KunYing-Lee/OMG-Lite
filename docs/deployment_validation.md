# DDIM20 deployment port validation — 2026-10-22

Source algorithm: OMG commit `38afa98948fe494895a11975d445ea36de6f083d`,
`pipeline/planner.py`, with the tested text-only, one-chunk, CFG batch2 path.
The port retains NumPy DDIM arithmetic/RNG and the guided-x0 cache rule. It omits
unused multimodal/continuation branches and auxiliary feature/history outputs.

On Mars5090 (RTX 5090, TensorRT 10.16.1.11, ORT 1.23.2), compared against that
backend using the same TensorRT FP16 engine, DDIM20 metadata, 38D/50Hz stats,
T5 features and initial history. Checkpoint step 75500 SHA256:
`8bdd19cc6a47131dacb8f29a2a440c6a3a405dfcd335ecfffc3054828bd59fe7`.

- Four prompts: walk forward, squat down and stand up, turn around, raise hands.
- Forty paired seeds 1234–1273, alternating backend order after warmup.
- All 40 generated qpos arrays bitwise identical; cache forward counts identical.
- Original planner median/p95: 20.26/20.91 ms. Port: 14.54/15.20 ms.
  This measures warm planning including output transfer, not robot transport.
- Default CLI server, localhost ZMQ: startup and subsequent native requests
  passed. The warm request took 14.48 ms planning, 16.69 ms roundtrip, with
  10 full forwards + 10 cached updates. Cold startup request took 2.49 s
  including server initialization. These two requests are a protocol smoke,
  not a sustained deadline benchmark.
- 43 local pipeline, realtime and training-configuration tests passed.
  Tests used a short `/tmp` base directory to fit macOS Unix socket path limits.

Raw records and scripts on Mars5090:
`/home/leeky/runs/lite_deploy_ddim20_20261022/{parity.json,verify.py,server_smoke.json,server_smoke.py}`.
No live planner was replaced and no robot motor output was enabled.
