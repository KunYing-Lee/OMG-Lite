"""Compare uncached denoising with exact CUDA graph replay; keep all50 steps."""

import argparse
import json
from pathlib import Path

import numpy as np
from omg.pipeline.checkpoint import CheckpointPlanner
from omg.pipeline.cuda_graph import CUDAGraphDenoiser

p = argparse.ArgumentParser()
for name in ["config", "checkpoint", "history", "output"]:
    p.add_argument("--" + name, required=True)
a = p.parse_args()
h = np.load(a.history)["qpos_36"][-10:]
planner = CheckpointPlanner(a.config, a.checkpoint, cuda_graph=False)
plain = planner.plan(h, "walk forward", 0)
planner.model.denoiser = CUDAGraphDenoiser(planner.model.denoiser)
planner.plan(h, "walk forward", 0)
replay = planner.plan(h, "walk forward", 0)
np.testing.assert_allclose(plain["qpos"], replay["qpos"], atol=1e-5, rtol=0)
r = {
    "max_qpos_abs_diff": float(abs(plain["qpos"] - replay["qpos"]).max()),
    "plain_seconds": plain["latency_s"],
    "replay_seconds": replay["latency_s"],
    "steps": 50,
}
Path(a.output).write_text(json.dumps(r, indent=2))
print(r)
