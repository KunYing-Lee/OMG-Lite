"""Planner service for the initial-noise policy (default port 5574).

This is intentionally a separate entry point from ``planner_server.py`` so the
residual/generator services on 5571-5573 are not modified.  It serves the same
50 Hz request/response protocol; only the plan generation is different:
history -> frozen noise actor -> sampled 100x38 initial noise -> frozen
generator (DDIM-20, FP32, CFG 2.5).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from omg.noise.planner import TRAIN_CAPTION, NoiseCheckpointPlanner
from omg.pipeline.schedule import HORIZON, Schedule
from omg.realtime.native import PlannerService
from omg.realtime.transport import ZmqPlanServer


def _check_plan(planner: NoiseCheckpointPlanner) -> None:
    """One synthetic standing-history smoke test; no robot or tracker needed."""
    history = np.zeros((10, 36), dtype=np.float32)
    history[:, 2] = 0.78
    history[:, 3] = 1.0
    result = planner.plan(history, planner.caption, 0)
    qpos = result["qpos"]
    if qpos.shape != (HORIZON, 36) or not np.isfinite(qpos).all():
        raise RuntimeError(f"invalid plan shape/values: {qpos.shape}")
    print(
        json.dumps(
            {
                "check": "PASS",
                "qpos_shape": list(qpos.shape),
                "qpos_min": float(qpos.min()),
                "qpos_max": float(qpos.max()),
                "latency_ms": round(result["latency_s"] * 1000.0, 2),
                "metadata": result["metadata"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="frozen SFT generator")
    parser.add_argument("--noise-checkpoint", required=True, help="noise PPO policy.pt")
    parser.add_argument("--bind", default="tcp://0.0.0.0:5574")
    parser.add_argument("--seed", type=int, default=202609150)
    parser.add_argument("--switch-seconds", type=float, default=0.5)
    parser.add_argument("--caption", default=TRAIN_CAPTION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check", action="store_true", help="load and plan once, then exit")
    args = parser.parse_args()

    torch.set_num_threads(1)
    planner = NoiseCheckpointPlanner(
        args.config,
        args.checkpoint,
        args.noise_checkpoint,
        device=args.device,
        seed=args.seed,
        caption=args.caption,
    )
    print(
        "noise planner loaded: generator_step=%d actor_iteration=%d ddim=%d"
        % (planner.step, planner.actor_iteration, planner.ddim_steps),
        flush=True,
    )
    if args.check:
        _check_plan(planner)
        return

    schedule = Schedule(args.switch_seconds)
    service = PlannerService(planner, schedule=schedule)
    print(
        "noise planner serving on %s (switch=%.2fs, seed=%d)"
        % (args.bind, args.switch_seconds, args.seed),
        flush=True,
    )
    with ZmqPlanServer(args.bind) as server:
        while True:
            request = server.recv_request()
            server.send_plan(service.plan(request))


if __name__ == "__main__":
    main()
