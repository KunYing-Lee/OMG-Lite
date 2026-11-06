"""Planner service for the initial-noise policy (default port 5574).

Two backends share the same request/response protocol:

* ``pytorch``: frozen SFT generator in FP32, DDIM-20, explicit actor noise;
* ``trt``: the same frozen generator exported to 20-step TensorRT/ONNX,
  explicit actor noise, **no DiT cache**.

The residual/generator services on 5571-5573 are not modified.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from omg.noise.planner import TRAIN_CAPTION, NoiseCheckpointPlanner
from omg.pipeline.schedule import HORIZON, Schedule
from omg.realtime.native import PlannerService
from omg.realtime.transport import ZmqPlanServer


def _check_plan(planner) -> None:
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
    parser.add_argument("--backend", choices=("pytorch", "trt"), default="pytorch")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint", help="frozen SFT generator checkpoint (pytorch)")
    parser.add_argument("--onnx", help="20-step SFT denoiser ONNX (trt)")
    parser.add_argument("--metadata", help="ONNX metadata sidecar (trt)")
    parser.add_argument("--engine-cache", help="TensorRT engine cache directory (trt)")
    parser.add_argument("--text-encoder", help="T5 model directory (trt)")
    parser.add_argument("--noise-checkpoint", required=True, help="noise PPO policy.pt")
    parser.add_argument("--bind", default="tcp://0.0.0.0:5574")
    parser.add_argument("--seed", type=int, default=202609150)
    parser.add_argument("--switch-seconds", type=float, default=0.5)
    parser.add_argument("--caption", default=TRAIN_CAPTION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check", action="store_true", help="load and plan once, then exit")
    args = parser.parse_args()

    torch.set_num_threads(1)
    if args.backend == "pytorch":
        if not args.config or not args.checkpoint:
            parser.error("pytorch backend requires --config and --checkpoint")
        if args.onnx or args.metadata or args.engine_cache:
            parser.error("pytorch backend does not accept TensorRT arguments")
        planner = NoiseCheckpointPlanner(
            args.config,
            args.checkpoint,
            args.noise_checkpoint,
            device=args.device,
            seed=args.seed,
            caption=args.caption,
        )
    else:
        if not args.onnx:
            parser.error("trt backend requires --onnx")
        from omg.noise.tensorrt import NoiseTensorRTPlanner

        planner = NoiseTensorRTPlanner(
            args.onnx,
            noise_checkpoint=args.noise_checkpoint,
            metadata=args.metadata,
            text_encoder=args.text_encoder,
            engine_cache=args.engine_cache,
            device=args.device,
            seed=args.seed,
            caption=args.caption,
        )

    step = getattr(planner, "step", getattr(planner, "generator_step", -1))
    print(
        "noise planner loaded: backend=%s generator_step=%s actor_iteration=%s ddim=%s"
        % (args.backend, step, planner.actor_iteration, planner.ddim_steps),
        flush=True,
    )
    if args.check:
        _check_plan(planner)
        return

    schedule = Schedule(args.switch_seconds)
    service = PlannerService(planner, schedule=schedule)
    print(
        "noise planner serving on %s (backend=%s, switch=%.2fs, seed=%d)"
        % (args.bind, args.backend, args.switch_seconds, args.seed),
        flush=True,
    )
    with ZmqPlanServer(args.bind) as server:
        while True:
            request = server.recv_request()
            started = time.perf_counter()
            plan = service.plan(request)
            total_ms = (time.perf_counter() - started) * 1000.0
            metadata = plan.metadata
            print(
                "plan %d | planner_ms=%.1f total_ms=%.1f | noise_seed=%s "
                "mean_norm=%s action_norm=%s"
                % (
                    plan.plan_id,
                    plan.planning_latency_seconds * 1000.0,
                    total_ms,
                    metadata.get("noise_seed"),
                    metadata.get("noise_mean_norm"),
                    metadata.get("noise_action_norm"),
                ),
                flush=True,
            )
            server.send_plan(plan)


if __name__ == "__main__":
    main()
