"""Native 50Hz sync/async simulation with raw future history."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from omg.pipeline.checkpoint import CheckpointPlanner
from omg.pipeline.execution import run
from omg.pipeline.schedule import motion
from omg.pipeline.tracker import Tracker
from omg.tracking.holomotion.runner import HoloMotionRolloutRunner
from omg.tracking.holomotion.runtime import g1_state_from_qpos_history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ["config", "checkpoint", "tracker", "seed-motion", "text", "output"]:
        p.add_argument("--" + k, required=True)
    p.add_argument("--mode", choices=["sync", "async"], default="sync")
    p.add_argument("--frames", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--video", action="store_true")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    seed = np.load(args.seed_motion)
    assert float(seed["fps"]) == 50
    history = motion(seed["qpos_36"][-10:], 10)
    planner = CheckpointPlanner(args.config, args.checkpoint, seed=args.seed)
    # Warm exact kernels before starting the async control clock.
    warm = planner.plan(history, args.text, 0)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        args.tracker, sess_options=opts, providers=["CUDAExecutionProvider"]
    )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA tracker required")
    runner = HoloMotionRolloutRunner(
        holomotion_onnx=args.tracker, session=session, video=False
    )
    initial, vel = g1_state_from_qpos_history(
        runner.model, runner.g1_handles, history, fps=50
    )
    runner.reset(initial, vel)
    tracker = Tracker(runner)
    tracker.activate(warm["qpos"])
    for frame in range(10):
        tracker.step(frame, 0)
    runner.clear_rollout_buffers()
    runner.reset(initial, vel)
    try:
        timeline, plans = run(
            planner, tracker, history, args.text, frames=args.frames, mode=args.mode
        )
        for i, plan in enumerate(plans):
            np.savez_compressed(out / f"plan_{i:03d}.npz", **plan)
        if args.video:
            from omg.render.mujoco import render_qpos_comparison_video

            render_qpos_comparison_video(
                np.asarray(runner.reference_qpos_36),
                np.asarray(runner.executed_qpos_36),
                out / "comparison.mp4",
                fps=50,
                left_title="Raw reference",
                right_title="Tracker execution",
                overlay_lines=[args.text, "request25 / raw40-49 / switch50"],
            )
        (out / "report.json").write_text(
            json.dumps(
                {
                    "config": vars(args),
                    "checkpoint_step": planner.step,
                    "events": timeline.events,
                },
                indent=2,
            )
        )
    except Exception as error:
        (out / "report.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(error),
                    "config": vars(args),
                    "completed_frames": len(runner.executed_qpos_36),
                    "checkpoint_step": planner.step,
                },
                indent=2,
            )
        )
        raise
    finally:
        np.savez_compressed(
            out / "rollout.npz",
            executed=np.asarray(runner.executed_qpos_36),
            reference=np.asarray(runner.reference_qpos_36),
            fps=50,
        )
        runner.close()


if __name__ == "__main__":
    main()
