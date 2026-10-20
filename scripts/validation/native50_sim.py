"""Validate the actual deployment observation/action path against MuJoCo tracking."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from omg.pipeline.schedule import PERIOD, REQUEST, PlanTimeline
from omg.pipeline.tracker import Tracker
from omg.realtime.policy import PreviewPolicy
from omg.realtime.reference_stream import ReferenceReceiver, encode_window
from omg.tracking.holomotion.runner import HoloMotionRolloutRunner
from omg.tracking.holomotion.runtime import (
    apply_holomotion_action_pd,
    extract_g1_qpos,
    g1_state_from_qpos_history,
)

p = argparse.ArgumentParser()
p.add_argument("--plans", required=True)
p.add_argument("--tracker", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
root = Path(a.plans)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
session = ort.InferenceSession(
    a.tracker, sess_options=opts, providers=["CUDAExecutionProvider"]
)
results = {}
for name in ["walk", "squat", "hands"]:
    plans = [np.load(root / name / f"plan_{i:03d}.npz")["qpos"] for i in range(7)]
    history = np.load(root / name / "rollout.npz")["initial_history"]
    states = []
    for latency in [0, 0.03, 0.05, 0.07]:
        runner = HoloMotionRolloutRunner(
            holomotion_onnx=a.tracker, session=session, video=False
        )
        initial, vel = g1_state_from_qpos_history(
            runner.model, runner.g1_handles, history, fps=50
        )
        runner.reset(initial, vel)
        timeline = PlanTimeline(plans[0])
        tracker = Tracker(runner)
        tracker.activate(timeline.qpos)
        receiver = ReferenceReceiver(runner.holomotion_handles.onnx_to_g1)
        policy = PreviewPolicy(
            runner.tracker, runner.metadata, runner.holomotion_handles
        )
        executed = []
        actions = []
        ready = None
        for frame in range(150):
            if ready is not None and frame / 50 >= ready:
                r = timeline.pending
                timeline.accept(r, plans[r.plan_id], current_frame=frame)
                tracker.activate(timeline.qpos)
                ready = None
            timeline.check_frame(frame)
            if frame % PERIOD == REQUEST:
                r = timeline.request(frame)
                if latency == 0:
                    timeline.accept(r, plans[r.plan_id], current_frame=frame)
                    tracker.activate(timeline.qpos)
                else:
                    ready = frame / 50 + latency
            if latency == 0:
                state = tracker.step(frame, frame // PERIOD)
                action = runner.last_action
            else:
                features, cursor = receiver.consume(
                    *encode_window(timeline.qpos, frame, "sim")
                )
                g = runner.g1_handles
                d = runner.data
                state_input = {
                    "joint_pos": d.qpos[g["joint_qpos_adr"]].copy(),
                    "joint_vel": d.qvel[g["joint_dof_adr"]].copy(),
                    "root_quat_wxyz": d.xquat[g["pelvis_body_id"]].copy(),
                    "gyro": d.sensordata[
                        g["pelvis_gyro_adr"] : g["pelvis_gyro_adr"] + 3
                    ].copy(),
                }
                action = policy.step(features, cursor, state_input)
                apply_holomotion_action_pd(
                    model=runner.model,
                    data=d,
                    holomotion_handles=runner.holomotion_handles,
                    action=action,
                    control_substeps=runner.control_substeps,
                    action_clip=10.0,
                )
                state = extract_g1_qpos(d, g)
            executed.append(state)
            actions.append(action.copy())
        states.append(np.asarray(executed))
        q = np.asarray(executed)
        np.savez_compressed(
            out / f"{name}_{int(latency * 1000)}ms.npz",
            executed=q,
            reference=timeline.qpos[:150],
            actions=actions,
            fps=50,
        )
        runner.close()
    diffs = [float(np.max(abs(q - states[0]))) for q in states[1:]]
    # Both paths execute the same policy on the same raw references.
    for q in states[1:]:
        np.testing.assert_allclose(q, states[0], atol=1e-6, rtol=0)
    results[name] = {
        "frames": 150,
        "latencies_ms": [30, 50, 70],
        "max_execution_abs_diff": diffs,
    }
    print(name, results[name], flush=True)
(out / "deployment_parity.json").write_text(json.dumps(results, indent=2))
