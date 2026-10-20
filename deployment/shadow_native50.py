"""Live sensors -> wired planner -> tracker; no actuator imports or writers."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from g1_native50 import NativeSensors, load_tracker

from omg.realtime.deployment import DeploymentBridge
from omg.realtime.native import RemotePlanner
from omg.realtime.reference_stream import ReferenceReceiver, encode_window


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracker", required=True)
    p.add_argument("--connect", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--text", default="raise hands")
    p.add_argument("--frames", type=int, default=500)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    policy = load_tracker(a.tracker)
    sensors = NativeSensors()
    planner = RemotePlanner(a.connect)
    bridge = DeploymentBridge(planner, a.text)
    actions, states, timings = [], [], []
    report = {
        "motor_commands_sent": False,
        "scope": "Live sensor shadow; candidate actions are never sent",
    }
    try:
        # DDS discovery occurs before the control clock. Sensor faults after
        # startup are fatal, never substituted with an earlier state.
        time.sleep(1)
        sensors.history.clear()
        deadline = time.monotonic() + 3
        while len(sensors.history) < 10:
            sensors.read(record_history=True)
            time.sleep(0.01)
            if time.monotonic() > deadline:
                raise TimeoutError("Native 50Hz startup history capture timed out")
        history = sensors.startup_history()
        warm = planner.plan(history, a.text, 0)
        receiver = ReferenceReceiver(policy.handles.onnx_to_g1)
        features, cursor = receiver.consume(*encode_window(warm["qpos"], 0, "warm"))
        for _ in range(10):
            state, _ = sensors.read()
            policy.step(features, cursor, state)
        policy.tracker.reset()
        policy.last_action.fill(0)
        policy.history.clear()
        sensors.history.clear()
        deadline = time.monotonic() + 3
        while len(sensors.history) < 10:
            sensors.read(record_history=True)
            time.sleep(0.01)
            if time.monotonic() > deadline:
                raise TimeoutError("Native 50Hz startup history capture timed out")
        history = sensors.startup_history()
        receiver = ReferenceReceiver(policy.handles.onnx_to_g1)
        bridge.activate(history)
        for frame in range(a.frames):
            target = bridge.origin + frame / 50
            time.sleep(max(0, target - time.monotonic()))
            start = time.monotonic()
            qpos, cursor = bridge.tick()
            features, local = receiver.consume(*encode_window(qpos, cursor, "shadow"))
            state, measured = sensors.read()
            action = policy.step(features, local, state)
            elapsed = time.monotonic() - start
            if time.monotonic() >= target + 0.02:
                raise TimeoutError(
                    f"Tracker control interval exceeded at frame {frame}"
                )
            actions.append(action)
            states.append(measured)
            timings.append(elapsed * 1000)
        report.update(
            status="passed",
            frames=len(actions),
            events=bridge.timeline.events,
            compute_ms={
                "p50": float(np.median(timings)),
                "max": float(np.max(timings)),
            },
        )
    except Exception as error:
        report.update(status="failed", error=str(error), frames=len(actions))
        raise
    finally:
        bridge.close()
        sensors.close()
        np.savez_compressed(
            out / "trace.npz", actions=actions, measured=states, compute_ms=timings
        )
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
