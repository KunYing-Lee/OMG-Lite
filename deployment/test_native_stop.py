"""Exercise the C++ controller only in localhost ROS domain213, with remapped I/O."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import rclpy
import yaml
from std_msgs.msg import Float32MultiArray, String
from unitree_hg.msg import LowState


def main():
    if (
        os.environ.get("ROS_DOMAIN_ID") != "213"
        or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
    ):
        raise RuntimeError("Stop tests require isolated localhost ROS domain 213")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    binary = args.controller.resolve(strict=True)
    config = str(args.config.resolve(strict=True))
    cfg = yaml.safe_load(Path(config).read_text())
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = rclpy.create_node("native50_stop_test")
    results = {}
    for case in ["explicit_stop", "action_watchdog", "nonfinite_action"]:
        prefix = "/omg_test_" + case
        log = (root / f"{case}.log").open("w")
        argv = [str(binary), "--ros-args", "-p", "config_path:=" + config]
        for topic in [
            "lowstate",
            "lowcmd",
            "robot_state",
            "humanoid/action",
            "humanoid/kps",
            "humanoid/kds",
            "humanoid/tracking_stop",
        ]:
            argv += ["-r", "/" + topic + ":=" + prefix + "/" + topic]
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        states = []
        sub = node.create_subscription(
            String,
            prefix + "/robot_state",
            lambda m, states=states: states.append((time.monotonic(), m.data)),
            10,
        )
        low = node.create_publisher(LowState, prefix + "/lowstate", 10)
        stop = node.create_publisher(String, prefix + "/humanoid/tracking_stop", 10)
        action = node.create_publisher(
            Float32MultiArray, prefix + "/humanoid/action", 10
        )
        kp = node.create_publisher(Float32MultiArray, prefix + "/humanoid/kps", 10)
        kd = node.create_publisher(Float32MultiArray, prefix + "/humanoid/kds", 10)
        msg = LowState()
        for i, n in enumerate(cfg["complete_dof_order"]):
            msg.motor_state[i].q = float(cfg["default_joint_angles"][n])
        zeros = Float32MultiArray(data=[0.0] * 29)
        start = time.monotonic()
        stopped_at = None
        try:
            while time.monotonic() - start < 3:
                t = time.monotonic() - start
                keys = (
                    (1 << 2) if 0.6 < t < 0.65 else ((1 << 8) if 0.8 < t < 0.85 else 0)
                )
                msg.wireless_remote[2] = keys & 255
                msg.wireless_remote[3] = keys >> 8
                low.publish(msg)
                kp.publish(zeros)
                kd.publish(zeros)
                if case == "explicit_stop" and t > 0.5 and stopped_at is None:
                    stop.publish(String(data="isolated test"))
                    stopped_at = time.monotonic()
                elif case == "nonfinite_action" and t > 0.5 and stopped_at is None:
                    action.publish(Float32MultiArray(data=[float("nan")] * 29))
                    stopped_at = time.monotonic()
                rclpy.spin_once(node, timeout_sec=0.002)
                if any(s == "EMERGENCY_STOP" for _, s in states):
                    break
            assert any(s == "EMERGENCY_STOP" for _, s in states), (case, states)
            if case == "action_watchdog":
                assert any(s == "POLICY" for _, s in states), states
                entered = next(t for t, s in states if s == "POLICY")
                stopped_at = entered
            stopped = next(t for t, s in states if s == "EMERGENCY_STOP")
            results[case] = {
                "passed": True,
                "stop_delay_ms": (stopped - stopped_at) * 1000,
            }
        finally:
            child.terminate()
            child.wait(timeout=5)
            log.close()
            for pub in [low, stop, action, kp, kd]:
                node.destroy_publisher(pub)
            node.destroy_subscription(sub)
    node.destroy_node()
    rclpy.shutdown()
    (root / "stop-tests.json").write_text(
        json.dumps(
            {"domain": 213, "motor_transport": False, "results": results}, indent=2
        )
    )
    print(results)


if __name__ == "__main__":
    main()
