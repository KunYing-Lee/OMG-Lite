"""Replace the official policy node; retain its remote gates and motor mapping.

Use only with the native50 controller stop/watchdog patch. Loading never
presses remote buttons. Faults stop tracking rather than switching policies.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rclpy
from g1_native50 import NativeSensors, load_tracker
from humanoid_policy.policy_node_29dof import HoloMotionPolicyNode
from humanoid_policy.policy_runtime import PolicyRuntime
from humanoid_policy.utils.remote_controller_filter import KeyMap
from std_msgs.msg import String

from omg.realtime.deployment import DeploymentBridge
from omg.realtime.native import RemotePlanner
from omg.realtime.reference_stream import ReferenceReceiver, encode_window


class Native50Runtime(PolicyRuntime):
    def __init__(self, port, old):
        super().__init__(port, num_actions=29)
        self.state = old.state
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.bridge = None
        self.active = False
        self.failed = False
        self.receiver = None

    def handle_low_state(self, msg):
        # Y must also cancel an initial plan before motion mode is entered.
        self.port.remote_controller.set(msg.wireless_remote)
        if self.pending is not None and self.port.remote_controller.button[KeyMap.Y]:
            self._close_episode()
        super().handle_low_state(msg)

    def switch_to_motion_mode(self):
        if self.failed or self.pending is not None or self.active:
            return False
        if not self.state.policy_enabled or not self.state.robot_state_ready:
            return False
        if self.port.count_publishers("/humanoid/action") != 1:
            self.stop("Nonexclusive policy publisher")
            return False
        if self.port.count_subscribers("/humanoid/tracking_stop") != 1:
            self.stop("Expected exactly one native stop subscriber")
            return False
        try:
            history = self.port.sensors.startup_history()
            self.bridge = DeploymentBridge(
                RemotePlanner(self.port.args.connect), self.port.args.text
            )
            self.pending = self.pool.submit(self.bridge.activate, history)
            self.port.get_logger().info(
                "Preparing native50 initial plan; existing velocity loop remains responsive"
            )
            return True
        except Exception as error:  # noqa: BLE001 - stop at the robot callback boundary
            self.stop(str(error))
            return False

    def _close_episode(self):
        self.active = False
        if self.pending is not None:
            # Initial inference is bounded by the transport timeout. Never
            # interpret a cancelled operator request as permission to activate.
            self.pending.cancel()
            self.pending = None
        if self.bridge is not None:
            bridge, self.bridge = self.bridge, None
            # Serialize cleanup behind initial inference without blocking remote
            # callbacks. A detached episode can never be activated by run().
            self.pool.submit(bridge.close)

    def stop(self, reason):
        self.port.stop_publisher.publish(String(data=reason))
        self.failed = True
        self.state.policy_enabled = False
        self.state.motion_in_progress = False
        self.port.get_logger().error("Native50 stopped: " + reason)
        self._close_episode()

    def switch_to_velocity_mode(self, reason=""):
        self._close_episode()
        super().switch_to_velocity_mode(reason)

    def enable_velocity_policy(self):
        if not self.failed:
            self._close_episode()
            super().enable_velocity_policy()

    def set_robot_state(self, state):
        super().set_robot_state(state)
        if state in {"ZERO_TORQUE", "EMERGENCY_STOP"}:
            self.state.policy_enabled = False
            self._close_episode()

    def run(self):
        if not getattr(self.port, "_setup_completed", False) or self.failed:
            return
        try:
            state, _measured = self.port.sensors.read(
                record_history=not self.active and self.pending is None
            )
        except Exception as error:  # noqa: BLE001 - stop at the robot callback boundary
            if self.active or self.pending is not None:
                self.stop(str(error))
                return
            self.port.sensor_error = str(error)
            self.port.sensors.history.clear()
            self.run_policy_step()
            return
        self.port.sensor_error = None
        if self.pending is not None and self.pending.done():
            try:
                self.pending.result()
                self.pending = None
                policy = self.port.native_policy
                policy.tracker.reset()
                policy.last_action.fill(0)
                policy.history.clear()
                self.receiver = ReferenceReceiver(policy.handles.onnx_to_g1)
                self.bridge.origin = (
                    time.monotonic()
                )  # initial activation, before frame0
                self.active = True
                self.state.current_policy_mode = "motion"
                self.state.motion_in_progress = True
                self.port._publish_control_params()
                self.port._publish_policy_mode()
            except Exception as error:  # noqa: BLE001 - stop at the robot callback boundary
                self.stop(str(error))
                return
        if not self.active:
            self.run_policy_step()
            return
        if not self.state.policy_enabled or not self.state.robot_state_ready:
            self.stop("Native arming state revoked")
            return
        try:
            frame = self.bridge.frame
            if frame >= self.port.args.frames:
                self.stop("Requested bounded episode completed")
                return
            target = self.bridge.origin + frame / 50
            time.sleep(max(0, target - time.monotonic()))
            state, _measured = self.port.sensors.read(record_history=False)
            qpos, cursor = self.bridge.tick()
            features, local = self.receiver.consume(
                *encode_window(qpos, cursor, "live")
            )
            action = self.port.native_policy.step(features, local, state)
            if time.monotonic() >= target + 0.02:
                raise TimeoutError("Tracker missed its20ms control interval")
            self.apply_policy_output(
                action,
                action_scale=self.port.motion_action_scale_onnx,
                default_angles=self.port.motion_default_angles_onnx,
                onnx_to_real=self.port.motion_onnx_to_real,
                is_motion=True,
            )
            self.publish_current_action()
        except Exception as error:  # noqa: BLE001 - stop at the robot callback boundary
            self.stop(str(error))

    def close(self):
        if self.active or self.pending is not None:
            self.stop("Policy process shutting down")
        self.pool.shutdown(wait=True, cancel_futures=True)


class Native50Node(HoloMotionPolicyNode):
    def __init__(self, args):
        self.args = args
        self.sensor_error = "Not initialized"
        self.attempted = False
        super().__init__()

    def _low_state_callback(self, msg):
        if getattr(self, "_setup_completed", False):
            super()._low_state_callback(msg)
        else:
            self._lowstate_msg = msg

    def _delayed_setup(self):
        if self.attempted:
            return
        self.attempted = True
        try:
            self.setup()
            self._setup_completed = True
        except Exception as error:  # noqa: BLE001 - stop at the robot callback boundary
            self.setup_error = str(error)
            self.runtime.state.policy_enabled = False
            self.get_logger().error("Native50 setup failed: " + str(error))

    def setup(self):
        super().setup()
        if float(self.dt) != 0.02:
            raise ValueError("Native50 requires a50Hz policy loop")
        self.native_policy = load_tracker(self.args.tracker)
        meta = self.native_policy.metadata
        np.testing.assert_allclose(
            self.motion_action_scale_onnx, meta.action_scale, atol=1e-7, rtol=0
        )
        np.testing.assert_allclose(
            self.motion_default_angles_onnx, meta.default_joint_pos, atol=1e-7, rtol=0
        )
        np.testing.assert_allclose(
            self.motion_kps_onnx, meta.joint_stiffness, atol=1e-7, rtol=0
        )
        np.testing.assert_allclose(
            self.motion_kds_onnx, meta.joint_damping, atol=1e-7, rtol=0
        )
        np.testing.assert_array_equal(
            self.motion_onnx_to_real, np.argsort(self.native_policy.handles.onnx_to_g1)
        )
        tracker = self.native_policy.tracker
        dim = next(i.shape[-1] for i in tracker.session.get_inputs() if "obs" in i.name)
        for _ in range(10):
            tracker.run(np.zeros((1, dim), np.float32))
        tracker.reset()
        self.sensors = NativeSensors()
        self.stop_publisher = self.create_publisher(
            String, "/humanoid/tracking_stop", 10
        )
        self.runtime = Native50Runtime(self, self.runtime)
        if self.runtime.state.policy_enabled:
            raise RuntimeError("Must initialize disabled")
        self.get_logger().info("Native50 loaded DISABLED; no remote buttons generated")

    def destroy_node(self):
        if isinstance(self.runtime, Native50Runtime):
            self.runtime.close()
        if hasattr(self, "sensors"):
            self.sensors.close()
        return super().destroy_node()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracker", required=True)
    p.add_argument("--connect", required=True)
    p.add_argument("--text", default="raise hands")
    p.add_argument("--frames", type=int, default=250)
    p.add_argument("--check", action="store_true")
    a, ros_args = p.parse_known_args()
    if a.frames <= 0:
        p.error("--frames must be positive")
    if a.check and (
        os.environ.get("ROS_DOMAIN_ID") != "213"
        or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
    ):
        raise ValueError("Load check requires isolated localhost domain213")
    rclpy.init(args=ros_args)
    node = Native50Node(a)
    try:
        if a.check:
            deadline = time.monotonic() + 120
            while not getattr(node, "_setup_completed", False):
                rclpy.spin_once(node, timeout_sec=0.02)
                if hasattr(node, "setup_error"):
                    raise RuntimeError(node.setup_error)
                if time.monotonic() > deadline:
                    raise TimeoutError("Setup timed out")
            assert not node.runtime.state.policy_enabled
            print(
                "PASS: disabled node, model mapping/gains, native50 runtime; isolated domain213"
            )
        else:
            rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
