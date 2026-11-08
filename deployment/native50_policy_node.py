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

from omg.pipeline.schedule import Schedule
from omg.realtime.deployment import DeploymentBridge
from omg.realtime.native import RemotePlanner
from omg.realtime.recording import Recorder
from omg.realtime.reference_stream import ReferenceReceiver, encode_window

from residual50 import (
    ACTION_FRAMES,
    ResidualActor,
    ResidualCorrector,
    ResidualRegistry,
    open_onnx_session,
)


class Native50Runtime(PolicyRuntime):
    def __init__(self, port, old):
        super().__init__(port, num_actions=29)
        self.state = old.state
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.bridge = None
        self.active = False
        self.failed = False
        self._operator_stand = False
        self._last_y = 0
        self.receiver = None
        self.corrector = None
        self.tape = None
        # Module-level recorder (src/omg/realtime/recording.py): records the four
        # module outputs (plan / residual delta / action / measured), the event
        # timeline and the transform parameters. Model-agnostic.
        self.recorder = None
        self._recorder_configured = False
        self._tau = np.zeros(29, np.float32)
        record_path = getattr(self.port.args, "record", "")
        if record_path:
            self.recorder = Recorder(
                record_path,
                {
                    "node_file": os.path.abspath(__file__),
                    "args": {k: v for k, v in vars(self.port.args).items() if k != "record"},
                },
            )
            self.recorder.update_meta(
                planner_endpoint=getattr(self.port.args, "connect", None),
                text=getattr(self.port.args, "text", None),
                switch_seconds=getattr(self.port.args, "switch_seconds", None),
                residual_arg=getattr(self.port.args, "residual", None),
            )

    def _configure_recorder(self):
        """One-shot: store the transform parameters the recording needs to be replayable."""
        if self.recorder is None or self._recorder_configured:
            return
        self._recorder_configured = True
        meta = dict(
            action_scale=np.asarray(self.port.motion_action_scale_onnx, np.float32).tolist(),
            default_angles=np.asarray(self.port.motion_default_angles_onnx, np.float32).tolist(),
            onnx_to_real=np.asarray(self.port.motion_onnx_to_real, np.int64).tolist(),
            residual_active=self.corrector is not None,
        )
        policy = getattr(self.port, "native_policy", None)
        metadata = getattr(policy, "metadata", None)
        if metadata is not None:
            meta["tracker_metadata"] = dict(
                obs_schema_version=getattr(metadata, "obs_schema_version", None),
                n_fut_frames=getattr(metadata, "n_fut_frames", None),
                context_length=getattr(metadata, "context_length", None),
            )
        if self.corrector is not None:
            manifest = getattr(getattr(self.corrector, "actor", None), "manifest", {})
            meta["residual_resolved"] = (
                manifest.get("name") or manifest.get("model") or manifest.get("checkpoint")
            )
            meta["residual_manifest"] = manifest
        self.recorder.update_meta(**meta)

    def handle_low_state(self, msg):
        # Y cancels an initial plan before activation, and is a controlled
        # stand request while walking: motion -> velocity/stand mode without
        # an immediate damping cutoff.  SELECT remains the hard stop.
        self.port.remote_controller.set(msg.wireless_remote)
        y = int(self.port.remote_controller.button[KeyMap.Y])
        if y and not self._last_y:
            if self.pending is not None:
                self._close_episode()
            elif self.active:
                self._operator_stand = True
        self._last_y = y
        try:
            self._tau = np.asarray(
                [float(m.tau_est) for m in msg.motor_state[:29]], np.float32
            )
        except Exception:  # noqa: BLE001 - recording must never break control
            pass
        super().handle_low_state(msg)

    def switch_to_motion_mode(self):
        if self.failed or self.pending is not None or self.active:
            return False
        if not self.state.policy_enabled or not self.state.robot_state_ready:
            return False
        if self.port.count_publishers("/humanoid/action") != 1:
            self.stop("Nonexclusive policy publisher")
            return False
        n_stop = self.port.count_subscribers("/humanoid/tracking_stop")
        if n_stop != 1:
            if getattr(self.port.args, "allow_no_native_stop", False) and n_stop == 0:
                self.port.get_logger().warn(
                    "No native tracking_stop subscriber (original controller); "
                    "stop path is operator SELECT/L1"
                )
            else:
                self.stop("Expected exactly one native stop subscriber")
                return False
        try:
            self.port.get_logger().info(
                "startup history frames: %d (last %.3fs ago)"
                % (
                    len(self.port.sensors.history),
                    -1.0
                    if self.port.sensors.last_history_time is None
                    else time.monotonic() - self.port.sensors.last_history_time,
                )
            )
            history = self.port.sensors.startup_history()
            schedule = Schedule(switch_seconds=self.port.args.switch_seconds)
            self.bridge = DeploymentBridge(
                RemotePlanner(self.port.args.connect, schedule),
                self.port.args.text,
                schedule=schedule,
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
            message = str(error)
            if message != getattr(self, "_last_sensor_error", None):
                self._last_sensor_error = message
                self.port.get_logger().warn(
                    "Native50 idle sensor read failed (history=%d): %s"
                    % (len(self.port.sensors.history), message)
                )
            self.port.sensor_error = message
            self.port.sensors.history.clear()
            self.run_policy_step()
            return
        self.port.sensor_error = None
        if self._operator_stand:
            self._operator_stand = False
            if self.active:
                self.port.get_logger().warn(
                    "Operator Y: controlled stop -> velocity/stand mode"
                )
                self.switch_to_velocity_mode("operator Y controlled stop")
                return
        if self.pending is not None and self.pending.done():
            try:
                self.pending.result()
                self.pending = None
                policy = self.port.native_policy
                policy.tracker.reset()
                policy.last_action.fill(0)
                policy.history.clear()
                self.receiver = ReferenceReceiver(policy.handles.onnx_to_g1)
                self.corrector = getattr(self.port, "residual_corrector", None)
                self.tape = None
                if self.corrector is not None:
                    self.corrector.reset()
                    initial = np.asarray(self.bridge.timeline.qpos, np.float32)
                    self.tape = np.zeros((self.port.args.frames + 111, 36), np.float32)
                    self.tape[: len(initial)] = initial
                    self.tape[len(initial):] = initial[-1]
                    self.port.get_logger().info("Residual enabled: " + self.port.residual_name)
                self.active = True
                self.state.current_policy_mode = "motion"
                self.state.motion_in_progress = True
                self.port._publish_control_params()
                self.port._publish_policy_mode()
                # Anchor the schedule after the activation work and one frame
                # ahead, so the first control tick lands inside its 20 ms window.
                self.bridge.origin = time.monotonic() + 0.02
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
            if frame < 3:
                # Startup ticks: first-call costs can exceed one 20 ms period.
                # Re-anchor per tick so tick() stays inside its own window.
                self.bridge.origin = time.monotonic() - frame / 50.0
            target = self.bridge.origin + frame / 50
            time.sleep(max(0, target - time.monotonic()))
            t0 = time.monotonic()
            state, _measured = self.port.sensors.read(record_history=False)
            t1 = time.monotonic()
            qpos, cursor = self.bridge.tick()
            t2 = time.monotonic()
            timeline = self.bridge.timeline
            pending = timeline.pending
            if pending is not None and getattr(self, "_plan_pending_id", None) != pending.plan_id:
                self._plan_pending_id = pending.plan_id
                self._plan_request_at = time.monotonic()
                self.port.get_logger().info(
                    "plan %d requested frame=%d switch=%d deadline=%d"
                    % (pending.plan_id, pending.request_frame,
                       pending.switch_frame, pending.deadline_frame)
                )
            seen = getattr(self, "_plan_events_seen", 0)
            if len(timeline.events) > seen:
                for event in timeline.events[seen:]:
                    started = getattr(self, "_plan_request_at", None)
                    took = -1.0 if started is None else (time.monotonic() - started) * 1000.0
                    self.port.get_logger().info(
                        "plan %d accepted req=%d accept=%d switch=%d (%.0f ms)"
                        % (event["plan_id"], event["request_frame"],
                           event["accepted_frame"], event["switch_frame"], took)
                    )
                    if self.tape is not None:
                        plan = timeline.raw[-1]
                        switch = int(event["switch_frame"])
                        self.tape[switch: switch + plan.shape[0], 0:7] = plan[:, 0:7]
                    if self.recorder is not None:
                        self.recorder.plan(
                            plan_id=event["plan_id"],
                            request_frame=event["request_frame"],
                            accept_frame=event["accepted_frame"],
                            switch_frame=event["switch_frame"],
                            deadline_frame=int(event.get("deadline_frame",
                                                         event["switch_frame"] - 11)),
                            block=timeline.raw[-1],
                            request_time=started,
                            accept_time=time.monotonic(),
                        )
                self._plan_events_seen = len(timeline.events)
            reference = qpos
            correction = None
            if self.corrector is not None:
                correction = self.corrector.step(
                    frame=frame,
                    raw_reference=qpos,
                    corrected_tape=self.tape,
                    robot_state=state,
                    last_action_onnx=self.port.native_policy.last_action,
                )
                if correction is not None:
                    self.tape[correction.start: correction.start + ACTION_FRAMES, 7:36] = (
                        correction.joints_g1
                    )
                    if self.recorder is not None:
                        self.recorder.decision(
                            frame=frame,
                            start=correction.start,
                            delta=correction.delta,
                            latent_norm=float(np.linalg.norm(correction.latent)),
                        )
                reference = self.tape
            packet = encode_window(reference, cursor, "live")
            t2a = time.monotonic()
            features, local = self.receiver.consume(*packet)
            t3 = time.monotonic()
            action = self.port.native_policy.step(features, local, state)
            t4 = time.monotonic()
            # The first residual/tracker tick is cold and can be 20.2 ms.
            # Allow a slightly wider startup window only for the first frames;
            # steady-state ticks stay on the strict 20 ms budget.
            slack = 0.035 if frame < 3 else 0.02
            if time.monotonic() >= target + slack:
                raise TimeoutError(
                    "Tracker missed its20ms control interval (frame=%d, "
                    "elapsed=%.1fms, slack=%.0fms, read=%.1f, tick=%.1f, "
                    "encode=%.1f, consume=%.1f, step=%.1f)"
                    % (
                        frame,
                        (time.monotonic() - target) * 1000.0,
                        slack * 1000.0,
                        (t1 - t0) * 1000.0,
                        (t2 - t1) * 1000.0,
                        (t2a - t2) * 1000.0,
                        (t3 - t2a) * 1000.0,
                        (t4 - t3) * 1000.0,
                    )
                )
            self.apply_policy_output(
                action,
                action_scale=self.port.motion_action_scale_onnx,
                default_angles=self.port.motion_default_angles_onnx,
                onnx_to_real=self.port.motion_onnx_to_real,
                is_motion=True,
            )
            self.publish_current_action()
            if self.recorder is not None:
                self._configure_recorder()
                self.recorder.frame(
                    frame=frame,
                    t=time.monotonic(),
                    action=action,
                    q=state["joint_pos"],
                    dq=state["joint_vel"],
                    tau=self._tau,
                    quat=state["root_quat_wxyz"],
                    gyro=state["gyro"],
                    residual=self.corrector is not None,
                    read_ms=(t1 - t0) * 1000.0,
                    tick_ms=(t2 - t1) * 1000.0,
                    encode_ms=(t2a - t2) * 1000.0,
                    consume_ms=(t3 - t2a) * 1000.0,
                    step_ms=(t4 - t3) * 1000.0,
                )
            if frame % 50 == 0:
                try:
                    pdo = self.port._policy_dof_order()
                    row = qpos[int(cursor)]
                    k = 7 + pdo.index("left_knee_joint")
                    w = 7 + pdo.index("waist_pitch_joint")
                    self.port.get_logger().info(
                        "ref frame=%d knee=%+.2f waist_pitch=%+.2f root_z=%.3f"
                        % (frame, row[k], row[w], row[2])
                    )
                except Exception:  # noqa: BLE001 - diagnostics must not stop motion
                    pass
        except Exception as error:  # noqa: BLE001 - stop at the robot callback boundary
            self.stop(str(error))

    def close(self):
        if self.active or self.pending is not None:
            self.stop("Policy process shutting down")
        if self.recorder is not None:
            self.recorder.close()
        self.pool.shutdown(wait=True, cancel_futures=True)


class Native50Node(HoloMotionPolicyNode):
    def __init__(self, args):
        self.args = args
        self.sensor_error = "Not initialized"
        self.attempted = False
        super().__init__()
        self.get_logger().info("native50 node build: first-tick-fix-670a04d")

    def _policy_dof_order(self):
        if not hasattr(self, "_pdo_cache"):
            import yaml

            cfg = yaml.safe_load(open(self.args.config))
            self._pdo_cache = cfg["policy_dof_order"]
        return self._pdo_cache

    def _warm_reference_stream(self):
        try:
            self._policy_dof_order()
            from omg.tracking.holomotion.reference import precompute_reference_features

            warm = np.zeros((13, 36), np.float32)
            warm[:, 2] = 0.78
            warm[:, 3] = 1.0
            warm_state = dict(
                joint_pos=np.zeros(29, np.float32),
                joint_vel=np.zeros(29, np.float32),
                root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], np.float32),
                gyro=np.zeros(3, np.float32),
            )
            t0 = time.monotonic()
            receiver = ReferenceReceiver(self.native_policy.handles.onnx_to_g1)
            for warm_frame in (0, 1):
                features, local = receiver.consume(
                    *encode_window(warm, warm_frame, "warm")
                )
                self.native_policy.step(features, local, warm_state)
            self.native_policy.tracker.reset()
            self.native_policy.last_action.fill(0)
            self.native_policy.history.clear()
            self.get_logger().info(
                "reference+tracker warm: %.1f ms (encode+consume+step x2)"
                % ((time.monotonic() - t0) * 1000.0)
            )
        except Exception as error:  # noqa: BLE001 - warmup must never block startup
            self.get_logger().warn("reference stream warm failed: %s" % error)

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
        self.sensors = NativeSensors(
            dq_source=getattr(self.args, "dq_source", "qdiff")
        )
        self.stop_publisher = self.create_publisher(
            String, "/humanoid/tracking_stop", 10
        )
        self.residual_name = None
        self.residual_corrector = None
        registry = ResidualRegistry.discover(self.args.residual_dir)
        entry = registry.resolve(
            self.args.residual, self.args.text, warn=self.get_logger().warn
        )
        if entry is not None:
            if abs(float(entry.manifest["switch_seconds"]) - float(self.args.switch_seconds)) > 1e-9:
                raise ValueError("residual switch_seconds does not match --switch-seconds")
            self.residual_corrector = ResidualCorrector(
                ResidualActor(entry.manifest, open_onnx_session(entry))
            )
            warm = np.zeros((200, 36), np.float32)
            warm[:, 2] = 0.78
            warm[:, 3] = 1.0
            warm_state = dict(
                joint_pos=warm[0, 7:].copy(),
                joint_vel=np.zeros(29, np.float32),
                root_quat_wxyz=np.array([1, 0, 0, 0], np.float32),
                gyro=np.zeros(3, np.float32),
            )
            for _ in range(2):
                self.residual_corrector.step(
                    frame=0, raw_reference=warm, corrected_tape=warm,
                    robot_state=warm_state, last_action_onnx=np.zeros(29, np.float32),
                )
                self.residual_corrector.reset()
            self.residual_name = entry.name
            self.get_logger().info("Residual model: " + entry.name)
        else:
            self.get_logger().info("Residual disabled")
        self.runtime = Native50Runtime(self, self.runtime)
        if self.runtime.state.policy_enabled:
            raise RuntimeError("Must initialize disabled")
        self._warm_reference_stream()
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
    p.add_argument("--switch-seconds", type=float, default=1.0)
    p.add_argument(
        "--dq-source",
        choices=("qdiff", "raw", "lowpass"),
        default="lowpass",
        help="joint velocity fed to the tracker: 15Hz causal low-pass (default), qdiff, or raw LowState dq",
    )
    p.add_argument("--residual", default="off", help="off | auto | <registry entry>")
    p.add_argument(
        "--residual-dir",
        default="/workspace/omglite-shadow-20261025/experiments/residuals",
    )
    p.add_argument("--allow-no-native-stop", action="store_true")
    p.add_argument("--record", default="", help="per-frame CSV: plan/tape/action/state/timings")
    p.add_argument(
        "--config",
        default="/workspace/omglite-shadow-20261025/config/g1_29dof_holomotion.yaml",
    )
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
