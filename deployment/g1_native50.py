"""Native G1 sensor and tracker adapters; this module creates no motor writers."""

import time
from collections import deque
from types import SimpleNamespace

import numpy as np

from omg.pipeline.schedule import motion
from omg.realtime.policy import PreviewPolicy
from omg.robots.g1.constants import G1_JOINT_NAMES
from omg.tracking.holomotion.runtime import (
    HoloMotionTrackerSession,
    load_holomotion_metadata,
)


class _RosReader:
    """Use ROS's compiled message conversion, not Python per-field DDS decoding."""

    def __init__(self, subscription, kind):
        self.subscription, self.kind = subscription, kind

    def take(self, count):
        rows = []
        for _ in range(count):
            with self.subscription.handle:
                item = self.subscription.handle.take_message(self.kind, False)
            if item is None:
                break
            msg, info = item
            fields = {
                name: getattr(msg, name) for name in msg.get_fields_and_field_types()
            }
            fields["sample_info"] = SimpleNamespace(
                valid_data=True,
                publication_handle=bytes(info["publisher_gid"]).hex(),
                source_timestamp=int(info["source_timestamp"]),
            )
            rows.append(SimpleNamespace(**fields))
        return rows


class NativeSensors:
    """Read native LowState and odommodestate; never synthesize root height.

    Startup history follows native 50Hz odometry timestamps, paired to the
    nearest actual 1kHz LowState sample (within two 1kHz periods). No sensor
    values are interpolated. Each source must be at most two 50Hz periods old.
    Active DDS writer identities are pinned on first receipt for this boot.
    """

    def __init__(self):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from unitree_go.msg import SportModeState
        from unitree_hg.msg import LowState

        if not rclpy.ok():
            rclpy.init(args=[])
        self.node = rclpy.create_node("omg_native50_sensors")

        def reader(topic, kind, depth):
            qos = QoSProfile(depth=depth, reliability=ReliabilityPolicy.BEST_EFFORT)
            sub = self.node.create_subscription(kind, topic, lambda msg: None, qos)
            return _RosReader(sub, kind)

        self.readers = {
            "low": reader("/lowstate", LowState, 1),
            "odom": reader("/odommodestate", SportModeState, 1),
        }
        self.history_low_reader = reader("/lowstate", LowState, 64)
        self.low_history = deque(maxlen=128)
        self.last_history_stamp = None
        self.latest = {}
        self.writers = {}
        self.stamps = {}
        self.history = deque(maxlen=10)
        self.last_history_time = None

    def read(self, *, record_history=False):
        if record_history:
            self.low_history.extend(
                m for m in self.history_low_reader.take(64) if m.sample_info.valid_data
            )
        for key, reader in self.readers.items():
            for msg in reader.take(1):
                if not msg.sample_info.valid_data:
                    continue
                writer = msg.sample_info.publication_handle
                if key in self.writers and self.writers[key] != writer:
                    raise RuntimeError(f"{key}: active DDS writer changed")
                self.writers[key] = writer
                stamp = (
                    int(msg.tick)
                    if key == "low"
                    else int(msg.stamp.sec) * 10**9 + int(msg.stamp.nanosec)
                )
                if stamp < self.stamps.get(key, stamp):
                    raise RuntimeError(f"{key}: sensor clock reversed")
                if stamp == self.stamps.get(key):
                    continue
                self.stamps[key] = stamp
                # DDS source timestamps are in the robot wall-clock domain.
                age = time.time() - msg.sample_info.source_timestamp / 1e9
                self.latest[key] = (msg, time.monotonic() - age)
        if len(self.latest) != 2:
            raise RuntimeError("Waiting for both native sensor streams")
        now = time.monotonic()
        low, lt = self.latest["low"]
        odom, ot = self.latest["odom"]
        if min(now - lt, now - ot) < -0.02 or max(now - lt, now - ot) > 0.04:
            raise RuntimeError(
                f"Native sensor age/skew exceeds sampling contract: low={now - lt:.4f}s odom={now - ot:.4f}s skew={abs(lt - ot):.4f}s"
            )
        if (
            int(odom.error_code) != 0
            or not np.isfinite(odom.body_height)
            or odom.body_height <= 0
        ):
            raise RuntimeError("Native body-height source is invalid")
        state = {
            "joint_pos": np.asarray([m.q for m in low.motor_state[:29]], np.float32),
            "joint_vel": np.asarray([m.dq for m in low.motor_state[:29]], np.float32),
            "root_quat_wxyz": np.asarray(low.imu_state.quaternion, np.float32),
            "gyro": np.asarray(low.imu_state.gyroscope, np.float32),
        }
        qpos = np.concatenate(
            (
                np.asarray(odom.position[:2], np.float32),
                [odom.body_height],
                state["root_quat_wxyz"],
                state["joint_pos"],
            )
        ).astype(np.float32)
        motion(qpos[None], 1)
        if record_history and self.stamps["odom"] != self.last_history_stamp:
            self.low_history.extend(
                m for m in self.history_low_reader.take(64) if m.sample_info.valid_data
            )
            stamp = int(odom.sample_info.source_timestamp)
            if not self.low_history:
                raise RuntimeError(
                    "LowState history unavailable for native timestamp pairing"
                )
            paired = min(
                self.low_history,
                key=lambda m: abs(int(m.sample_info.source_timestamp) - stamp),
            )
            skew = abs(int(paired.sample_info.source_timestamp) - stamp) / 1e9
            if skew > 0.002:
                raise RuntimeError(
                    f"No LowState within two 1kHz periods of odometry: {skew:.4f}s"
                )
            if paired.sample_info.publication_handle != self.writers["low"]:
                raise RuntimeError("Startup LowState writer identity mismatch")
            paired_qpos = np.concatenate(
                (
                    np.asarray(odom.position[:2], np.float32),
                    [odom.body_height],
                    paired.imu_state.quaternion,
                    [m.q for m in paired.motor_state[:29]],
                )
            ).astype(np.float32)
            if (
                self.last_history_stamp is not None
                and (self.stamps["odom"] - self.last_history_stamp) / 1e9 > 0.04
            ):
                self.history.clear()
            self.history.append(motion(paired_qpos[None], 1)[0])
            self.last_history_time = now
            self.last_history_stamp = self.stamps["odom"]
        return state, qpos

    def close(self):
        self.node.destroy_node()

    def startup_history(self):
        if len(self.history) != 10:
            raise RuntimeError("Ten fresh 50Hz sensor history frames are required")
        if time.monotonic() - self.last_history_time > 0.04:
            raise RuntimeError("Startup history is stale")
        return motion(np.asarray(self.history), 10)


def load_tracker(path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=opts, providers=["CUDAExecutionProvider"]
    )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA tracker required")
    metadata = load_holomotion_metadata(session)
    mapping = np.asarray([G1_JOINT_NAMES.index(n) for n in metadata.joint_names])
    handles = SimpleNamespace(
        onnx_to_g1=mapping, default_joint_pos=metadata.default_joint_pos
    )
    tracker = HoloMotionTrackerSession(session, metadata)
    return PreviewPolicy(tracker, metadata, handles)
