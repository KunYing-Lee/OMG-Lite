"""Native G1 sensor and tracker adapters; this module creates no motor writers.

dsh patch 2026-10-31 (delta vs /workspace/native50/deployment/g1_native50.py):
  * LowState is consumed in raw CDR form (take_message(..., True)) into a
    timestamp ring, then only the sample nearest the odometry timestamp is
    deserialized (rclpy.serialization.deserialize_message). Measured cost:
    raw take 0.007 ms/message, single deserialize 1.06 ms, versus 1.06 ms x 64
    when draining the whole queue.
  * The duplicate 64-message drain in read() is removed.
  * `publisher_gid` is absent in this rclpy build; writer pinning degrades to a
    one-time warning instead of KeyError.
  * The 2 ms pairing contract is kept unchanged (measured nearest-sample skew
    is p50 0.03 ms, p95 0.42 ms on this robot).
"""
import bisect
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import numpy.testing  # preload: PlanTimeline.request() asserts history equality;
# the first numpy.testing use otherwise costs ~18 ms inside the control tick.

from omg.pipeline.schedule import motion
from omg.realtime.policy import PreviewPolicy
from omg.robots.g1.constants import G1_JOINT_NAMES
from omg.tracking.holomotion.runtime import (
    HoloMotionTrackerSession, load_holomotion_metadata,
)
from joint_velocity import JointVelocityEstimator

MAX_PAIR_SKEW_S = 0.002          # unchanged contract: two 1 kHz periods


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
            fields = {name: getattr(msg, name) for name in msg.get_fields_and_field_types()}
            fields['sample_info'] = SimpleNamespace(
                valid_data=True,
                publication_handle=(bytes(info['publisher_gid']).hex()
                                    if info.get('publisher_gid') is not None else 'no-gid'),
                source_timestamp=int(info['source_timestamp']))
            rows.append(SimpleNamespace(**fields))
        return rows

    def take_raw(self, count):
        """Return (source_timestamp_ns, serialized_bytes) without deserializing."""
        rows = []
        for _ in range(count):
            with self.subscription.handle:
                item = self.subscription.handle.take_message(self.kind, True)
            if item is None:
                break
            payload, info = item
            rows.append((int(info['source_timestamp']), payload))
        return rows


class NativeSensors:
    """Read native LowState and odommodestate; never synthesize root height.

    Startup history follows native 50Hz odometry timestamps, paired to the
    nearest actual 1kHz LowState sample (within two 1kHz periods). No sensor
    values are interpolated. Each source must be at most two 50Hz periods old.
    Active DDS writer identities are pinned on first receipt for this boot.

    Pairing path: every read drains LowState as raw bytes into `low_raw`
    (timestamp ring, 256 x ~2 KB). When odometry advances, the sample closest
    to the odometry source_timestamp is deserialized 鈥?one message, not 64.
    """
    def __init__(self, dq_source="qdiff"):
        if dq_source not in {"qdiff", "raw"}:
            raise ValueError(f"unsupported dq_source={dq_source!r}")
        self.dq_source = dq_source
        self._joint_velocity = (
            JointVelocityEstimator(joints=29, dt=0.02)
            if dq_source == "qdiff"
            else None
        )
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from unitree_hg.msg import LowState
        from unitree_go.msg import SportModeState
        if not rclpy.ok():
            rclpy.init(args=[])
        self.node = rclpy.create_node('omg_native50_sensors')
        self._low_type = LowState

        def reader(topic, kind, depth):
            qos = QoSProfile(depth=depth, reliability=ReliabilityPolicy.BEST_EFFORT)
            sub = self.node.create_subscription(kind, topic, lambda msg: None, qos)
            return _RosReader(sub, kind)

        self.readers = {
            'low': reader('/lowstate', LowState, 1),
            'odom': reader('/odommodestate', SportModeState, 1),
        }
        self.history_low_reader = reader('/lowstate', LowState, 64)
        self.low_raw = deque(maxlen=256)     # (source_timestamp_ns, cdr_bytes)
        self.low_history = deque(maxlen=128)  # legacy, unused
        self.last_history_stamp = None
        self.latest = {}
        self.writers = {}
        self.stamps = {}
        self.history = deque(maxlen=10)
        self.last_history_time = None
        self.last_skew = None
        self._warned_no_gid = False

    def _drain_raw_lowstate(self):
        for ts, payload in self.history_low_reader.take_raw(512):
            self.low_raw.append((ts, payload))

    def _pair_lowstate(self, odom_ts_ns):
        """Return (deserialized LowState, skew_seconds) nearest to odom timestamp."""
        if not self.low_raw:
            raise RuntimeError('LowState history unavailable for native timestamp pairing')
        stamps = [ts for ts, _ in self.low_raw]
        i = bisect.bisect_left(stamps, odom_ts_ns)
        candidates = []
        for j in (i - 1, i):
            if 0 <= j < len(stamps):
                candidates.append((abs(stamps[j] - odom_ts_ns), stamps[j]))
        if not candidates:
            raise RuntimeError('LowState history unavailable for native timestamp pairing')
        delta_ns, chosen_ts = min(candidates)
        skew = delta_ns / 1e9
        if skew > MAX_PAIR_SKEW_S:
            raise RuntimeError(
                f'No LowState within two 1kHz periods of odometry: {skew:.4f}s')
        for ts, payload in self.low_raw:
            if ts == chosen_ts:
                from rclpy.serialization import deserialize_message
                return deserialize_message(payload, self._low_type), skew
        raise RuntimeError('LowState history unavailable for native timestamp pairing')

    def read(self, *, record_history=False):
        # Cheap: raw bytes only, ~0.007 ms/message.
        self._drain_raw_lowstate()
        for key, reader in self.readers.items():
            for msg in reader.take(1):
                if not msg.sample_info.valid_data:
                    continue
                writer = msg.sample_info.publication_handle
                if writer == 'no-gid':
                    if not self._warned_no_gid:
                        print('[g1_native50] rclpy exposes no publisher_gid; '
                              'writer pinning disabled', flush=True)
                        self._warned_no_gid = True
                elif key in self.writers and self.writers[key] != writer:
                    raise RuntimeError(f'{key}: active DDS writer changed')
                self.writers[key] = writer
                stamp = int(msg.tick) if key == 'low' else int(msg.stamp.sec)*10**9+int(msg.stamp.nanosec)
                if stamp < self.stamps.get(key, stamp):
                    raise RuntimeError(f'{key}: sensor clock reversed')
                if stamp == self.stamps.get(key):
                    continue
                self.stamps[key] = stamp
                # DDS source timestamps are in the robot wall-clock domain.
                age = time.time() - msg.sample_info.source_timestamp / 1e9
                self.latest[key] = (msg, time.monotonic()-age)
        if len(self.latest) != 2:
            raise RuntimeError('Waiting for both native sensor streams')
        now = time.monotonic()
        low, lt = self.latest['low']; odom, ot = self.latest['odom']
        if min(now-lt, now-ot) < -.02 or max(now-lt, now-ot) > .04:
            raise RuntimeError(f'Native sensor age/skew exceeds sampling contract: low={now-lt:.4f}s odom={now-ot:.4f}s skew={abs(lt-ot):.4f}s')
        if int(odom.error_code) != 0 or not np.isfinite(odom.body_height) or odom.body_height <= 0:
            raise RuntimeError('Native body-height source is invalid')
        joint_pos = np.asarray([m.q for m in low.motor_state[:29]], np.float32)
        raw_joint_vel = np.asarray([m.dq for m in low.motor_state[:29]], np.float32)
        state = {
            'joint_pos': joint_pos,
            'joint_vel': (
                self._joint_velocity.update(joint_pos)
                if self._joint_velocity is not None
                else raw_joint_vel
            ),
            'root_quat_wxyz': np.asarray(low.imu_state.quaternion, np.float32),
            'gyro': np.asarray(low.imu_state.gyroscope, np.float32),
        }
        qpos = np.concatenate((np.asarray(odom.position[:2], np.float32),
                               [odom.body_height], state['root_quat_wxyz'], state['joint_pos'])).astype(np.float32)
        motion(qpos[None], 1)
        if record_history and self.stamps['odom'] != self.last_history_stamp:
            odom_ts = int(odom.sample_info.source_timestamp)
            # Drain again *after* odometry was taken: the odometry sample can be
            # a couple of milliseconds newer than the ring content drained at
            # the top of this call, which would otherwise show up as extra skew.
            self._drain_raw_lowstate()
            paired, skew = self._pair_lowstate(odom_ts)
            self.last_skew = skew
            paired_qpos = np.concatenate((np.asarray(odom.position[:2], np.float32),
                                          [odom.body_height], paired.imu_state.quaternion,
                                          [m.q for m in paired.motor_state[:29]])).astype(np.float32)
            if self.last_history_stamp is not None and (self.stamps['odom']-self.last_history_stamp)/1e9 > .04:
                self.history.clear()
                if self._joint_velocity is not None:
                    self._joint_velocity.reset()
            self.history.append(motion(paired_qpos[None], 1)[0])
            self.last_history_time = now
            self.last_history_stamp = self.stamps['odom']
        return state, qpos

    def close(self):
        self.node.destroy_node()

    def startup_history(self):
        # DDS can drop one or more 50Hz odometry samples between the operator
        # pressing B and this callback.  Fill the bounded 10-frame window here
        # instead of failing the whole motion activation.
        if len(self.history) != 10:
            deadline = time.monotonic() + 1.0
            while len(self.history) < 10 and time.monotonic() < deadline:
                self.read(record_history=True)
                time.sleep(0.005)
        if len(self.history) != 10:
            raise RuntimeError(
                f'Ten fresh 50Hz sensor history frames are required '
                f'(have {len(self.history)})')
        if time.monotonic()-self.last_history_time > .04:
            raise RuntimeError('Startup history is stale')
        return motion(np.asarray(self.history), 10)


def load_tracker(path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=opts, providers=['CUDAExecutionProvider'])
    if session.get_providers()[0] != 'CUDAExecutionProvider':
        raise RuntimeError('CUDA tracker required')
    metadata = load_holomotion_metadata(session)
    mapping = np.asarray([G1_JOINT_NAMES.index(n) for n in metadata.joint_names])
    handles = SimpleNamespace(onnx_to_g1=mapping, default_joint_pos=metadata.default_joint_pos)
    tracker = HoloMotionTrackerSession(session, metadata)
    return PreviewPolicy(tracker, metadata, handles)

