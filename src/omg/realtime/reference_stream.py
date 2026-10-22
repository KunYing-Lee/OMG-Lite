"""Native preview packets for a 50Hz HoloMotion policy loop.

A packet carries one preceding sample, current +10 preview samples, and one
extra sample for finite differences. No resampling or extrapolation occurs.
"""

import json
import time

import numpy as np

from omg.pipeline.schedule import motion
from omg.tracking.holomotion.reference import precompute_reference_features


def encode_window(qpos, frame, session):
    start = max(0, frame - 1)
    end = frame + 12
    if len(qpos) < end:
        raise ValueError("Missing preview/derivative support")
    window = motion(qpos[start:end])
    header = {
        "version": 2,
        "fps": 50,
        "session": session,
        "frame": frame,
        "cursor": frame - start,
        "rows": len(window),
        "sent_monotonic": time.monotonic(),
    }
    return json.dumps(header).encode(), window.astype("<f4").tobytes()


class ReferenceReceiver:
    def __init__(self, onnx_to_g1):
        self.onnx_to_g1 = onnx_to_g1
        self.session = None
        self.next_frame = 0

    def consume(self, header_bytes, payload, *, now=None):
        h = json.loads(header_bytes)
        now = time.monotonic() if now is None else now
        if h["version"] != 2 or h["fps"] != 50:
            raise ValueError("Wrong reference protocol")
        if not 0 <= now - h["sent_monotonic"] <= 0.02:
            raise TimeoutError("Stale reference packet")
        if self.session is None:
            self.session = h["session"]
        if h["session"] != self.session or h["frame"] != self.next_frame:
            raise ValueError("Reference sequence mismatch")
        cursor = 0 if h["frame"] == 0 else 1
        if h["cursor"] != cursor or h["rows"] != cursor + 12:
            raise ValueError("Invalid preview window")
        q = motion(np.frombuffer(payload, dtype="<f4").reshape(h["rows"], 36))
        features = precompute_reference_features(
            q, fps=50.0, onnx_to_g1=self.onnx_to_g1
        )
        self.next_frame += 1
        return features, cursor


class ReferencePublisher:
    """Local Orin IPC: bridge and HoloMotion consumer share the monotonic clock."""

    def __init__(self, bind, session):
        import zmq

        self.session = session
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PAIR)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDTIMEO, 20)
        self.socket.bind(bind)

    def wait_armed_history(self):
        frames = self.socket.recv_multipart()
        if len(frames) != 2 or frames[0] != b"armed-history-50hz-v2":
            raise ValueError("Expected armed native history")
        return motion(np.frombuffer(frames[1], dtype="<f4").reshape(10, 36), 10)

    def activate(self, qpos):
        self.qpos = motion(qpos)

    def step(self, frame, plan_id):
        self.socket.send_multipart(encode_window(self.qpos, frame, self.session))

    def close(self):
        self.socket.close()
        self.ctx.term()


class ReferenceSubscriber:
    def __init__(self, connect, onnx_to_g1):
        import zmq

        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PAIR)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 20)
        self.socket.connect(connect)
        self.receiver = ReferenceReceiver(onnx_to_g1)

    def next_reference(self):
        frames = self.socket.recv_multipart()
        if len(frames) != 2:
            raise ValueError("Invalid packet")
        result = self.receiver.consume(*frames)
        self.socket.setsockopt(__import__("zmq").RCVTIMEO, 20)
        return result

    def close(self):
        self.socket.close()
        self.ctx.term()

    def arm(self, measured_history):
        """Called only after the robot controller's existing arming gate."""
        history = motion(measured_history, 10)
        self.socket.send_multipart(
            [b"armed-history-50hz-v2", history.astype("<f4").tobytes()]
        )
        # The first plan is computed before active tracking starts.
        self.socket.setsockopt(__import__("zmq").RCVTIMEO, 30000)
