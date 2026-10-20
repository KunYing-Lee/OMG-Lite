import threading

import numpy as np
import pytest
from omg.realtime.native import PlannerService, RemotePlanner
from omg.realtime.reference_stream import ReferenceReceiver, encode_window
from omg.realtime.transport import ZmqPlanServer


def qpos(n):
    q = np.zeros((n, 36), np.float32)
    q[:, 3] = 1
    return q


def test_native_planner_roundtrip(tmp_path):
    endpoint = "ipc://" + str(tmp_path / "planner")
    ready = threading.Event()
    errors = []

    class Planner:
        def plan(self, h, t, p):
            np.testing.assert_array_equal(h, qpos(10))
            assert t == "walk"
            return {"qpos": qpos(100), "latency_s": 0.01}

    def serve():
        try:
            with ZmqPlanServer(endpoint) as server:
                ready.set()
                service = PlannerService(Planner())
                for _ in range(2):
                    server.send_plan(service.plan(server.recv_request()))
        except Exception as e:  # noqa: BLE001 - propagate failure to test thread
            errors.append(e)

    thread = threading.Thread(target=serve)
    thread.start()
    assert ready.wait(3), errors
    client = RemotePlanner(endpoint)
    for i in [0, 1]:
        np.testing.assert_array_equal(
            client.plan(qpos(10), "walk", i)["qpos"], qpos(100)
        )
    thread.join(3)
    assert not thread.is_alive()
    assert not errors


def test_reference_stale_and_reordered_rejected():
    import json

    h, p = encode_window(qpos(100), 0, "session")
    sent = json.loads(h)["sent_monotonic"]
    receiver = ReferenceReceiver(list(range(29)))
    with pytest.raises(TimeoutError):
        receiver.consume(h, p, now=sent + 0.03)
    receiver.consume(h, p, now=sent + 0.001)
    with pytest.raises(ValueError, match="sequence"):
        receiver.consume(h, p, now=sent + 0.002)
