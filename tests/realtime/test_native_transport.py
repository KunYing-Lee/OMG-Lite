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
    client.close()
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


def test_connection_and_close_share_one_worker(monkeypatch):
    from types import SimpleNamespace

    from omg.realtime import transport

    calls = []

    class Client:
        def __init__(self, endpoint):
            calls.append(("open", threading.get_ident()))

        def request_plan(self, request, timeout_ms):
            calls.append((timeout_ms, threading.get_ident()))
            return SimpleNamespace(
                fps=50,
                plan_id=request.metadata["plan_id"],
                request_tracker_frame=request.tracker_frame,
                metadata=request.metadata,
                qpos_36=qpos(100),
                planning_latency_seconds=0.01,
            )

        def close(self):
            calls.append(("close", threading.get_ident()))

    monkeypatch.setattr(transport, "ZmqPlanClient", Client)
    planner = RemotePlanner("unused")
    try:
        planner.plan(qpos(10), "walk", 0)
        planner.plan(qpos(10), "walk", 1)
    finally:
        planner.close()
    assert [x[0] for x in calls] == ["open", 30000, 280, "close"]
    assert len({x[1] for x in calls}) == 1
    assert calls[0][1] != threading.get_ident()
