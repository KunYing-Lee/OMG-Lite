import numpy as np
import pytest
from omg.pipeline.schedule import PlanTimeline
from omg.realtime.reference_stream import ReferenceReceiver, encode_window
from omg.tracking.holomotion.reference import precompute_reference_features


def qpos(n=100):
    q = np.zeros((n, 36), np.float32)
    q[:, 3] = 1
    q[:, 0] = np.arange(n) * 0.01
    return q


def test_future_history_and_fixed_switch():
    t = PlanTimeline(qpos())
    r = t.request(25)
    np.testing.assert_array_equal(r.history, qpos()[40:50])
    assert r.deadline_frame == 39
    new = qpos()
    new[:, 0] += 2
    t.accept(r, new, current_frame=38)
    np.testing.assert_array_equal(t.qpos[:50], qpos()[:50])
    np.testing.assert_array_equal(t.qpos[50:], new)
    np.testing.assert_array_equal(t.request(75).history, new[40:50])


def test_deadline_no_late_splice():
    t = PlanTimeline(qpos())
    r = t.request(25)
    with pytest.raises(TimeoutError):
        t.check_frame(39)
    with pytest.raises(TimeoutError):
        t.accept(r, qpos(), current_frame=40)
    assert len(t.raw) == 1


def test_packet_features_equal_full_timeline_at_seam():
    t = PlanTimeline(qpos())
    r = t.request(25)
    new = qpos()
    new[:, 0] += 1
    t.accept(r, new, current_frame=25)
    order = list(range(29))
    full = precompute_reference_features(t.qpos, fps=50, onnx_to_g1=order)
    receiver = ReferenceReceiver(order)
    for frame in range(70):
        header, payload = encode_window(t.qpos, frame, "test")
        features, cursor = receiver.consume(header, payload)
        for key in [
            "ref_base_linvel",
            "ref_base_angvel",
            "ref_dof_pos_onnx",
            "ref_root_height",
        ]:
            np.testing.assert_allclose(
                features[key][cursor : cursor + 11],
                full[key][frame : frame + 11],
                atol=1e-6,
            )


def test_future_derivative_reaches_switch_one_frame_early():
    old = qpos()
    new = old.copy()
    new[50, 0] += 1
    a = precompute_reference_features(old, fps=50, onnx_to_g1=list(range(29)))
    b = precompute_reference_features(new, fps=50, onnx_to_g1=list(range(29)))
    assert not np.array_equal(a["ref_base_linvel"][49], b["ref_base_linvel"][49])


def test_sync_and_deployment_have_identical_preview_for_30_50_70ms():
    from concurrent.futures import Future

    from omg.pipeline.execution import run
    from omg.realtime.deployment import DeploymentBridge

    class Planner:
        def plan(self, history, text, plan_id):
            q = qpos()
            q[:, 0] += plan_id * 0.5
            return {"qpos": q, "latency_s": 0}

    class Tracker:
        def __init__(self):
            self.views = []

        def activate(self, q):
            self.q = q

        def step(self, f, p):
            self.views.append(self.q[f : f + 12].copy())

    p = Planner()
    t = Tracker()
    run(p, t, qpos(10), "walk", frames=150)
    # Compare effective observation features, not unused distant raw suffixes.
    for latency in [0.03, 0.05, 0.07]:
        now = [0.0]
        b = DeploymentBridge(p, "walk", clock=lambda now=now: now[0])
        b.activate(qpos(10))

        class ControlledPool:
            def __init__(self, now, latency):
                self.now, self.latency = now, latency

            def submit(self, fn, request):
                self.future = Future()
                self.request = request
                self.ready = self.now[0] + self.latency
                return self.future

            def shutdown(self, **kwargs):
                pass

        b.pool.shutdown()
        b.pool = ControlledPool(now, latency)
        for frame in range(150):
            now[0] = frame / 50
            if b.future is not None and not b.future.done() and now[0] >= b.pool.ready:
                r = b.pool.request
                b.future.set_result(
                    (p.plan(r.history, "walk", r.plan_id), b.pool.ready)
                )
            timeline, _ = b.tick()
            a = precompute_reference_features(
                timeline, fps=50, onnx_to_g1=list(range(29))
            )
            z = precompute_reference_features(
                t.views[frame], fps=50, onnx_to_g1=list(range(29))
            )
            # Future features are not affected by the missing previous sample in z.
            for key in ["ref_dof_pos_onnx", "ref_base_linvel", "ref_base_angvel"]:
                np.testing.assert_allclose(
                    a[key][frame + 1 : frame + 11], z[key][1:11], atol=1e-6
                )
        b.close()
