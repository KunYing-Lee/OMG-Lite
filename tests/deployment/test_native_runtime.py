"""Exercise cancellation without ROS, a controller, or any motor transport."""

import importlib.util
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def runtime_class(monkeypatch):
    class Base:
        def handle_low_state(self, msg):
            self.handled = msg

    modules = {
        "rclpy": {},
        "std_msgs.msg": {"String": object},
        "humanoid_policy.policy_node_29dof": {"HoloMotionPolicyNode": object},
        "humanoid_policy.policy_runtime": {"PolicyRuntime": Base},
        "humanoid_policy.utils.remote_controller_filter": {
            "KeyMap": SimpleNamespace(Y=11)
        },
        "g1_native50": {"NativeSensors": object, "load_tracker": object},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[2] / "deployment/native50_policy_node.py"
    spec = importlib.util.spec_from_file_location("native_runtime_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Native50Runtime


def test_y_detaches_inflight_plan_without_blocking(runtime_class):
    runtime = runtime_class.__new__(runtime_class)
    runtime.pool = ThreadPoolExecutor(max_workers=1)
    started, release, closed = threading.Event(), threading.Event(), threading.Event()

    def inference():
        started.set()
        assert release.wait(3)

    runtime.pending = runtime.pool.submit(inference)
    runtime.active = False
    runtime.bridge = SimpleNamespace(close=closed.set)
    runtime.port = SimpleNamespace(
        remote_controller=SimpleNamespace(set=lambda _: None, button={11: True})
    )
    assert started.wait(1)
    try:
        msg = SimpleNamespace(wireless_remote=b"")
        runtime.handle_low_state(msg)
        assert runtime.pending is None
        assert runtime.bridge is None
        assert runtime.handled is msg
        assert not closed.is_set()
    finally:
        release.set()
        runtime.pool.shutdown(wait=True)
    assert closed.is_set()
