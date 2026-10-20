"""Orin-local native preview bridge. Requires the v2 HoloMotion consumer hook."""

import argparse
import time
import uuid

from omg.realtime.deployment import DeploymentBridge
from omg.realtime.native import RemotePlanner
from omg.realtime.reference_stream import ReferencePublisher


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--connect", required=True, help="GPU planner endpoint")
    p.add_argument("--publish-bind", default="ipc:///tmp/omg-preview-v2")
    p.add_argument("--text", required=True)
    args = p.parse_args()
    publisher = ReferencePublisher(args.publish_bind, uuid.uuid4().hex)
    bridge = DeploymentBridge(RemotePlanner(args.connect), args.text)
    try:
        # The policy process supplies measured history only after its hardware arming gate.
        history = publisher.wait_armed_history()
        bridge.activate(history)
        while True:
            time.sleep(max(0, bridge.origin + bridge.frame / 50 - time.monotonic()))
            qpos, frame = bridge.tick()
            publisher.activate(qpos)
            publisher.step(frame, frame // 50)
    finally:
        bridge.close()
        publisher.close()


if __name__ == "__main__":
    main()
