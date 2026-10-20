"""Call from the G1 HoloMotion controller after its existing remote-B gate.

read_state returns native measured G1 joints, pelvis quaternion and body gyro.
apply_action applies the existing ONNX-order action-to-PD/motor conversion.
stop_tracking must revoke active motor tracking on error or termination.
History must be ten 50Hz measured qpos36 samples, including the controller's
root pose estimate. LowState alone does not supply root translation: do not
invent it from a nominal pose or the generated reference.
"""

from omg.realtime.policy import PreviewPolicy
from omg.realtime.reference_stream import ReferenceSubscriber


def run_armed_tracker(
    *,
    tracker,
    metadata,
    handles,
    measured_history,
    read_state,
    apply_action,
    stop_tracking,
    connect="ipc:///tmp/omg-preview-v2",
):
    stream = ReferenceSubscriber(connect, handles.onnx_to_g1)
    policy = PreviewPolicy(tracker, metadata, handles)
    try:
        stream.arm(measured_history)
        while True:
            features, cursor = stream.next_reference()
            state = read_state()
            action = policy.step(features, cursor, state)
            apply_action(action)
    finally:
        # Includes missed reference packets, stale state reported by read_state,
        # late planner output, and keyboard interruption. Never hold the last action.
        stop_tracking()
        stream.close()
