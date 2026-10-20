"""HoloMotion policy-loop adapter with the exact simulator observation builder.

The caller supplies measured joints (G1 order), pelvis IMU, and initialized
HoloMotion session/handles. Output is a policy action in ONNX order, passed to
the deployment's existing PD/motor layer; this module does not arm the robot.
"""

import numpy as np

from omg.tracking.holomotion.reference import build_holomotion_obs


class PreviewPolicy:
    def __init__(self, tracker, metadata, handles):
        if metadata.n_fut_frames != 10 or metadata.context_length != 1:
            raise ValueError("Expected preview10/context1 HoloMotion policy")
        self.tracker = tracker
        self.metadata = metadata
        self.handles = handles
        self.last_action = np.zeros(29, np.float32)
        self.history = {}

    def step(self, features, cursor, robot_state):
        for key, shape in [
            ("joint_pos", (29,)),
            ("joint_vel", (29,)),
            ("root_quat_wxyz", (4,)),
            ("gyro", (3,)),
        ]:
            value = np.asarray(robot_state[key])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid robot state: {key}")
        obs = build_holomotion_obs(
            features,
            cursor,
            10,
            None,
            None,
            self.handles,
            self.last_action,
            context_length=1,
            obs_schema_version=self.metadata.obs_schema_version,
            robot_history=self.history,
            robot_state=robot_state,
        )
        action = np.asarray(self.tracker.run(obs), np.float32).reshape(29)
        if not np.isfinite(action).all():
            raise FloatingPointError("Nonfinite tracker action")
        action = np.clip(action, -10.0, 10.0)
        self.last_action = action.copy()
        return action
