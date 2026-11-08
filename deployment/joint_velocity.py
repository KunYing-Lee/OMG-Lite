"""Causal joint-velocity estimator for tracker observations.

The real G1 LowState `motor_state.dq` carries a strong 24-25 Hz component that
is absent from the simulator qvel used to train the HoloMotion tracker.  This
helper reconstructs a clean velocity from measured joint positions with a
one-frame causal difference, with an explicit first-frame zero and a reset
hook for sensor discontinuities.
"""

from __future__ import annotations

import numpy as np


class JointVelocityEstimator:
    def __init__(self, joints: int = 29, dt: float = 0.02, max_abs: float = 60.0):
        self.joints = int(joints)
        self.dt = float(dt)
        self.max_abs = float(max_abs)
        self._prev = None

    def reset(self):
        self._prev = None

    def update(self, joint_pos):
        q = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
        if q.shape != (self.joints,):
            raise ValueError(f"expected {self.joints} joints, got {q.shape}")
        if self._prev is None:
            self._prev = q.copy()
            return np.zeros(self.joints, dtype=np.float32)
        dq = ((q - self._prev) / self.dt).astype(np.float32)
        self._prev = q.copy()
        if self.max_abs > 0:
            np.clip(dq, -self.max_abs, self.max_abs, out=dq)
        return dq
