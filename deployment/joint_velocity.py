"""Joint-velocity helpers for tracker observations.

The real G1 LowState `motor_state.dq` can contain a strong 24-25 Hz component
that is absent from the simulator qvel used to train the HoloMotion tracker.
Two fallbacks are provided:

* JointVelocityEstimator: causal 50 Hz q difference, first frame zero.
* JointVelocityLowPass: causal 15 Hz low-pass on the reported dq, preserving
  more of the low-frequency motion than the difference estimator.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


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


class JointVelocityLowPass:
    def __init__(self, joints: int = 29, fs: float = 50.0, cutoff_hz: float = 15.0, order: int = 4):
        self.joints = int(joints)
        self.sos = butter(order, float(cutoff_hz) / (float(fs) / 2.0), btype="low", output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2, self.joints), np.float32)

    def reset(self):
        self.zi = np.zeros((self.sos.shape[0], 2, self.joints), np.float32)

    def update(self, joint_vel):
        x = np.asarray(joint_vel, dtype=np.float32).reshape(1, self.joints)
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y[0].astype(np.float32)
