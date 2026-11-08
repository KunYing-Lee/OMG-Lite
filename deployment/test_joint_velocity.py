import numpy as np

from deployment.joint_velocity import JointVelocityEstimator


def test_first_frame_is_zero_and_then_difference():
    est = JointVelocityEstimator(joints=3, dt=0.02, max_abs=0)
    q0 = np.array([0.1, -0.2, 0.3], np.float32)
    assert np.array_equal(est.update(q0), np.zeros(3, np.float32))
    q1 = q0 + np.array([0.02, -0.04, 0.06], np.float32)
    out = est.update(q1)
    np.testing.assert_allclose(out, [1.0, -2.0, 3.0], atol=1e-6)


def test_reset_rezeros_velocity():
    est = JointVelocityEstimator(joints=2, dt=0.02, max_abs=0)
    est.update(np.zeros(2, np.float32))
    est.update(np.ones(2, np.float32))
    est.reset()
    assert np.array_equal(est.update(np.ones(2, np.float32)), np.zeros(2, np.float32))


def test_clip_is_optional_and_bounded():
    est = JointVelocityEstimator(joints=1, dt=0.02, max_abs=5.0)
    est.update(np.zeros(1, np.float32))
    assert float(est.update(np.array([1.0], np.float32))[0]) == 5.0
    est2 = JointVelocityEstimator(joints=1, dt=0.02, max_abs=0)
    est2.update(np.zeros(1, np.float32))
    assert float(est2.update(np.array([1.0], np.float32))[0]) == 50.0
