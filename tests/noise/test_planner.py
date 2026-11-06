import torch

from omg.noise.actor import JointDiagonalGaussian, NOISE_DIM
from omg.noise.sampling import sample_action_noise


def _distribution():
    return JointDiagonalGaussian(
        mean=torch.zeros(1, NOISE_DIM),
        log_std=torch.zeros(1, NOISE_DIM),
    )


def test_same_seed_and_plan_id_reproduce():
    distribution = _distribution()
    action_a, base_a, seed_a = sample_action_noise(
        distribution, seed=202609150, plan_id=7, device="cpu"
    )
    action_b, base_b, seed_b = sample_action_noise(
        distribution, seed=202609150, plan_id=7, device="cpu"
    )
    assert seed_a == seed_b == 202609150 + 7
    assert torch.equal(action_a, action_b)
    assert torch.equal(base_a, base_b)
    assert action_a.shape == (1, NOISE_DIM)


def test_different_plan_id_changes_noise():
    distribution = _distribution()
    action_a, _, _ = sample_action_noise(
        distribution, seed=202609150, plan_id=0, device="cpu"
    )
    action_b, _, _ = sample_action_noise(
        distribution, seed=202609150, plan_id=1, device="cpu"
    )
    assert not torch.equal(action_a, action_b)


def test_mean_and_std_are_used():
    mean = torch.full((1, NOISE_DIM), 0.25)
    distribution = JointDiagonalGaussian(
        mean=mean, log_std=torch.full((1, NOISE_DIM), float(torch.log(torch.tensor(2.0))))
    )
    action, base, _ = sample_action_noise(
        distribution, seed=1, plan_id=0, device="cpu"
    )
    assert torch.allclose(action, mean + 2.0 * base)
