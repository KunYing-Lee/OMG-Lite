import pytest
import torch

from omg.noise.actor import HISTORY_DIM, NOISE_DIM, NoiseActor


def test_zero_initialization_is_standard_normal():
    actor = NoiseActor()
    observation = torch.zeros(2, HISTORY_DIM)
    distribution = actor.distribution(observation)
    assert distribution.mean.shape == (2, NOISE_DIM)
    assert torch.count_nonzero(distribution.mean) == 0
    assert torch.allclose(distribution.std, torch.ones_like(distribution.std))


def test_fixed_base_noise_is_identity_at_zero_mean():
    actor = NoiseActor()
    base = torch.randn(1, NOISE_DIM)
    action, logp, mean, logstd = actor.sample(
        torch.zeros(1, HISTORY_DIM), base_noise=base
    )
    assert torch.equal(action, base)
    assert torch.equal(mean, torch.zeros_like(mean))
    assert torch.equal(logstd, torch.zeros_like(logstd))
    assert torch.isfinite(logp).all()


def test_rejects_wrong_observation_shape():
    actor = NoiseActor()
    with pytest.raises(ValueError):
        actor.distribution(torch.zeros(1, HISTORY_DIM - 1))
